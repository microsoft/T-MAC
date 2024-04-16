from t_mac.ops import QGeMMLUTBitsCodegen, QGeMMLUTBitsPreprocessorCodegen
from t_mac.utils import get_default_device_kwargs

import logging
import os
import argparse
from typing import Optional
import configparser

import tvm
from tvm import relay


logger = logging.getLogger("compile")


# Please set for more configs
MKNs = [
    # llama-7b
    # M, K, N, m_groups
    # [12288, 4096, 1, -1],
    # [4096, 4096, 1, -1],
    # [11008, 4096, 1, -1],
    # [4096, 11008, 1, -1],
    # BitNet
    # [12288, 4096, 1, 3],
    # [4096, 4096, 1, 1],
    # [11008, 4096, 1, 1],
    # [4096, 11008, 1, 1],
    # llama-70b
    [1024, 8192, 1, -1],
    [8192, 8192, 1, -1],
    [28672, 8192, 1, -1],
    [8192, 28672, 1, -1],
]


def compile(
    target: str,
    remote_kwargs: Optional[dict] = None,
    dtype: str = "int8",
    cc_opts: Optional[list] = None,
    eval_kwargs: Optional[dict] = None,
    out_dtype: str = "float16",
):
    bits = FLAGS.bits

    if target == "opencl" or target == "vulkan":
        target_host = "llvm -mtriple=arm64-linux-android -mattr=+neon"
    else:
        target_host = None

    codegen_kwargs = {
        "dtype": dtype,
        "target": target,
        "save_dir": FLAGS.out_path,
        "verify": True,
        "target_host": target_host,
        "tune": FLAGS.tune,
        "reuse_tuned": FLAGS.reuse_tuned,
        "remote_kwargs": remote_kwargs,
        "bits": bits,
        "cc_opts": cc_opts,
        "out_dtype": out_dtype,
        "act_group_size": FLAGS.act_group_size,
        "num_threads": 1 if FLAGS.one_thread_block else FLAGS.num_threads,
    }

    def insert(all: tvm.IRModule, new: tvm.IRModule):
        if all is None:
            return new
        else:
            all.update(new)
            return all

    mod = None
    qgemm_lut = QGeMMLUTBitsCodegen(
        name="qgemm_lut",
        group_size=FLAGS.group_size,
        fast_aggregation=FLAGS.fast_aggregation,
        **codegen_kwargs,
    )
    preprocessor = QGeMMLUTBitsPreprocessorCodegen(
        name="preprocessor",
        **codegen_kwargs,
    )
    config = configparser.ConfigParser()
    for M, K, N, m_groups in MKNs:
        M = M * bits
        qgemm_lut.m_groups = m_groups
        preprocessor.M = M
        if FLAGS.act_group_size == -1:
            qgemm_lut.act_group_size = K
            preprocessor.act_group_size = K

        qgemm_mod = qgemm_lut.compile(
            M, N, K,
            thread_affinity=FLAGS.thread_affinity,
            return_lower=True,
            **eval_kwargs,
        )
        template_name = qgemm_lut.get_template_name(M, N, K)
        if FLAGS.one_thread_block:
            # Reuse tuned configs set by the complete M, N, K
            # The section name in kcfg.ini will be the same as the complete M, N, K
            # The kernel name will be constructed from tiled m, n, k
            qgemm_mod = qgemm_lut.compile(
                qgemm_lut.bm, N, K,
                thread_affinity=FLAGS.thread_affinity,
                return_lower=True,
                preserve_cfg=True,
                **eval_kwargs,
            )
        mod = insert(mod, qgemm_mod)
        # Write kcfg
        config[template_name] = {
            "bm": str(qgemm_lut.bm),
            "simd_n_in": str(qgemm_lut.simd_n_in),
            "simd_n_out": str(qgemm_lut.simd_n_out),
            "kfactor": str(qgemm_lut.kfactor),
            "group_size": str(qgemm_lut.group_size),
            "lut_scales_size": str(N * K // qgemm_lut.act_group_size),
            "scales_size": str(qgemm_lut.m_groups if qgemm_lut.m_groups > 1 else (M // bits * K // qgemm_lut.group_size)),
            "n_tile_num": str(M // qgemm_lut.bm),
        }

        if FLAGS.fast_aggregation:
            if qgemm_lut.do_scale_final:
                fast_aggregation_k = qgemm_lut.kfactor
            else:
                fast_aggregation_k = FLAGS.act_group_size // 4
        else:
            fast_aggregation_k = 0
        preprocessor.fast_aggregation_k = fast_aggregation_k

        mod = insert(mod, preprocessor.compile(
            N, K,
            thread_affinity=FLAGS.thread_affinity,
            return_lower=True,
            **eval_kwargs,
        ))

    with tvm.transform.PassContext(config={"tir.disable_assert": FLAGS.disable_assert}):
        with tvm.target.Target(target, host=target_host):
            syslib = tvm.build(
                mod,
                runtime=relay.backend.Runtime("cpp", {"system-lib": True}),
            )
            syslib.save(os.path.join(FLAGS.out_path, "kernels.o"))
            dylib = tvm.build(mod)
            dylib.export_library(os.path.join(FLAGS.out_path, "kernels.dll"))

    with open(os.path.join(FLAGS.out_path, "kcfg.ini"), "w") as f:
        config.write(f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out_path", type=str, default="out")
    parser.add_argument("-d", "--device", type=str, choices=["m2", "android", "intel_win"], default="m2")
    parser.add_argument("-tgt", "--target", type=str, choices=["llvm", "opencl", "vulkan"], default="llvm")
    parser.add_argument("-t", "--tune", action="store_true")
    parser.add_argument("-r", "--reuse_tuned", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-b", "--bits", type=int, default=2)
    parser.add_argument("-tb", "--one_thread_block", action="store_true")
    parser.add_argument("-da", "--disable_assert", action="store_true")

    parser.add_argument("-nt", "--num_threads", type=int, default=16)  # 16 big cores for M2-ultra
    parser.add_argument("-ta", "--thread_affinity", type=int, default=1)
    parser.add_argument("-gs", "--group_size", type=int, default=128)
    parser.add_argument("-ags", "--act_group_size", type=int, default=64, help="-1 for BitNet-like unified scale")
    parser.add_argument("-fa", "--fast_aggregation", action="store_true")
    return parser.parse_args()


def main():
    device_kwargs = get_default_device_kwargs(FLAGS.device)
    compile(**device_kwargs)


if __name__ == "__main__":
    FLAGS = parse_args()

    if FLAGS.verbose:
        logging.basicConfig()
        logging.getLogger().setLevel(logging.INFO)

    main()
