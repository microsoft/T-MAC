from t_mac.ops import QGeMMLUTBitsCodegen, QGeMMLUTBitsPreprocessorCodegen
from t_mac.utils import get_default_device_kwargs, get_devices
from t_mac.model_utils import get_preset_models, extract_kernel_shapes, get_quantization_config

import logging
import os
import argparse
from typing import Optional, Union, Tuple
import configparser
import re

import tvm
from tvm import relay


logger = logging.getLogger("compile")


def compile(
    target: str,
    remote_kwargs: Optional[dict] = None,
    dtype: str = "int8",
    cc: Optional[str] = None,
    cc_opts: Optional[list] = None,
    eval_kwargs: Optional[dict] = None,
    out_dtype: str = "float16",
    aggregation_dtype: str = "int32",
):
    if target == "opencl" or target == "vulkan":
        target_host = "llvm -mtriple=arm64-linux-android -mattr=+neon"
    else:
        target_host = None

    codegen_kwargs = {
        "dtype": dtype,
        "target": target,
        "save_dir": FLAGS.out_path,
        "verify": False,
        "target_host": target_host,
        "tune": FLAGS.tune,
        "reuse_tuned": FLAGS.reuse_tuned,
        "remote_kwargs": remote_kwargs,
        "cc": cc,
        "cc_opts": cc_opts,
        "out_dtype": out_dtype,
        "act_group_size": FLAGS.act_group_size,
        "num_threads": 1 if FLAGS.one_thread_block else FLAGS.num_threads,
    }

    wrapper_func_defs = {"qgemm_lut": "", "preprocessor": ""}
    wrapper_func_calls = {"qgemm_lut": "", "preprocessor": ""}
    def make_call(kernel_def: str) -> bool:
        kernel_def_ptn = r"int32_t (\w+)_t1_int8_m(\d+)_k(\d+)_n(\d+)_b(\d+)"
        match = re.search(kernel_def_ptn, kernel_def)
        if not match:
            raise RuntimeError("Wrong kernel definition: {}".format(kernel_def))
        name, m, k, n, b = match[1], match[2], match[3], match[4], match[5]
        ptr_arg_ptn = r"void\* (\w+)"
        args = [(m[0], m[1]) for m in re.finditer(ptr_arg_ptn, kernel_def)]
        wrapper_func_defs[name] = "inline int {}_int8(int m, int k, int n, int b, {})".format(
            name,
            ", ".join(arg[0] for arg in args)
        )
        wrapper_func_call = ", ".join(arg[1] for arg in args)
        wrapper_func_call = """
    if (m == {m} && k == {k} && n == {n} && b == {b}) return {name}_t1_int8_m{m}_k{k}_n{n}_b{b}({wrapper_func_call});
""".format(m=m, k=k, n=n, b=b, name=name, wrapper_func_call=wrapper_func_call)
        if wrapper_func_call in wrapper_func_calls[name]:
            return True  # already called
        wrapper_func_calls[name] += wrapper_func_call
        return False


    def insert(all: Union[tvm.IRModule, Tuple[str]], new: Union[tvm.IRModule, Tuple[str]]):
        if isinstance(new, tuple):
            if make_call(new[0]):
                return all
        if all is None:
            return new
        elif isinstance(all, tvm.IRModule):
            all.update(new)
            return all
        elif isinstance(all, tuple):
            return (all[0] + "\n" + new[0], all[1] + "\n" + new[1])


    return_type = "lower" if not FLAGS.gen_c_code else "c"
    mod = None
    body_code = ""
    config = configparser.ConfigParser()
    kernel_shapes = extract_kernel_shapes(FLAGS.preset_model, FLAGS.model_dir)

    group_size = FLAGS.group_size
    zero_point = FLAGS.zero_point
    quantization_config = None
    if FLAGS.model_dir is not None:
        quantization_config = get_quantization_config(FLAGS.model_dir)
        if quantization_config["quant_method"] not in ["gptq", "gptq_v2"]:
            quantization_config = None

        if quantization_config is not None:
            group_size = quantization_config["group_size"]
            zero_point = not quantization_config["sym"]

    for bits, M, K, N, m_groups in kernel_shapes:
        if quantization_config is not None and bits != quantization_config["bits"]:
            logger.warning("Invalid kernels")
        qgemm_lut = QGeMMLUTBitsCodegen(
            name="qgemm_lut",
            group_size=group_size,
            fast_aggregation=FLAGS.fast_aggregation,
            bits=bits,
            aggregation_dtype=aggregation_dtype,
            zero_point=zero_point,
            **codegen_kwargs,
        )
        preprocessor = QGeMMLUTBitsPreprocessorCodegen(
            name="preprocessor",
            bits=bits,
            **codegen_kwargs,
        )
        M = M * bits
        qgemm_lut.m_groups = m_groups
        preprocessor.M = M
        if FLAGS.act_group_size == -1:
            qgemm_lut.act_group_size = K
            preprocessor.act_group_size = K

        qgemm_lut.num_threads = FLAGS.num_threads
        qgemm_mod = qgemm_lut.compile(
            M, N, K,
            thread_affinity=FLAGS.thread_affinity,
            return_type=return_type,
            **eval_kwargs,
        )
        qgemm_lut.num_threads = 1 if FLAGS.one_thread_block else FLAGS.num_threads
        template_name = qgemm_lut.get_template_name(M, N, K)
        if FLAGS.one_thread_block:
            # Reuse tuned configs set by the complete M, N, K
            # The section name in kcfg.ini will be the same as the complete M, N, K
            # The kernel name will be constructed from tiled m, n, k
            qgemm_mod = qgemm_lut.compile(
                qgemm_lut.bm, N, K,
                thread_affinity=FLAGS.thread_affinity,
                return_type=return_type,
                preserve_cfg=True,
                **eval_kwargs,
            )
        if qgemm_lut.extra_cc_body not in body_code:
            body_code += qgemm_lut.extra_cc_body
        mod = insert(mod, qgemm_mod)
        # Write kcfg
        scales_size = qgemm_lut.m_groups if qgemm_lut.m_groups != -1 else (M // bits * K // qgemm_lut.group_size)
        if zero_point:
            scales_size *= 2
        config[template_name] = {
            "bm": str(qgemm_lut.bm),
            "simd_n_in": str(qgemm_lut.simd_n_in),
            "simd_n_out": str(qgemm_lut.simd_n_out),
            "kfactor": str(qgemm_lut.kfactor),
            "group_size": str(qgemm_lut.group_size),
            "lut_scales_size": str(N * K // qgemm_lut.act_group_size),
            "scales_size": str(scales_size),
            "n_tile_num": str(M // qgemm_lut.bm),
        }

        if FLAGS.fast_aggregation:
            if qgemm_lut.do_scale_final(K):
                fast_aggregation_k = qgemm_lut.kfactor
            else:
                fast_aggregation_k = FLAGS.act_group_size // 4
        else:
            fast_aggregation_k = 0
        preprocessor.fast_aggregation_k = fast_aggregation_k

        mod = insert(mod, preprocessor.compile(
            N, K,
            thread_affinity=FLAGS.thread_affinity,
            return_type=return_type,
            **eval_kwargs,
        ))
        if preprocessor.extra_cc_body not in body_code:
            body_code += preprocessor.extra_cc_body

    if return_type == "lower":
        with tvm.transform.PassContext(config={"tir.disable_assert": FLAGS.disable_assert}):
            with tvm.target.Target(target, host=target_host):
                syslib = tvm.build(
                    mod,
                    runtime=relay.backend.Runtime("cpp", {"system-lib": True}),
                )
                syslib.save(os.path.join(FLAGS.out_path, "kernels.o"))
                dylib = tvm.build(mod)
                dylib.export_library(os.path.join(FLAGS.out_path, "kernels.dll"))
    else:
        with open(os.path.join(FLAGS.out_path, "kernels.h"), "w") as f:
            wrapper_func = (wrapper_func_defs["qgemm_lut"] + " {\n" + wrapper_func_calls["qgemm_lut"] + "\n    return -1;\n}\n" +
                            wrapper_func_defs["preprocessor"] + " {\n" + wrapper_func_calls["preprocessor"] + "\n    return -1;\n}\n")
            f.write('#include "stdint.h"\n' + mod[0] + wrapper_func)
        with open(os.path.join(FLAGS.out_path, "kernels.cc"), "w") as f:
            f.write('#include "t-mac/kernels.h"\n' + qgemm_lut.extra_cc_header + preprocessor.extra_cc_header + body_code + mod[1])

    with open(os.path.join(FLAGS.out_path, "kcfg.ini"), "w") as f:
        config.write(f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out_path", type=str, default="out")
    parser.add_argument("-d", "--device", type=str, choices=get_devices(), default="")
    parser.add_argument("-tgt", "--target", type=str, choices=["llvm", "opencl", "vulkan"], default="llvm")
    parser.add_argument("-t", "--tune", action="store_true")
    parser.add_argument("-r", "--reuse_tuned", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-tb", "--one_thread_block", action="store_true")
    parser.add_argument("-gc", "--gen_c_code", action="store_true")
    parser.add_argument("-da", "--disable_assert", action="store_true")

    parser.add_argument("-nt", "--num_threads", type=int, default=16)  # 16 big cores for M2-ultra
    parser.add_argument("-ta", "--thread_affinity", type=int, default=1)
    parser.add_argument("-gs", "--group_size", type=int, default=128, help="128 for w2g128/w4g128 or 64 for w2g64/w4g64")
    parser.add_argument("-ags", "--act_group_size", type=int, default=64, help="-1 for BitNet-like unified scale")
    parser.add_argument("-fa", "--fast_aggregation", action="store_true")
    parser.add_argument("-zp", "--zero_point", action="store_true")

    parser.add_argument("-m", "--preset_model", type=str, choices=get_preset_models(), default="hf-bitnet-3b",
                        help="Use gptq-auto only if your models are not in presets. gptq-auto will automatically detect kernel shapes for you. Only support quantized models in GPTQ format.")
    parser.add_argument("-md", "--model_dir", type=str, default=None, help="Set this if using gptq-auto")
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
