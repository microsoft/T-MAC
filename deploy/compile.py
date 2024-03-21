from t_mac.ops import QGeMMLUTBitsCodegen, QGeMMLUTBitsPreprocessorCodegen
import logging
import os
import argparse

from typing import Optional

import tvm
from tvm import relay


logger = logging.getLogger("compile")


# Please set for more configs
MKNs = [
    # llama-7b
    [12288, 4096, 1],
    [4096, 4096, 1],
    [11008, 4096, 1],
    [4096, 11008, 1],
]
KNs = [
    [4096, 1],
    [11008, 1],
]
bits = 4


def compile(
    target: str,
    remote_kwargs: Optional[dict] = None,
    dtype: str = "int8",
    **eval_kwargs,
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
    }

    def insert(all: tvm.IRModule, new: tvm.IRModule):
        if all is None:
            return new
        else:
            all.update(new)
            return all

    mod = None
    qgemm_lut = QGeMMLUTBitsCodegen(name="qgemm_lut", bits=bits, **codegen_kwargs)
    preprocessor = QGeMMLUTBitsPreprocessorCodegen(name="preprocessor", **codegen_kwargs)
    for M, K, N in MKNs:
        M = M * bits
        if FLAGS.one_thread_block:
            M = M // FLAGS.num_threads

        template_name = f"qgemm_lut_{M}_{K}_{N}_{FLAGS.num_threads}_{dtype}_{bits}"
        mod = insert(mod, qgemm_lut.compile(
            M, N, K,
            template_name=template_name,
            num_threads=FLAGS.num_threads,
            thread_affinity=FLAGS.thread_affinity,
            return_lower=True,
            **eval_kwargs,
        ))
    for KN in KNs:
        K, N = KN

        template_name = f"preprocessor_{K}_{N}_{FLAGS.num_threads}_{dtype}"
        mod = insert(mod, preprocessor.compile(
            N, K,
            template_name=template_name,
            num_threads=FLAGS.num_threads,
            thread_affinity=FLAGS.thread_affinity,
            return_lower=True,
            **eval_kwargs,
        ))

    with tvm.target.Target(target, host=target_host):
        syslib = tvm.build(
            mod,
            runtime=relay.backend.Runtime("cpp", {"system-lib": True}),
        )
        syslib.save(os.path.join(FLAGS.out_path, f"kernels.o"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out_path", type=str, default="out")
    parser.add_argument("-d", "--device", type=str, choices=["m2", "android"], default="m2")
    parser.add_argument("-tgt", "--target", type=str, choices=["llvm", "opencl", "vulkan"], default="llvm")
    parser.add_argument("-t", "--tune", action="store_true")
    parser.add_argument("-r", "--reuse_tuned", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-b", "--bits", type=int, default=2)
    parser.add_argument("-tb", "--one_thread_block", action="store_true")

    parser.add_argument("-nt", "--num_threads", type=int, default=16)  # 16 big cores for M2-ultra
    parser.add_argument("-ta", "--thread_affinity", type=int, default=1)
    return parser.parse_args()


def main():
    dtype = "int8"

    if FLAGS.device == "m2":
        target = "llvm -mtriple=arm64-apple-darwin23.1.0 -mcpu=apple-m2"
        eval_kwargs = {
            "number": 1000,
            "repeat": 100,
        }
        remote_kwargs = {
            "key": "local",
            "host": os.environ["TVM_TRACKER_HOST"],
            "port": int(os.environ["TVM_TRACKER_PORT"]),
            "build_func": "default",
            "timeout": 600,
        }
    elif FLAGS.device == "android":
        if FLAGS.target == "llvm":
            target = "llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+v8.2a,+fullfp16,+fp-armv8,+neon"
        else:
            target = FLAGS.target
        eval_kwargs = {
            "number": 10,
            "repeat": 10,
        }
        remote_kwargs = {
            "key": "android",
            "host": os.environ["TVM_TRACKER_HOST"],
            "port": int(os.environ["TVM_TRACKER_PORT"]),
            "build_func": "ndk",
            "timeout": 600,
        }

    compile(
        target,
        remote_kwargs=remote_kwargs,
        dtype=dtype,
        **eval_kwargs,
    )


if __name__ == "__main__":
    FLAGS = parse_args()

    if FLAGS.verbose:
        logging.basicConfig()
        logging.getLogger().setLevel(logging.INFO)

    main()
