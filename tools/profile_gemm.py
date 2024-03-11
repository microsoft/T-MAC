from t_mac.ops import GeMMCodegen, GeMMCLCodegen, QGeMMLUTBitsCodegen
import logging
import os
from typing import Tuple
import pandas as pd
import argparse

from typing import Optional
import time
import gc

logger = logging.getLogger("profile_gemm")


def profile_gemv_codegen(
    MKN: Tuple[int, int, int],
    bits: int,
    num_threads: int,
    target: str,
    remote_kwargs: Optional[dict] = None,
    dtype: str = "int8",
    **eval_kwargs,
):
    M, K, N = MKN
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
        "bits": bits,
    }

    codegen_keys = [
        # "gemm",
        "qgemm_lut",
    ]

    if target == "opencl":
        codegen_keys = [k + "_cl" for k in codegen_keys]
    elif target == "vulkan":
        codegen_keys = [k + "_vulkan" for k in codegen_keys]

    codegen_cls = {
        "gemm": GeMMCodegen,
        "gemm_cl": GeMMCLCodegen,
        "qgemm_lut": QGeMMLUTBitsCodegen,
    }

    template_names = {
        k: f"{k}_{M}_{K}_{N}_{num_threads}_{dtype}_{bits}"
        for k in codegen_keys
    }

    def _eval(codegen_key):
        codegen = codegen_cls[codegen_key](name=codegen_key, **codegen_kwargs)
        return 1000 * codegen.evaluate(
            M, N, K,
            template_name=template_names[codegen_key],
            num_threads=num_threads,
            thread_affinity=FLAGS.thread_affinity,
            **eval_kwargs,
        )

    return {
        k: _eval(k)
        for k in codegen_keys
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out_path", type=str, default="out")
    parser.add_argument("-d", "--device", type=str, choices=["m2", "android", "intel_win"], default="m2")
    parser.add_argument("-tgt", "--target", type=str, choices=["llvm", "opencl", "vulkan"], default="llvm")
    parser.add_argument("-ta", "--thread_affinity", type=int, default=1)
    parser.add_argument("-t", "--tune", action="store_true")
    parser.add_argument("-r", "--reuse_tuned", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def main():
    MKNs = [
        # (8192, 16384, 1),
        # (8192, 16384, 32),
        [12288, 4096, 1],
        [4096, 4096, 1],
        [11008, 4096, 1],
        [4096, 11008, 1],
        # [12288, 4096, 16],
        # [4096, 4096, 16],
        # [11008, 4096, 16],
        # [4096, 11008, 16],
        # [4096, 4096, 1],
    ]

    threads = [
        # 1,
        # 4,
        # 8,
        16,
    ]

    dtypes = [
        "int8",
        # "float16",
    ]
    header = True
    bits = 4

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
    elif FLAGS.device == "intel_win":
        target = "llvm -mtriple=x86_64-pc-windows-msvc"
        eval_kwargs = {
            "number": 100,
            "repeat": 10,
        }
        remote_kwargs = None

    for MKN in MKNs:
        for num_threads in threads:
            for dtype in dtypes:
                results = {
                    "M": MKN[0],
                    "K": MKN[1],
                    "N": MKN[2],
                    "num_threads": num_threads,
                    "dtype": dtype,
                }
                _MKN = [MKN[0] * bits, MKN[1], MKN[2]]
                results.update(
                    profile_gemv_codegen(
                        _MKN, bits, num_threads, target,
                        remote_kwargs=remote_kwargs,
                        dtype=dtype,
                        **eval_kwargs,
                    )
                )
                logger.info(results)

                pd.DataFrame([results]).to_csv(
                    os.path.join(FLAGS.out_path, "results.csv"),
                    mode="a",
                    header=header,
                    index=False,
                )
                header = False

                gc.collect()
                time.sleep(60)


if __name__ == "__main__":
    FLAGS = parse_args()

    if FLAGS.verbose:
        logging.basicConfig()
        logging.getLogger().setLevel(logging.INFO)

    main()
