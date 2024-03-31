from t_mac.ops import GeMMCodegen, GeMMCLCodegen, QGeMMLUTBitsCodegen, QGeMMLUTBitsPreprocessorCodegen
import t_mac.utils
import logging
import os
from typing import Tuple
import pandas as pd
import argparse

from typing import Optional
import time
import gc

logger = logging.getLogger("profile")


def profile_codegen(
    MKN: Tuple[int, int, int],
    bits: int,
    num_threads: int,
    target: str,
    remote_kwargs: Optional[dict] = None,
    dtype: str = "int8",
    cc_opts: Optional[list] = None,
    eval_kwargs: Optional[dict] = None,
    out_dtype: str = "float16",
):
    M, K, N = MKN
    if target == "opencl" or target == "vulkan":
        target_host = "llvm -mtriple=arm64-linux-android -mattr=+neon"
    else:
        target_host = None

    codegen_keys = [
        FLAGS.kernel,
    ]

    if "gemm" in FLAGS.kernel:
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
        }
        template_names = {
            k: f"{k}_{M}_{K}_{N}_{num_threads}_{dtype}_{bits}"
            for k in codegen_keys
        }
        args = (M, N, K)
    elif "preprocessor" in FLAGS.kernel:
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
            "fast_aggregation_k": 0,
        }
        template_names = {
            k: f"{k}_{M}_{K}_{N}_{num_threads}_{dtype}"
            for k in codegen_keys
        }
        args = (N, K)

    if target == "opencl":
        codegen_keys = [k + "_cl" for k in codegen_keys]
    elif target == "vulkan":
        codegen_keys = [k + "_vulkan" for k in codegen_keys]

    codegen_cls = {
        "gemm": GeMMCodegen,
        "gemm_cl": GeMMCLCodegen,
        "qgemm_lut": QGeMMLUTBitsCodegen,
        "preprocessor": QGeMMLUTBitsPreprocessorCodegen,
    }

    def _eval(codegen_key):
        codegen = codegen_cls[codegen_key](name=codegen_key, **codegen_kwargs)
        return 1000 * codegen.evaluate(
            *args,
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
    parser.add_argument("-k", "--kernel", type=str, choices=["qgemm_lut", "preprocessor"], default="qgemm_lut")
    parser.add_argument("-t", "--tune", action="store_true")
    parser.add_argument("-r", "--reuse_tuned", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def main():
    MKNs = [
        # (8192, 16384, 1),
        # (8192, 16384, 32),
        # [12288, 4096, 1],
        [4096, 4096, 1],
        # [11008, 4096, 1],
        # [4096, 11008, 1],
        # [12288, 4096, 16],
        # [4096, 4096, 16],
        # [11008, 4096, 16],
        # [4096, 11008, 16],
        # [4096, 4096, 1],
        # [128, 128, 1],
    ]

    threads = [
        1,
        # 2,
        # 4,
        # 8,
        # 16,
    ]

    dtypes = [
        "int8",
        # "float16",
    ]
    header = True
    bits = 2

    device_kwargs = t_mac.utils.get_default_device_kwargs(FLAGS.device)

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
                    profile_codegen(
                        _MKN, bits, num_threads,
                        dtype=dtype,
                        **device_kwargs,
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


if __name__ == "__main__":
    FLAGS = parse_args()

    if FLAGS.verbose:
        logging.basicConfig()
        logging.getLogger().setLevel(logging.INFO)

    main()
