import os


def get_default_device_kwargs(device: str):
    if device == "m2":
        return {
            "target": "llvm -mtriple=arm64-apple-darwin23.1.0 -mcpu=apple-m2",
            "eval_kwargs": {
                "number": 100,
                "min_repeat_ms": 50,
                "repeat": 100,
            },
            "remote_kwargs": {
                "key": "local",
                "host": os.environ["TVM_TRACKER_HOST"],
                "port": int(os.environ["TVM_TRACKER_PORT"]),
                "build_func": "default",
                "timeout": 600,
            },
            # "remote_kwargs": None,
            "cc_opts": ["-O3", "-std=c++17", "-mcpu=apple-m2"],
            "out_dtype": "float16",
        }
    elif device == "android":
        return {
            "target": "llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+v8.2a,+fullfp16,+fp-armv8,+neon",
            "eval_kwargs": {
                "number": 10,
                "repeat": 10,
            },
            "remote_kwargs": {
                "key": "android",
                "host": os.environ["TVM_TRACKER_HOST"],
                "port": int(os.environ["TVM_TRACKER_PORT"]),
                "build_func": "ndk",
                "timeout": 600,
            },
            "cc_opts": ["-O3", "-march=armv8.2a+fp16"],
            "out_dtype": "float16",
        }
    elif device == "intel_win":
        return {
            "target": "llvm -mtriple=x86_64-pc-windows-msvc -mcpu=core-avx2",
            "eval_kwargs": {
                "number": 100,
                "repeat": 10,
            },
            "remote_kwargs": None,
            # TODO: check if inline-threshold is needed for other devices
            "cc_opts": ["-O3", "-march=native", "-mllvm", "-inline-threshold=10000"],
            "out_dtype": "float32",
        }
    else:
        raise ValueError(f"Unknown device: {device}")


def get_bits_alphas(bits: int):
    alphas = [1 / 2, 1, 2, 4]
    return alphas[:bits]