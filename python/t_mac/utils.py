import os
import copy
import numpy as np

from .platform import get_osx_isysroot, get_system_info


_device_kwargs = {
    "m2": {
        "target": "llvm -mtriple=arm64-apple-darwin23.1.0 -mcpu=apple-m2",
        "eval_kwargs": {
            "min_repeat_ms": 50,
            "repeat": 100,
        },
        # "remote_kwargs": {
        #     "key": "local",
        #     "host": os.environ.get("TVM_TRACKER_HOST", "0.0.0.0"),
        #     "port": int(os.environ.get("TVM_TRACKER_PORT", 9190)),
        #     "build_func": "default",
        #     "timeout": 600,
        # },
        "remote_kwargs": None,
        "cc_opts": ["-O3", "-std=c++17", "-mcpu=apple-m2", "-mllvm", "-inline-threshold=10000"] + get_osx_isysroot(),
        "out_dtype": "float16",
        "aggregation_dtype": "int32",
    },
    "android": {
        "target": "llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+v8.2a,+fullfp16,+fp-armv8,+neon",
        "eval_kwargs": {
            "number": 10,
            "repeat": 10,
        },
        "remote_kwargs": {
            "key": "android",
            "host": os.environ.get("TVM_TRACKER_HOST", "0.0.0.0"),
            "port": int(os.environ.get("TVM_TRACKER_PORT", 9190)),
            "build_func": "ndk",
            "timeout": 600,
        },
        "cc_opts": ["-O3", "-march=armv8.2a+fp16", "-mllvm", "-inline-threshold=10000"],
        "out_dtype": "float16",
        "aggregation_dtype": "int32",
    },
    "intel_win": {
        "target": "llvm -mtriple=x86_64-pc-windows-msvc -mcpu=core-avx2",
        "eval_kwargs": {
            "number": 10,
            "repeat": 10,
        },
        "remote_kwargs": None,
        # TODO: check if inline-threshold is needed for other devices
        "cc_opts": ["-O3", "-march=native", "-mllvm", "-inline-threshold=10000"],
        "out_dtype": "float32",
        "aggregation_dtype": "int32",
    },
    "intel_linux": {
        "target": "llvm -mtriple=x86_64-unknown-linux-gnu -mcpu=core-avx2",
        "eval_kwargs": {
            "number": 10,
            "repeat": 10,
        },
        "remote_kwargs": None,
        # TODO: check if inline-threshold is needed for other devices
        "cc_opts": ["-O3", "-march=native", "-mllvm", "-inline-threshold=10000"],
        "out_dtype": "float32",
        "aggregation_dtype": "int32",
    },
    "jetson": {
        "target": "llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+v8.2a,+fullfp16,+fp-armv8,+neon",
        "eval_kwargs": {
            "number": 10,
            "repeat": 10,
        },
        "remote_kwargs": None,
        "cc_opts": ["-O3", "-std=c++17", "-march=armv8.2a+fp16", "-mllvm", "-inline-threshold=10000"],
        "out_dtype": "float16",
        "aggregation_dtype": "int32",
    },
    "arm_win": {
        "target": "llvm -device=arm_cpu -mtriple=aarch64-pc-windows-msvc -mattr=+v8.2a,+fullfp16,+fp-armv8,+neon",
        "eval_kwargs": {
            "number": 10,
            "repeat": 10,
        },
        "remote_kwargs": None,
        "cc_opts": ["-O3", "-std=c++17", "-march=armv8.2a+fp16", "-mllvm", "-inline-threshold=10000"],
        "out_dtype": "float16",
        "aggregation_dtype": "int32",
    },
}


def get_devices():
    return list(_device_kwargs.keys())


_platform_device_default_map = {
    ("Darwin", "aarch64"): "m2",
    ("Linux", "aarch64"): "jetson",
    ("Windows", "x86_64"): "intel_win",
    ("Linux", "x86_64"): "intel_linux",
    ("Windows", "aarch64"): "arm_win",
}


def get_default_device_kwargs(device: str = ""):
    if device == "":
        device = _platform_device_default_map[get_system_info()]
    return copy.deepcopy(_device_kwargs.get(device, {}))


def get_bits_alphas(bits: int):
    alphas = [1 / 2, 1, 2, 4]
    return alphas[:bits]


def nmse(a: np.ndarray, b: np.ndarray):
    a, b = a.astype(np.float32), b.astype(np.float32)
    return np.mean(np.square(a - b)) / np.mean(np.square(a))
