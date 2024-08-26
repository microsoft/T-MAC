import subprocess
import platform
from typing import Tuple, List
import logging
import copy
import os

logger = logging.getLogger("platform")


ARCH_MAP = {
    "arm64": "aarch64",
    "AMD64": "x86_64",
    "ARM64": "aarch64",
    "x86": "x86_64",
}


def get_system_info() -> Tuple[str, str]:
    """Get OS and processor architecture"""
    system = platform.system()
    processor = platform.machine()
    if system == "Windows":
        # https://github.com/python/cpython/issues/98962
        # A workaround as TVM doesn't support python 3.12
        try:
            import wmi
            arch = wmi.WMI().Win32_Processor()[0].Architecture
            WIN_ARCH_MAP = {
                0: "x86",
                9: "AMD64",
                12: "ARM64",
            }
            processor = WIN_ARCH_MAP[arch]
        except ImportError:
            import os
            arch = os.environ.get("TMAC_NATIVE_CPU_ARCH")
            if arch is None:
                logger.warn("Install wmi through `pip install wmi` to get accurate CPU architecture on Windows. "
                            "Otherwise, please specify CPU architecture manually through environment `TMAC_NATIVE_CPU_ARCH`. "
                            "Ignore this warning if you are not using Windows on ARM.")
            else:
                processor = arch
    processor = ARCH_MAP.get(processor, processor)
    return system, processor


def get_osx_sdk_root() -> str:
    if get_system_info()[0] == "Darwin":
        try:
            return subprocess.check_output(["xcrun", "--show-sdk-path"]).decode().strip()
        except subprocess.CalledProcessError:
            return ""
    return ""


def get_osx_isysroot() -> List[str]:
    sdk_root = get_osx_sdk_root()
    if sdk_root:
        return ["-isysroot", sdk_root]
    else:
        return []


def is_win() -> bool:
    """Check if is windows or not"""
    return get_system_info()[0] == "Windows"


def is_arm() -> bool:
    """Check if is windows or not"""
    return get_system_info()[1] == "aarch64"


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
        "cc": os.environ.get("TVM_NDK_CC", "clang++"),
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
    if not device:
        device = _platform_device_default_map[get_system_info()]
    return copy.deepcopy(_device_kwargs.get(device, {}))


def get_arch(device: str = ""):
    if not device:
        return get_system_info()[1]
    elif device == "android":
        return "aarch64"
    else:
        _, arch = next(key for key, value in _platform_device_default_map.items() if value == device)
        return arch
