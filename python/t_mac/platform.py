import subprocess
import platform
from typing import Tuple, List
import logging

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
