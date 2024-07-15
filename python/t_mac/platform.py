import subprocess
import platform
from typing import Tuple, List


ARCH_MAP = {
    "arm64": "aarch64",
    "AMD64": "x86_64",
}


def get_system_info() -> Tuple[str, str]:
    """Get OS and processor architecture"""
    system = platform.system()
    processor = platform.machine()
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
