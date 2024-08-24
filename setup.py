# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import subprocess
import shutil
from setuptools import setup
from setuptools.command.build_py import build_py
from typing import Tuple
import tarfile
from io import BytesIO
import os
import urllib.request
import platform

ROOT_DIR = os.path.dirname(__file__)
PLATFORM_LLVM_MAP = {
    # (system, processor): (llvm_version, file_suffix)
    ("Darwin", "aarch64"): ("17.0.6", "arm64-apple-darwin22.0.tar.xz"),
    ("Linux", "aarch64"): ("17.0.6", "aarch64-linux-gnu.tar.xz"),
    # ("Windows", "x86_64"): ("18.1.6", "x86_64-pc-windows-msvc.tar.xz"),
    ("Linux", "x86_64"): ("17.0.6", "x86_64-linux-gnu-ubuntu-22.04.tar.xz"),
}
MANUAL_BUILD = bool(int(os.getenv("MANUAL_BUILD", "0")))


def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)


def is_win() -> bool:
    """Check if is windows or not"""
    return get_system_info()[0] == "Windows"


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
    processor = ARCH_MAP.get(processor, processor)
    return system, processor


def download_and_extract_llvm(extract_path="build"):
    """
    Downloads and extracts the specified version of LLVM for the given platform.
    Args:
        extract_path (str): The directory path where the archive will be extracted.

    Returns:
        str: The path where the LLVM archive was extracted.
    """

    llvm_version, file_suffix = PLATFORM_LLVM_MAP[get_system_info()]
    base_url = (f"https://github.com/llvm/llvm-project/releases/download/llvmorg-{llvm_version}")
    file_name = f"clang+llvm-{llvm_version}-{file_suffix}"

    download_url = f"{base_url}/{file_name}"

    # Download the file
    print(f"Downloading {file_name} from {download_url}")
    with urllib.request.urlopen(download_url) as response:
        if response.status != 200:
            raise Exception(f"Download failed with status code {response.status}")
        file_content = response.read()
    # Ensure the extract path exists
    os.makedirs(extract_path, exist_ok=True)

    # if the file already exists, remove it
    if os.path.exists(os.path.join(extract_path, file_name)):
        os.remove(os.path.join(extract_path, file_name))

    # Extract the file
    print(f"Extracting {file_name} to {extract_path}")
    with tarfile.open(fileobj=BytesIO(file_content), mode="r:xz") as tar:
        tar.extractall(path=extract_path)

    print("Download and extraction completed successfully.")
    return os.path.abspath(os.path.join(extract_path, file_name.replace(".tar.xz", "")))


def update_submodules():
    """Updates git submodules."""
    try:
        subprocess.check_call(["git", "submodule", "update", "--init", "--recursive"])
    except subprocess.CalledProcessError as error:
        raise RuntimeError("Failed to update submodules") from error


def build_tvm(llvm_config_path):
    """Configures and builds TVM."""
    os.chdir("3rdparty/tvm")
    if not os.path.exists("build"):
        os.makedirs("build")
    os.chdir("build")
    # Copy the config.cmake as a baseline
    if not os.path.exists("config.cmake"):
        shutil.copy("../cmake/config.cmake", "config.cmake")
    # Set LLVM path and enable CUDA in config.cmake
    with open("config.cmake", "a") as config_file:
        if is_win():
            import posixpath
            llvm_config_path = llvm_config_path.replace(os.sep, posixpath.sep)
        config_file.write(f"set(USE_LLVM {llvm_config_path})\n")
    # Run CMake and make
    try:
        subprocess.check_call(["cmake", ".."])
        if is_win():
            subprocess.check_call(["cmake", "--build", ".", "--config", "Release"])
        else:
            subprocess.check_call(["make", "-j4"])
    except subprocess.CalledProcessError as error:
        raise RuntimeError("Failed to build TVM") from error
    finally:
        # Go back to the original directory
        os.chdir("../../..")


def setup_llvm_for_tvm():
    """Downloads and extracts LLVM, then configures TVM to use it."""
    # Assume the download_and_extract_llvm function and its dependencies are defined elsewhere in this script
    extract_path = download_and_extract_llvm()
    llvm_config_path = os.path.join(extract_path, "bin", "llvm-config")
    return extract_path, llvm_config_path


class TMACBuilPydCommand(build_py):
    """Customized setuptools install command - builds TVM after setting up LLVM."""

    def run(self):
        build_py.run(self)
        if not MANUAL_BUILD:
            if get_system_info() not in PLATFORM_LLVM_MAP:
                raise RuntimeError(
                    "T-MAC hasn't supported auto-build for your operating system or CPU."
                    " Please refer to https://github.com/kaleid-liner/T-MAC/blob/main/docs/codegen.md to prepare TVM and Clang+LLVM,"
                    " and set environment variable MANUAL_BUILD=1 before pip install."
                )
            # custom build tvm
            update_submodules()
            # Set up LLVM for TVM
            _, llvm_path = setup_llvm_for_tvm()
            # Build TVM
            build_tvm(llvm_path)

            llvm_bin_path = os.path.abspath(os.path.dirname(llvm_path))
            tvm_python_path = os.path.abspath(os.path.join("3rdparty/tvm", "python"))

            envs = "export PATH={}:$PATH\nexport PYTHONPATH={}:$PYTHONPATH\n".format(llvm_bin_path, tvm_python_path)
            env_file_path = os.path.abspath(os.path.join("build", "t-mac-envs.sh"))
            with open(env_file_path, "w") as env_file:
                env_file.write(envs)
            print("Installation success. Please set environment variables through `source {}`".format(env_file_path))


setup(
    cmdclass={
        "build_py": TMACBuilPydCommand,
    },
)
