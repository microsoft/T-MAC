# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import subprocess
import shutil
from setuptools import setup
from setuptools.command.install import install
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist
import distutils.dir_util
from typing import List, Tuple
import tarfile
from io import BytesIO
import os
import urllib.request
import platform

ROOT_DIR = os.path.dirname(__file__)
SUPPORTED_SYSTEM = [
    ("darwin", "arm"),
    # TODO: test and add linux/win, intel cpu
]
LLVM_VERSION = "17.0.6"


def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)


def get_system_info() -> Tuple[str, str]:
    """Get OS and processor architecture"""
    system = platform.system().lower()
    processor = platform.processor()
    return system, processor


assert get_system_info() in SUPPORTED_SYSTEM, "T-MAC hasn't supported your operating system or CPU"


def download_and_extract_llvm(llvm_version, extract_path="3rdparty"):
    """
    Downloads and extracts the specified version of LLVM for the given platform.
    Args:
        version (str): The version of LLVM to download.
        is_aarch64 (bool): True if the target platform is aarch64, False otherwise.
        extract_path (str): The directory path where the archive will be extracted.

    Returns:
        str: The path where the LLVM archive was extracted.
    """
    ubuntu_versions = {
        "17.0.6": "22.04",
    }
    darwin_versions = {
        "17.0.6": "22.0",
    }
    ubuntu_version = ubuntu_versions.get(llvm_version, "22.04")
    darwin_version = darwin_versions.get(llvm_version, "22.0")

    file_prefix = {
        ("darwin", "arm"): f"arm64-apple-darwin{darwin_version}.tar.xz",
    }

    base_url = (f"https://github.com/llvm/llvm-project/releases/download/llvmorg-{llvm_version}")
    file_name = f"clang+llvm-{llvm_version}-{file_prefix[get_system_info()]}"

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
        config_file.write(f"set(USE_LLVM {llvm_config_path})\n")
    # Run CMake and make
    try:
        subprocess.check_call(["cmake", ".."])
        subprocess.check_call(["make", "-j"])
    except subprocess.CalledProcessError as error:
        raise RuntimeError("Failed to build TVM") from error
    finally:
        # Go back to the original directory
        os.chdir("../../..")


def setup_llvm_for_tvm():
    """Downloads and extracts LLVM, then configures TVM to use it."""
    # Assume the download_and_extract_llvm function and its dependencies are defined elsewhere in this script
    extract_path = download_and_extract_llvm(LLVM_VERSION)
    llvm_config_path = os.path.join(extract_path, "bin", "llvm-config")
    return extract_path, llvm_config_path


class TMACInstallCommand(install):
    """Customized setuptools install command - builds TVM after setting up LLVM."""

    def run(self):
        # Recursively update submodules
        # update_submodules()
        # Set up LLVM for TVM
        _, llvm_path = setup_llvm_for_tvm()
        # Build TVM
        build_tvm(llvm_path)
        # Continue with the standard installation process
        install.run(self)


class TMACBuilPydCommand(build_py):
    """Customized setuptools install command - builds TVM after setting up LLVM."""

    def run(self):
        build_py.run(self)
        # custom build tvm
        update_submodules()
        # Set up LLVM for TVM
        _, llvm_path = setup_llvm_for_tvm()
        # Build TVM
        build_tvm(llvm_path)
        # Copy the built TVM to the package directory
        TVM_PREBUILD_ITEMS = [
            "3rdparty/tvm/build/libtvm_runtime.so",
            "3rdparty/tvm/build/libtvm.so",
            "3rdparty/tvm/build/libtvm_runtime.dylib",
            "3rdparty/tvm/build/libtvm.dylib",
            "3rdparty/tvm/python",
        ]
        for item in TVM_PREBUILD_ITEMS:
            source_dir = os.path.join(ROOT_DIR, item)
            target_dir = os.path.join(self.build_lib, "t_mac", item)
            if not os.path.exists(source_dir):
                continue
            if os.path.isdir(source_dir):
                self.mkpath(target_dir)
                distutils.dir_util.copy_tree(source_dir, target_dir)
            else:
                target_dir = os.path.dirname(target_dir)
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                shutil.copy2(source_dir, target_dir)


class TMACSdistCommand(sdist):
    """Customized setuptools sdist command - includes the pyproject.toml file."""

    def make_distribution(self):
        super().make_distribution()


setup(
    cmdclass={
        "install": TMACInstallCommand,
        "build_py": TMACBuilPydCommand,
        "sdist": TMACSdistCommand,
    },
)
