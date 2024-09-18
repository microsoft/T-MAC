import argparse
import subprocess
import os
from datetime import datetime
import shutil
import logging

from t_mac.platform import is_win, is_arm, get_arch, get_devices, get_default_device_kwargs
from t_mac.model_utils import get_preset_models


logger = logging.getLogger("run_pipeline")


def run_command(command, pwd):
    print(f"  Running command in {pwd}:")
    print(f"    {' '.join(command)}")
    os.makedirs(FLAGS.logs_dir, exist_ok=True)
    log_file = os.path.join(FLAGS.logs_dir, datetime.now().strftime("%Y-%m-%d-%H-%M-%S.log"))
    with open(log_file, "w") as fp:
        try:
            subprocess.check_call(command, cwd=pwd, stdout=fp, stderr=fp)
        except subprocess.CalledProcessError as err:
            print(RED + f"Please check {log_file} for what's wrong" + RESET)
            exit(-1)
    return log_file


def run_adb_command(command, pwd):
    new_command = ['adb']
    if FLAGS.adb_serial:
        new_command.append(f'-s {FLAGS.adb_serial}')
    new_command = new_command + command
    return run_command(new_command, pwd)


def is_cross_compiling():
    return get_default_device_kwargs()["target"] != get_default_device_kwargs(FLAGS.device)["target"]


def get_llamacpp_build_dir():
    llamacpp_dir = os.path.join(ROOT_DIR, "3rdparty", "llama.cpp")
    if is_cross_compiling():
        return os.path.join(llamacpp_dir, f"build-{FLAGS.device}")
    else:
        return os.path.join(llamacpp_dir, f"build")


def compile_kernels():
    deploy_dir = os.path.join(ROOT_DIR, "deploy")
    tuned_dir = os.path.join(deploy_dir, "tuned")
    prebuilt_dir = os.path.join(tuned_dir, f"{get_arch(FLAGS.device)}-{FLAGS.model}")
    if FLAGS.use_prebuilt and os.path.isdir(prebuilt_dir):
        print(f"  Copy prebuilt kernels from {prebuilt_dir} to {tuned_dir}")
        shutil.copytree(prebuilt_dir, tuned_dir, dirs_exist_ok=True)
        return

    qargs = get_quant_args()
    command = [
        'python', 'compile.py',
        '-o', 'tuned',
        '-da',
        '-nt', f'{FLAGS.num_threads}',
        '-tb',
        '-gc',
        '-gs', f'{qargs["group_size"]}',
        '-ags', f'{qargs["act_group_size"]}',
        '-m', f'{FLAGS.model}',
        '-md', f'{FLAGS.model_dir}',
    ]
    if not FLAGS.disable_tune:
        command.append('-t')
    if qargs["zero_point"]:
        command.append('-zp')
    if FLAGS.reuse_tuned:
        command.append('-r')
    if FLAGS.device:
        command.append('-d')
        command.append(f'{FLAGS.device}')
    if FLAGS.verbose:
        command.append('-v')
    run_command(command, deploy_dir)


def _clean_cmake(build_dir):
    command = ['cmake', '--build', '.', '--target', 'clean']
    run_command(command, build_dir)
    shutil.rmtree(os.path.join(build_dir, "CMakeFiles"), ignore_errors=True)
    shutil.rmtree(os.path.join(build_dir, "CMakeCache.txt"), ignore_errors=True)


def cmake_t_mac():
    build_dir = os.path.join(ROOT_DIR, "build")
    install_dir = os.path.join(ROOT_DIR, "install")
    os.makedirs(build_dir, exist_ok=True)
    _clean_cmake(build_dir)
    command = [
        'cmake',
        f'-DCMAKE_INSTALL_PREFIX={install_dir}',
        '..',
    ]
    run_command(command, build_dir)


def install_t_mac():
    build_dir = os.path.join(ROOT_DIR, "build")
    install_dir = os.path.join(ROOT_DIR, "install")
    shutil.rmtree(install_dir, ignore_errors=True)
    command = [
        'cmake',
        '--build',
        '.',
        '--target',
        'install',
        '--config',
        'Release',
    ]
    run_command(command, build_dir)


def convert_models():
    model_dir = FLAGS.model_dir
    if not os.path.exists(model_dir):
        raise FileNotFoundError(model_dir)
    out_path = os.path.join(model_dir, f"ggml-model.{FLAGS.quant_type}.gguf")
    kcfg_path = os.path.join(ROOT_DIR, "install", "lib", "kcfg.ini")
    llamacpp_dir = os.path.join(ROOT_DIR, "3rdparty", "llama.cpp")
    command = [
        'python',
        'convert_hf_to_gguf.py',
        f'{model_dir}',
        '--outtype', f'{FLAGS.quant_type}',
        '--outfile', f'{out_path}',
        '--kcfg', f'{kcfg_path}',
        '--enable-t-mac',
        '--verbose',
    ]
    run_command(command, llamacpp_dir)


def cmake_llamacpp():
    build_dir = get_llamacpp_build_dir()
    cmake_prefix_path = os.path.join(ROOT_DIR, "install", "lib", "cmake", "t-mac")
    command = [
        'cmake', '..',
        '-DGGML_TMAC=ON',
        f'-DCMAKE_PREFIX_PATH={cmake_prefix_path}',
        '-DCMAKE_BUILD_TYPE=Release',
    ]
    if FLAGS.device == "android":
        try:
            ndk_home = FLAGS.ndk_home or os.environ["NDK_HOME"]
        except KeyError:
            raise KeyError("Missing NDK_HOME. Please either specify by -ndk or set environ NDK_HOME")
        command.append(f"-DCMAKE_TOOLCHAIN_FILE={ndk_home}/build/cmake/android.toolchain.cmake")
        command.append("-DANDROID_ABI=arm64-v8a")
        command.append("-DANDROID_PLATFORM=android-23")
        command.append("-DCMAKE_C_FLAGS=-march=armv8.2a+dotprod+fp16")
        command.append("-DGGML_METAL=OFF")
        command.append("-DGGML_ACCELERATE=OFF")
        command.append("-DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=BOTH")
        command.append("-DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=BOTH")
    elif is_win():
        if is_arm():
            command.append("-DCMAKE_C_COMPILER=clang")
            command.append("-DCMAKE_CXX_COMPILER=clang++")
            command.append("-G Ninja")
        else:
            command.append("-T ClangCL")
    else:
        command.append("-DCMAKE_C_COMPILER=clang")
        command.append("-DCMAKE_CXX_COMPILER=clang++")

    os.makedirs(build_dir, exist_ok=True)
    _clean_cmake(build_dir)
    run_command(command, build_dir)


def build_llamacpp():
    build_dir = get_llamacpp_build_dir()
    command = ['cmake', '--build', '.', '--target', 'llama-cli', 'llama-bench', '--config', 'Release']
    run_command(command, build_dir)


def run_inference():
    build_dir = get_llamacpp_build_dir()
    out_path = os.path.join(FLAGS.model_dir, f"ggml-model.{FLAGS.quant_type}.gguf")
    if is_win():
        main_path = os.path.join(build_dir, "bin", "Release", "llama-cli.exe")
        if not os.path.exists(main_path):
            main_path = os.path.join(build_dir, "bin", "llama-cli")
    else:
        main_path = os.path.join(build_dir, "bin", "llama-cli")
    prompt = "Microsoft Corporation is an American multinational corporation and technology company headquartered in Redmond, Washington."
    if FLAGS.device == "android":
        remote_bin_path = os.path.join(FLAGS.remote_dir, "bin")
        # TODO: verify in Windows
        command = ['push', os.path.join(build_dir, "bin"), FLAGS.remote_dir]
        run_adb_command(command, build_dir)
        remote_main_path = os.path.join(remote_bin_path, "llama-cli")
        command = ['shell', 'chmod', '-R', '+x', remote_bin_path]
        run_adb_command(command, build_dir)
        remote_out_path = os.path.join(
            FLAGS.remote_dir,
            f"{os.path.basename(FLAGS.model_dir)}-{os.path.basename(out_path)}",
        )
        if not FLAGS.skip_push_model:
            command = ['push', out_path, remote_out_path]
            run_adb_command(command, build_dir)
        kcfg_path = os.path.join(ROOT_DIR, "install", "lib", "kcfg.ini")
        remote_kcfg_path = os.path.join(FLAGS.remote_dir, "kcfg.ini")
        command = ['push', kcfg_path, remote_kcfg_path]
        run_adb_command(command, build_dir)
        command = [
            'shell',
            f'TMAC_KCFG_FILE={remote_kcfg_path}',
            f'{remote_main_path}',
            '-m', f'{remote_out_path}',
            '-n', '128',
            '-t', f'{FLAGS.num_threads}',
            '-p', f'"{prompt}"',
            '-ngl', '0',
            '-c', '2048'
        ]
        log_file = run_adb_command(command, build_dir)
    else:
        command = [
            f'{main_path}',
            '-m', f'{out_path}',
            '-n', '128',
            '-t', f'{FLAGS.num_threads}',
            '-p', prompt,
            '-ngl', '0',
            '-c', '2048'
        ]
        log_file = run_command(command, build_dir)
    print(GREEN + f"Check {log_file} for inference output" + RESET)


STEPS = [
    ("Compile kernels", compile_kernels),
    ("Build T-MAC C++ CMakeFiles", cmake_t_mac),
    ("Install T-MAC C++", install_t_mac),
    ("Convert HF to GGUF", convert_models),
    ("Build llama.cpp CMakeFiles", cmake_llamacpp),
    ("Build llama.cpp", build_llamacpp),
    ("Run inference", run_inference),
]


STEPS_PRESETS = {
    "all": [0, 1, 2, 3, 4, 5, 6],
    "fast": [0, 2, 3, 5, 6],
    "compile": [0, 2, 5],
}


MODELS = get_preset_models()


RED = '\033[31m'
GREEN = '\033[32m'
RESET = '\033[0m'


ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--model_dir", type=str)
    parser.add_argument("-nt", "--num_threads", type=int, default=4)
    parser.add_argument("-m", "--model", type=str, choices=MODELS, default="hf-bitnet-3b")
    parser.add_argument("-p", "--steps_preset", type=str, choices=STEPS_PRESETS.keys(), default="all",
                        help="Will be overridden by --steps. `fast` is recommended if you are not building the first time.")
    steps_str = ", ".join(f"{i}: {step}" for i, (step, _) in enumerate(STEPS))
    parser.add_argument("-s", "--steps", type=str, default=None, help="Select steps from " + steps_str + ". E.g., --steps 0,2,3,5,6")
    parser.add_argument("-gs", "--group_size", type=int, default=None, help="Don't set this argument if you don't know its meaning.")
    parser.add_argument("-ags", "--act_group_size", type=int, default=None, help="Don't set this argument if you don't know its meaning.")
    parser.add_argument("-ld", "--logs_dir", type=str, default="logs")
    parser.add_argument("-q", "--quant_type", type=str, choices=["int_n", "f16", "f32"], default="int_n")
    parser.add_argument("-zp", "--zero_point", action="store_true", help="Enforce enable zero_point. Required by EfficientQAT models.")
    parser.add_argument("-nzp", "--no_zero_point", action="store_false", help="Enforce disable zero_point. Don't set this argument if you don't know its meaning.")

    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-dt", "--disable_tune", action="store_true")
    parser.add_argument("-r", "--reuse_tuned", action="store_true")
    parser.add_argument("-u", "--use_prebuilt", action="store_true")

    parser.add_argument("-d", "--device", type=str, choices=get_devices(), default="", help="Set this argument if you are cross compiling for another device.")
    parser.add_argument("-as", "--adb_serial", type=str, default="", help="ADB serial number. Set this argument if there are multiple adb devices connected.")
    parser.add_argument("-rd", "--remote_dir", type=str, default="/data/local/tmp", help="Remote path to store bin and models.")
    parser.add_argument("-ndk", "--ndk_home", type=str, default="", help="NDK home")
    parser.add_argument("-spm", "--skip_push_model", action="store_true", help="Suppose the model is unchanged to skip pushing the model file")

    parser.set_defaults(zero_point=None)
    return parser.parse_args()


def get_quant_args():
    group_size = 128
    act_group_size = 64
    zero_point = False
    if FLAGS.model == "hf-bitnet-3b":
        act_group_size = -1
        if is_arm():
            act_group_size = 64
    elif FLAGS.model.endswith("2bit"):
        zero_point = True
    group_size = FLAGS.group_size or group_size
    act_group_size = FLAGS.act_group_size or act_group_size
    if FLAGS.zero_point is not None:
        zero_point = FLAGS.zero_point
    return {"group_size": group_size, "act_group_size": act_group_size, "zero_point": zero_point}


def main():
    steps_to_run = STEPS_PRESETS[FLAGS.steps_preset]
    if FLAGS.steps is not None:
        steps_to_run = [int(s) for s in FLAGS.steps.split(",")]

    for step in steps_to_run:
        desc, func = STEPS[step]
        print(f"Running STEP.{step}: {desc}")
        func()


if __name__ == "__main__":
    FLAGS = parse_args()

    if FLAGS.verbose:
        logging.basicConfig()
        logging.getLogger().setLevel(logging.INFO)

    main()
