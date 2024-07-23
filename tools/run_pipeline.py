import argparse
import subprocess
import os
from datetime import datetime
import shutil
import logging

from t_mac.platform import get_system_info, is_win, is_arm


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


def compile_kernels():
    deploy_dir = os.path.join(ROOT_DIR, "deploy")
    tuned_dir = os.path.join(deploy_dir, "tuned")
    prebuilt_dir = os.path.join(tuned_dir, f"{get_system_info()[1]}-{FLAGS.model}")
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
        '-t',
        '-m', f'{FLAGS.model}',
    ]
    if qargs["zero_point"]:
        command.append('-zp')
    if FLAGS.reuse_tuned:
        command.append('-r')
    run_command(command, deploy_dir)


def _clean_cmake(build_dir):
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
        'convert-hf-to-gguf-t-mac.py',
        f'{model_dir}',
        '--outtype',
        f'{FLAGS.quant_type}',
        '--outfile', f'{out_path}',
        '--kcfg', f'{kcfg_path}',
    ]
    run_command(command, llamacpp_dir)


def cmake_llamacpp():
    build_dir = os.path.join(ROOT_DIR, "3rdparty", "llama.cpp", "build")
    cmake_prefix_path = os.path.join(ROOT_DIR, "install", "lib", "cmake", "t-mac")
    command = [
        'cmake', '..',
        '-DLLAMA_TMAC=ON',
        f'-DCMAKE_PREFIX_PATH={cmake_prefix_path}',
        '-DCMAKE_BUILD_TYPE=Release',
        '-DLLAMA_LLAMAFILE_DEFAULT=OFF',
    ]
    if is_win():
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
    build_dir = os.path.join(ROOT_DIR, "3rdparty", "llama.cpp", "build")
    command = ['cmake', '--build', '.', '--target', 'main', 'llama-bench', '--config', 'Release']
    run_command(command, build_dir)


def run_inference():
    build_dir = os.path.join(ROOT_DIR, "3rdparty", "llama.cpp", "build")
    out_path = os.path.join(FLAGS.model_dir, f"ggml-model.{FLAGS.quant_type}.gguf")
    if is_win():
        main_path = os.path.join(build_dir, "bin", "Release", "main.exe")
        if not os.path.exists(main_path):
            main_path = os.path.join(build_dir, "bin", "main")
    else:
        main_path = os.path.join(build_dir, "bin", "main")
    prompt = "Microsoft Corporation is an American multinational corporation and technology company headquartered in Redmond, Washington."
    command = [
        f'{main_path}',
        '-m', f'{out_path}',
        '-n', '128',
        '-t', f'{FLAGS.num_threads}',
        '-p', prompt,
        '-b', '1',
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


MODELS = [
    "llama-2-7b-4bit",
    "llama-2-7b-2bit",
    "llama-2-13b-2bit",
    "llama-3-8b-2bit",
    "hf-bitnet-3b",
    "test",
]


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
                        help="Will be overriden by --steps. `fast` is recommended if you are not building the first time.")
    steps_str = ", ".join(f"{i}: {step}" for i, (step, _) in enumerate(STEPS))
    parser.add_argument("-s", "--steps", type=str, default=None, help="Select steps from " + steps_str + ". E.g., --steps 0,2,3,5,6")
    parser.add_argument("-gs", "--group_size", type=int, default=None, help="Don't set this argument if you don't know its meaning.")
    parser.add_argument("-ags", "--act_group_size", type=int, default=None, help="Don't set this argument if you don't know its meaning.")
    parser.add_argument("-ld", "--logs_dir", type=str, default="logs")
    parser.add_argument("-q", "--quant_type", type=str, choices=["i2"], default="i2")

    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-r", "--reuse_tuned", action="store_true")
    parser.add_argument("-u", "--use_prebuilt", action="store_true")
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
