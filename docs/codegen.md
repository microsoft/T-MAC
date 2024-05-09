# Codegen and Tuning

## Prepare

We suggest to use conda to install all build tools.

```bash
cd 3rdparty/tvm
conda env create --file conda/build-environment.yaml
conda activate tvm-build
```

If you are using Windows, install Visual Studio and toggle Clang tools on, then execute from `Developer Command Prompt for VS`:
```
cd 3rdparty/tvm
conda env create --file conda/build-environment-win.yaml
conda activate tvm-build
```

Then install additional dependencies for codegen:

```bash
cd ../../
pip install -e .  # install t-mac
pip install -r requirements-codegen.txt
```

## Install TVM from Source

First, install TVM from source:
```bash
cd 3rdparty/tvm
mkdir build
cp cmake/config.cmake build
cd build
cmake ..
make -j4
```

Or if you are using Visual Studio:
```
mkdir build
copy cmake\config.cmake build
cd build
cmake ..
cmake --build . --config Release -- /m
```

After that, set `PYTHONPATH` environment variables:
```
export PYTHONPATH=/path/to/T-MAC/3rdparty/tvm/python:${PYTHONPATH}
```

## Setup RPC (macos)

This step is only required for macos.

```bash
python -m tvm.exec.rpc_tracker --host=0.0.0.0 --port=9190 
cd 3rdparty/tvm/build
./tvm_rpc server --host=0.0.0.0 --port=9000 --port-end=9090 --tracker=0.0.0.0:9190 --key=local
```

## Setup RPC (android)

Follow https://github.com/apache/tvm/blob/main/apps/android_rpc/README.md to setup RPC and cross compilation tools for android.

## Compile and Tuning

Then, compile T-MAC kernels with:

```bash
cd ../../../  # navigate to project root
cd deploy
python compile.py -t -o tuned -da -d m2/intel_win/android/jetson -b 4 -nt 1 -tb -gc -gs 32 -ags 32
```

For `compile.py`:

- Comment/Uncomment lines in `MKNs = [...]` to select kernels to build. The defaults are for llama-2-7b.
- `-d` to select device type.
    - m2 for mac device equipped with m1/m2/m3 cpu
    - intel_win for intel devices. It should also work for linux with intel cpu (untested).
    - jetson for nvidia jetson nano. It should also work for any aarch64 device.
- `-b` to select bits. `-b 4` for llama-2-7b-chat.Q4_0
- `-nt` to specify num threads. Better set it to number of CPU cores.
- `-tb` to compile works for only one threadblock. Enable this if you want to integrate this kernel without using TVM threadpool (e.g., llama.cpp).
- `-gc` to generate portable c++ code instead of dynamic/static libraries. It's recommended for cross-platform compilation (e.g., llama.cpp).
- `-gs` to specify group_size. 32 for llama-2-7b-chat.Q4_0, or 128 for default settings of GPTQ.
- `-ags` to specify activation group size. It should be less than the value of `-gs`. Recommended settings are `-ags 32`, `-ags 64` or `-ags -1` (for BitNet)
- `-fa` to toggle fast aggregation.
