## Usage (Deprecated)

We currently supports mainstream int4 quantization (e.g., GGUF, GPTQ) on ARM CPU (e.g., M1/M2 Mac, Snapdragon CPUs) and Intel CPU (with AVX2).

### Installation

Install this project from source with:

```
git clone --recursive https://github.com/microsoft/T-MAC.git
cd T-MAC
pip install -e .
```

### Integration into llama.cpp (exprimental)

Currently, we have integrated T-MAC into llama.cpp on windows/linux/osx.

> We have provided prebuilt kernels at `deploy/tuned/kernels.cc` for fast test. To tune kernels on your own device for maximum performance or generate kernels of different shapes, follow [this document](./codegen.md).

If you are using Intel CPUs, first replace prebuilt kernels:
```
cd deploy\tuned
copy avx2\* .\
```

Build T-MAC with:

```bash
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=${TMAC_PROJECT_DIR}/install ..
cmake --build . --target install --config Release
```

Build llama.cpp with T-MAC:

```bash
cd ../3rdparty/llama.cpp
mkdir build
cd build
cmake .. -DLLAMA_TMAC=ON -DCMAKE_PREFIX_PATH=${TMAC_PROJECT_DIR}/install/lib/cmake/t-mac -DCMAKE_BUILD_TYPE=Release -DLLAMA_TMAC_TVM_THREADPOOL=OFF -DLLAMA_LLAMAFILE_DEFAULT=OFF
cmake --build . --config Release --target llama-bench
```

If your device is not equipped with clang (if you are using OSX or Visual Studio on Windows, you already have clang), please follow [Prepare section of this document](docs/codegen.md) to install clang from conda, and replace with:
```bash
# cmake .. -DLLAMA_TMAC=ON -DCMAKE_PREFIX_PATH=${TMAC_PROJECT_DIR}/install/lib/cmake/t-mac -DCMAKE_BUILD_TYPE=Release -DLLAMA_TMAC_TVM_THREADPOOL=OFF
cmake .. -DLLAMA_TMAC=ON -DCMAKE_PREFIX_PATH=${TMAC_PROJECT_DIR}/install/lib/cmake/t-mac -DCMAKE_BUILD_TYPE=Release -DLLAMA_TMAC_TVM_THREADPOOL=OFF -DLLAMA_LLAMAFILE_DEFAULT=OFF -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
```

In Visual Studio, you should add `-T ClangCL`:
```bash
# cmake .. -DLLAMA_TMAC=ON -DCMAKE_PREFIX_PATH=${TMAC_PROJECT_DIR}/install/lib/cmake/t-mac -DCMAKE_BUILD_TYPE=Release -DLLAMA_TMAC_TVM_THREADPOOL=OFF
cmake .. -DLLAMA_TMAC=ON -DCMAKE_PREFIX_PATH=${TMAC_PROJECT_DIR}/install/lib/cmake/t-mac -DCMAKE_BUILD_TYPE=Release -DLLAMA_TMAC_TVM_THREADPOOL=OFF -DLLAMA_LLAMAFILE_DE
FAULT=OFF -T ClangCL
```

Get the test model `llama-2-7b-chat-Q4_0.gguf` from https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/tree/main, then evaluate token-generation throughput with:

```bash
./bin/llama-bench -m ${MODEL_DIR}/llama-2-7b-chat.Q4_0.gguf -n 128 -ngl 0 -b 1 -t 1 -p 0
```

> Add `LD_LIBRARY_PATH=/path/to/conda/envs/tvm-build/${arch}/lib:${LD_LIBRARY_PATH}` before `./bin/llama-bench` if you are using clang from conda and encounter errors like version `GLIBCXX_3.4.32' not found`
