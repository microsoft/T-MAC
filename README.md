# T-MAC

demo_gif

## Introduction

T-MAC is a kernel library to directly support mixed-precision matrix multiplication (int1/2/4 x int8/fp16) without the need for dequantization by utilizing lookup tables. T-MAC aims to boost low-bit LLM inference on CPUs. T-MAC already offers support for various low-bit models, including W4A16 from GPTQ/gguf, W2A16 from [BitDistiller](https://github.com/DD-DuDa/BitDistiller) and W1(.58)A8 from [BitNet](https://huggingface.co/1bitLLM/bitnet_b1_58-3B) on OSX/Linux/Windows equipped with ARM/Intel CPUs.

T-MAC achieves a token generation throughput of 22 tokens/sec with a single core and 54 tokens/sec with four cores on M2-Ultra for 3B BitNet, which is a 3x speedup compared to SOTA CPU low-bit framework ([llama.cpp](https://github.com/ggerganov/llama.cpp)). T-MAC can even reach 11 tokens/sec on lower-end devices like Raspberry Pi 5.

## End-2-End Speedup

We evaluate the token generation performance of difference models on four different devices: Apple M2-Ultra, Jetson AGX Orin, Raspberry Pi 5 and Surface Book 3.

> We evaluate BitNet-3B and Llama-2-7B (W2) with T-MAC 2-bit and llama.cpp Q2_K, and evaluate Llama-2-7B (W4) with T-MAC 4-bit and llama.cpp Q4_0.

| Model              | Device         | NUM_THREADS | llama.cpp (CPU) (tokens/sec) | T-MAC (CPU) |
|--------------------|----------------|-------------|------------------------------|-------------|
| BitNet-3B          | M2-Ultra       | 1           | 6.49                         | 22.08       |
| BitNet-3B          | M2-Ultra       | 4           | 22.09                        | 54.46       |
| Llama-2-7B (W2)    | M2-Ultra       | 1           | 3.82                         | 16.68       |
| Llama-2-7B (W2)    | M2-Ultra       | 8           | 22.06                        | 51.01       |
| Llama-2-7B (W4)    | M2-Ultra       | 1           | 5.65                         | 8.97        |
| Llama-2-7B (W4)    | M2-Ultra       | 8           | 31.57                        | 35.65       |
|                    |                |             |                              |             |
| BitNet-3B          | AGX Orin       | 1           | 1.62                         | 8.18        |
| BitNet-3B          | AGX Orin       | 12          | 12.34                        | 26.02       |
| Llama-2-7B (W2)    | AGX Orin       | 1           | 0.79                         | 4.36        |
| Llama-2-7B (W2)    | AGX Orin       | 12          | 7.08                         | 15.62       |
| Llama-2-7B (W4)    | AGX Orin       | 1           | 1.04                         | 2.46        |
| Llama-2-7B (W4)    | AGX Orin       | 12          | 7.42                         | 8.09        |
|                    |                |             |                              |             |
| BitNet-3B          | Raspberry Pi 5 | 1           | 1.37                         | 8.03        |
| BitNet-3B          | Raspberry Pi 5 | 2           | 2.71                         | 11.09       |
| Llama-2-7B (W2)    | Raspberry Pi 5 | 1           | 0.66                         | 4.40        |
| Llama-2-7B (W2)    | Raspberry Pi 5 | 2           | 1.31                         | 5.92        |
| Llama-2-7B (W4)    | Raspberry Pi 5 | 1           | 0.85                         | 2.42        |
| Llama-2-7B (W4)    | Raspberry Pi 5 | 2           | 1.63                         | 3.35        |
|                    |                |             |                              |             |
| BitNet-3B          | Surface Book 3 | 1           | 5.65                         | 12.65       |
| BitNet-3B          | Surface Book 3 | 4           | 14.85                        | 28.60       |
| Llama-2-7B (W2)    | Surface Book 3 | 1           | 2.70                         | 6.77        |
| Llama-2-7B (W2)    | Surface Book 3 | 4           | 7.50                         | 16.82       |
| Llama-2-7B (W4)    | Surface Book 3 | 1           | 2.50                         | 3.74        |
| Llama-2-7B (W4)    | Surface Book 3 | 4           | 6.52                         | 9.34        |

## Kernel-level Speedup

Our kernels demonstrate superior performance over SOTA low-bit GEMM on CPU. The following figure shows the speedup compared to llama.cpp for llama-7b kernels during token generation (NUM_THREADS=1):

![](assets/gemv_t1.png)

> llama.cpp doesn't provide 1-bit kernel implementation, but we can deduce it from the 2-bit, as it won't bring additional speedup according to the 2/3/4-bit results.

Although we haven't integrated multi-batch (N>1) GEMM into llama.cpp, T-MAC can achieve significant speedup due to reduced computaional cost, which ensures superior performance on prompt evaluation and multi-batch token generation. The following figures shows the speedup compared to llama.cpp using OpenBLAS backend (NUM_THREADS=1):

![](assets/gemm.png)

## Usage

TODO

## Techniques

LLM inference incurs significant computational cost. Low-bit quantization, a widely adopted technique, introduces the challenge of mixed-precision GEMM (mpGEMM), which is not directly supported by hardware and requires convert/dequant operations.

We propose the use of a lookup table (LUT) to support mpGEMM. Our method involves the following key technniques:

1. Given the low precision of weights, we group one-bit weights (e.g., into groups of 4), precompute all possible partial sums, and then use a LUT to store them.
2. We employ shift and accumulate operations to support scalable bits from 1 to 4.
3. On a CPU, we utilize tbl/pshuf instructions for fast table lookup.
4. We reduce the table size from $2^n$ to $2^{n-1}$, incorporating a sign bit to accelerate LUT precomputation.

Our method exhibits several notable characteristics:

1. T-MAC shows a linear scaling ratio of FLOPs and inference latency relative to the number of bits. This contrasts with traditional convert-based methods, which fail to achieve additional speedup when reducing from 4 bits to lower bits.
2. T-MAC inherently supports bit-wise computation for int1/2/3/4, eliminating the need for dequantization. Furthermore, it accommodates all types of activations (e.g., fp8, fp16, int8) using fast table lookup and add instructions, bypassing the need for poorly supported fused-multiply-add instructions.
3. T-MAC holds the potential to realize performance gains across all processing units (PUs).

## Cite
If you find this repository useful, please use the following BibTeX entry for citation.
```
@misc{t-mac,
      title={T-MAC: CPU Renaissance via Table Lookup for Low-Bit LLM Inference},
      author={Jianyu Wei and Shijie Cao and Ting Cao and Lei Wang and Lingxiao Ma},
      year={2024},
      url={https://github.com/microsoft/T-MAC/}
}
```

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

> We have provided prebuilt kernels at `deploy/tuned/kernels.cc` for fast test. To tune kernels on your own device for maximum performance or generate kernels of different shapes, follow [this document](docs/codegen.md).

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
