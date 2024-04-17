# T-MAC

## Introduction

T-MAC is a kernel library supporting mixed-precission GeMM.

LLM inference incurs significant computational cost. Low-bit quantization, a widely adopted technique, introduces the challenge of mixed-precision GeMM (mpGeMM), which is not directly supported by hardware and requires convert/dequant operations.

We propose the use of a lookup table (LUT) to support mpGeMM. Our method involves the following key technniques:

1. Given the low precision of weights, we group one-bit weights (e.g., into groups of 4), precompute all possible partial sums, and then use a LUT to store them.
2. We employ shift and accumulate operations to support scalable bits from 1 to 4.
3. On a CPU, we utilize tbl/pshuf instructions for fast table lookup.
4. We reduce the table size from $2^n$ to $2^{n-1}$, incorporating a sign bit to accelerate LUT precomputation.

Our method exhibits several notable characteristics:

1. It shows a linear scaling ratio of FLOPs and inference latency relative to the number of bits. This contrasts with traditional convert-based methods, which fail to achieve additional speedup when reducing from 4 bits to lower bits.
2. T-MAC inherently supports bit-wise computation for int1/2/3/4, eliminating the need for dequantization. Furthermore, it accommodates all types of activations (e.g., fp8, fp16, int8) using fast table lookup and add instructions, bypassing the need for poorly supported fused-multiply-add instructions.
3. T-MAC holds the potential to realize performance gains across all processing units (PUs).

## Speedup to SOTA Low-Bit GeMM

Our kernels demonstrate superior performance over SOTA low-bit GeMM on CPU. Due to the linear scaling characteristic of T-MAC, our kernels deliver improved results at 2 bits, even surpassing the performance of Metal GPUs.

The following table shows the speedup on M2-Ultra compared to llama.cpp for llama-7b kernels during token generation (NUM_THREADS=16):

| Bits | M     | N | K     | T-MAC (CPU) (ms) | llama.cpp (CPU) | llama.cpp (METAL GPU) |
|------|-------|---|-------|-------------|-----------------|-------------------|
| 4    | 12288 | 1 | 4096  | 0.059  | 0.096         | 0.033           |
| 4    | 4096  | 1 | 4096  | 0.022 | 0.034         | 0.014            |
| 4    | 11008 | 1 | 4096  | 0.053 | 0.093         | 0.030           |
| 4    | 4096  | 1 | 11008 | 0.052 | 0.091         | 0.031           |
|      |       |   |       |             |                 |                   |
| 2    | 12288 | 1 | 4096  | 0.031 | 0.117          | 0.039           |
| 2    | 4096  | 1 | 4096  | 0.013   | 0.048          | 0.016           |
| 2    | 11008 | 1 | 4096  | 0.029 | 0.105         | 0.035           |
| 2    | 4096  | 1 | 11008 | 0.028    | 0.116         | 0.037           |

## E2E Speedup to llama.cpp

By integrating T-MAC kernels to llama.cpp, we obtain the following table to show the speedup on M2-Ultra for llama-7b duing token generation (NUM_THREADS=1):

| Bits | T-MAC (CPU) (tokens/sec) | llama.cpp (CPU) |
|------|--------------------------|-----------------|
| 4    | 7.80                     | 5.56            |
| 2    | 16.17                    | 3.63            |

*We will release multi-threading performance soon.*

## Usage

We currently supports mainstream int4 quantization (e.g., GGUF, GPTQ) on ARM CPU (e.g., M1/M2 Mac, Snapdragon CPUs) and Intel CPU (with AVX2).

### Requirements

This project used TVM for kernel code-generation and auto-tuning.

- tvm
- llvm
- tvm-rpc: If you want to tune the kernels yourself on M1/M2 Mac or Android instead of provided tuned configurations, please setup tvm-rpc following [the official documentation](https://github.com/apache/tvm/tree/main/apps/cpp_rpc).

### Installation

Install this project from source with:

```
git clone --recursive https://github.com/microsoft/T-MAC.git
cd T-MAC
pip install -e .
```

### Integration into llama.cpp (exprimental)

Currently, we have integrated T-MAC into llama.cpp on windows/linux/osx.

First, compile T-MAC kernels with.

```bash
cd deploy
python compile.py -t -o tuned -da -d m2 -b 4 -nt 1
```

Then, build T-MAC with:

```bash
cd ..
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=${TMAC_PROJECT_DIR}/install ..
cmake --build . --target install --config Release
```

After that, build llama.cpp with T-MAC:

```bash
cd ../3rdparty/llama.cpp
mkdir build
cd build
cmake .. -DLLAMA_TMAC=ON -DCMAKE_PREFIX_PATH=${TMAC_PROJECT_DIR}/install/lib/cmake/t-mac -DCMAKE_BUILD_TYPE=Release -DLLAMA_TMAC_TVM_THREADPOOL=ON
cmake --build . --config Release --target llama-bench
```

Evaluate token-generation throughput with:

```bash
./bin/llama-bench -m ${MODEL_DIR}/llama-2-7b-chat.Q4_0.gguf -n 128 -ngl 0 -b 1 -t 1 -p 0
```

#### Issues

- We are working on resolving conflicts between TVM threadpool and llama.cpp threadpool. The current multi-threading performance is suboptimal.

- We are converting 2-bit models to support correct token generation for 2bit.

## TODO List

- [ ] Pre-built release
- [ ] E2E inference integration into llama.cpp
- [x] BitNet kernel support
- [ ] BitNet E2E integration
- [x] Intel CPU support
- [ ] Release performance data for more devices
