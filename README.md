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

The following table shows the speedup on M2-Ultra compared to llama.cpp for llama-7b kernels during token generation:

| Bits | M     | N | K     | T-MAC (CPU) (ms) | llama.cpp (CPU) | llama.cpp (METAL GPU) |
|------|-------|---|-------|-------------|-----------------|-------------------|
| 4    | 12288 | 1 | 4096  | 0.059  | 0.09689         | 0.03317           |
| 4    | 4096  | 1 | 4096  | 0.022 | 0.03407         | 0.0145            |
| 4    | 11008 | 1 | 4096  | 0.053850583 | 0.09397         | 0.03062           |
| 4    | 4096  | 1 | 11008 | 0.052825458 | 0.09132         | 0.03107           |
|      |       |   |       |             |                 |                   |
| 2    | 12288 | 1 | 4096  | 0.031785083 | 0.1197          | 0.03965           |
| 2    | 4096  | 1 | 4096  | 0.0134594   | 0.0486          | 0.01678           |
| 2    | 11008 | 1 | 4096  | 0.029358291 | 0.10573         | 0.03512           |
| 2    | 4096  | 1 | 11008 | 0.028356    | 0.11614         | 0.03749           |

## Usage

We currently supports mainstream int4 quantization (e.g., GGUF, GPTQ) on ARM CPU (e.g., M1/M2 Mac, Snapdragon CPUs).

### Requirements

This project used TVM for kernel code-generation and auto-tuning.

- tvm
- tvm-rpc: If you want to tune the kernels yourself on M1/M2 Mac or Android instead of provided tuned configurations, please setup tvm-rpc following [the official documentation](https://github.com/apache/tvm/tree/main/apps/cpp_rpc).

### From Python

Compared to normal GeMM, the weights need to be first offline preprocessed. Please refer to [the E2E inference example](./tests/test_e2e.py).

### From C++

We provide a wrapper for TMAC-GeMM. Please refer to [benchmark](./deploy/benchmark.cc) for usage.

## TODO List

- [ ] E2E inference integration into llama.cpp
- [ ] GGUF Q2_K/Q3_K support
- [ ] Build scripts
- [ ] Intel CPU support
- [ ] Release performance data for more devices
