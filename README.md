# T-MAC

## Introduction

T-MAC is a kernel library supporting mixed-precission GeMM on CPUs.

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

| Model      | Bits | T-MAC (CPU) (tokens/sec) | llama.cpp (CPU) |
|------------|------|--------------------------|-----------------|
| llama-2-7b | 4    | 7.80                     | 5.56            |
| llama-2-7b | 2    | 16.17                    | 3.63            |
|            |      |                          |                 |
| BitNet-3b  | 2    | 24.36                    | 7.38            |

*We will release multi-threading performance soon.*

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

## TODO List

- [ ] Pre-built release
- [ ] Add llama.cpp build instructions
- [x] BitNet kernel support
- [ ] BitNet E2E integration
- [x] Intel CPU support
- [ ] Release performance data for more devices
