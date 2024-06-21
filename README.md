# T-MAC

<h3 align="center">
    <img src="assets/demo.gif">
    <p><a href=https://huggingface.co/1bitLLM/bitnet_b1_58-3B>BitNet</a> on T-MAC (LUT-based) vs llama.cpp (dequantization-based)</p>
</h3>


## Introduction

T-MAC is a kernel library to directly support mixed-precision matrix multiplication (int1/2/3/4 x int8/fp16/fp32) without the need for dequantization by utilizing lookup tables. T-MAC aims to boost low-bit LLM inference on CPUs. T-MAC already offers support for various low-bit models, including W4A16 from GPTQ/gguf, W2A16 from [BitDistiller](https://github.com/DD-DuDa/BitDistiller) and W1(.58)A8 from [BitNet](https://huggingface.co/1bitLLM/bitnet_b1_58-3B) on OSX/Linux/Windows equipped with ARM/Intel CPUs.

T-MAC achieves a token generation throughput of 22 tokens/sec with a single core and 54 tokens/sec with four cores on M2-Ultra for 3B BitNet, which is a 3x speedup compared to SOTA CPU low-bit framework ([llama.cpp](https://github.com/ggerganov/llama.cpp)). T-MAC can even reach 11 tokens/sec on lower-end devices like Raspberry Pi 5.

## End-2-End Speedup

We evaluate the token generation performance of different models on four different devices: Apple M2-Ultra, Jetson AGX Orin, Raspberry Pi 5 and Surface Book 3.

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

Our GEMM kernels demonstrate superior performance over SOTA low-bit GEMM on CPU. The following figure shows the speedup compared to llama.cpp for llama-7b kernels during token generation (NUM_THREADS=1):

![](assets/gemv_t1.png)

> llama.cpp doesn't provide 1-bit kernel implementation, but we can deduce it from the 2-bit, as it won't bring additional speedup according to the 2/3/4-bit results.

Although we haven't integrated multi-batch (N>1) GEMM into llama.cpp, T-MAC can achieve significant speedup due to reduced computaional cost, which ensures superior performance on prompt evaluation and multi-batch token generation. The following figures shows the speedup compared to llama.cpp using OpenBLAS backend (NUM_THREADS=1):

![](assets/gemm.png)

> M2-Ultra is an exception as it is equipped with a specially designed [AMX coprocessor](https://github.com/corsix/amx) to accelerate multi-batch GEMM. However, T-MAC can still achieve comparable performance at 2-bit.

## Energy and Power Saving

By replacing heavy fused-multiply-add instructions with table lookup instructions, T-MAC significantly reduces power consumption. Combined with the speedup, T-MAC ultimately results in a substantial decrease in total energy consumption.

<p align="center">
    <img src="assets/e2e_power.png">
    <p align="center">Multi-threading power/energy consumption on M2-Ultra for three models, M1: Llama-2-7B (W4), M2: Llama-2-7B (W2) and M3: BitNet-3B</p>
</p>

> Data sampled with [powermetrics](https://www.unix.com/man-page/osx/1/powermetrics/).

### Compared to CUDA GPU

T-MAC achieves comparable 2-bit mpGEMM performance compared to CUDA GPU on Jetson AGX Orin. While the CUDA GPU outperforms the CPU in executing kernels other than mpGEMM, making the end-to-end performance of T-MAC (CPU) slightly slower, T-MAC can deliver considerable savings in power and energy consumption.

| Framework       | Throughput (tokens/sec) | Power (W)   | Energy (J/token) |
|-----------------|:------------------------|:------------|:-----------------|
| llama.cpp (CPU) |         7.08            |     15.0    | 2.12             |
| llama.cpp (GPU) |        <b>20.03</b>     |     30.8    | 1.54             |
| T-MAC (CPU)     |         15.62           | <b>10.4</b> | <b>0.66</b>      |

<p align="center">
<b>Throughput/power/energy comparison for Llama-2-7B (W2) on NVIDIA Jetson AGX Orin (NUM_THREADS=12 for CPU)</b>
</p>

> Data sampled with [jetson-stats](https://github.com/rbonghi/jetson_stats) under power mode MAXN.

## Installation

### OSX

First, install `cmake`, `zstd` (dependency of llvm) and `libomp` (dependency of tvm). Homebrew is recommended:

```bash
brew install cmake zlib libomp
```

> If `zstd` is installed through homebrew, than `cmake` should also be installed through homebrew to ensure that `zstd` can be found by `cmake`.

Install `t_mac` from the source (please run in a `virtualenv`):

```bash
git clone --recursive https://github.com/microsoft/T-MAC.git
# in virtualenv
pip install -e .
source build/t-mac-envs.sh
```

The command will download clang+llvm and build tvm from source. So it might take a bit of time.

### Verification

After that, you can verify the installation through: `python -c "import t_mac; print(t_mac.__version__); from tvm.contrib.clang import find_clang; print(find_clang())"`.

## Usage

Currently, we supports end-to-end inference through llama.cpp integration.

### Prepare models

> The following guide use BitNet-3B. We will add instructions how to use other models or even your customized kernels.

First, download the model `huggingface-cli download 1bitLLM/bitnet_b1_58-3B --local-dir ${model_dir}`.

Then, compile kernels for the model. There are two options:

- Use prebuilt kernels for ARM CPUs:
    ```bash
    cd deploy
    cp tuned/arm-hf-bitnet-3b/* tuned/
    ```
- Compile the kernels yourself:
    ```bash
    cd deploy
    python compile.py -o tuned -da -d m2 -nt 4 -tb -gc -ags 64 -t -m hf-bitnet-3b
    ```

Build T-MAC C++ source:

```bash
cd ..  # back to project root directory
export TMAC_ROOT_DIR=$(pwd)
cd build
cmake -DCMAKE_INSTALL_PREFIX=${TMAC_ROOT_DIR}/install ..
cmake --build . --target install --config Release
```

Convert the huggingface model to gguf:

```bash
cd ../3rdparty/llama.cpp/gguf-py
pip install .
cd ..
python convert-hf-to-gguf-bitnet.py ${model_dir} --outtype i2 --outfile ${model_dir}/hf-bitnet-3B.i2.gguf --kcfg ${TMAC_ROOT_DIR}/install/lib/kcfg.ini
```

Build llama.cpp:

```bash
# in 3rdparty/llama.cpp
mkdir build
cd build
cmake .. -DLLAMA_TMAC=ON -DCMAKE_PREFIX_PATH=${TMAC_ROOT_DIR}/install/lib/cmake/t-mac -DCMAKE_BUILD_TYPE=Release -DLLAMA_LLAMAFILE_DEFAULT=OFF -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
cmake --build . --target main llama-bench --config Release
```

Run inference through:

```bash
./bin/main -m ~/Downloads/test_models/hf-bitnet-3B.i2.gguf -n 128 -t 1 -p "Microsoft Corporation is an American multinational corporation and technology company headquartered in Redmond, Washington." -b 1 -ngl 0 -c 2048
```

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
