# Transparency Responsible FAQ for T-MAC

## What is T-MAC?

T-MAC is a kernel library supporting mixed-precission GeMM for Low bit LLM inference on CPUs. LLM inference incurs significant computational cost. Low-bit quantization, a widely adopted technique, introduces the challenge of mixed-precision GEMM (mpGEMM), which is not directly supported by hardware and requires convert/dequant operations. We propose the use of a lookup table (LUT) to support mpGEMM. Our kernels demonstrate superior performance over SOTA low-bit GeMM on CPU. 

## What can T-MAC do?

T-MAC offers efficient mixed precison GEMM kernels on CPU. It can speedup up the inference of low-bit large language model on CPU.

## What are T-MAC's intended uses?

The intended use cases for T-MAC involve enhancing computational efficiency in low-bit Large language models on CPU. It's designed to serve developers and researchers requiring advanced mixed precision matrix multiplication.

## How was T-MAC evaluated?

T-MAC was evaluated based on its performance and correctness in generating and executing optimized code for mixed precision GEMM. Performance metrics include the speed of execution, computational efficiency. The evaluation involved comparing T-MAC's performance with other existing frameworks, such as llama.cpp. The accuracy verification through simulations with PyTorch operators.

## What are the limitations of T-MAC?

While T-MAC is designed to be highly efficient, there are limitations to its applicability and performance. It is optimized for CPUs, which means its use is restricted to systems with compatible CPUs.

## Operational factors and settings for effective and responsible use

T-MAC is expected to perform reliably within the operational factors and settings of CPU architectures. Users can influence the system's behavior through customization of the DSL scripts, selection of data types and precision and operator configurations, and tuning of performance parameters to suit their specific computational needs. These choices impact the efficiency and accuracy of the computations, making it essential for users to understand the trade-offs involved.

## Plugins and Extensibility

T-MAC doesn't allow for plugins or extensibility.
