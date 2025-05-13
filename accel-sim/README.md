<!-- 
## Notes on A100 Configuration and Accuracy

To ensure that GEMM operations on tensor cores in Accel-Sim more closely resemble real hardware behavior, we fine-tuned several simulation parameters. These adjustments were validated using selected microbenchmarks from [YH’s Samples](https://github.com/Yinghan-Li/YHs_Sample).

### Adjustments

1. **Tensor Core Throughput**  
   The NVIDIA A100 includes four 8-4-8 tensor cores per cycle. For HMMA instructions with a 16-8-16 data format, the initiation interval (II) is 8 cycles. This behavior is reflected in our customized `trace.config` file to more accurately emulate real execution latency.

2. **L2 Cache Behavior**  
   The A100 features a 40MB L2 cache with larger-than-default bank sizes. Since Accel-Sim’s default IPOLY model does not scale to this size, we adopted a hash-based approximation as suggested by the framework authors. While this method does not represent actual hardware architecture, it provides a reasonable estimate of expected performance.  
   For more details, see this [GitHub issue](https://github.com/accel-sim/accel-sim-framework/issues/126).

3. **Memory Configuration Constraints**  
   Accel-Sim currently does not support non-power-of-two memory sizes. For example, shared memory (smem) and L1 cache must be set to 256 KiB instead of the real-world 192 KiB. This is a known limitation. To avoid related issues, we recommend not allocating memory sizes that depend on the exact hardware capacity.

### Notes on workflow

As Accel-Sim needs running on real cards to generate trace, we have to run the inital version of our gemm code first on real hardware to generate the basic version of trace.
For end-to-end mpgemm tiling, please refer to Ladder repo.
In our paper, our flow follows a equivalent method to simulate on top of current Accel-Sim workflow(which needs running on real card to generate trace to send it to simulation engine, while there is no off-the-shelf LUT Tensor Core -equipped GPU). We use an example to illustrate how we simulate the whole process.

Take an GEMM with M=4096, N= 27648 ,K= 5120, Act fp16, Weight Int2 with 4x basic fp16-fp16 tensor core array size as an exmaple.  
For weight data, as INT2 is 1/8 of FP16 size, 27648 Int2 weight equals to 3456 Fp16 weights in data volume. Thus we trasform the GEMM first to 1/8 of the weights. Similarly, we do the tiling with a 8x scale, like if you would like to do a tile_N=512 tiling for Weight Int2, you should transform it into tile_N=64 tile in FP16 format.
After we generate the trace on real A100 cards, we need to post-process the trace as the HMMA instruction count has not been scaled. First as there the N dimension is scaled to 1/8, we need to duplicate the hmma instructions 8 times. However, as our tensor core array size is four times of the original one, so we only need to duplicate the hmma instructions 8/4=2 times.

Please refer to flow folder for detailed information -->


# Accel-Sim: A100 Configuration and Simulation Workflow

This directory provides simulation configurations and workflows for evaluating LUT-based tensor core behavior using Accel-Sim. The setup is based on the NVIDIA A100 GPU and includes fine-tuned parameters to better match real hardware behavior.

## Key Adjustments for A100 Simulation

1. Tensor Core Throughput

   The A100 GPU includes four 8-4-8 tensor cores. For HMMA instructions using the 16-8-16 instruction size, the initiation interval (II) is 8 cycles. We adjusted the `trace.config` file to reflect this timing behavior.

2. L2 Cache Behavior

   The A100 features a 40MB L2 cache with larger-than-default banks. Accel-Sim’s default IPOLY cache model cannot scale to this size, so we apply a hash-based approximation as suggested by the Accel-Sim authors. This does not match the exact microarchitecture but provides a reasonable performance estimate.

   Reference: https://github.com/accel-sim/accel-sim-framework/issues/126

3. Memory Size Constraints

   Accel-Sim currently does not support non-power-of-two memory sizes. For instance, shared memory and L1 cache must be configured as 256 KiB instead of the real-world 192 KiB. This is a known limitation — please avoid relying on exact memory sizes when configuring simulations.

## Simulating LUT Tensor Core Behavior on Accel-Sim

Accel-Sim requires traces generated on actual GPUs to run simulations. Since no off-the-shelf GPU currently includes LUT-based tensor cores, we simulate this behavior using a trace transformation approach.

We begin by executing a baseline version of our GEMM kernel on a real A100 GPU to produce the initial trace. We then apply post-processing to approximate how the kernel would execute on a custom LUT Tensor Core array.

For full end-to-end matrix multiplication (mpGEMM) tiling, please refer to the Ladder/BitBLAS (https://github.com/microsoft/BitBLAS) repository. In our paper, this method serves as an equivalent simulation flow built on top of the existing Accel-Sim framework, compensating for the lack of real LUT-based tensor core hardware.

### Example Workflow

Consider a GEMM with the following configuration:

- M = 4096, N = 27648, K = 5120  
- Activation = FP16  
- Weight = INT2  
- Simulated array size = 4× standard FP16-FP16 tensor core array

1. Weight Representation

   INT2 is 1/8 the size of FP16. So 27648 INT2 weights are equivalent to 3456 FP16 values in data volume. In our simulation, we convert the GEMM to use FP16 format and scale the number of weights accordingly.

2. Tiling Transformation

   For tiling, if the original plan uses tile_N = 512 for INT2 weights, it should be converted to tile_N = 64 in the scaled FP16 simulation — a factor of 1/8.

3. Post-processing Trace Adjustments

   After generating traces on a real A100, we adjust the HMMA instructions:

   - N was scaled to 1/8, so we duplicate the HMMA instructions 8 times.
   - Our custom tensor core array is 4× wider, so only 8 / 4 = 2 duplications are required.

This approach approximates the execution characteristics of our proposed architecture using standard GPU traces.

See the `flow` directory for automation scripts and examples.

## Simulation Steps

1. Install Accel-Sim following the official guide:  
   https://github.com/accel-sim/accel-sim-framework

2. Navigate to the Accel-Sim root directory:

   cd /path/to/accel-sim/

3. Set up the simulation environment:

   source ./gpu-simulator/setup_environment.sh

4. Compile your CUDA GEMM kernel.  
   In our case, the source is in:  
   accel-sim/flow/cuda_source

   Example:

   nvcc -std=c++17 -arch=sm_80 -I/path_to_cutlass/cutlass-3.1.0/include cutlass_gemm_tcore.cu -o cutlass_gemm_tcore  
   ./cutlass_gemm_tcore

5. Run the compiled binary with the tracer to collect execution traces:

   LD_PRELOAD=./tracer_tool/tracer_tool.so ./cutlass_gemm_tcore

6. Post-process the collected traces:

   ./tracer_tool/traces-processing/post-traces-processing ./traces/kernelslist

7. Apply additional transformations (e.g., HMMA duplication) to simulate your custom tensor core design.

8. Launch the Accel-Sim simulator:

   ./gpu-simulator/bin/release/accel-sim.out \
     -trace ./util/tracer_nvbit/traces/kernelslist.g \
     -config ./gpu-simulator/configs/tested-cfgs/xxxxx/gpgpusim.config \
     -config ./gpu-simulator/configs/tested-cfgs/xxxxx/trace.config \
     > xxx.log 2>&1

Replace `xxxxx` with your selected configuration directory. 