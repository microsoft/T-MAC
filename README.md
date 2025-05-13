# LUT Tensor Core

This repository provides code for exploring LUT-based tensor core acceleration, including hardware synthesis and GPU-based simulation.

## Hardware: `sv_lut_array`
A synthesizable LUT-based tensor core array design.

To run synthesis:
- Set your own Design Compiler and TSMC library paths in `./scripts/dc.tcl`.
- Follow the instructions in the `sv` directory.

## Accelerator Simulator: `accel-sim`
Includes a customized configuration targeting NVIDIA A100 GPUs.

For tuning details and simulation workflow, see `accel-sim/README.md`.

## End-to-End Simulator
To be released in our upcoming work.