
# Hardware Synthesis: sv_lut_array

This directory contains the synthesizable Verilog implementation of our LUT-based tensor core array.

## Overview

The synthesis flow is based on Synopsys Design Compiler. You will need to configure paths to your local toolchain and standard cell library before running.

## Prerequisites

- Synopsys Design Compiler (dc_shell)
- Access to a TSMC standard cell library (e.g., 28nm)

## Instructions

Follow the steps below to run synthesis:

1. Launch Design Compiler:

   dc_shell

2. Navigate to the synthesis script directory:

   cd sv/scripts

3. Run the synthesis script:

   source dc.tcl

## Configuration

Before running the script, make sure to update the following in scripts/dc.tcl:

- The path to your Design Compiler setup (license, binary, etc.)
- The path to your local TSMC technology library

Example:

   set target_library 
   set link_library 

## Output

After synthesis, output files will be generated in the following locations (unless modified in the script):

- Netlist: sv/output/
- Reports (timing, area, power): sv/reports/

## Notes

- Timing and area estimates depend heavily on the specific standard cell library used.
- Ensure that your library constraints match the target process node and synthesis goals.