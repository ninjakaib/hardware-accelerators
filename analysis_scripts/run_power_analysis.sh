#!/bin/bash

# Create results directory
mkdir -p results

# Synthesize designs using Yosys
echo "Synthesizing accelerator design..."
yosys -s synth_accelerator.tcl
echo "Synthesizing systolic array design..."
yosys -s synth_systolic.tcl

# Run power analysis using OpenSTA
echo "Running power analysis..."
sta power_analysis.tcl

# Move results to results directory
mv *_power.rpt results/
mv *_timing.rpt results/
mv *_area.rpt results/

echo "Power analysis complete. Results are in the results directory."
echo "Generated files:"
ls -l results/ 