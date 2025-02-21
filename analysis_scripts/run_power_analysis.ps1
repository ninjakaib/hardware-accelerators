# PowerShell script for power analysis

# Create results directory if it doesn't exist
New-Item -ItemType Directory -Force -Path ".\results"

# Synthesize designs using Yosys
Write-Host "Synthesizing accelerator design..."
yosys -s .\synth_accelerator.tcl

Write-Host "Synthesizing systolic array design..."
yosys -s .\synth_systolic.tcl

# Run power analysis using OpenSTA
Write-Host "Running power analysis..."
sta .\power_analysis.tcl

# Move results to results directory
Move-Item -Force *_power.rpt .\results\
Move-Item -Force *_timing.rpt .\results\
Move-Item -Force *_area.rpt .\results\

Write-Host "Power analysis complete. Results are in the results directory."
Write-Host "Generated files:"
Get-ChildItem .\results\ | Format-Table Name, Length, LastWriteTime