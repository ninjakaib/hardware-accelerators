yosys -import

# Read design
read_verilog ..\accelerator.v

# Generic synthesis
synth -top accelerator

# Map to standard cell library
# Note: Replace liberty_file.lib with your actual library path
dfflibmap -liberty .\liberty_file.lib
abc -liberty .\liberty_file.lib

# Write synthesized netlist
write_verilog .\synth_accelerator.v
write_spef .\accelerator.spef

# Print statistics
stat -liberty .\liberty_file.lib