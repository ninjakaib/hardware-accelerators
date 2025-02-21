yosys -import

# Read design
read_verilog ..\systolic_array.v

# Generic synthesis
synth -top systolic_array

# Map to standard cell library
# Note: Replace liberty_file.lib with your actual library path
dfflibmap -liberty .\liberty_file.lib
abc -liberty .\liberty_file.lib

# Write synthesized netlist
write_verilog .\synth_systolic.v
write_spef .\systolic.spef

# Print statistics
stat -liberty .\liberty_file.lib