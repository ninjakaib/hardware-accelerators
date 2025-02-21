# OpenSTA power analysis script

proc analyze_power {design} {
    # Read liberty file
    read_liberty .\liberty_file.lib
    
    # Read synthesized netlist
    read_verilog .\synth_${design}.v
    
    # Link design
    link_design ${design}
    
    # Read SDC constraints
    read_sdc .\constraints.sdc
    
    # Read SPEF file for parasitic information
    read_spef .\${design}.spef
    
    # Set operating conditions
    set_operating_conditions typical
    
    # Set switching activity
    # Assuming 20% switching activity and 50% static probability for all nets
    set_switching_activity -static 0.5 -toggle_rate 0.2 [all_nets]
    
    # Report power
    report_power > .\${design}_power.rpt
    
    # Report timing
    report_checks -path_delay max > .\${design}_timing.rpt
    
    # Report area
    report_area > .\${design}_area.rpt
}

# Analyze both designs
analyze_power accelerator
analyze_power systolic