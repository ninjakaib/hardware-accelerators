export DESIGN_NAME = adder_pipelined_fp8_pretty
export PLATFORM    = nangate45
export VERILOG_FILES = $(DESIGN_DIR)/src/adder_pipelined_fp8_pretty.v
export SDC_FILE      = $(DESIGN_DIR)/constraint.sdc

# These values must be multiples of placement site
export DIE_AREA    = 0 0 60 60
export CORE_AREA   = 5 5 55 55
export PLACE_DENSITY = 0.34

export CLOCK_PERIOD = 1.0
