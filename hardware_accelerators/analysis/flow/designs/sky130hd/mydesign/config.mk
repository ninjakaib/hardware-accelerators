export DESIGN_NAME = toplevel
export PLATFORM    = nangate45
export VERILOG_FILES = ./designs/pipelined_adder_BF16/src/pipelined_adder_BF16.v
export SDC_FILE      = ./designs/pipelined_adder_BF16/constraint/constraint.sdc

# These values must be multiples of placement site
export DIE_AREA    = 0 0 100 100
export CORE_AREA   = 10 10 90 90

export CLOCK_PERIOD = 10.0
