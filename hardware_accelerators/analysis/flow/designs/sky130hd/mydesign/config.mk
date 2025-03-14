export DESIGN_NAME = lmul_pipelined_fast
export PLATFORM    = nangate45
export VERILOG_FILES = $(DESIGN_DIR)/src/lmul_pipelined_fast.v
export SDC_FILE      = $(DESIGN_DIR)/constraint.sdc

# These values must be multiples of placement site
export DIE_AREA    = 0 0 100 100
export CORE_AREA   = 10 10 90 90

export CLOCK_PERIOD = 1.0
