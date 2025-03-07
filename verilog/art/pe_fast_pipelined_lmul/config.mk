export DESIGN_NAME = pe_fast_pipelined_lmul
export PLATFORM    = nangate45
export VERILOG_FILES = $(DESIGN_DIR)/src/pe_fast_pipelined_lmul.v
export SDC_FILE      = $(DESIGN_DIR)/constraint.sdc

# These values must be multiples of placement site
export DIE_AREA    = 0 0 200 200
export CORE_AREA   = 10 10 190 190
export PLACE_DENSITY = 0.1

export CLOCK_PERIOD = 3.0
