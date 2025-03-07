export DESIGN_NAME = lmul_full_accelerator_fast
export PLATFORM    = nangate45
export VERILOG_FILES = $(DESIGN_DIR)/src/lmul_full_accelerator_fast.v
export SDC_FILE      = $(DESIGN_DIR)/constraint.sdc

# These values must be multiples of placement site
export DIE_AREA    = 0 0 1500 1500
export CORE_AREA   = 10 10 1490 1490
export PLACE_DENSITY = 0.0
export SYNTH_MEMORY_MAX_BITS = 1000000
export SKIP_CTS_REPAIR_TIMING = 1

export CLOCK_PERIOD = 5.0
