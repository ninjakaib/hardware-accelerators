export DESIGN_NAME = toplevel
export PLATFORM    = sky130hd

export VERILOG_FILES = $(DESIGN_DIR)/src/mydesign/toplevel.v
export SDC_FILE      = $(DESIGN_DIR)/$(PLATFORM)/mydesign/constraint.sdc

# These values must be multiples of placement site
export DIE_AREA    = 0 0 100 100
export CORE_AREA   = 10 10 90 90

export CLOCK_PERIOD = 10.0
export CLOCK_PORT   = clk