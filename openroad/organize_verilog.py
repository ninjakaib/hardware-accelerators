#!/usr/bin/env python3
import os
import shutil
import re

# Base directory
base_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(base_dir, "..", "verilog")
all_dir = os.path.join(base_dir, "all")

# Create the "all" directory if it doesn't exist
if not os.path.exists(all_dir):
    os.makedirs(all_dir)

# List of hardware types and dtypes
hw_types = ["adder", "lmul", "multiplier"]
dtypes = ["bf16", "fp8", "fp32"]


# Function to rename the toplevel module in a Verilog file
def rename_toplevel_module(file_content, new_module_name):
    # Find the module declaration line
    module_pattern = re.compile(r"module\s+toplevel\s*\(")
    return re.sub(module_pattern, f"module {new_module_name}(", file_content)


# Function to create a config.mk file
def create_config_mk(target_dir, design_name, verilog_file):
    # Set clock period based on data type
    clock_period = 8.0  # default value
    # Default die and core area values
    die_area = "0 0 100 100"
    core_area = "10 10 90 90"

    if "_fp8" in design_name:
        clock_period = 0.75
    elif "_bf16" in design_name:
        clock_period = 1.0
    elif "_fp32" in design_name:
        clock_period = 1.5
        # Special die and core area for fp32
        die_area = "0 0 150 150"
        core_area = "5 5 145 145"

    config_content = f"""export DESIGN_NAME = {design_name}
export PLATFORM    = nangate45
export VERILOG_FILES = $(DESIGN_DIR)/src/{verilog_file}
export SDC_FILE      = $(DESIGN_DIR)/constraint.sdc

# These values must be multiples of placement site
export DIE_AREA    = {die_area}
export CORE_AREA   = {core_area}

export CLOCK_PERIOD = {clock_period}
"""
    with open(os.path.join(target_dir, "config.mk"), "w") as f:
        f.write(config_content)


# Process each hardware type and dtype
for hw_type in hw_types:
    hw_dir = os.path.join(base_dir, hw_type)
    if not os.path.exists(hw_dir):
        print(f"Warning: {hw_dir} does not exist, skipping...")
        continue

    for dtype in dtypes:
        dtype_dir = os.path.join(hw_dir, dtype)
        if not os.path.exists(dtype_dir):
            print(f"Warning: {dtype_dir} does not exist, skipping...")
            continue

        # Process each Verilog file in the dtype directory
        for filename in os.listdir(dtype_dir):
            if filename.endswith(".v"):
                # Original file path
                orig_file_path = os.path.join(dtype_dir, filename)

                # New file name with dtype suffix
                base_name = os.path.splitext(filename)[0]
                new_file_name = f"{base_name}_{dtype}.v"

                # Create directory for this specific design
                design_dir_name = f"{base_name}_{dtype}"
                design_dir = os.path.join(all_dir, design_dir_name)
                src_dir = os.path.join(design_dir, "src")

                if not os.path.exists(design_dir):
                    os.makedirs(design_dir)
                if not os.path.exists(src_dir):
                    os.makedirs(src_dir)

                # Read the original file
                with open(orig_file_path, "r") as f:
                    content = f.read()

                # Rename the toplevel module
                new_module_name = f"{base_name}_{dtype}"
                modified_content = rename_toplevel_module(content, new_module_name)

                # Write the modified content to the new file
                new_file_path = os.path.join(src_dir, new_file_name)
                with open(new_file_path, "w") as f:
                    f.write(modified_content)

                # Create config.mk
                create_config_mk(design_dir, new_module_name, new_file_name)

                # Copy constraint.sdc
                constraint_src = os.path.join(base_dir, "constraint.sdc")
                constraint_dst = os.path.join(design_dir, "constraint.sdc")

                # Set clock period based on data type
                clock_period = 1.0  # default value
                if dtype == "fp8":
                    clock_period = 0.75
                elif dtype == "bf16":
                    clock_period = 1.0
                elif dtype == "fp32":
                    clock_period = 1.5

                if os.path.exists(constraint_src):
                    # Read the original constraint file
                    with open(constraint_src, "r") as f:
                        constraint_content = f.read()

                    # Replace the clock period
                    constraint_content = re.sub(
                        r"-period\s+\d+\.?\d*",
                        f"-period {clock_period}",
                        constraint_content,
                    )

                    # Write the modified constraint file
                    with open(constraint_dst, "w") as f:
                        f.write(constraint_content)
                else:
                    print(
                        f"Warning: {constraint_src} does not exist, creating constraint file with appropriate clock period"
                    )
                    with open(constraint_dst, "w") as f:
                        f.write(
                            f"create_clock -name clk -period {clock_period} [get_ports {{clk}}]\n"
                        )

                print(f"Processed: {filename} -> {new_file_path}")

print("Organization complete!")
