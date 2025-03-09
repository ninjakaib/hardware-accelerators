#!/bin/bash
# Script to run make with specific config for all design subfolders
# Check if designs directory exists
if [ ! -d "./designs" ]; then
    echo "Error: designs directory not found in current location"
    exit 1
fi
# Counter for successful and failed builds
success_count=0
fail_count=0
failed_designs=()
echo "Starting build process for all designs..."
echo "----------------------------------------"
# Loop through all subdirectories in the designs directory
for design_dir in ./designs/*/; do
    # Remove trailing slash and get just the subfolder name
    subfolder=$(basename "$design_dir")
    config_path="./designs/$subfolder/config.mk"
    # Check if config.mk exists in this subfolder
    if [ -f "$config_path" ]; then
        echo "Building design: $subfolder"
        # Clean first with the specific config
        echo "Cleaning previous build..."
        make clean_all DESIGN_CONFIG="$config_path"
        # Run make with the specific config
        if make DESIGN_CONFIG="$config_path"; then
            echo "✓ Successfully built $subfolder"
            ((success_count++))
        else
            echo "✗ Failed to build $subfolder"
            ((fail_count++))
            failed_designs+=("$subfolder")
        fi
        echo "----------------------------------------"
    else
        echo "Skipping $subfolder (no config.mk found)"
        echo "----------------------------------------"
    fi
done
# Print summary
echo "Build Summary:"
echo "  - Total designs processed: $((success_count + fail_count))"
echo "  - Successfully built: $success_count"
echo "  - Failed to build: $fail_count"
# List failed designs if any
if [ $fail_count -gt 0 ]; then
    echo "Failed designs:"
    for design in "${failed_designs[@]}"; do
        echo "  - $design"
    done
fi
exit 0