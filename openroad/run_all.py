#!/usr/bin/env python3
import os
import subprocess
import sys


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Step 1: Run the organize_verilog.py script
    print("Step 1: Organizing Verilog files...")
    organize_script = os.path.join(script_dir, "organize_verilog.py")

    try:
        result = subprocess.run(
            ["python", organize_script],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        print(result.stdout)
        if result.stderr:
            print("Warnings/Errors:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error running organize_verilog.py: {e}")
        print(e.stdout)
        print(e.stderr)
        return 1

    # Step 2: Run the copy_to_openroad.py script
    print("\nStep 2: Copying files to OpenROAD-flow-scripts...")
    copy_script = os.path.join(script_dir, "copy_to_openroad.py")

    try:
        result = subprocess.run(
            ["python", copy_script],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        print(result.stdout)
        if result.stderr:
            print("Warnings/Errors:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error running copy_to_openroad.py: {e}")
        print(e.stdout)
        print(e.stderr)
        return 1

    print("\nAll operations completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
