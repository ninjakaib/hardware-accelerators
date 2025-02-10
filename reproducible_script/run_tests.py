#!/usr/bin/env python3
import subprocess
import sys

def run_tests():
    """Run all tests in the project"""
    print("Running hardware-accelerators tests...")
    
    # Run pytest with verbose output and show test progress
    cmd = [sys.executable, "-m", "pytest", 
           "-v",                     # verbose output
           "--capture=no",          # show print statements
           "--tb=short",           # shorter traceback format
           "tests/"]               # test directory
    
    try:
        subprocess.run(cmd, check=True)
        print("\nAll tests completed successfully! ✅")
    except subprocess.CalledProcessError as e:
        print("\nSome tests failed! ❌")
        sys.exit(1)

if __name__ == "__main__":
    run_tests() 
    