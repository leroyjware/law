#!/usr/bin/env python3
"""Run tests from any directory. Usage: python run_tests.py"""
import os
import subprocess
import sys

# Find project root (directory containing src/ and tests/)
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Ensure project root is on path
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/test_topologies.py", "-v"],
    cwd=script_dir,
)
sys.exit(result.returncode)
