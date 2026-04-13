#!/usr/bin/env python3
"""Test sandbox: Python availability, pip install, and package usage."""
import sys
import subprocess

def main():
    print("=== 1. Python ===")
    print(f"Python: {sys.executable}")
    print(f"Version: {sys.version}")
    assert sys.version_info >= (3, 6), "Need Python 3.6+"

    print("\n=== 2. Pip install (simple package: six) ===")
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "six"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if r.returncode != 0:
        print("stderr:", r.stderr)
        print("stdout:", r.stdout)
        raise SystemExit(f"pip install failed: {r.returncode}")
    print("pip install six: OK")

    print("\n=== 3. Use installed package ===")
    import six
    assert six.PY3
    print(f"six.PY3 = {six.PY3}, six.__version__ = {six.__version__}")
    print("Import and use six: OK")

    print("\n=== Sandbox test passed ===")
    return 0

if __name__ == "__main__":
    sys.exit(main())
