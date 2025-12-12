#!/usr/bin/env python3
"""
Clean up the test array created by test_create_array.py.

Usage:
    python test_cleanup_array.py
"""

import shutil
import os

# Array URI
array_uri = "test_toy_array"

def cleanup():
    """Remove the test array."""
    if os.path.exists(array_uri):
        shutil.rmtree(array_uri)
        print(f"✓ Cleaned up: {array_uri}")
    else:
        print(f"  Array not found: {array_uri}")


def main():
    print("=" * 80)
    print("CLEANUP TEST ARRAY")
    print("=" * 80)
    print()

    cleanup()

    print()
    print("=" * 80)
    print("Done")
    print("=" * 80)


if __name__ == "__main__":
    main()
