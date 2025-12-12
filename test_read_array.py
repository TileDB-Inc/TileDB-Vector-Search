#!/usr/bin/env python3
"""
Read a toy TileDB array created by test_create_array.py.
Use this to test if different TileDB-Py versions can read arrays created by other versions.

Usage:
    python test_read_array.py
"""

import numpy as np
import tiledb
import os

# Array URI
array_uri = "test_toy_array"

def read_data():
    """Read data from the array."""

    if not os.path.exists(array_uri):
        print(f"ERROR: Array not found at: {array_uri}")
        print("Run test_create_array.py first to create the array")
        return False

    try:
        with tiledb.open(array_uri, 'r') as A:
            # Check schema
            print(f"✓ Array schema:")
            print(f"  Domain: {A.schema.domain}")
            print(f"  Attributes: {[attr.name for attr in A.schema]}")
            print(f"  Cell order: {A.schema.cell_order}")
            print(f"  Tile order: {A.schema.tile_order}")

            # Check nonempty domain
            ned = A.nonempty_domain()
            print(f"\n✓ Nonempty domain: {ned}")

            if ned is None:
                print("  WARNING: Array is empty!")
                return False

            # Read using nonempty domain
            slices = [slice(*x) for x in ned]
            print(f"  Slices: {slices}")

            data = A[tuple(slices)]
            print(f"\n✓ Read data successfully:")
            for key, value in data.items():
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                if len(value.shape) == 2:
                    print(f"    First row: {value[0]}")
                    print(f"    Last row:  {value[-1]}")
                else:
                    print(f"    Values: {value}")

        return True

    except Exception as e:
        print(f"\n✗ ERROR reading array:")
        print(f"  {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 80)
    print(f"READ ARRAY (TileDB version: {tiledb.version()})")
    print("=" * 80)
    print()

    # Read data
    print("Reading array...")
    print()

    success = read_data()

    print()
    print("=" * 80)
    if success:
        print("SUCCESS: Array read successfully!")
    else:
        print("FAILURE: Could not read array")
    print("=" * 80)


if __name__ == "__main__":
    main()
