#!/usr/bin/env python3
"""
Create a toy TileDB array with similar schema to backwards-compatibility-data.
This helps test if TileDB-Py can read/write arrays with this schema.
"""

import numpy as np
import tiledb
import shutil
import os

# Array URI
array_uri = "test_toy_array"

def create_array():
    """Create a toy array with similar schema to backwards-compat data."""

    # Clean up if exists
    if os.path.exists(array_uri):
        shutil.rmtree(array_uri)

    # Create schema - same structure but 10x10 instead of 128x2147483647
    dom = tiledb.Domain(
        tiledb.Dim(
            name='rows',
            domain=(0, 9),  # 10 rows instead of 128
            tile=10,
            dtype='int32',
            filters=tiledb.FilterList([tiledb.ZstdFilter(level=-1)])
        ),
        tiledb.Dim(
            name='cols',
            domain=(0, 9),  # 10 cols instead of 2147483647
            tile=10,
            dtype='int32',
            filters=tiledb.FilterList([tiledb.ZstdFilter(level=-1)])
        ),
    )

    attr = tiledb.Attr(
        name='values',
        dtype='float32',
        var=False,
        nullable=False,
        filters=tiledb.FilterList([tiledb.ZstdFilter(level=-1)])
    )

    schema = tiledb.ArraySchema(
        domain=dom,
        attrs=[attr],
        cell_order='col-major',
        tile_order='col-major',
        sparse=False
    )

    # Create the array
    tiledb.Array.create(array_uri, schema)
    print(f"✓ Created array: {array_uri}")
    print(f"  Schema: {schema}")


def write_data():
    """Write some test data to the array."""

    # Create 10x10 data (rows x cols)
    data = np.arange(100, dtype=np.float32).reshape(10, 10)

    with tiledb.open(array_uri, 'w') as A:
        A[:] = data

    print(f"✓ Wrote data shape: {data.shape}")
    print(f"  Data:\n{data}")


def read_data():
    """Read data back from the array."""

    with tiledb.open(array_uri, 'r') as A:
        # Check schema
        print(f"\n✓ Array schema:")
        print(f"  {A.schema}")

        # Check nonempty domain
        ned = A.nonempty_domain()
        print(f"\n✓ Nonempty domain: {ned}")

        if ned is None:
            print("  WARNING: Array is empty!")
            return

        # Read using nonempty domain
        slices = [slice(*x) for x in ned]
        print(f"  Slices: {slices}")

        data = A[tuple(slices)]
        print(f"\n✓ Read data:")
        for key, value in data.items():
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            print(f"  Values:\n{value}")


def cleanup():
    """Remove the test array."""
    if os.path.exists(array_uri):
        shutil.rmtree(array_uri)
        print(f"✓ Cleaned up: {array_uri}")


def main():
    print("=" * 80)
    print(f"Testing TileDB array schema (TileDB version: {tiledb.version()})")
    print("=" * 80)
    print()

    try:
        # Create array
        print("Step 1: Creating array...")
        create_array()
        print()

        # Write data
        print("Step 2: Writing data...")
        write_data()
        print()

        # Read data back
        print("Step 3: Reading data...")
        read_data()
        print()

        print("=" * 80)
        print("SUCCESS: Array created, written, and read successfully!")
        print("=" * 80)

    finally:
        # Always cleanup
        print()
        print("Step 4: Cleaning up...")
        cleanup()


if __name__ == "__main__":
    main()
