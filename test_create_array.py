#!/usr/bin/env python3
"""
Create and write a toy TileDB array with similar schema to backwards-compatibility-data.
Use this to test if different TileDB-Py versions can read arrays created by other versions.

Usage:
    python test_create_array.py
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
            domain=(0, 9),  # 10 rows
            tile=10,
            dtype='int32',
            filters=tiledb.FilterList([tiledb.ZstdFilter(level=-1)])
        ),
        tiledb.Dim(
            name='cols',
            domain=(0, 9),  # 10 cols
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
    print(f"  First row: {data[0]}")
    print(f"  Last row:  {data[-1]}")


def main():
    print("=" * 80)
    print(f"CREATE ARRAY (TileDB version: {tiledb.version()})")
    print("=" * 80)
    print()

    # Create array
    print("Step 1: Creating array...")
    create_array()
    print()

    # Write data
    print("Step 2: Writing data...")
    write_data()
    print()

    print("=" * 80)
    print(f"SUCCESS: Array created at: {array_uri}")
    print("Run test_read_array.py to test reading it with another version")
    print("=" * 80)


if __name__ == "__main__":
    main()
