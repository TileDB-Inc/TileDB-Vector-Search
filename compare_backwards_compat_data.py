#!/usr/bin/env python3
"""
Compare two exports of backwards-compatibility-data.

This script compares NumPy exports created by export_backwards_compat_data.py
to verify that different versions of tiledb-py read the same data identically.

Usage:
    python compare_backwards_compat_data.py output_folder_v1 output_folder_v2
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
try:
    with tiledb.open(uri, 'r') as A:
        print(f"Schema: {A.schema}")
        ned = A.nonempty_domain()
        print(f"Nonempty domain: {ned}")

        if ned:
            slices = [slice(*x) for x in ned]
            data = A[tuple(slices)]
            print(f"Success! Data keys: {list(data.keys())}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()




class ComparisonResult:
    """Stores the results of a comparison."""

    def __init__(self):
        self.identical_count = 0
        self.different_count = 0
        self.missing_in_dir1 = []
        self.missing_in_dir2 = []
        self.differences = []

    def add_difference(self, path: str, reason: str, details: Any = None):
        """Add a difference to the results."""
        self.different_count += 1
        self.differences.append({
            "path": path,
            "reason": reason,
            "details": details
        })

    def add_identical(self):
        """Increment the count of identical items."""
        self.identical_count += 1

    def print_summary(self):
        """Print a summary of the comparison."""
        print("\n" + "=" * 80)
        print("COMPARISON SUMMARY")
        print("=" * 80)
        print(f"Identical items: {self.identical_count}")
        print(f"Different items: {self.different_count}")
        print(f"Missing in dir1: {len(self.missing_in_dir1)}")
        print(f"Missing in dir2: {len(self.missing_in_dir2)}")
        print("=" * 80)

        if self.missing_in_dir1:
            print("\nMissing in first directory:")
            for item in self.missing_in_dir1:
                print(f"  - {item}")

        if self.missing_in_dir2:
            print("\nMissing in second directory:")
            for item in self.missing_in_dir2:
                print(f"  - {item}")

        if self.differences:
            print("\nDifferences found:")
            for diff in self.differences:
                print(f"\n  Path: {diff['path']}")
                print(f"  Reason: {diff['reason']}")
                if diff['details']:
                    print(f"  Details: {diff['details']}")

        print("\n" + "=" * 80)
        if self.different_count == 0 and not self.missing_in_dir1 and not self.missing_in_dir2:
            print("SUCCESS: All data is identical!")
        else:
            print("FAILURE: Differences detected!")
        print("=" * 80 + "\n")

        return self.different_count == 0 and not self.missing_in_dir1 and not self.missing_in_dir2


def compare_json_files(file1: Path, file2: Path, result: ComparisonResult, rel_path: str) -> bool:
    """
    Compare two JSON files.

    Returns True if identical, False otherwise.
    """
    try:
        with open(file1, "r") as f:
            data1 = json.load(f)
        with open(file2, "r") as f:
            data2 = json.load(f)

        if data1 == data2:
            result.add_identical()
            return True
        else:
            # Find specific differences
            diff_keys = []
            all_keys = set(data1.keys()) | set(data2.keys())
            for key in all_keys:
                if key not in data1:
                    diff_keys.append(f"'{key}' missing in dir1")
                elif key not in data2:
                    diff_keys.append(f"'{key}' missing in dir2")
                elif data1[key] != data2[key]:
                    diff_keys.append(f"'{key}': {data1[key]} != {data2[key]}")

            result.add_difference(
                rel_path,
                "JSON content differs",
                "; ".join(diff_keys)
            )
            return False

    except Exception as e:
        result.add_difference(rel_path, f"Error comparing JSON: {e}")
        return False


def compare_numpy_files(file1: Path, file2: Path, result: ComparisonResult, rel_path: str) -> bool:
    """
    Compare two NumPy files.

    Returns True if arrays are equal, False otherwise.
    """
    try:
        arr1 = np.load(file1, allow_pickle=True)
        arr2 = np.load(file2, allow_pickle=True)

        # Check shapes
        if arr1.shape != arr2.shape:
            result.add_difference(
                rel_path,
                "Array shape mismatch",
                f"{arr1.shape} != {arr2.shape}"
            )
            return False

        # Check dtypes
        if arr1.dtype != arr2.dtype:
            result.add_difference(
                rel_path,
                "Array dtype mismatch",
                f"{arr1.dtype} != {arr2.dtype}"
            )
            return False

        # Check values
        try:
            if np.array_equal(arr1, arr2, equal_nan=True):
                result.add_identical()
                return True
            else:
                # Calculate statistics about differences
                if np.issubdtype(arr1.dtype, np.number):
                    diff = np.abs(arr1 - arr2)
                    max_diff = np.max(diff)
                    mean_diff = np.mean(diff)
                    num_different = np.sum(arr1 != arr2)

                    result.add_difference(
                        rel_path,
                        "Array values differ",
                        f"max_diff={max_diff}, mean_diff={mean_diff}, "
                        f"num_different={num_different}/{arr1.size}"
                    )
                else:
                    num_different = np.sum(arr1 != arr2)
                    result.add_difference(
                        rel_path,
                        "Array values differ",
                        f"num_different={num_different}/{arr1.size}"
                    )
                return False

        except Exception as e:
            # For complex objects, try direct comparison
            if arr1.dtype == object or arr2.dtype == object:
                if all(np.array_equal(a, b, equal_nan=True) if isinstance(a, np.ndarray) else a == b
                       for a, b in zip(arr1.flat, arr2.flat)):
                    result.add_identical()
                    return True
                else:
                    result.add_difference(rel_path, "Array object values differ")
                    return False
            else:
                raise

    except Exception as e:
        result.add_difference(rel_path, f"Error comparing NumPy arrays: {e}")
        return False


def compare_directories(dir1: Path, dir2: Path, result: ComparisonResult, rel_path: str = "") -> None:
    """
    Recursively compare two directories.

    Args:
        dir1: First directory to compare
        dir2: Second directory to compare
        result: ComparisonResult object to store results
        rel_path: Relative path for reporting
    """
    # Get all files and subdirectories
    items1 = set(p.name for p in dir1.iterdir())
    items2 = set(p.name for p in dir2.iterdir())

    # Find missing items
    missing_in_dir2 = items1 - items2
    missing_in_dir1 = items2 - items1

    for item in missing_in_dir2:
        result.missing_in_dir2.append(f"{rel_path}/{item}" if rel_path else item)

    for item in missing_in_dir1:
        result.missing_in_dir1.append(f"{rel_path}/{item}" if rel_path else item)

    # Compare common items
    common_items = items1 & items2

    for item in sorted(common_items):
        path1 = dir1 / item
        path2 = dir2 / item
        item_rel_path = f"{rel_path}/{item}" if rel_path else item

        if path1.is_dir() and path2.is_dir():
            # Recursively compare directories
            compare_directories(path1, path2, result, item_rel_path)

        elif path1.is_file() and path2.is_file():
            # Skip export_info.json - versions and timestamps are expected to differ
            if item == "export_info.json":
                continue

            # Compare files based on extension
            if item.endswith(".json"):
                compare_json_files(path1, path2, result, item_rel_path)
            elif item.endswith(".npy") or item.endswith(".pkl"):
                compare_numpy_files(path1, path2, result, item_rel_path)
            elif item.endswith(".txt"):
                # Text file comparison
                try:
                    with open(path1, "r") as f:
                        content1 = f.read()
                    with open(path2, "r") as f:
                        content2 = f.read()

                    if content1 == content2:
                        result.add_identical()
                    else:
                        result.add_difference(item_rel_path, "Text content differs")
                except Exception as e:
                    result.add_difference(item_rel_path, f"Error comparing text: {e}")
            else:
                # Binary comparison for other files
                try:
                    with open(path1, "rb") as f:
                        content1 = f.read()
                    with open(path2, "rb") as f:
                        content2 = f.read()

                    if content1 == content2:
                        result.add_identical()
                    else:
                        result.add_difference(item_rel_path, "Binary content differs")
                except Exception as e:
                    result.add_difference(item_rel_path, f"Error comparing binary: {e}")

        else:
            # Type mismatch (one is file, other is directory)
            result.add_difference(
                item_rel_path,
                "Type mismatch",
                f"dir1={'dir' if path1.is_dir() else 'file'}, "
                f"dir2={'dir' if path2.is_dir() else 'file'}"
            )


def compare_exports(dir1: str, dir2: str) -> bool:
    """
    Compare two export directories.

    Args:
        dir1: First export directory
        dir2: Second export directory

    Returns:
        True if identical, False if differences found
    """
    dir1_path = Path(dir1)
    dir2_path = Path(dir2)

    if not dir1_path.exists():
        print(f"Error: Directory not found: {dir1_path}")
        sys.exit(1)

    if not dir2_path.exists():
        print(f"Error: Directory not found: {dir2_path}")
        sys.exit(1)

    print("Comparing backwards-compatibility-data exports")
    print(f"  Directory 1: {dir1_path}")
    print(f"  Directory 2: {dir2_path}")
    print()

    # Load and display export info (for informational purposes only)
    print("Export metadata (not compared):")
    for i, d in enumerate([dir1_path, dir2_path], 1):
        info_file = d / "export_info.json"
        if info_file.exists():
            with open(info_file, "r") as f:
                info = json.load(f)
            print(f"  Directory {i}:")
            for key, value in info.items():
                print(f"    {key}: {value}")
    print()

    # Perform comparison
    result = ComparisonResult()
    compare_directories(dir1_path, dir2_path, result)

    # Print results
    return result.print_summary()


def main():
    parser = argparse.ArgumentParser(
        description="Compare two exports of backwards-compatibility-data"
    )
    parser.add_argument(
        "dir1",
        help="First export directory"
    )
    parser.add_argument(
        "dir2",
        help="Second export directory"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    success = compare_exports(args.dir1, args.dir2)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
