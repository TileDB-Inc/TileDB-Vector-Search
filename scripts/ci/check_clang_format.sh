#!/bin/bash

# Check test formatting. Should only be run on linux.

set -e pipefail

# Install clang-format.
ls -la
sudo ./scripts/install_clang_format.sh

# Run clang-format.
src=$GITHUB_WORKSPACE
cd $src
$src/scripts/run_clang_format.sh $src clang-format-17 0
