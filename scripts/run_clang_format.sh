#!/bin/bash

# Runs clang format in the given directory
# Arguments:
#   $1 - Path to the source tree
#   $2 - Path to the clang format binary
#   $3 - Apply fixes (will raise an error if false and there are changes)
#   $ARGN - Files to run clang format on

SOURCE_DIR=$1
shift
CLANG_FORMAT=$1
shift
APPLY_FIXES=$1
shift

echo "Running clang-format version:" `$CLANG_FORMAT --version`

# clang format will only find its configuration if we are in
# the source tree or in a path relative to the source tree
pushd $SOURCE_DIR

src=$SOURCE_DIR
SOURCE_PATHS=($src/tiledb $src/test $src/examples $src/tools $src/experimental)
FIND_FILES=(-name "*.cc" -or -name "*.c" -or -name "*.h")

if [ "$APPLY_FIXES" == "1" ]; then
  find "${SOURCE_PATHS[@]}" \( "${FIND_FILES[@]}" \) -print0 | xargs -0 -P8 $CLANG_FORMAT -i

else
  NUM_CORRECTIONS=`find "${SOURCE_PATHS[@]}" \( "${FIND_FILES[@]}" \) -print0 | xargs -0 -P8 $CLANG_FORMAT -output-replacements-xml | grep offset | wc -l`

  if [ "$NUM_CORRECTIONS" -gt "0" ]; then
    echo "clang-format suggested changes, please run 'make format'!!!!"

    # If running on CI, print out the change-set
    if [ "$CI" = true ]; then
      echo "-------- see list of files to update below --------"
      find "${SOURCE_PATHS[@]}" "${FIND_FILES[@]}" -print0 | xargs -P8 -0 $CLANG_FORMAT -i
      git diff --name-only

      echo "-------- see diff below --------"
      git diff
    fi

    # Fail the job
    exit 1
  fi
fi
popd
