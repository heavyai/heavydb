#!/usr/bin/env bash

set -ex

this_dir=$(dirname "${BASH_SOURCE[0]}")
RECIPE_DIR=${RECIPE_DIR:-${this_dir}}

# Omnisci UDF support uses CLangTool for parsing Load-time UDF C++
# code to AST. If the C++ code uses C++ std headers, we need to
# specify the locations of include directories:
. ${RECIPE_DIR}/get_cxx_include_path.sh
export CPLUS_INCLUDE_PATH=$(get_cxx_include_path)

mkdir -p build
cd build

make sanity_tests

