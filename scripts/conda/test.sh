#!/bin/bash

set -ex

cd $(dirname "$0")

# Omnisci UDF support uses CLangTool for parsing Load-time UDF C++
# code to AST. If the C++ code uses C++ std headers, we need to
# specify the locations of include directories
export CXX=g++
. ./get_cxx_include_path.sh
export CPLUS_INCLUDE_PATH=$(get_cxx_include_path)

mkdir -p ../../build
cd ../../build

make sanity_tests

