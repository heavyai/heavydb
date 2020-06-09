#!/usr/bin/env bash

set -ex
[ -z "$PREFIX" ] && export PREFIX=$CONDA_PREFIX

# Make sure -fPIC is not in CXXFLAGS (that some conda packages may
# add):
export CXXFLAGS="`echo $CXXFLAGS | sed 's/-fPIC//'`"

# go overwrites CC and CXX with nonsense (see
# https://github.com/conda-forge/go-feedstock/issues/47), hence we
# redefine these below. Reset GO env variables for omniscidb build
# (IIRC, it is needed for CUDA support):
export CGO_ENABLED=1
export CGO_LDFLAGS=
export CGO_CFLAGS=$CFLAGS
export CGO_CPPFLAGS=

if [ $(uname) == Darwin ]; then
    # Darwin has only clang, must use clang++ from clangdev
    # All these must be picked up from $PREFIX/bin
    export CC=clang
    export CXX=clang++
    export CMAKE_CC=clang
    export CMAKE_CXX=clang++
    export MACOSX_DEPLOYMENT_TARGET=10.12
else
    # Linux
    echo "uname=${uname}"
    # must use gcc compiler as llvmdev is built with gcc and there
    # exists ABI incompatibility between llvmdev-7 built with gcc and
    # clang.
    COMPILERNAME=gcc                      # options: clang, gcc

    if [ "$COMPILERNAME" == "clang" ]; then
        # All these must be picked up from $PREFIX/bin
        export CC=clang
        export CXX=clang++
        export CMAKE_CC=clang
        export CMAKE_CXX=clang++
    else
        export CC=$HOST-gcc
        export CXX=$HOST-g++
        export CMAKE_CC=$HOST-gcc
        export CMAKE_CXX=$HOST-g++
    fi

    GXX=$HOST-g++         # replace with $GXX
    GCCVERSION=$(basename $(dirname $($GXX -print-libgcc-file-name)))
fi

set CMAKE_COMPILERS="-DCMAKE_C_COMPILER=$CMAKE_CC -DCMAKE_CXX_COMPILER=$CMAKE_CXX"

cmake -S . -B build -Wno-dev \
    -DCMAKE_INSTALL_PREFIX="$PREFIX"\
    -DCMAKE_C_COMPILER=$CMAKE_CC\
    -DCMAKE_CXX_COMPILER=$CMAKE_CXX\
    -DCMAKE_BUILD_TYPE=release\
    -DMAPD_DOCS_DOWNLOAD=OFF\
    -DENABLE_AWS_S3=OFF\
    -DENABLE_CUDA=OFF\
    -DENABLE_FOLLY=OFF\
    -DENABLE_JAVA_REMOTE_DEBUG=OFF\
    -DENABLE_PROFILER=OFF\
    -DENABLE_TESTS=OFF\
    -DPREFER_STATIC_LIBS=OFF\
    -DENABLE_FSI=ON\
    -DENABLE_TBB=ON\
    -DENABLE_DBE=ON\
    || exit 1

pushd build
# preventing races by executing vulnerable targets first
make PatchParser PatchScanner thrift_gen -j && make -j || \
     make --trace  # running sequentualy and enabling trace in case of failures
make install || exit 1

# copy initdb to mapd_initdb to avoid conflict with psql initdb
mv $PREFIX/bin/initdb $PREFIX/bin/omnisci_initdb
cd ..
rm -rf data
mkdir data
# do lightweight testing here, make sanity_tests should go to under test env
omnisci_initdb -f data
omnisci_server --enable-fsi --db-query-list SampleData/db-query-list-flights.sql --exit-after-warmup
rm -rf data
popd
