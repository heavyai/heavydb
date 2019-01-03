#!/usr/bin/env bash

set -e
set -x

cd "$( dirname "${BASH_SOURCE[0]}" )"

hash yay || { echo >&2 "yay is required but is not installed. Aborting."; exit 1; }

# Install all normal dependencies
yay -S \
    blosc \
    boost \
    clang \
    cmake \
    cuda \
    doxygen \
    gcc \
    gdal \
    git \
    glbinding \
    glslang \
    go \
    google-glog \
    jdk-openjdk \
    llvm \
    lz4 \
    maven \
    python-numpy \
    snappy \
    thrift \
    wget \
    zlib

# Install Arrow
pushd arch/arrow
makepkg -si
popd

# Install Bison++ from source
wget --continue https://dependencies.mapd.com/thirdparty/bisonpp-1.21-45.tar.gz
tar xvf bisonpp-1.21-45.tar.gz
pushd bison++-1.21
./configure
make -j $(nproc)
sudo make install
popd
