#!/usr/bin/env bash

set -e
set -x

cd "$( dirname "${BASH_SOURCE[0]}" )"

hash yay || { echo >&2 "yay is required but is not installed. Aborting."; exit 1; }

unset CMAKE_GENERATOR

# Install all normal dependencies
yay -S \
    aws-sdk-cpp \
    blosc \
    boost \
    clang \
    cmake \
    cuda \
    doxygen \
    gcc \
    gdal \
    geos \
    git \
    glbinding \
    go \
    google-glog \
    intel-tbb \
    jdk-openjdk \
    librdkafka \
    llvm \
    lz4 \
    maven \
    ninja \
    python-numpy \
    snappy \
    thrift \
    vulkan-headers \
    wget \
    zlib

# Install Arrow
pushd arch/arrow
makepkg -cis
popd

# Install SPIRV-Cross
pushd arch/spirv-cross
makepkg -cis
popd

# Install glslang
pushd arch/glslang
makepkg -cis
popd

# Install Bison++ from source
wget --continue https://dependencies.mapd.com/thirdparty/bisonpp-1.21-45.tar.gz
tar xvf bisonpp-1.21-45.tar.gz
pushd bison++-1.21
./configure
make -j $(nproc)
sudo make install
popd
