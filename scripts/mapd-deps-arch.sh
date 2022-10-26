#!/usr/bin/env bash

# Must be run from the scripts/ directory as the non-root user.
# Since we use an older version of Apache Arrow, automatic updates to arrow can be avoided by
# adding it to the uncommented IgnorePkg line in /etc/pacman.conf. Example:
# IgnorePkg   = arrow

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
    c-ares \
    clang \
    cmake \
    cuda \
    doxygen \
    flex \
    fmt \
    gcc \
    gdal \
    geos \
    git \
    glslang \
    go \
    google-glog \
    intel-tbb \
    jdk-openjdk \
    libiodbc \
    librdkafka \
    libuv \
    llvm \
    lz4 \
    maven \
    ninja \
    python-numpy \
    snappy \
    spirv-cross \
    thrift \
    vulkan-headers \
    wget \
    zlib

# Install Arrow
pushd arch/arrow
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
