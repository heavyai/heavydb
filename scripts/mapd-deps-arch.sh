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
# Package cannot be built in a path that incudes "internal" as a substring.
ARROW_PKG_DIR=$HOME/omnisci_tmp_arrow
mkdir -p $ARROW_PKG_DIR
cp arch/arrow/PKGBUILD $ARROW_PKG_DIR
pushd $ARROW_PKG_DIR
makepkg -cis
rm -f PKGBUILD
popd
mv $ARROW_PKG_DIR/{apache-arrow-*.tar.gz,arrow-*.pkg.tar.xz} arch/arrow/
rmdir $ARROW_PKG_DIR

# Install SPIRV-Cross
pushd arch/spirv-cross
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
