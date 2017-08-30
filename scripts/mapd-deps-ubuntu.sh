#!/usr/bin/env bash

set -e
set -x

PREFIX=/usr/local/mapd-deps

sudo mkdir -p $PREFIX
sudo chown -R $(id -u) $PREFIX

sudo apt update
sudo apt install -y \
    build-essential \
    ccache \
    cmake \
    cmake-curses-gui \
    git \
    wget \
    curl \
    clang \
    llvm \
    llvm-dev \
    clang-format \
    gcc-5 \
    g++-5 \
    libboost-all-dev \
    libgoogle-glog-dev \
    golang \
    libssl-dev \
    libevent-dev \
    default-jre \
    default-jre-headless \
    default-jdk \
    default-jdk-headless \
    maven \
    libncurses5-dev \
    libldap2-dev \
    binutils-dev \
    google-perftools \
    libdouble-conversion-dev \
    libevent-dev \
    libgdal-dev \
    libgflags-dev \
    libgoogle-perftools-dev \
    libiberty-dev \
    libjemalloc-dev \
    libglu1-mesa-dev \
    libglewmx-dev \
    liblz4-dev \
    liblzma-dev \
    libsnappy-dev \
    zlib1g-dev \
    autoconf \
    autoconf-archive \
    automake \
    bison \
    flex-old

VERS=0.10.0
wget --continue http://apache.claz.org/thrift/$VERS/thrift-$VERS.tar.gz
tar xvf thrift-$VERS.tar.gz
pushd thrift-$VERS
patch -p1 < ../thrift-3821-tmemorybuffer-overflow-check.patch
patch -p1 < ../thrift-3821-tmemorybuffer-overflow-test.patch
JAVA_PREFIX=$PREFIX/lib ./configure \
    --with-lua=no \
    --with-python=no \
    --with-php=no \
    --with-ruby=no \
    --prefix=$PREFIX
make -j $(nproc)
make install
popd

VERS=1.11.3
wget --continue https://github.com/Blosc/c-blosc/archive/v$VERS.tar.gz
tar xvf v$VERS.tar.gz
BDIR="c-blosc-$VERS/build"
rm -rf "$BDIR"
mkdir -p "$BDIR"
pushd "$BDIR"
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=$PREFIX \
    -DBUILD_BENCHMARKS=off \
    -DBUILD_TESTS=off \
    -DPREFER_EXTERNAL_SNAPPY=off \
    -DPREFER_EXTERNAL_ZLIB=off \
    -DPREFER_EXTERNAL_ZSTD=off \
    ..
make -j $(nproc)
make install
popd

VERS=2017.04.10.00
wget --continue https://github.com/facebook/folly/archive/v$VERS.tar.gz
tar xvf v$VERS.tar.gz
pushd folly-$VERS/folly
/usr/bin/autoreconf -ivf
./configure --prefix=$PREFIX
make -j $(nproc)
make install
popd

VERS=1.21-45
wget --continue https://github.com/jarro2783/bisonpp/archive/$VERS.tar.gz
tar xvf $VERS.tar.gz
pushd bisonpp-$VERS
./configure --prefix=$PREFIX
make -j $(nproc)
make install
popd

VERS=0.4.1
wget --continue https://github.com/apache/arrow/archive/apache-arrow-$VERS.tar.gz
tar -xf apache-arrow-$VERS.tar.gz
mkdir -p arrow-apache-arrow-$VERS/cpp/build
pushd arrow-apache-arrow-$VERS/cpp/build
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DARROW_BUILD_SHARED=off \
    -DARROW_BUILD_STATIC=on \
    -DCMAKE_INSTALL_PREFIX=$PREFIX \
    -DARROW_BOOST_USE_SHARED=off \
    -DARROW_JEMALLOC_USE_SHARED=off \
    ..
make -j $(nproc)
make install
popd

cat >> $PREFIX/mapd-deps.sh <<EOF
PREFIX=$PREFIX

LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH
LD_LIBRARY_PATH=\$PREFIX/lib:\$LD_LIBRARY_PATH
LD_LIBRARY_PATH=\$PREFIX/lib64:\$LD_LIBRARY_PATH

PATH=/usr/local/cuda/bin:\$PATH
PATH=\$PREFIX/bin:\$PATH

export LD_LIBRARY_PATH PATH
EOF

echo
echo "Done. Be sure to source the 'mapd-deps.sh' file to pick up the required environment variables:"
echo "    source $PREFIX/mapd-deps.sh"
