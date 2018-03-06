#!/usr/bin/env bash

set -e
set -x

PREFIX=/usr/local/mapd-deps

SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $SCRIPTS_DIR/common-functions.sh

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
    liblz4-dev \
    liblzma-dev \
    libbz2-dev \
    libarchive-dev \
    libcurl4-openssl-dev \
    uuid-dev \
    libsnappy-dev \
    zlib1g-dev \
    autoconf \
    autoconf-archive \
    automake \
    bison \
    flex-old

# install AWS core and s3 sdk
install_awscpp -j $(nproc)

VERS=0.10.0
wget --continue http://apache.claz.org/thrift/$VERS/thrift-$VERS.tar.gz
tar xvf thrift-$VERS.tar.gz
pushd thrift-$VERS
patch -p1 < $SCRIPTS_DIR/thrift-3821-tmemorybuffer-overflow-check.patch
patch -p1 < $SCRIPTS_DIR/thrift-3821-tmemorybuffer-overflow-test.patch
JAVA_PREFIX=$PREFIX/lib ./configure \
    --with-lua=no \
    --with-python=no \
    --with-php=no \
    --with-ruby=no \
    --with-qt4=no \
    --with-qt5=no \
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

VERS=2017.10.16.00
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

# Apache Arrow (see common-functions.sh)
ARROW_BOOST_USE_SHARED="ON"
install_arrow

VERS=2.1.4_egl
wget --continue https://github.com/vastcharade/glbinding/archive/v$VERS.tar.gz
tar xvf v$VERS.tar.gz
mkdir -p glbinding-$VERS/build
pushd glbinding-$VERS/build
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=$PREFIX \
    -DOPTION_BUILD_DOCS=OFF \
    -DOPTION_BUILD_EXAMPLES=OFF \
    -DOPTION_BUILD_GPU_TESTS=OFF \
    -DOPTION_BUILD_TESTS=OFF \
    -DOPTION_BUILD_TOOLS=OFF \
    -DOPTION_BUILD_WITH_BOOST_THREAD=OFF \
    ..
make -j $(nproc)
make install
popd

cat > $PREFIX/mapd-deps.sh <<EOF
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
