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
    software-properties-common \
    build-essential \
    libtool \
    ccache \
    cmake \
    cmake-curses-gui \
    git \
    wget \
    curl \
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
    libedit-dev \
    uuid-dev \
    libsnappy-dev \
    zlib1g-dev \
    autoconf \
    autoconf-archive \
    automake \
    bison \
    flex-old \
    libpng-dev \
    rsync \
    unzip \
    jq \
    python-dev \
    python-yaml \
    swig

# Install gcc 6
sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y
sudo apt update
sudo apt install -y g++-6
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-6 60 \
                         --slave /usr/bin/g++ g++ /usr/bin/g++-6
sudo update-alternatives --config gcc

# Needed to find xmltooling and xml_security_c
export PKG_CONFIG_PATH=$PREFIX/lib/pkgconfig:$PREFIX/lib64/pkgconfig:$PKG_CONFIG_PATH

# GEO STUFF
# expat
download_make_install https://github.com/libexpat/libexpat/releases/download/R_2_2_5/expat-2.2.5.tar.bz2
# kml
download ${HTTP_DEPS}/libkml-master.zip
unzip -u libkml-master.zip
pushd libkml-master
./autogen.sh || true
CXXFLAGS="-std=c++03" ./configure --with-expat-include-dir=$PREFIX/include/ --with-expat-lib-dir=$PREFIX/lib --prefix=$PREFIX --disable-java --disable-python --disable-swig
makej
make install
popd
# proj.4
download_make_install ${HTTP_DEPS}/proj-5.2.0.tar.gz
# gdal
download_make_install ${HTTP_DEPS}/gdal-2.3.2.tar.xz "" "--without-geos --with-libkml=$PREFIX --with-proj=$PREFIX"


VERS=1_67_0
# http://downloads.sourceforge.net/project/boost/boost/${VERS//_/.}/boost_$VERS.tar.bz2
download https://internal-dependencies.mapd.com/thirdparty/boost_$VERS.tar.bz2
extract boost_$VERS.tar.bz2
pushd boost_$VERS
./bootstrap.sh --prefix=$PREFIX
./b2 cxxflags=-fPIC install --prefix=$PREFIX || true
popd
 # Needed for folly to find boost
export LIBRARY_PATH=$PREFIX/lib:$LIBRARY_PATH

# llvm
# (see common-functions.sh)
install_llvm

# install AWS core and s3 sdk
install_awscpp -j $(nproc)

VERS=0.11.0
wget --continue http://apache.claz.org/thrift/$VERS/thrift-$VERS.tar.gz
tar xvf thrift-$VERS.tar.gz
pushd thrift-$VERS
CFLAGS="-fPIC" CXXFLAGS="-fPIC" JAVA_PREFIX=$PREFIX/lib ./configure \
    --with-lua=no \
    --with-python=no \
    --with-php=no \
    --with-ruby=no \
    --with-qt4=no \
    --with-qt5=no \
    --prefix=$PREFIX \
    --with-boost=$PREFIX
make -j $(nproc)
make install
popd

# c-blosc
VERS=1.14.4
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

VERS=2018.05.07.00
wget --continue https://github.com/facebook/folly/archive/v$VERS.tar.gz
tar xvf v$VERS.tar.gz
pushd folly-$VERS/folly
/usr/bin/autoreconf -ivf
./configure --prefix=$PREFIX --with-boost=$PREFIX/
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

VERS=1.11
ARCH=$(uname -m)
ARCH=${ARCH//x86_64/amd64}
ARCH=${ARCH//aarch64/arm64}
# https://dl.google.com/go/go$VERS.linux-$ARCH.tar.gz
download ${HTTP_DEPS}/go$VERS.linux-$ARCH.tar.gz
extract go$VERS.linux-$ARCH.tar.gz
rm -rf $PREFIX/go || true
mv go $PREFIX

VERS=3.0.2
wget --continue https://github.com/cginternals/glbinding/archive/v$VERS.tar.gz
tar xvf v$VERS.tar.gz
mkdir -p glbinding-$VERS/build
pushd glbinding-$VERS/build
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=$PREFIX \
    -DOPTION_BUILD_DOCS=OFF \
    -DOPTION_BUILD_EXAMPLES=OFF \
    -DOPTION_BUILD_TESTS=OFF \
    -DOPTION_BUILD_TOOLS=OFF \
    -DOPTION_BUILD_WITH_BOOST_THREAD=OFF \
    ..
make -j $(nproc)
make install
popd

# OpenSAML
VERS=3.2.2
download_make_install ${HTTP_DEPS}/xerces-c-3.2.2.tar.gz
download_make_install ${HTTP_DEPS}/xml-security-c-2.0.2.tar.gz "" "--without-xalan"
download_make_install ${HTTP_DEPS}/xmltooling-3.0.4-nolog4shib.tar.gz "" "--with-boost=$PREFIX"
download_make_install ${HTTP_DEPS}/opensaml-3.0.1-nolog4shib.tar.gz "" "--with-boost=$PREFIX"

cat > $PREFIX/mapd-deps.sh <<EOF
PREFIX=$PREFIX

LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH
LD_LIBRARY_PATH=\$PREFIX/lib:\$LD_LIBRARY_PATH
LD_LIBRARY_PATH=\$PREFIX/lib64:\$LD_LIBRARY_PATH

PATH=/usr/local/cuda/bin:\$PATH
PATH=\$PREFIX/go/bin:\$PATH
PATH=\$PREFIX/bin:\$PATH

CMAKE_PREFIX_PATH=\$PREFIX:\$CMAKE_PREFIX_PATH

GOROOT=\$PREFIX/go

export LD_LIBRARY_PATH PATH CMAKE_PREFIX_PATH GOROOT
EOF

echo
echo "Done. Be sure to source the 'mapd-deps.sh' file to pick up the required environment variables:"
echo "    source $PREFIX/mapd-deps.sh"
