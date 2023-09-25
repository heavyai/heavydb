#!/usr/bin/env bash

set -e
set -x

SUFFIX=${SUFFIX:=$(date +%Y%m%d)}
PREFIX=/usr/local/mapd-deps

SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $SCRIPTS_DIR/common-functions.sh
source /etc/os-release

sudo mkdir -p $PREFIX
sudo chown -R $(id -u) $PREFIX

sudo apt update
sudo apt install -y \
    software-properties-common \
    build-essential \
    libtool \
    ccache \
    git \
    wget \
    curl \
    libgoogle-glog-dev \
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
    flex \
    libpng-dev \
    rsync \
    unzip \
    jq \
    python-dev \
    python-yaml \
    pkg-config \
    swig \
    patchelf

# Install gcc 8
sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y
sudo apt update
sudo apt install -y g++-8
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 80 \
                         --slave /usr/bin/g++ g++ /usr/bin/g++-8
sudo update-alternatives --config gcc

# Needed to find sqlite3, xmltooling, and xml_security_c
export PKG_CONFIG_PATH=$PREFIX/lib/pkgconfig:$PREFIX/lib64/pkgconfig:$PKG_CONFIG_PATH
export PATH=$PREFIX/bin:$PATH

install_ninja

install_cmake

# c-blosc
install_blosc

# Geo Support
install_gdal

VERS=1_72_0
# http://downloads.sourceforge.net/project/boost/boost/${VERS//_/.}/boost_$VERS.tar.bz2
download ${HTTP_DEPS}/boost_$VERS.tar.bz2
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

# thrift
install_thrift

install_folly

install_iwyu

download_make_install ${HTTP_DEPS}/bisonpp-1.21-45.tar.gz bison++-1.21

# TBB
install_tbb

# Apache Arrow (see common-functions.sh)
ARROW_BOOST_USE_SHARED="ON"
install_arrow

# Go
install_go

# librdkafka
install_rdkafka

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

if [ "$1" = "--compress" ] ; then
    tar acf mapd-deps-ubuntu-$VERSION_ID-$SUFFIX.tar.xz -C $PREFIX .
fi
