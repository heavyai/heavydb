#!/bin/bash

set -e
set -x

SUFFIX=${SUFFIX:=$(date +%Y%m%d)}
PREFIX=${MAPD_PATH:="/usr/local/mapd-deps/$SUFFIX"}
if [ ! -w $(dirname $PREFIX) ] ; then
    SUDO=sudo
fi
$SUDO mkdir -p $PREFIX
$SUDO chown -R $USER $PREFIX

export PATH=$PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$PREFIX/lib64:$PREFIX/lib:$LD_LIBRARY_PATH

# Needed to find xmltooling and xml_security_c
export PKG_CONFIG_PATH=$PREFIX/lib/pkgconfig:$PREFIX/lib64/pkgconfig:$PKG_CONFIG_PATH

SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $SCRIPTS_DIR/common-functions.sh


sudo yum groupinstall -y "Development Tools"
sudo yum install -y \
    zlib-devel \
    epel-release \
    libssh \
    openssl-devel \
    ncurses-devel \
    git \
    maven \
    java-1.8.0-openjdk-devel \
    java-1.8.0-openjdk-headless \
    gperftools \
    gperftools-devel \
    gperftools-libs \
    python-devel \
    wget \
    curl \
    openldap-devel
sudo yum install -y \
    jq

# gmp, mpc, mpfr, autoconf, automake
# note: if gmp fails on POWER8:
# wget https://gmplib.org/repo/gmp/raw-rev/4a6d258b467f
# patch -p1 < 4a6d258b467f
# https://gmplib.org/download/gmp/gmp-6.1.2.tar.xz
download_make_install ${HTTP_DEPS}/gmp-6.1.2.tar.xz "" "--enable-fat"
# http://www.mpfr.org/mpfr-current/mpfr-3.1.5.tar.xz
download_make_install ${HTTP_DEPS}/mpfr-4.0.1.tar.xz "" "--with-gmp=$PREFIX"
download_make_install ftp://ftp.gnu.org/gnu/mpc/mpc-1.1.0.tar.gz "" "--with-gmp=$PREFIX"
download_make_install ftp://ftp.gnu.org/gnu/autoconf/autoconf-2.69.tar.xz # "" "--build=powerpc64le-unknown-linux-gnu"
download_make_install ftp://ftp.gnu.org/gnu/automake/automake-1.16.1.tar.xz

# gcc
VERS=7.4.0
download ftp://ftp.gnu.org/gnu/gcc/gcc-$VERS/gcc-$VERS.tar.xz
extract gcc-$VERS.tar.xz
pushd gcc-$VERS
export CPPFLAGS="-I$PREFIX/include"
./configure \
    --prefix=$PREFIX \
    --disable-multilib \
    --enable-bootstrap \
    --enable-shared \
    --enable-threads=posix \
    --enable-checking=release \
    --with-system-zlib \
    --enable-__cxa_atexit \
    --disable-libunwind-exceptions \
    --enable-gnu-unique-object \
    --enable-languages=c,c++ \
    --with-tune=generic \
    --with-gmp=$PREFIX \
    --with-mpc=$PREFIX \
    --with-mpfr=$PREFIX #replace '--with-tune=generic' with '--with-tune=power8' for POWER8
makej
make install
popd

export CC=$PREFIX/bin/gcc
export CXX=$PREFIX/bin/g++

install_ninja

download_make_install ftp://ftp.gnu.org/gnu/libtool/libtool-2.4.6.tar.gz

# http://zlib.net/zlib-1.2.8.tar.xz
download_make_install ${HTTP_DEPS}/zlib-1.2.8.tar.xz

VERS=1.0.6
# http://bzip.org/$VERS/bzip2-$VERS.tar.gz
download ${HTTP_DEPS}/bzip2-$VERS.tar.gz
extract bzip2-$VERS.tar.gz
pushd bzip2-$VERS
makej
make install PREFIX=$PREFIX
popd

# https://www.openssl.org/source/openssl-1.0.2u.tar.gz
download_make_install ${HTTP_DEPS}/openssl-1.0.2u.tar.gz "" "linux-$(uname -m) no-shared no-dso -fPIC"

# libarchive
download_make_install ${HTTP_DEPS}/xz-5.2.4.tar.xz "" "--disable-shared"
download_make_install ${HTTP_DEPS}/libarchive-3.3.2.tar.gz "" "--without-openssl --disable-shared"

CFLAGS="-fPIC" download_make_install ftp://ftp.gnu.org/pub/gnu/ncurses/ncurses-6.1.tar.gz # "" "--build=powerpc64le-unknown-linux-gnu"

download_make_install ftp://ftp.gnu.org/gnu/bison/bison-3.0.4.tar.xz # "" "--build=powerpc64le-unknown-linux-gnu"

# https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/flexpp-bisonpp/bisonpp-1.21-45.tar.gz
download_make_install ${HTTP_DEPS}/bisonpp-1.21-45.tar.gz bison++-1.21

CFLAGS="-fPIC" download_make_install ftp://ftp.gnu.org/gnu/readline/readline-7.0.tar.gz

VERS=1_67_0
# http://downloads.sourceforge.net/project/boost/boost/${VERS//_/.}/boost_$VERS.tar.bz2
download ${HTTP_DEPS}/boost_$VERS.tar.bz2
extract boost_$VERS.tar.bz2
pushd boost_$VERS
./bootstrap.sh --prefix=$PREFIX
./b2 cxxflags=-fPIC install --prefix=$PREFIX || true
popd

# https://github.com/Kitware/CMake/releases/download/v3.14.5/cmake-3.14.5.tar.gz
CXXFLAGS="-pthread" CFLAGS="-pthread" download_make_install ${HTTP_DEPS}/cmake-3.14.5.tar.gz

VERS=3.1.5
download https://github.com/google/double-conversion/archive/v$VERS.tar.gz
extract v$VERS.tar.gz
mkdir -p double-conversion-$VERS/build
pushd double-conversion-$VERS/build
cmake -DCMAKE_CXX_FLAGS="-fPIC" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$PREFIX ..
makej
make install
popd

VERS=2.2.2
download https://github.com/gflags/gflags/archive/v$VERS.tar.gz
extract v$VERS.tar.gz
mkdir -p gflags-$VERS/build
pushd gflags-$VERS/build
cmake -DCMAKE_CXX_FLAGS="-fPIC" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$PREFIX ..
makej
make install
popd

VERS=0.3.5
CXXFLAGS="-fPIC" download_make_install https://github.com/google/glog/archive/v$VERS.tar.gz glog-$VERS "--enable-shared=no" # --build=powerpc64le-unknown-linux-gnu"

# folly
VERS=2.1.10
download_make_install https://github.com/libevent/libevent/releases/download/release-$VERS-stable/libevent-$VERS-stable.tar.gz

VERS=2019.04.29.00
download https://github.com/facebook/folly/archive/v$VERS.tar.gz
extract v$VERS.tar.gz
pushd folly-$VERS/build/
CXXFLAGS="-fPIC -pthread" cmake -DCMAKE_INSTALL_PREFIX=$PREFIX ..
makej
make install
popd

# llvm
# http://thrysoee.dk/editline/libedit-20170329-3.1.tar.gz
download_make_install ${HTTP_DEPS}/libedit-20170329-3.1.tar.gz
# (see common-functions.sh)
install_llvm

VERS=7.69.0
# https://curl.haxx.se/download/curl-$VERS.tar.xz
download_make_install ${HTTP_DEPS}/curl-$VERS.tar.xz "" "--disable-ldap --disable-ldaps"

# thrift
VERS=0.11.0
# http://apache.claz.org/thrift/$VERS/thrift-$VERS.tar.gz
download ${HTTP_DEPS}/thrift-$VERS.tar.gz
extract thrift-$VERS.tar.gz
pushd thrift-$VERS
CFLAGS="-fPIC" CXXFLAGS="-fPIC" JAVA_PREFIX=$PREFIX/lib ./configure \
    --prefix=$PREFIX \
    --with-lua=no \
    --with-python=no \
    --with-php=no \
    --with-ruby=no \
    --with-qt4=no \
    --with-qt5=no \
    --with-boost-libdir=$PREFIX/lib
makej
make install
popd

# librdkafka
install_rdkafka static

# backend rendering
VERS=1.6.21
# http://download.sourceforge.net/libpng/libpng-$VERS.tar.xz
download_make_install ${HTTP_DEPS}/libpng-$VERS.tar.xz

VERS=3.0.2
download https://github.com/cginternals/glbinding/archive/v$VERS.tar.gz
extract v$VERS.tar.gz
BDIR="glbinding-$VERS/build"
mkdir -p $BDIR
pushd $BDIR
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DOPTION_BUILD_DOCS=OFF \
    -DOPTION_BUILD_EXAMPLES=OFF \
    -DOPTION_BUILD_TESTS=OFF \
    -DOPTION_BUILD_TOOLS=OFF \
    -DOPTION_BUILD_WITH_BOOST_THREAD=OFF \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_INSTALL_PREFIX=$PREFIX \
    ..
makej
make install
popd

install_snappy

# c-blosc
VERS=1.14.4
download https://github.com/Blosc/c-blosc/archive/v$VERS.tar.gz
extract v$VERS.tar.gz
BDIR="c-blosc-$VERS/build"
rm -rf "$BDIR"
mkdir -p "$BDIR"
pushd "$BDIR"
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$PREFIX -DBUILD_BENCHMARKS=off -DBUILD_TESTS=off -DPREFER_EXTERNAL_SNAPPY=off -DPREFER_EXTERNAL_ZLIB=off -DPREFER_EXTERNAL_ZSTD=off ..
makej
make install
popd

# Geo Support
install_gdal

# Apache Arrow (see common-functions.sh)
install_arrow

# Go
install_go

# install AWS core and s3 sdk
install_awscpp -j $(nproc)

# glslang (with spirv-tools)
VERS=7.12.3352 # 8/20/19
rm -rf glslang
mkdir -p glslang
pushd glslang
wget --continue https://github.com/KhronosGroup/glslang/archive/$VERS.tar.gz
tar xvf $VERS.tar.gz
pushd glslang-$VERS
./update_glslang_sources.py
mkdir build
pushd build
cmake \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_INSTALL_PREFIX=$PREFIX \
    ..
make -j $(nproc)
make install
popd # build
popd # glslang-$VERS
popd # glslang

# spirv-cross
VERS=2019-09-04
rm -rf spirv-cross
mkdir -p spirv-cross
pushd spirv-cross
wget --continue https://github.com/KhronosGroup/SPIRV-Cross/archive/$VERS.tar.gz
tar xvf $VERS.tar.gz
pushd SPIRV-Cross-$VERS
mkdir build
pushd build
cmake \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_INSTALL_PREFIX=$PREFIX \
    -DCMAKE_POSITION_INDEPENDENT_CODE=on \
    -DSPIRV_CROSS_ENABLE_TESTS=off \
    ..
make -j $(nproc)
make install
popd # build
popd # SPIRV-Cross-$VERS
popd # spirv-cross

# Vulkan
# Custom tarball which excludes the spir-v toolchain
VERS=1.1.126.0 # 11/1/19
rm -rf vulkan
mkdir -p vulkan
pushd vulkan
wget --continue ${HTTP_DEPS}/vulkansdk-linux-x86_64-no-spirv-$VERS.tar.gz -O vulkansdk-linux-x86_64-no-spirv-$VERS.tar.gz
tar xvf vulkansdk-linux-x86_64-no-spirv-$VERS.tar.gz
rsync -av $VERS/x86_64/* $PREFIX
popd # vulkan

# install opensaml and its dependencies
VERS=3.2.2
download ${HTTP_DEPS}/xerces-c-$VERS.tar.gz
extract xerces-c-$VERS.tar.gz
XERCESCROOT=$PWD/xerces-c-$VERS
mkdir $XERCESCROOT/build
pushd $XERCESCROOT/build
cmake -DCMAKE_INSTALL_PREFIX=$PREFIX -DBUILD_SHARED_LIBS=off -Dnetwork=off -DCMAKE_BUILD_TYPE=release ..
makej
make install
popd

download_make_install ${HTTP_DEPS}/xml-security-c-2.0.2.tar.gz "" "--without-xalan --enable-static --disable-shared"
download_make_install ${HTTP_DEPS}/xmltooling-3.0.4-nolog4shib.tar.gz "" "--enable-static --disable-shared"
download_make_install ${HTTP_DEPS}/opensaml-3.0.1-nolog4shib.tar.gz "" "--enable-static --disable-shared"

sed -e "s|%MAPD_DEPS_ROOT%|$PREFIX|g" mapd-deps.modulefile.in > mapd-deps-$SUFFIX.modulefile
sed -e "s|%MAPD_DEPS_ROOT%|$PREFIX|g" mapd-deps.sh.in > mapd-deps-$SUFFIX.sh

cp mapd-deps-$SUFFIX.sh mapd-deps-$SUFFIX.modulefile $PREFIX

if [ "$1" = "--compress" ] ; then
    tar acvf mapd-deps-$SUFFIX.tar.xz -C $(dirname $PREFIX) $SUFFIX
fi
