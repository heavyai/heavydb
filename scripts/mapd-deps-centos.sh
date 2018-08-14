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

SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $SCRIPTS_DIR/common-functions.sh


# gmp, mpc, mpfr, autoconf, automake
# note: if gmp fails on POWER8:
# wget https://gmplib.org/repo/gmp/raw-rev/4a6d258b467f
# patch -p1 < 4a6d258b467f
# https://gmplib.org/download/gmp/gmp-6.1.2.tar.xz
download_make_install https://dependencies.mapd.com/thirdparty/gmp-6.1.2.tar.xz "" "--enable-fat"
# http://www.mpfr.org/mpfr-current/mpfr-3.1.5.tar.xz
download_make_install https://dependencies.mapd.com/thirdparty/mpfr-4.0.1.tar.xz "" "--with-gmp=$PREFIX"
download_make_install ftp://ftp.gnu.org/gnu/mpc/mpc-1.1.0.tar.gz "" "--with-gmp=$PREFIX"
download_make_install ftp://ftp.gnu.org/gnu/autoconf/autoconf-2.69.tar.xz # "" "--build=powerpc64le-unknown-linux-gnu"
download_make_install ftp://ftp.gnu.org/gnu/automake/automake-1.16.1.tar.xz

# gcc
VERS=6.4.0
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

download_make_install ftp://ftp.gnu.org/gnu/libtool/libtool-2.4.6.tar.gz
# http://zlib.net/zlib-1.2.8.tar.xz
download_make_install https://dependencies.mapd.com/thirdparty/zlib-1.2.8.tar.xz

VERS=1.0.6
download http://bzip.org/$VERS/bzip2-$VERS.tar.gz
extract bzip2-$VERS.tar.gz
pushd bzip2-$VERS
makej
make install PREFIX=$PREFIX
popd

download_make_install https://www.openssl.org/source/openssl-1.0.2o.tar.gz "" "linux-$(uname -m) no-shared no-dso -fPIC"

# libarchive
download_make_install https://dependencies.mapd.com/thirdparty/xz-5.2.4.tar.xz "" "--disable-shared"
download_make_install http://libarchive.org/downloads/libarchive-3.3.2.tar.gz "" "--without-openssl --disable-shared"

CFLAGS="-fPIC" download_make_install ftp://ftp.gnu.org/pub/gnu/ncurses/ncurses-6.1.tar.gz # "" "--build=powerpc64le-unknown-linux-gnu"

download_make_install ftp://ftp.gnu.org/gnu/bison/bison-3.0.4.tar.xz # "" "--build=powerpc64le-unknown-linux-gnu"

# https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/flexpp-bisonpp/bisonpp-1.21-45.tar.gz
download_make_install https://dependencies.mapd.com/thirdparty/bisonpp-1.21-45.tar.gz bison++-1.21

CFLAGS="-fPIC" download_make_install ftp://ftp.gnu.org/gnu/readline/readline-7.0.tar.gz

VERS=1_67_0
# http://downloads.sourceforge.net/project/boost/boost/${VERS//_/.}/boost_$VERS.tar.bz2
download https://dependencies.mapd.com/thirdparty/boost_$VERS.tar.bz2
extract boost_$VERS.tar.bz2
pushd boost_$VERS
./bootstrap.sh --prefix=$PREFIX
./b2 cxxflags=-fPIC install --prefix=$PREFIX || true
popd

# https://cmake.org/files/v3.9/cmake-3.9.6.tar.gz
download_make_install https://dependencies.mapd.com/thirdparty/cmake-3.9.6.tar.gz

# folly
VERS=3.0.0
download https://github.com/google/double-conversion/archive/v$VERS.tar.gz
extract v$VERS.tar.gz
mkdir -p double-conversion-$VERS/build
pushd double-conversion-$VERS/build
cmake -DCMAKE_CXX_FLAGS="-fPIC" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$PREFIX ..
makej
make install
popd

VERS=2.2.1
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

VERS=2.1.8
download_make_install https://github.com/libevent/libevent/releases/download/release-$VERS-stable/libevent-$VERS-stable.tar.gz

VERS=2018.05.07.00
download https://github.com/facebook/folly/archive/v$VERS.tar.gz
extract v$VERS.tar.gz
pushd folly-$VERS/folly
OLDPATH=$PATH
PATH=/usr/bin
/usr/bin/autoreconf -ivf
PATH=$OLDPATH
CXXFLAGS="-fPIC -pthread" ./configure --prefix=$PREFIX --with-boost=$PREFIX --with-boost-libdir=$PREFIX/lib --enable-shared=no
makej
make install
popd

# llvm
download_make_install http://thrysoee.dk/editline/libedit-20170329-3.1.tar.gz
VERS=6.0.1
# http://releases.llvm.org
download https://dependencies.mapd.com/thirdparty/llvm/$VERS/llvm-$VERS.src.tar.xz
download https://dependencies.mapd.com/thirdparty/llvm/$VERS/cfe-$VERS.src.tar.xz
download https://dependencies.mapd.com/thirdparty/llvm/$VERS/compiler-rt-$VERS.src.tar.xz
download https://dependencies.mapd.com/thirdparty/llvm/$VERS/lldb-$VERS.src.tar.xz
download https://dependencies.mapd.com/thirdparty/llvm/$VERS/lld-$VERS.src.tar.xz
download https://dependencies.mapd.com/thirdparty/llvm/$VERS/libcxx-$VERS.src.tar.xz
download https://dependencies.mapd.com/thirdparty/llvm/$VERS/libcxxabi-$VERS.src.tar.xz
download https://dependencies.mapd.com/thirdparty/llvm/$VERS/clang-tools-extra-$VERS.src.tar.xz
rm -rf llvm-$VERS.src
extract llvm-$VERS.src.tar.xz
extract cfe-$VERS.src.tar.xz
extract compiler-rt-$VERS.src.tar.xz
extract lld-$VERS.src.tar.xz
extract lldb-$VERS.src.tar.xz
extract libcxx-$VERS.src.tar.xz
extract libcxxabi-$VERS.src.tar.xz
extract clang-tools-extra-$VERS.src.tar.xz
mv cfe-$VERS.src llvm-$VERS.src/tools/clang
mv compiler-rt-$VERS.src llvm-$VERS.src/projects/compiler-rt
mv lld-$VERS.src llvm-$VERS.src/tools/lld
mv lldb-$VERS.src llvm-$VERS.src/tools/lldb
mv libcxx-$VERS.src llvm-$VERS.src/projects/libcxx
mv libcxxabi-$VERS.src llvm-$VERS.src/projects/libcxxabi
mkdir -p llvm-$VERS.src/tools/clang/tools
mv clang-tools-extra-$VERS.src llvm-$VERS.src/tools/clang/tools/extra
rm -rf build.llvm-$VERS
mkdir build.llvm-$VERS
pushd build.llvm-$VERS
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$PREFIX -DLLVM_ENABLE_RTTI=on ../llvm-$VERS.src
makej
if [ ! -d "lib/python2.7" ]; then
    cp -R lib64/python2.7 lib/python2.7
fi
make install
popd

VERS=7.60.0
# https://curl.haxx.se/download/curl-$VERS.tar.xz
download_make_install https://dependencies.mapd.com/thirdparty/curl-$VERS.tar.xz "" "--disable-ldap --disable-ldaps"

# thrift
VERS=0.11.0
download http://apache.claz.org/thrift/$VERS/thrift-$VERS.tar.gz
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

# backend rendering
VERS=1.6.21
# http://download.sourceforge.net/libpng/libpng-$VERS.tar.xz
download_make_install https://dependencies.mapd.com/thirdparty/libpng-$VERS.tar.xz

VERS=2.1.4_egl
download https://github.com/vastcharade/glbinding/archive/v$VERS.tar.gz
extract v$VERS.tar.gz
BDIR="glbinding-$VERS/build"
mkdir -p $BDIR
pushd $BDIR
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DOPTION_BUILD_DOCS=OFF \
    -DOPTION_BUILD_EXAMPLES=OFF \
    -DOPTION_BUILD_GPU_TESTS=OFF \
    -DOPTION_BUILD_TESTS=OFF \
    -DOPTION_BUILD_TOOLS=OFF \
    -DOPTION_BUILD_WITH_BOOST_THREAD=OFF \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_INSTALL_PREFIX=$PREFIX \
    ..
makej
make install
popd

# c-blosc
VERS=1.14.3
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

# geo
download_make_install https://github.com/libexpat/libexpat/releases/download/R_2_2_5/expat-2.2.5.tar.bz2

# https://github.com/google/libkml/archive/master.zip
download https://dependencies.mapd.com/thirdparty/libkml-master.zip
unzip -u libkml-master.zip
pushd libkml-master
./autogen.sh || true
CXXFLAGS="-std=c++03" ./configure --with-expat-include-dir=$PREFIX/include/ --with-expat-lib-dir=$PREFIX/lib --prefix=$PREFIX --enable-static --disable-java --disable-python --disable-swig
makej
make install
popd

download_make_install https://github.com/OSGeo/proj.4/releases/download/5.0.1/proj-5.0.1.tar.gz
download_make_install http://download.osgeo.org/gdal/2.3.1/gdal-2.3.1.tar.xz "" "--without-geos --with-libkml=$PREFIX --with-static-proj4=$PREFIX"

# Apache Arrow (see common-functions.sh)
install_arrow

VERS=1.10.3
ARCH=$(uname -m)
ARCH=${ARCH//x86_64/amd64}
ARCH=${ARCH//aarch64/arm64}
# https://dl.google.com/go/go$VERS.linux-$ARCH.tar.gz
download https://dependencies.mapd.com/thirdparty/go$VERS.linux-$ARCH.tar.gz
extract go$VERS.linux-amd64.tar.gz
mv go $PREFIX

# install AWS core and s3 sdk
install_awscpp -j $(nproc)

sed -e "s|%MAPD_DEPS_ROOT%|$PREFIX|g" mapd-deps.modulefile.in > mapd-deps-$SUFFIX.modulefile
sed -e "s|%MAPD_DEPS_ROOT%|$PREFIX|g" mapd-deps.sh.in > mapd-deps-$SUFFIX.sh

cp mapd-deps-$SUFFIX.sh mapd-deps-$SUFFIX.modulefile $PREFIX

if [ "$1" = "--compress" ] ; then
    tar acvf mapd-deps-$SUFFIX.tar.xz -C $(dirname $PREFIX) $SUFFIX
fi

