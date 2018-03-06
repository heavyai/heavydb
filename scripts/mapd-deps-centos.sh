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
download_make_install https://internal-dependencies.mapd.com/thirdparty/gmp-6.1.2.tar.xz "" "--enable-fat"
# http://www.mpfr.org/mpfr-current/mpfr-3.1.5.tar.xz
download_make_install https://internal-dependencies.mapd.com/thirdparty/mpfr-3.1.5.tar.xz "" "--with-gmp=$PREFIX"
download_make_install ftp://ftp.gnu.org/gnu/mpc/mpc-1.0.3.tar.gz "" "--with-gmp=$PREFIX"
download_make_install ftp://ftp.gnu.org/gnu/autoconf/autoconf-2.69.tar.xz # "" "--build=powerpc64le-unknown-linux-gnu"
download_make_install ftp://ftp.gnu.org/gnu/automake/automake-1.14.1.tar.xz

# gcc
VERS=5.5.0
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
download_make_install https://internal-dependencies.mapd.com/thirdparty/zlib-1.2.8.tar.xz

download http://bzip.org/1.0.6/bzip2-1.0.6.tar.gz
extract bzip2-1.0.6.tar.gz
pushd bzip2-1.0.6
makej
make install PREFIX=$PREFIX
popd

# libarchive
download_make_install https://www.openssl.org/source/openssl-1.0.2n.tar.gz "" "linux-$(uname -m) no-shared no-dso -fPIC"
download_make_install https://tukaani.org/xz/xz-5.2.3.tar.gz "" "--disable-shared"
download_make_install http://libarchive.org/downloads/libarchive-3.3.2.tar.gz "" "--without-openssl --disable-shared"

CFLAGS="-fPIC" download_make_install ftp://ftp.gnu.org/pub/gnu/ncurses/ncurses-6.0.tar.gz # "" "--build=powerpc64le-unknown-linux-gnu"

download_make_install ftp://ftp.gnu.org/gnu/bison/bison-2.5.1.tar.xz # "" "--build=powerpc64le-unknown-linux-gnu"

# https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/flexpp-bisonpp/bisonpp-1.21-45.tar.gz
download_make_install https://internal-dependencies.mapd.com/thirdparty/bisonpp-1.21-45.tar.gz bison++-1.21

CFLAGS="-fPIC" download_make_install ftp://ftp.gnu.org/gnu/readline/readline-6.3.tar.gz

# http://downloads.sourceforge.net/project/boost/boost/1.62.0/boost_1_62_0.tar.bz2
download https://internal-dependencies.mapd.com/thirdparty/boost_1_62_0.tar.bz2
extract boost_1_62_0.tar.bz2
pushd boost_1_62_0
./bootstrap.sh --prefix=$PREFIX
./b2 cxxflags=-fPIC install --prefix=$PREFIX || true
popd

# http://www.cmake.org/files/v3.7/cmake-3.7.2.tar.gz
download_make_install https://internal-dependencies.mapd.com/thirdparty/cmake-3.7.2.tar.gz

# folly
download https://github.com/google/double-conversion/archive/4abe3267170fa52f39460460456990dbae803f4d.tar.gz
extract 4abe3267170fa52f39460460456990dbae803f4d.tar.gz
mv double-conversion-4abe3267170fa52f39460460456990dbae803f4d google-double-conversion-4abe326
mkdir -p google-double-conversion-4abe326/build
pushd google-double-conversion-4abe326/build
cmake -DCMAKE_CXX_FLAGS="-fPIC" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$PREFIX ..
makej
make install
popd

download https://github.com/gflags/gflags/archive/v2.2.0.tar.gz
extract v2.2.0.tar.gz
mkdir -p gflags-2.2.0/build
pushd gflags-2.2.0/build
cmake -DCMAKE_CXX_FLAGS="-fPIC" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$PREFIX ..
makej
make install
popd

CXXFLAGS="-fPIC" download_make_install https://github.com/google/glog/archive/v0.3.4.tar.gz glog-0.3.4 "--enable-shared=no" # --build=powerpc64le-unknown-linux-gnu"

download_make_install https://github.com/libevent/libevent/releases/download/release-2.0.22-stable/libevent-2.0.22-stable.tar.gz

VERS=2017.10.16.00
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
download_make_install http://thrysoee.dk/editline/libedit-20160903-3.1.tar.gz
VERS=3.9.1
# http://releases.llvm.org
download https://internal-dependencies.mapd.com/thirdparty/llvm/$VERS/llvm-$VERS.src.tar.xz
download https://internal-dependencies.mapd.com/thirdparty/llvm/$VERS/cfe-$VERS.src.tar.xz
download https://internal-dependencies.mapd.com/thirdparty/llvm/$VERS/compiler-rt-$VERS.src.tar.xz
download https://internal-dependencies.mapd.com/thirdparty/llvm/$VERS/lldb-$VERS.src.tar.xz
download https://internal-dependencies.mapd.com/thirdparty/llvm/$VERS/lld-$VERS.src.tar.xz
download https://internal-dependencies.mapd.com/thirdparty/llvm/$VERS/libcxx-$VERS.src.tar.xz
download https://internal-dependencies.mapd.com/thirdparty/llvm/$VERS/libcxxabi-$VERS.src.tar.xz
rm -rf llvm-$VERS.src
extract llvm-$VERS.src.tar.xz
extract cfe-$VERS.src.tar.xz
extract compiler-rt-$VERS.src.tar.xz
extract lld-$VERS.src.tar.xz
extract lldb-$VERS.src.tar.xz
extract libcxx-$VERS.src.tar.xz
extract libcxxabi-$VERS.src.tar.xz
mv cfe-$VERS.src llvm-$VERS.src/tools/clang
mv compiler-rt-$VERS.src llvm-$VERS.src/projects/compiler-rt
mv lld-$VERS.src llvm-$VERS.src/tools/lld
mv lldb-$VERS.src llvm-$VERS.src/tools/lldb
mv libcxx-$VERS.src llvm-$VERS.src/projects/libcxx
mv libcxxabi-$VERS.src llvm-$VERS.src/projects/libcxxabi
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

# openssl (probably not a good idea)
#download https://www.openssl.org/source/openssl-1.0.2c.tar.gz
#x openssl-1.0.2c.tar.gz
#pushd openssl-1.0.2c
#./Configure shared --prefix=$PREFIX linux-x86_64
#makej
#make install
#popd

# https://curl.haxx.se/download/curl-7.50.0.tar.bz2
download_make_install https://internal-dependencies.mapd.com/thirdparty/curl-7.50.0.tar.bz2 "" "--disable-ldap --disable-ldaps"

# http://www.cryptopp.com/cryptopp563.zip
download https://internal-dependencies.mapd.com/thirdparty/cryptopp563.zip
unzip -a -d cryptopp563 cryptopp563
pushd cryptopp563
PREFIX=$PREFIX make all shared
PREFIX=$PREFIX make install
popd

# thrift
VERS=0.10.0
download http://apache.claz.org/thrift/$VERS/thrift-$VERS.tar.gz
extract thrift-$VERS.tar.gz
pushd thrift-$VERS
patch -p1 < $SCRIPTS_DIR/thrift-3821-tmemorybuffer-overflow-check.patch
patch -p1 < $SCRIPTS_DIR/thrift-3821-tmemorybuffer-overflow-test.patch
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
# http://download.sourceforge.net/libpng/libpng-1.6.21.tar.xz
download_make_install https://internal-dependencies.mapd.com/thirdparty/libpng-1.6.21.tar.xz

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
VERS=1.11.3
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
# https://downloads.sourceforge.net/project/expat/expat/2.2.0/expat-2.2.0.tar.bz2
download_make_install https://internal-dependencies.mapd.com/thirdparty/expat-2.2.0.tar.bz2

# https://github.com/google/libkml/archive/master.zip
download https://internal-dependencies.mapd.com/thirdparty/libkml-master.zip
unzip -u libkml-master.zip
pushd libkml-master
./autogen.sh || true
./configure --with-expat-include-dir=$PREFIX/include/ --with-expat-lib-dir=$PREFIX/lib --prefix=$PREFIX --enable-static --disable-java --disable-python --disable-swig
makej
make install
popd

download_make_install http://download.osgeo.org/proj/proj-4.9.3.tar.gz
download_make_install http://download.osgeo.org/gdal/2.0.3/gdal-2.0.3.tar.xz "" "--without-curl --without-geos --with-libkml=$PREFIX --with-static-proj4=$PREFIX"

# Apache Arrow (see common-functions.sh)
install_arrow

# https://storage.googleapis.com/golang/go1.8.1.linux-amd64.tar.gz
download https://internal-dependencies.mapd.com/thirdparty/go1.8.1.linux-amd64.tar.gz
extract go1.8.1.linux-amd64.tar.gz
mv go $PREFIX

# install AWS core and s3 sdk
install_awscpp -j $(nproc)

sed -e "s|%MAPD_DEPS_ROOT%|$PREFIX|g" mapd-deps.modulefile.in > mapd-deps-$SUFFIX.modulefile
sed -e "s|%MAPD_DEPS_ROOT%|$PREFIX|g" mapd-deps.sh.in > mapd-deps-$SUFFIX.sh

cp mapd-deps-$SUFFIX.sh mapd-deps-$SUFFIX.modulefile $PREFIX

if [ "$1" = "--compress" ] ; then
    tar acvf mapd-deps-$SUFFIX.tar.xz -C $(dirname $PREFIX) $SUFFIX
fi

