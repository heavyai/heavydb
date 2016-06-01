#!/bin/bash

set -e
set -x

PREFIX=${MAPD_PATH:="/usr/local/mapd-deps"}
sudo mkdir -p $PREFIX
sudo chown -R $USER $PREFIX

export PATH=$PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$PREFIX/lib64:$PREFIX/lib:$LD_LIBRARY_PATH


download() {
    wget --continue "$1"
}

extract() {
    tar xvf "$1"
}

makej() {
    make -j $(nproc)
}

download_make_install() {
    name="$(basename $1)"
    download "$1"
    extract $name
    if [ -z "$2" ]; then
        pushd ${name%%.tar*}
    else
        pushd $2
    fi
    ./configure --prefix=$PREFIX $3
    makej
    make install
    popd
}


# gmp, mpc, mpfr, autoconf, automake
# note: if gmp fails on POWER8:
# wget https://gmplib.org/repo/gmp/raw-rev/4a6d258b467f
# patch -p1 < 4a6d258b467f
download_make_install https://gmplib.org/download/gmp/gmp-6.1.0.tar.xz
download_make_install http://www.mpfr.org/mpfr-current/mpfr-3.1.4.tar.xz "" "--with-gmp=$PREFIX"
download_make_install ftp://ftp.gnu.org/gnu/mpc/mpc-1.0.3.tar.gz "" "--with-gmp=$PREFIX"
download_make_install ftp://ftp.gnu.org/gnu/autoconf/autoconf-2.69.tar.xz # "" "--build=powerpc64le-unknown-linux-gnu"
download_make_install ftp://ftp.gnu.org/gnu/automake/automake-1.14.1.tar.xz

# gcc
VERS=4.9.3
download ftp://ftp.gnu.org/gnu/gcc/gcc-$VERS/gcc-$VERS.tar.bz2
extract gcc-$VERS.tar.bz2
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
download_make_install http://zlib.net/zlib-1.2.8.tar.xz

download http://bzip.org/1.0.6/bzip2-1.0.6.tar.gz
extract bzip2-1.0.6.tar.gz
pushd bzip2-1.0.6
makej
make install PREFIX=$PREFIX
popd

download_make_install ftp://ftp.gnu.org/pub/gnu/ncurses/ncurses-5.9.tar.gz # "" "--build=powerpc64le-unknown-linux-gnu"

download_make_install ftp://ftp.gnu.org/gnu/bison/bison-2.5.1.tar.xz # "" "--build=powerpc64le-unknown-linux-gnu"

download_make_install https://flexpp-bisonpp.googlecode.com/files/bisonpp-1.21-45.tar.gz bison++-1.21

download_make_install https://github.com/google/glog/archive/v0.3.4.tar.gz glog-0.3.4 # --build=powerpc64le-unknown-linux-gnu"

download_make_install ftp://ftp.gnu.org/gnu/readline/readline-6.3.tar.gz

download http://downloads.sourceforge.net/project/boost/boost/1.57.0/boost_1_57_0.tar.bz2
extract boost_1_57_0.tar.bz2
pushd boost_1_57_0
./bootstrap.sh --prefix=$PREFIX
./b2 install --prefix=$PREFIX || true
popd

download_make_install http://www.cmake.org/files/v3.4/cmake-3.4.1.tar.gz

# llvm
VERS=3.5.2
download http://llvm.org/releases/$VERS/llvm-$VERS.src.tar.xz
download http://llvm.org/releases/$VERS/cfe-$VERS.src.tar.xz
download http://llvm.org/releases/$VERS/compiler-rt-$VERS.src.tar.xz
rm -rf llvm-$VERS.src
extract llvm-$VERS.src.tar.xz
extract cfe-$VERS.src.tar.xz
extract compiler-rt-$VERS.src.tar.xz
mv cfe-$VERS.src llvm-$VERS.src/tools/clang
mv compiler-rt-$VERS.src llvm-$VERS.src/projects/compiler-rt
rm -rf build.llvm-$VERS
mkdir build.llvm-$VERS
pushd build.llvm-$VERS
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$PREFIX -DLLVM_ENABLE_RTTI=on ../llvm-$VERS.src
makej
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

# thrift
download_make_install https://github.com/libevent/libevent/releases/download/release-2.0.22-stable/libevent-2.0.22-stable.tar.gz
download_make_install http://apache.claz.org/thrift/0.9.3/thrift-0.9.3.tar.gz "" "--with-lua=no --with-python=no --with-php=no --with-boost-libdir=$PREFIX/lib"

# backend rendering
download https://sourceforge.net/projects/glew/files/glew/1.13.0/glew-1.13.0.tgz
extract glew-1.13.0.tgz
pushd glew-1.13.0
makej DESTDIR=$PREFIX GLEW_DEST="" install
popd

download https://github.com/glfw/glfw/releases/download/3.1.2/glfw-3.1.2.zip
unzip glfw-3.1.2.zip
mkdir -p glfw-3.1.2/build
pushd glfw-3.1.2/build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$PREFIX ..
makej
make install
popd

download_make_install http://download.sourceforge.net/libpng/libpng-1.6.21.tar.xz
