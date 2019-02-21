#!/bin/bash

HTTP_DEPS="https://dependencies.mapd.com/thirdparty"

function download() {
    wget --continue "$1"
}

function extract() {
    tar xvf "$1"
}

function makej() {
  os=$(uname)
  if [ "$os" = "Darwin" ]; then
    nproc=$(sysctl -n hw.ncpu)
  else
    nproc=$(nproc)
  fi
  make -j ${nproc:-8}
}

function make_install() {
  # sudo is needed on osx
  os=$(uname)
  if [ "$os" = "Darwin" ]; then
    sudo make install
  else
    make install
  fi
}

function download_make_install() {
    name="$(basename $1)"
    download "$1"
    extract $name
    if [ -z "$2" ]; then
        pushd ${name%%.tar*}
    else
        pushd $2
    fi

    if [ -x ./Configure ]; then
        ./Configure --prefix=$PREFIX $3
    else
        ./configure --prefix=$PREFIX $3
    fi
    makej
    make_install
    popd
}

ARROW_VERSION=apache-arrow-0.11.1

function install_arrow() {
  download https://github.com/apache/arrow/archive/$ARROW_VERSION.tar.gz
  extract $ARROW_VERSION.tar.gz
  mkdir -p arrow-$ARROW_VERSION/cpp/build
  pushd arrow-$ARROW_VERSION/cpp/build
  cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=$PREFIX \
    -DARROW_BUILD_SHARED=ON \
    -DARROW_BUILD_STATIC=ON \
    -DARROW_BUILD_TESTS=OFF \
    -DARROW_BUILD_BENCHMARKS=OFF \
    -DARROW_WITH_BROTLI=OFF \
    -DARROW_WITH_ZLIB=OFF \
    -DARROW_WITH_LZ4=OFF \
    -DARROW_WITH_SNAPPY=ON \
    -DARROW_WITH_ZSTD=OFF \
    -DARROW_BOOST_USE_SHARED=${ARROW_BOOST_USE_SHARED:="OFF"} \
    -DARROW_PARQUET=ON \
    -DTHRIFT_HOME=${THRIFT_HOME:-$PREFIX} \
    ..
  makej
  make_install
  popd

}

SNAPPY_VERSION=1.1.7
function install_snappy() {
  download https://github.com/google/snappy/archive/$SNAPPY_VERSION.tar.gz
  extract $SNAPPY_VERSION.tar.gz
  mkdir -p snappy-$SNAPPY_VERSION/build
  pushd snappy-$SNAPPY_VERSION/build
  cmake \
    -DCMAKE_INSTALL_PREFIX=$PREFIX \
    -DCMAKE_BUILD_TYPE=Release \
    -DSNAPPY_BUILD_TESTS=OFF \
    ..
  makej
  make_install
  popd
}

AWSCPP_VERSION=1.7.83

function install_awscpp() {
    # default c++ standard support
    CPP_STANDARD=14
    # check c++17 support
    GNU_VERSION1=$(g++ --version|head -n1|awk '{print $4}'|cut -d'.' -f1)
    if [ "$GNU_VERSION1" = "7" ]; then
        CPP_STANDARD=17
    fi
    rm -rf aws-sdk-cpp-${AWSCPP_VERSION}
    download https://github.com/aws/aws-sdk-cpp/archive/${AWSCPP_VERSION}.tar.gz
    tar xvfz ${AWSCPP_VERSION}.tar.gz
    pushd aws-sdk-cpp-${AWSCPP_VERSION}
    mkdir build
    cd build
    cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=$PREFIX \
        -DBUILD_ONLY="s3" \
        -DBUILD_SHARED_LIBS=0 \
        -DCUSTOM_MEMORY_MANAGEMENT=0 \
        -DCPP_STANDARD=$CPP_STANDARD \
        -DENABLE_TESTING=off \
        ..
    make $*
    make_install
    popd
}

LLVM_VERSION=8.0.0

function install_llvm() {
    VERS=${LLVM_VERSION}
    download ${HTTP_DEPS}/llvm/$VERS/llvm-$VERS.src.tar.xz
    download ${HTTP_DEPS}/llvm/$VERS/cfe-$VERS.src.tar.xz
    download ${HTTP_DEPS}/llvm/$VERS/compiler-rt-$VERS.src.tar.xz
    download ${HTTP_DEPS}/llvm/$VERS/lldb-$VERS.src.tar.xz
    download ${HTTP_DEPS}/llvm/$VERS/lld-$VERS.src.tar.xz
    download ${HTTP_DEPS}/llvm/$VERS/libcxx-$VERS.src.tar.xz
    download ${HTTP_DEPS}/llvm/$VERS/libcxxabi-$VERS.src.tar.xz
    download ${HTTP_DEPS}/llvm/$VERS/clang-tools-extra-$VERS.src.tar.xz
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
}
