#!/bin/bash

function download() {
    wget --continue "$1"
}

function extract() {
    tar xvf "$1"
}

function makej() {
    make -j $(nproc)
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
    make install
    popd
}

ARROW_VERSION=apache-arrow-0.7.1

function install_arrow() {
  download https://github.com/apache/arrow/archive/$ARROW_VERSION.tar.gz
  extract $ARROW_VERSION.tar.gz
  mkdir -p arrow-$ARROW_VERSION/cpp/build
  pushd arrow-$ARROW_VERSION/cpp/build
  cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DARROW_BUILD_SHARED=ON \
    -DARROW_BUILD_STATIC=ON \
    -DARROW_BUILD_TESTS=OFF \
    -DARROW_BUILD_BENCHMARKS=OFF \
    -DARROW_WITH_BROTLI=OFF \
    -DARROW_WITH_ZLIB=OFF \
    -DARROW_WITH_LZ4=OFF \
    -DARROW_WITH_SNAPPY=OFF \
    -DARROW_WITH_ZSTD=OFF \
    -DCMAKE_INSTALL_PREFIX=$PREFIX \
    -DARROW_BOOST_USE_SHARED=${ARROW_BOOST_USE_SHARED:="OFF"} \
    ..
  make -j $(nproc)
  make install
  popd

}

AWSCPP_VERSION=1.3.10

function install_awscpp() {
    # default c++ standard support
    CPP_STANDARD=14
    # check c++17 support
    GNU_VERSION1=`g++ --version|head -n1|awk '{print $4}'|cut -d'.' -f1`
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
        ..
    make $*

    # sudo is needed on osx
    os=`uname`
    if [ "$os" = "Darwin" ]; then
        sudo make install
    else 
        make install
    fi

    popd
}
