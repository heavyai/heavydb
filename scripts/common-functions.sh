#!/bin/bash

# master as of 2017-09-07. Will want to update to 0.7.0 final as soon as it's
# released
ARROW_VERSION=6f27a6447171353427a129a5ce88dba181bd8af6

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
    -DARROW_BOOST_USE_SHARED=off \
    ..
  make -j $(nproc)
  make install
  popd

}
