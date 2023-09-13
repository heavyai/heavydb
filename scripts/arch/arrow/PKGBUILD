# Maintainer: Bruno Pagani <archange@archlinux.org>
# Contributor: Guillaume Horel <guillaume.horel@gmail.com>

pkgname=arrow
pkgver=13.0.0
pkgrel=2
pkgdesc="Columnar in-memory analytics layer for big data."
arch=(x86_64)
url="https://arrow.apache.org"
license=(Apache)
depends=(apache-orc brotli bzip2 gflags grpc google-glog libutf8proc
         lz4 openssl protobuf re2 snappy thrift zlib zstd)
provides=(parquet-cpp)
conflicts=(parquet-cpp)
makedepends=(boost clang cmake flatbuffers git gmock rapidjson xsimd)
source=(https://archive.apache.org/dist/${pkgname}/${pkgname}-${pkgver}/apache-${pkgname}-${pkgver}.tar.gz{,.asc}
        git+https://github.com/apache/parquet-testing.git
        git+https://github.com/apache/arrow-testing.git)
sha512sums=('3314d79ef20ac2cfc63f2c16fafb30c3f6187c10c6f5ea6ff036f6db766621d7c65401d85bf1e979bd0ecf831fbb0a785467642792d6bf77016f9807243c064e'
            'SKIP'
            'SKIP'
            'SKIP')
validpgpkeys=(265F80AB84FE03127E14F01125BCCA5220D84079  # Krisztian Szucs (apache) <szucs.krisztian@gmail.com>
              08D3564B7C6A9CAFBFF6A66791D18FCF079F8007) # Kouhei Sutou <kou@cozmixng.org>

build(){
  CC=clang \
  CXX=clang++ \
  cmake \
    -B build -S apache-${pkgname}-${pkgver}/cpp \
    -DCMAKE_INSTALL_PREFIX="/usr" \
    -DCMAKE_INSTALL_LIBDIR="lib" \
    -DCMAKE_BUILD_TYPE=Release \
    -DARROW_BUILD_STATIC=OFF \
    -DARROW_DEPENDENCY_SOURCE=SYSTEM \
    -DARROW_BUILD_TESTS=ON \
    -DARROW_COMPUTE=ON \
    -DARROW_CSV=ON \
    -DARROW_SUBSTRAIT=ON \
    -DARROW_FLIGHT=ON \
    -DARROW_FLIGHT_SQL=ON \
    -DARROW_GANDIVA=OFF \
    -DARROW_HDFS=ON \
    -DARROW_IPC=ON \
    -DARROW_JEMALLOC=ON \
    -DARROW_ORC=ON \
    -DARROW_PARQUET=ON \
    -DARROW_TENSORFLOW=ON \
    -DARROW_USE_GLOG=ON \
    -DARROW_WITH_BROTLI=ON \
    -DARROW_WITH_BZ2=ON \
    -DARROW_WITH_LZ4=ON \
    -DARROW_WITH_SNAPPY=ON \
    -DARROW_WITH_ZLIB=ON \
    -DARROW_WITH_ZSTD=ON \
    -DPARQUET_REQUIRE_ENCRYPTION=ON \
    -Wno-dev
  make -C build
}

check(){
  PARQUET_TEST_DATA="${srcdir}"/parquet-testing/data \
  ARROW_TEST_DATA="${srcdir}"/arrow-testing/data \
  ctest --test-dir build --output-on-failure
}

package(){
  DESTDIR="$pkgdir" cmake --install build
  find "${pkgdir}"/usr/lib/ -name '*testing*' -delete
}
