#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Make $ID, $VERSION_ID, etc. available to all functions (e.g. download_make_install).
source /etc/os-release

function generate_deps_version_file() {
  # SUFFIX, BRANCH_NAME, GIT_COMMIT and BUILD_CONTAINER_NAME are set as environment variables not as parameters and
  # are generally set 'on' the calling docker container.
  echo "Public Release:Deps generated for prefix [$PREFIX], commit [$GIT_COMMIT] and SUFFIX [$SUFFIX]" > $PREFIX/mapd_deps_version.txt
  # BUILD_CONTAINER_IMAGE will only be set if called from heavyai-dependency-tar-builder.sh
  if [[ -n $BUILD_CONTAINER_IMAGE_ID ]] ; then
    echo "Public Release:Using build image id [${BUILD_CONTAINER_IMAGE_ID}]" >> $PREFIX/mapd_deps_version.txt
  fi
  if [[ -n $BUILD_CONTAINER_IMAGE ]] ; then
    # Not copied to released version of this file
    echo "Using build image [${BUILD_CONTAINER_IMAGE}]" >> $PREFIX/mapd_deps_version.txt
  fi
  echo "LIBRARY_TYPE=$LIBRARY_TYPE" >> $PREFIX/mapd_deps_version.txt
  echo "TSAN=$TSAN" >> $PREFIX/mapd_deps_version.txt
  echo "Component version information:" >> $PREFIX/mapd_deps_version.txt
  # Record dependency versions and source/patch provenance in the dependency image.
  # This isn't a complete list of all software and versions.  For example openssl either uses
  # the version that ships with the OS or it is installed from the OS specific file and
  # doesn't use an _VERSION variable.
  # Not to be copied to released version of this file
  for i in $(compgen -A variable | grep -E '(_VERSION|_SOURCE_SHA256)$') ; do echo  $i "${!i}" ; done >> $PREFIX/mapd_deps_version.txt
}

##
## Rocky specific Functions
##
function install_required_rockylinux_packages() {
  # TODO(scb) python3-devel may no longer be required
  sudo dnf install -y dnf-plugins-core
  sudo dnf config-manager --set-enabled devel
  sudo dnf makecache
  # --nobest: the nvcr.io CUDA container pre-installs UBI packages at version X,
  # but ubi-8-appstream can drift to X+1 between container builds. Any -devel or
  # -static package that requires an arch-specific capability (e.g.
  # libxml2(x86-64) = X+1) from the pre-installed base will hard-fail without
  # --nobest. The security-update pass (--update-packages --update-options=--nobest)
  # keeps packages current separately.
  sudo dnf install -y --nobest \
      ca-certificates \
      curl \
      epel-release \
      git \
      java-21-openjdk-devel \
      libglvnd-devel \
      libssh \
      libxml2-devel \
      libxml2-static \
      perl-IPC-Cmd \
      perl-Pod-Html \
      perl-Time-Piece \
      python3 \
      python3-devel \
      rsync \
      wget \
      which \
      xz \
      xz-static \
      zlib-devel \
      zlib-static

  # Install packages from EPEL
  sudo dnf install -y \
      jq \
      pxz
}

function install_gcc_rocky() {
  ## ONLY RUN THIS ON ROCKY!!
  local GCC_VERSION=11.4.0
  download ftp://ftp.gnu.org/gnu/gcc/gcc-${GCC_VERSION}/gcc-${GCC_VERSION}.tar.xz
  extract gcc-${GCC_VERSION}.tar.xz
  pushd gcc-${GCC_VERSION}
  export CPPFLAGS="-I$PREFIX/include"
  # --with-tune=generic is x86_64-only; aarch64 has no "generic" tune so omit it there
  local TUNE_FLAG="--with-tune=generic"
  [ "$(uname -m)" = "aarch64" ] && TUNE_FLAG=""
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
    $TUNE_FLAG \
    --with-gmp=$PREFIX \
    --with-mpc=$PREFIX \
    --with-mpfr=$PREFIX
  makej
  make install
  popd
  check_artifact_cleanup gcc-${GCC_VERSION}.tar.xz gcc-${GCC_VERSION}
}

function install_libmd() {
  local LIBMD_VERSION=1.0.0
  download https://libbsd.freedesktop.org/releases/libmd-${LIBMD_VERSION}.tar.xz
  tar xvf libmd-$LIBMD_VERSION.tar.xz
  pushd libmd-$LIBMD_VERSION
  ./autogen
  ./configure --prefix=$PREFIX --disable-shared --enable-static
  makej
  make install
  popd
  check_artifact_cleanup libmd-$LIBMD_VERSION.tar.gz libmd-$LIBMD_VERSION
}

# tukaani.org/xz identifies tukaani-project/xz as the primary repository.
# v5.8.3 is signed by maintainer Lasse Collin; this digest is the official
# release asset digest and prevents the downloaded archive from changing.
XZ_VERSION=5.8.3
XZ_SOURCE_SHA256=fff1ffcf2b0da84d308a14de513a1aa23d4e9aa3464d17e64b9714bfdd0bbfb6

function install_xz() {
  CFLAGS="${CFLAGS}" download_make_install \
    https://github.com/tukaani-project/xz/releases/download/v${XZ_VERSION}/xz-${XZ_VERSION}.tar.xz \
    xz-${XZ_VERSION}.tar.xz \
    "" \
    "${CONFIGURE_OPTS}" \
    "${XZ_SOURCE_SHA256}"
}

function install_cmake_rocky() {
  ## CMAKE_VERSION and CMAKE_DLOAD are set in common-functions.sh
  download ${CMAKE_DLOAD}
  extract cmake-${CMAKE_VERSION}.tar.gz
  pushd cmake-${CMAKE_VERSION}
  # patch the FindCURL.cmake script to work around the apparent
  # long-standing bug that breaks find_package(CURL PROPERTIES HTTP HTTPS)
  # in the libcpr build that follows
  patch -p0 < ${SCRIPTS_DIR}/cmake_find_curl_fix.patch
  CXXFLAGS="-pthread" CFLAGS="-pthread" ./configure --prefix=${PREFIX}
  makej
  make install
  popd
  check_artifact_cleanup cmake-${CMAKE_VERSION}.tar.gz cmake-${CMAKE_VERSION}
}


function install_icu() {
  local ICU_VERSION=60_3
  local ICU_VERSION_HYP=$(echo $ICU_VERSION | tr '_' '-')
  download https://github.com/unicode-org/icu/releases/download/release-${ICU_VERSION_HYP}/icu4c-${ICU_VERSION}-src.tgz
  extract icu4c-$ICU_VERSION-src.tgz
  pushd icu/source
  chmod +x runConfigureICU configure install-sh
  mkdir -p build
  pushd build
  CXXFLAGS=-std=c++11 ../runConfigureICU --enable-debug Linux/gcc --prefix=$PREFIX --enable-static --disable-shared --disable-dyload
  makej
  make install
  popd
  popd
  check_artifact_cleanup icu4c-$ICU_VERSION-src.tgz icu4c-$ICU_VERSION-src
}

function install_uriparser() {
  local URIPARSER_VERSION=0.9.8
  local URI_NAME="uriparser-$URIPARSER_VERSION"
  download https://github.com/uriparser/uriparser/archive/refs/tags/$URI_NAME.tar.gz
  extract $URI_NAME.tar.gz
  mkdir uriparser-$URI_NAME/build
  ( cd uriparser-$URI_NAME/build
    cmake .. \
        -DBUILD_SHARED_LIBS=off \
        -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
        -DCMAKE_CXX_FLAGS="$CXXFLAGS" \
        -DCMAKE_C_FLAGS="$CFLAGS" \
        -DCMAKE_INSTALL_PREFIX="$PREFIX" \
        -DCMAKE_POSITION_INDEPENDENT_CODE=on \
        -DURIPARSER_BUILD_DOCS=off \
        -DURIPARSER_BUILD_TESTS=off
    makej
    make install
  )
  check_artifact_cleanup $URI_NAME.tar.gz uriparser-$URI_NAME
}


function install_xerces_c() {
  local XERCES_C_VERS=3.2.5
  download https://archive.apache.org/dist/xerces/c/3/sources/xerces-c-${XERCES_C_VERS}.tar.gz
  extract xerces-c-$XERCES_C_VERS.tar.gz
  XERCESCROOT=$PWD/xerces-c-$XERCES_C_VERS
  mkdir -p $XERCESCROOT/build
  pushd $XERCESCROOT/build
  cmake \
    -DCMAKE_INSTALL_PREFIX=$PREFIX \
    -DBUILD_SHARED_LIBS=off \
    -DPREFER_STATIC_LIBS=on \
    -Dnetwork=off \
    -Dtranscoder=iconv \
    -Dmessage-loader=inmemory \
    -DCMAKE_BUILD_TYPE=release \
    ..
  makej
  make install
  popd
}
##
## End Rocky specific Funtions
##

function install_required_ubuntu_packages() {
  # Please keep this list sorted via the sort command.
  DEBIAN_FRONTEND=noninteractive sudo apt install -y \
      autoconf \
      autoconf-archive \
      automake \
      binutils-dev \
      bison \
      build-essential \
      ccache \
      curl \
      flex \
      git \
      google-perftools \
      groff-base \
      jq \
      libbz2-dev \
      libdouble-conversion-dev \
      libedit-dev \
      libegl-dev \
      libgflags-dev \
      libgoogle-perftools-dev \
      libiberty-dev \
      libicu-dev \
      liblzma-dev \
      libmd-dev \
      libncurses5-dev \
      libsnappy-dev \
      libtool \
      libxerces-c-dev \
      libxml2-dev \
      patchelf \
      pkg-config \
      python3-dev \
      python3-yaml \
      rsync \
      software-properties-common \
      swig \
      unzip \
      uuid-dev \
      valgrind \
      wget \
      zlib1g-dev

  DEBIAN_FRONTEND=noninteractive sudo apt install -y \
      openjdk-21-jdk \
      openjdk-21-jdk-headless \
      openjdk-21-jre \
      openjdk-21-jre-headless

  if [ "$LIBRARY_TYPE" != "static" ]; then
    DEBIAN_FRONTEND=noninteractive sudo apt install -y \
        libglu1-mesa-dev \
        libldap2-dev \
        libxcursor-dev \
        libxi-dev \
        libxinerama-dev \
        libxrandr-dev
  fi
}

function download() {
  local TARGET_FILE=
  if [[ $# -eq 2 ]]; then
    TARGET_FILE=$2
  else
    TARGET_FILE=$(basename $1)
  fi
  echo ${CACHE}/${TARGET_FILE}
  if [[ -s ${CACHE}/${TARGET_FILE} ]]; then
    \cp ${CACHE}/${TARGET_FILE} .
  else
    wget --continue "$1" --output-document=${TARGET_FILE}
  fi
  if [[ -n "${CACHE}" && ! -e "${CACHE}/${TARGET_FILE}" ]]; then
    \cp ${TARGET_FILE} ${CACHE}
  fi
}

function extract() {
    tar xvf "$1"
}

function cmake_build_and_install() {
  cmake --build . --parallel ${NPROC} && cmake --install .
}

function makej() {
  make -j ${NPROC} $1
}

function make_install() {
  make install
}

function check_artifact_cleanup() {
  download_file=$1
  build_dir=$2
  [[ -z $build_dir || -z $download_file ]] && echo "Invalid args remove_install_artifacts" && return
  if [[ $SAVE_SPACE == 'true' ]] ; then
    rm -f $download_file
    rm -rf $build_dir
  fi
}

function force_artifact_cleanup() {
  download_file=$1
  build_dir=$2
  [[ -z $build_dir || -z $download_file ]] && echo "Invalid args remove_install_artifacts" && return
  rm -f $download_file
  rm -rf $build_dir
}

function download_make_install() {
    local target_file=
    local source_sha256="${5:-}"
    if [[ $# -eq 1 ]] ; then
      target_file="$(basename $1)"
    elif [[ $# -ge 2 ]] ; then
	    target_file=$2
    fi
    download "$1" $target_file
    if [[ -n "${source_sha256}" ]]; then
      echo "${source_sha256}  ${target_file}" | sha256sum --check -
    fi
    extract $target_file
    build_dir=${target_file%%.tar*}
    [[ -n "$3" ]] && build_dir="${3}"
    pushd ${build_dir}

    # Packages with config.guess older than ~2012 can't detect aarch64 (e.g. glog-0.3.5
    # ships a config.guess from 2007). We can't know in advance which tarballs are affected,
    # so we apply the fix defensively here for every package. The overhead is just a find+cp
    # per package, which is negligible compared to compile time. Restricted to Rocky on aarch64
    # because that is the only platform where libtool is built from source (providing the
    # replacement config.guess); on Ubuntu, libtool comes from apt and is not under $PREFIX.
    if [[ ${ARCH} == "aarch64" && ${ID} == "rocky" ]]; then
      for cfg_file in config.guess config.sub; do
        if [ -f "$PREFIX/share/libtool/build-aux/$cfg_file" ]; then
          find . -name "$cfg_file" -exec cp "$PREFIX/share/libtool/build-aux/$cfg_file" {} \;
        fi
      done
    fi

    if [ -x ./Configure ]; then
        ./Configure --prefix=$PREFIX $4
    else
        ./configure --prefix=$PREFIX $4
    fi
    makej
    make_install
    popd
    check_artifact_cleanup $target_file $build_dir
}

## These variables are also used explicitly by the rocky deps builder
## in a rock specific cmake install.
CMAKE_VERSION=3.26.5
CMAKE_DLOAD=https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}.tar.gz
function install_cmake() {
  CXXFLAGS="-pthread" CFLAGS="-pthread" download_make_install ${CMAKE_DLOAD}
}

BOOST_VERSION=1_86_0
function install_boost() {
  download http://downloads.sourceforge.net/project/boost/boost/${BOOST_VERSION//_/.}/boost_${BOOST_VERSION}.tar.bz2
  extract boost_${BOOST_VERSION}.tar.bz2
  pushd boost_${BOOST_VERSION}
  ./bootstrap.sh --prefix=$PREFIX
  ./b2 cxxflags=-fPIC install --prefix=$PREFIX || true
  popd
  check_artifact_cleanup boost_${BOOST_VERSION}.tar.bz2 boost_${BOOST_VERSION}
}

OPENSSL_VERSION=3.6.4
function install_openssl() {
  download_make_install https://www.openssl.org/source/old/3.0/openssl-${OPENSSL_VERSION}.tar.gz "openssl-${OPENSSL_VERSION}.tar.gz" "" "linux-${ARCH} no-shared no-dso -fPIC"
}

LDAP_VERSION=2.5.16
function install_openldap2() {
  download https://www.openldap.org/software/download/OpenLDAP/openldap-release/openldap-$LDAP_VERSION.tgz
  extract openldap-$LDAP_VERSION.tgz
  mkdir -p openldap-$LDAP_VERSION/build
  pushd openldap-$LDAP_VERSION/build
  if [[ ${ID} == "rocky" ]]; then
    ../configure --prefix=$PREFIX --disable-shared --enable-static
  else
    ../configure --prefix=$PREFIX --disable-shared --enable-static --without-cyrus-sasl
  fi
  make depend
  make -j ${NPROC}
  make install
  popd
  check_artifact_cleanup openldap-$LDAP_VERSION.tar.gz openldap-$LDAP_VERSION
}

ARROW_VERSION=apache-arrow-18.1.0

function install_arrow() {
  if [ "$TSAN" = "true" ]; then
    ARROW_TSAN="-DARROW_USE_TSAN=ON"
    ARROW_JEMALLOC="-DARROW_JEMALLOC=OFF"
  elif [ "$TSAN" = "false" ]; then
    ARROW_TSAN="-DARROW_USE_TSAN=OFF"
    ARROW_JEMALLOC="-DARROW_JEMALLOC=ON"
    if [ "$ARCH" == "aarch64" ]; then
      # build bundled jemalloc with 64K system page size to support GH200
      ARROW_JEMALLOC="${ARROW_JEMALLOC} -DARROW_JEMALLOC_LG_PAGE=16"
    fi
  fi

  ARROW_USE_CUDA="-DARROW_CUDA=ON"
  if [ "$NOCUDA" = "true" ]; then
    ARROW_USE_CUDA="-DARROW_CUDA=OFF"
  fi

  if [ "$LIBRARY_TYPE" == "static" ]; then
    ARROW_BUILD_STATIC=on
    ARROW_BUILD_SHARED=off
  else
    ARROW_BUILD_STATIC=off
    ARROW_BUILD_SHARED=on
  fi

  download https://github.com/apache/arrow/archive/$ARROW_VERSION.tar.gz
  extract $ARROW_VERSION.tar.gz

  mkdir -p arrow-$ARROW_VERSION/cpp/build
  pushd arrow-$ARROW_VERSION/cpp/build

  # Use installed liburiparser instead.
  sed -Ei 's/^\s*vendored\/uriparser\/.*\)/)/' ../src/arrow/CMakeLists.txt
  sed -Ei  '/^\s*vendored\/uriparser\//d'      ../src/arrow/CMakeLists.txt

  # Use Thrift 0.24.0 instead of 0.20.0
  sed -i 's/ARROW_THRIFT_BUILD_VERSION=0.20.0/ARROW_THRIFT_BUILD_VERSION=0.24.0/' ../thirdparty/versions.txt
  sed -i 's/ARROW_THRIFT_BUILD_SHA256_CHECKSUM=b5d8311a779470e1502c027f428a1db542f5c051c8e1280ccd2163fa935ff2d6/ARROW_THRIFT_BUILD_SHA256_CHECKSUM=1859d932d2ae1f13d16c5a196931208c116310a5ff50f2bfd11d3db03be8f46f/' ../thirdparty/versions.txt

  # Arrow 16+ requires the latest liblz4 (1.10.0) and libzstd (1.5.6) or it won't
  # find them. Also, a little known factoid (only shown in some older versions of
  # the documentation) is that although ARROW_DEPENDENCY_USE_SHARED=ON will
  # correctly default all sub-dependencies to shared, setting it to OFF will
  # *not* reliably default them all to static! Here, ZSTD, LZ4, and BZ2 must be
  # individually forced to static using their respective ARROW_*_USE_SHARED=OFF.
  cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=$PREFIX \
    -DARROW_BUILD_SHARED=${ARROW_BUILD_SHARED} \
    -DARROW_BUILD_STATIC=${ARROW_BUILD_STATIC} \
    -DARROW_DEPENDENCY_USE_SHARED=${ARROW_BUILD_SHARED} \
    -DARROW_CSV=ON \
    -DARROW_JSON=ON \
    -DARROW_WITH_BROTLI=ON \
    -DARROW_WITH_SNAPPY=ON \
    -DARROW_WITH_ZLIB=ON \
    -DARROW_WITH_ZSTD=ON \
    -DARROW_WITH_LZ4=ON \
    -DARROW_WITH_BZ2=ON \
    -DARROW_ZSTD_USE_SHARED=${ARROW_BUILD_SHARED} \
    -DARROW_LZ4_USE_SHARED=${ARROW_BUILD_SHARED} \
    -DARROW_BZ2_USE_SHARED=${ARROW_BUILD_SHARED} \
    -DARROW_USE_GLOG=OFF \
    -DARROW_PARQUET=ON \
    -DARROW_FILESYSTEM=ON \
    -DARROW_S3=ON \
    ${ARROW_USE_CUDA} \
    ${ARROW_JEMALLOC} \
    ${ARROW_TSAN} \
    ..
  makej
  make_install
  popd
  check_artifact_cleanup $ARROW_VERSION.tar.gz arrow-$ARROW_VERSION
}

SNAPPY_VERSION=1.1.7
function install_snappy() {
  download https://github.com/google/snappy/archive/$SNAPPY_VERSION.tar.gz snappy-$SNAPPY_VERSION.tar.gz
  extract snappy-$SNAPPY_VERSION.tar.gz
  mkdir -p snappy-$SNAPPY_VERSION/build
  pushd snappy-$SNAPPY_VERSION/build
  cmake \
    -DCMAKE_CXX_FLAGS="-fPIC" \
    -DCMAKE_INSTALL_PREFIX=$PREFIX \
    -DCMAKE_BUILD_TYPE=Release \
    -DSNAPPY_BUILD_TESTS=OFF \
    ..
  makej
  make_install
  popd
  check_artifact_cleanup snappy-$SNAPPY_VERSION.tar.gz snappy-$SNAPPY_VERSION
}

# latest as of 2/28/25
AWSCPP_VERSION=1.11.517

function install_awscpp() {
  rm -rf aws-sdk-cpp-${AWSCPP_VERSION}
  download https://github.com/aws/aws-sdk-cpp/archive/${AWSCPP_VERSION}.tar.gz aws-sdk-cpp-${AWSCPP_VERSION}.tar.gz
  tar xvfz aws-sdk-cpp-${AWSCPP_VERSION}.tar.gz
  pushd aws-sdk-cpp-${AWSCPP_VERSION}
  ./prefetch_crt_dependency.sh
  sed -i 's/-Werror//g' cmake/compiler_settings.cmake
  mkdir build
  cd build
  cmake \
      -GNinja \
      -DAUTORUN_UNIT_TESTS=off \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=$PREFIX \
      -DBUILD_ONLY="s3;transfer;config;sts;cognito-identity;identity-management" \
      -DBUILD_SHARED_LIBS=${BUILD_SHARED_LIBS} \
      -DCUSTOM_MEMORY_MANAGEMENT=0 \
      -DCPP_STANDARD=17 \
      -DENABLE_TESTING=off \
      ..
  cmake --build . --parallel ${NPROC}
  sed -i 's/PARENT_SCOPE//g' AWSSDK/AWSSDKConfigVersion.cmake
  cmake --install .
  popd
  check_artifact_cleanup aws-sdk-cpp-${AWSCPP_VERSION}.tar.gz aws-sdk-cpp-${AWSCPP_VERSION}
}

LLVM_VERSION=14.0.6

function install_llvm() {
    local VERS=${LLVM_VERSION}
    local remote_repo="https://github.com/llvm/llvm-project/releases/download/llvmorg-${VERS}"
    download ${remote_repo}/llvm-$VERS.src.tar.xz
    download ${remote_repo}/clang-$VERS.src.tar.xz
    download ${remote_repo}/compiler-rt-$VERS.src.tar.xz
    download ${remote_repo}/clang-tools-extra-$VERS.src.tar.xz
    rm -rf llvm-$VERS.src
    extract llvm-$VERS.src.tar.xz
    extract clang-$VERS.src.tar.xz
    extract compiler-rt-$VERS.src.tar.xz
    extract clang-tools-extra-$VERS.src.tar.xz
    mv clang-$VERS.src llvm-$VERS.src/tools/clang
    mv compiler-rt-$VERS.src llvm-$VERS.src/projects/compiler-rt
    mkdir -p llvm-$VERS.src/tools/clang/tools
    mv clang-tools-extra-$VERS.src llvm-$VERS.src/tools/clang/tools/extra

    rm -rf build.llvm-$VERS
    mkdir build.llvm-$VERS
    pushd build.llvm-$VERS

    LLVM_SHARED=""
    if [ "$LIBRARY_TYPE" == "shared" ]; then
      LLVM_SHARED="-DLLVM_BUILD_LLVM_DYLIB=ON -DLLVM_LINK_LLVM_DYLIB=ON"
    fi

    if [ "$ARCH" == "x86_64" ]; then
      LLVM_TARGETS_TO_BUILD="X86"
    elif [ "$ARCH" == "aarch64" ]; then
      LLVM_TARGETS_TO_BUILD="AArch64"
    else
      echo "ERROR - Unsupported ARCH: ${ARCH}"
      exit 1
    fi
    LLVM_TARGETS_TO_BUILD="${LLVM_TARGETS_TO_BUILD};NVPTX"

    cmake \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=$PREFIX \
      -DLLVM_ENABLE_RTTI=on \
      -DLLVM_USE_INTEL_JITEVENTS=on \
      -DLLVM_ENABLE_LIBEDIT=off \
      -DLLVM_ENABLE_ZLIB=off \
      -DLLVM_INCLUDE_BENCHMARKS=off \
      -DLLVM_ENABLE_LIBXML2=off \
      -DLLVM_TARGETS_TO_BUILD="${LLVM_TARGETS_TO_BUILD}" \
      $LLVM_SHARED \
      ../llvm-$VERS.src
    makej
    make install
    popd
    check_artifact_cleanup clang-$VERS.src.tar.xz llvm-$VERS.src/tools/clang
    check_artifact_cleanup compiler-rt-$VERS.src.tar.xz llvm-$VERS.src/projects/compiler-rt
    check_artifact_cleanup clang-tools-extra-$VERS.src.tar.xz llvm-$VERS.src/tools/clang/tools/extra
    check_artifact_cleanup llvm-$VERS.src.tar.xz  llvm-$VERS.src
    if [[ $SAVE_SPACE == 'true' ]]; then
      rm -rf build.llvm-$VERS
    fi
}

THRIFT_VERSION=0.24.0

function install_thrift() {
    download https://archive.apache.org/dist/thrift/$THRIFT_VERSION/thrift-$THRIFT_VERSION.tar.gz
    extract thrift-$THRIFT_VERSION.tar.gz
    pushd thrift-$THRIFT_VERSION

    # For a future TSAN build, add -fsanitize=thread and -fno-omit-frame-pointer
    THRIFT_CFLAGS="-fPIC"
    THRIFT_CXXFLAGS="-fPIC"
    
    source /etc/os-release
    if [ "$ID" == "ubuntu"  ] ; then
      BOOST_LIBDIR="--with-boost=$PREFIX/include --with-boost-libdir=$PREFIX/lib"
    else
      BOOST_LIBDIR="--with-boost-libdir=$PREFIX/lib"
    fi
    CFLAGS="$THRIFT_CFLAGS" CXXFLAGS="$THRIFT_CXXFLAGS" JAVA_PREFIX=$PREFIX/lib ./configure \
        --prefix=$PREFIX \
        --enable-libs=yes \
        --enable-tests=no \
        --with-cpp \
        --without-go \
        --without-python \
        $BOOST_LIBDIR
    makej
    make install
    popd
    check_artifact_cleanup thrift-$THRIFT_VERSION.tar.gz thrift-$THRIFT_VERSION
}

SQLITE3_YEAR_DIR=2026
SQLITE3_VERSION=3530200

function install_sqlite3() {
    # SQLite (HeavyDB catalog, PROJ, GDAL)
    CFLAGS="-O2 -DSQLITE_ENABLE_RTREE=1" download_make_install https://sqlite.org/${SQLITE3_YEAR_DIR}/sqlite-autoconf-${SQLITE3_VERSION}.tar.gz
}

EXPAT_VERSION=2.8.3
PROJ_VERSION=9.6.0
GDAL_VERSION=3.10.3
TIFF_VERSION=4.7.2
GEOTIFF_VERSION=1.7.4
PDAL_VERSION=2.4.2
OPENJPEG_VERSION=2.5.4
OPENJPEG_SOURCE_SHA256=a695fbe19c0165f295a8531b1e4e855cd94d0875d2f88ec4b61080677e27188a
LCMS_VERSION=2.16
WEBP_VERSION=1.4.0
ZSTD_VERSION=1.5.6 # required by Arrow 16, also used by GDAL
HDF5_VERSION=2.2.0
NETCDF_VERSION=4.10.0
KML_VERSION=1.3.0

function install_gdal_and_pdal() {
    if [ "$LIBRARY_TYPE" == "static" ]; then
      BUILD_STATIC_LIBS=on
    else
      BUILD_STATIC_LIBS=off
    fi

    # expat (for gdal)
    local EXPAT_VERSION_DIR=R_$(echo ${EXPAT_VERSION} | tr '.' '_')
    download_make_install https://github.com/libexpat/libexpat/releases/download/${EXPAT_VERSION_DIR}/expat-${EXPAT_VERSION}.tar.bz2

    # kml (for gdal)
    download https://github.com/libkml/libkml/archive/refs/tags/${KML_VERSION}.tar.gz
    tar xvf ${KML_VERSION}.tar.gz
    pushd libkml-${KML_VERSION}
    mkdir build
    pushd build
    cmake .. \
      -DCMAKE_INSTALL_PREFIX=$PREFIX \
      -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
      -DBUILD_SHARED_LIBS=${BUILD_SHARED_LIBS}
    cmake_build_and_install
    popd
    popd
    check_artifact_cleanup ${KML_VERSION}.tar.gz libkml-${KML_VERSION}

    # hdf5 (for gdal)
    download https://support.hdfgroup.org/releases/hdf5/${HDF5_VERSION}/downloads/hdf5-${HDF5_VERSION}.tar.gz
    tar xzvf hdf5-${HDF5_VERSION}.tar.gz
    mkdir hdf5-${HDF5_VERSION}-build
    pushd hdf5-${HDF5_VERSION}-build
    cmake ../hdf5-${HDF5_VERSION} \
      -DCMAKE_INSTALL_PREFIX=${PREFIX} \
      -DCMAKE_INSTALL_LIBDIR=lib \
      -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
      -DBUILD_SHARED_LIBS=${BUILD_SHARED_LIBS} \
      -DBUILD_STATIC_LIBS=${BUILD_STATIC_LIBS} \
      -DBUILD_TESTING:BOOL=OFF \
      -DHDF5_BUILD_TOOLS:BOOL=OFF \
      -DHDF5_BUILD_FORTRAN:BOOL=OFF \
      -DHDF5_BUILD_JAVA:BOOL=OFF \
      -DHDF5_ENABLE_ZLIB_SUPPORT:BOOL=ON
    cmake_build_and_install          
    popd

    # netcdf (for gdal)
    download https://github.com/Unidata/netcdf-c/archive/refs/tags/v${NETCDF_VERSION}.tar.gz
    tar xzvf v${NETCDF_VERSION}.tar.gz
    pushd netcdf-c-${NETCDF_VERSION}
    CPPFLAGS=-I${PREFIX}/include LDFLAGS=-L${PREFIX}/lib ./configure --prefix=$PREFIX
    makej
    make install
    popd
    check_artifact_cleanup v${NETCDF_VERSION}.tar.gz netcdf-c-${NETCDF_VERSION}

    # webp (for tiff and openjpeg)
    download https://github.com/webmproject/libwebp/archive/refs/tags/v$WEBP_VERSION.tar.gz
    extract v$WEBP_VERSION.tar.gz
    ( cd libwebp-$WEBP_VERSION
      ./autogen.sh
      ./configure --prefix=$PREFIX
      makej
      make install
    )
    check_artifact_cleanup v$WEBP_VERSION.tar.gz libwebp-$WEBP_VERSION

    # tiff (for proj, geotiff, gdal)
    download http://download.osgeo.org/libtiff/tiff-${TIFF_VERSION}.tar.gz
    extract tiff-$TIFF_VERSION.tar.gz
    mkdir tiff-$TIFF_VERSION/build2
    ( cd tiff-$TIFF_VERSION/build2
      if [[ ${ID} == "rocky" ]]; then
        rm -f CMakeCache.txt
        cmake .. \
          -DBUILD_SHARED_LIBS=off \
          -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
          -DCMAKE_C_FLAGS="-fPIC" \
          -DCMAKE_INSTALL_PREFIX="$PREFIX" \
          -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
          -DCMAKE_PREFIX_PATH="$PREFIX" \
          -Dtiff-contrib=OFF \
          -Dtiff-docs=OFF \
          -Dtiff-tests=OFF \
          -Dtiff-tools=OFF
        cmake_build_and_install
      else
        # Build and install both libtiff.so and libtiff.a.
        # Static build requires libtiff.a and proj+gdal apps like ogrinfo require libtiff.so.
        for build_shared_libs in ON OFF; do
          rm -f CMakeCache.txt
          cmake .. \
            -DBUILD_SHARED_LIBS=$build_shared_libs \
            -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
            -DCMAKE_C_FLAGS="-fPIC" \
            -DCMAKE_INSTALL_PREFIX="$PREFIX" \
            -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
            -DCMAKE_PREFIX_PATH="$PREFIX" \
            -Dtiff-contrib=OFF \
            -Dtiff-docs=OFF \
            -Dtiff-tests=OFF \
            -Dtiff-tools=OFF
          cmake_build_and_install
        done
      fi
    )
    check_artifact_cleanup tiff-$TIFF_VERSION.tar.gz tiff-$TIFF_VERSION

    # proj (for geotiff, gdal)
    download https://download.osgeo.org/proj/proj-${PROJ_VERSION}.tar.gz
    tar xzvf proj-${PROJ_VERSION}.tar.gz
    mkdir proj-${PROJ_VERSION}/build
    ( cd proj-${PROJ_VERSION}/build
      cmake .. \
          -DBUILD_APPS=${BUILD_SHARED_LIBS} \
          -DBUILD_SHARED_LIBS=$BUILD_SHARED_LIBS \
          -DBUILD_TESTING=off \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_INSTALL_PREFIX=$PREFIX \
          -DCMAKE_PREFIX_PATH=$PREFIX \
          -DENABLE_TIFF=on
      cmake_build_and_install
    )
    check_artifact_cleanup proj-${PROJ_VERSION}.tar.gz proj-${PROJ_VERSION}

    # geotiff (for gdal, pdal)
    download https://github.com/OSGeo/libgeotiff/releases/download/${GEOTIFF_VERSION}/libgeotiff-$GEOTIFF_VERSION.tar.gz
    extract libgeotiff-$GEOTIFF_VERSION.tar.gz
    pushd libgeotiff-$GEOTIFF_VERSION
    sed -i 's/CHECK_FUNCTION_EXISTS(TIFFOpen HAVE_TIFFOPEN)/SET(HAVE_TIFFOPEN TRUE)/g' CMakeLists.txt
    sed -i 's/CHECK_FUNCTION_EXISTS(TIFFMergeFieldInfo HAVE_TIFFMERGEFIELDINFO)/SET(HAVE_TIFFMERGEFIELDINFO TRUE)/g' CMakeLists.txt
    mkdir build
    pushd build
    cmake .. -DCMAKE_INSTALL_PREFIX=$PREFIX -DBUILD_SHARED_LIBS=${BUILD_SHARED_LIBS} -DWITH_UTILITIES=off
    cmake_build_and_install
    popd
    popd
    check_artifact_cleanup libgeotiff-$GEOTIFF_VERSION.tar.gz libgeotiff-$GEOTIFF_VERSION

    # little cms (for openjpeg)
    download_make_install https://github.com/mm2/Little-CMS/archive/refs/tags/lcms${LCMS_VERSION}.tar.gz lcms${LCMS_VERSION}.tar.gz "Little-CMS-lcms${LCMS_VERSION}"

    # openjpeg (for gdal JP2/Sentinel2 support)
    download https://github.com/uclouvain/openjpeg/archive/refs/tags/v${OPENJPEG_VERSION}.tar.gz
    echo "${OPENJPEG_SOURCE_SHA256}  v${OPENJPEG_VERSION}.tar.gz" | sha256sum --check -
    tar xzvf v${OPENJPEG_VERSION}.tar.gz
    pushd openjpeg-${OPENJPEG_VERSION}
    # Backport upstream commit 839936aa33eb8899bbbd80fda02796bb65068951,
    # merged via uclouvain/openjpeg PR #1628. The patch is vendored in this
    # repository and is not downloaded during the build.
    patch -p1 < ${SCRIPTS_DIR}/openjpeg-cve-2026-6192.patch
    mkdir build
    pushd build
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${PREFIX} -DBUILD_CODEC=off -DBUILD_SHARED_LIBS=${BUILD_SHARED_LIBS} -DBUILD_STATIC_LIBS=${BUILD_STATIC_LIBS}
    makej
    make install
    popd
    popd
    check_artifact_cleanup v${OPENJPEG_VERSION}.tar.gz openjpeg-${OPENJPEG_VERSION}

    # gdal
    # use tiff and geotiff already built
    # disable geos, parquet and arrow (as before)
    # disable pcre (Perl Regex for SQLite3 driver which we don't use anyway)
    # disable opencl image processing acceleration (new in 3.7, don't need it)
    # disable use of libarchive (for 7z and RAR compression, but CentOS version is too old to use)
    download https://github.com/OSGeo/gdal/releases/download/v${GDAL_VERSION}/gdal-${GDAL_VERSION}.tar.gz
    tar xzvf gdal-${GDAL_VERSION}.tar.gz
    pushd gdal-${GDAL_VERSION}
    mkdir -p build
    pushd build
    cmake .. -DCMAKE_BUILD_TYPE=Release \
             -DCMAKE_C_FLAGS="$CFLAGS" \
             -DCMAKE_CXX_FLAGS="$CXXFLAGS" \
             -DCMAKE_INSTALL_PREFIX=$PREFIX \
             -DCMAKE_DISABLE_FIND_PACKAGE_Arrow=on \
             -DBUILD_SHARED_LIBS=${BUILD_SHARED_LIBS} \
             -DGDAL_USE_GEOS=off \
             -DGDAL_USE_ARROW=off \
             -DGDAL_USE_PARQUET=off \
             -DGDAL_USE_TIFF=${PREFIX} \
             -DGDAL_USE_GEOTIFF=${PREFIX} \
             -DGDAL_USE_ARCHIVE=off \
             -DGDAL_USE_PCRE=off \
             -DGDAL_USE_OPENCL=off \
             -DGDAL_USE_MYSQL=off \
             -DGDAL_USE_POSTGRESQL=off \
             -DGDAL_USE_XERCESC=off \
             -DBUILD_APPS=${BUILD_SHARED_LIBS} \
             -DBUILD_PYTHON_BINDINGS=off
    cmake_build_and_install
    popd
    popd
    check_artifact_cleanup gdal-${GDAL_VERSION}.tar.gz gdal-${GDAL_VERSION}

    # pdal
    download https://github.com/PDAL/PDAL/releases/download/${PDAL_VERSION}/PDAL-${PDAL_VERSION}-src.tar.bz2
    extract PDAL-${PDAL_VERSION}-src.tar.bz2
    pushd PDAL-${PDAL_VERSION}-src
    patch -p1 < $SCRIPTS_DIR/pdal-asan-leak-4be888818861d34145aca262014a00ee39c90b29.patch
    patch -p1 < $SCRIPTS_DIR/pdal-gdal-3.7.2-const-ogrspatialreference.patch
    if [ "$LIBRARY_TYPE" == "static" ] ; then
      # Build static libraries
      build_static_libs=$(printf "/%s/c %s%s" \
          'set(PDAL_LIB_TYPE "SHARED")' \
          'set(PDAL_LIB_TYPE "STATIC")\n' \
          'set(CMAKE_FIND_LIBRARY_SUFFIXES .a)')
      sed -i "$build_static_libs" cmake/libraries.cmake
      sed -Ei 's/PDAL_ADD_FREE_LIBRARY\((\S+) (SHARED|STATIC)/PDAL_ADD_LIBRARY(\1/' \
          pdal/util/CMakeLists.txt \
          vendor/arbiter/CMakeLists.txt \
          vendor/kazhdan/CMakeLists.txt \
          vendor/lazperf/CMakeLists.txt
      export_libs=$(printf '/^ *export( *$/,/^ *FILE *$/ { /^ *FILE *$/i\\\n%s%s%s\n}' \
          '        ${PDAL_ARBITER_LIB_NAME}\n' \
          '        ${PDAL_KAZHDAN_LIB_NAME}\n' \
          '        ${PDAL_LAZPERF_LIB_NAME}')
      sed -i "$export_libs" CMakeLists.txt
      # System libunwind.a has R_X86_64_32 code and causes linking problems w/ heavydb.
      sed -i 's|^include(${PDAL_CMAKE_DIR}/unwind.cmake)|#&|' pdal/util/CMakeLists.txt
      sed -i 's|^include(${PDAL_CMAKE_DIR}/execinfo.cmake)|#&|' pdal/util/CMakeLists.txt
      # Don't build libpdal_plugin_kernel_fauxplugin.so or bin/pdal
      sed -i 's/^/#/' plugins/faux/CMakeLists.txt
      sed -i '/# Configure build targets/,/# Targets installation/s/^/#/' apps/CMakeLists.txt
    fi
    mkdir build
    pushd build
    cmake .. -DCMAKE_CXX_FLAGS="$CXXFLAGS" \
             -DCMAKE_INSTALL_PREFIX=$PREFIX \
             -DCMAKE_POSITION_INDEPENDENT_CODE=$CMAKE_POSITION_INDEPENDENT_CODE \
             -DBUILD_PLUGIN_PGPOINTCLOUD=off \
             -DBUILD_SHARED_LIBS=$BUILD_SHARED_LIBS \
             -DWITH_TESTS=off
    cmake_build_and_install
    popd
    popd
    check_artifact_cleanup PDAL-${PDAL_VERSION}-src.tar.bz2 PDAL-${PDAL_VERSION}-src
}

GEOS_VERSION=3.11.1

function install_geos() {
    download https://download.osgeo.org/geos/geos-${GEOS_VERSION}.tar.bz2
    tar xvf geos-${GEOS_VERSION}.tar.bz2
    pushd geos-${GEOS_VERSION}
    mkdir build
    pushd build
    cmake .. -DCMAKE_BUILD_TYPE=Release \
             -DCMAKE_INSTALL_PREFIX=${PREFIX} \
             -DBUILD_SHARED_LIBS=on \
             -DBUILD_GEOSOP=off \
             -DBUILD_TESTING=off
    cmake_build_and_install
    popd
    popd
    check_artifact_cleanup geos-${GEOS_VERSION}.tar.bz2 geos-${GEOS_VERSION}
}

IWYU_VERSION=0.18
LLVM_VERSION_USED_FOR_IWYU=14.0.6
if [ "$LLVM_VERSION" != "$LLVM_VERSION_USED_FOR_IWYU" ]; then
  # NOTE: If you get this error, somebody upgraded LLVM, but they need to go
  # to https://include-what-you-use.org/ then scroll down, figure out which
  # iwyu version goes with the new LLVM_VERSION we're now using, then update
  # IWYU_VERSION and LLVM_VERSION_USED_FOR_IWYU above, appropriately.
  echo "ERROR: IWYU_VERSION of $IWYU_VERSION must be updated because LLVM_VERSION of $LLVM_VERSION_USED_FOR_IWYU was changed to $LLVM_VERSION"
  exit 1
fi
function install_iwyu() {
  download https://include-what-you-use.org/downloads/include-what-you-use-${IWYU_VERSION}.src.tar.gz
  extract include-what-you-use-${IWYU_VERSION}.src.tar.gz
  BUILD_DIR=include-what-you-use/build
  mkdir -p $BUILD_DIR
  pushd $BUILD_DIR
  cmake -G "Unix Makefiles" \
        -DCMAKE_PREFIX_PATH=${PREFIX}/lib \
        -DCMAKE_INSTALL_PREFIX=${PREFIX} \
        ..
  cmake_build_and_install
  popd
  check_artifact_cleanup "include-what-you-use-${IWYU_VERSION}.src.tar.gz" "include-what-you-use"
}

RDKAFKA_VERSION=2.14.2
function install_rdkafka() {
    if [ "$LIBRARY_TYPE" == "static" ]; then
      RDKAFKA_BUILD_STATIC="ON"
    else
      RDKAFKA_BUILD_STATIC="OFF"
    fi
    download https://github.com/confluentinc/librdkafka/archive/refs/tags/v$RDKAFKA_VERSION.tar.gz
    extract v$RDKAFKA_VERSION.tar.gz
    BDIR="librdkafka-$RDKAFKA_VERSION/build"
    mkdir -p $BDIR
    pushd $BDIR
    cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=$PREFIX \
        -DRDKAFKA_BUILD_STATIC=$RDKAFKA_BUILD_STATIC \
        -DRDKAFKA_BUILD_EXAMPLES=OFF \
        -DRDKAFKA_BUILD_TESTS=OFF \
        -DWITH_SASL=OFF \
        -DWITH_SSL=ON \
        ..
    makej
    make install
    popd
    check_artifact_cleanup  v$RDKAFKA_VERSION.tar.gz "librdkafka-$RDKAFKA_VERSION"
}

NINJA_VERSION=1.11.1

function install_ninja() {
  if [ "$ARCH" == "aarch64" ]; then
    # build from source as precompiled version not available for ARM
    download https://github.com/ninja-build/ninja/archive/refs/tags/v${NINJA_VERSION}.tar.gz
    tar xzvf v${NINJA_VERSION}.tar.gz
    pushd ninja-${NINJA_VERSION}
    cmake -Bbuild-cmake
    cmake --build build-cmake
    cp -f build-cmake/ninja ${PREFIX}/bin
    popd
    check_artifact_cleanup v${NINJA_VERSION}.tar.gz ninja-${NINJA_VERSION}
  else
    # download precompiled for x86
    download https://github.com/ninja-build/ninja/releases/download/v${NINJA_VERSION}/ninja-linux.zip
    unzip -u ninja-linux.zip
    mkdir -p $PREFIX/bin/
    mv ninja $PREFIX/bin/
    if [[ $SAVE_SPACE == 'true' ]]; then
      rm  ninja-linux.zip
    fi
  fi
}

MAVEN_VERSION=3.9.16

function install_maven() {
    download https://archive.apache.org/dist/maven/maven-3/${MAVEN_VERSION}/binaries/apache-maven-${MAVEN_VERSION}-bin.tar.gz
    extract apache-maven-${MAVEN_VERSION}-bin.tar.gz
    rm -rf $PREFIX/maven || true
    mv apache-maven-${MAVEN_VERSION} $PREFIX/maven
    # Configure the GCS Maven Central mirror in Maven's global settings so it also
    # applies during bootstrap. Core extensions (java/.mvn/extensions.xml, e.g.
    # project-settings-extension) are resolved before the project-level
    # java/.mvn/settings.xml mirror loads, so without a global mirror Maven tries
    # repo.maven.apache.org directly — unreachable from the build runner.
    cat > $PREFIX/maven/conf/settings.xml <<'MVN_SETTINGS_EOF'
<settings>
  <mirrors>
    <mirror>
      <id>gcs-maven-central</id>
      <name>Cloud Storage Maven Central</name>
      <url>https://maven-central.storage-download.googleapis.com/maven2/</url>
      <mirrorOf>central</mirrorOf>
    </mirror>
  </mirrors>
</settings>
MVN_SETTINGS_EOF
    if [[ $SAVE_SPACE == 'true' ]]; then
      rm apache-maven-${MAVEN_VERSION}-bin.tar.gz
    fi
}

TBB_VERSION=2021.9.0

function install_tbb() {
  download https://github.com/oneapi-src/oneTBB/archive/v${TBB_VERSION}.tar.gz
  extract v${TBB_VERSION}.tar.gz
  pushd oneTBB-${TBB_VERSION}
  mkdir -p build
  pushd build
  if [ "$TSAN" == "false" ]; then
    TBB_CFLAGS=""
    TBB_CXXFLAGS=""
    TBB_TSAN=""
  elif [ "$TSAN" = "true" ]; then
    TBB_CFLAGS="-fPIC -fsanitize=thread -fPIC -O1 -fno-omit-frame-pointer"
    TBB_CXXFLAGS="-fPIC -fsanitize=thread -fPIC -O1 -fno-omit-frame-pointer"
    TBB_TSAN="-DTBB_SANITIZE=thread"
  fi
  cmake -E env CFLAGS="$TBB_CFLAGS" CXXFLAGS="$TBB_CXXFLAGS" \
  cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=$PREFIX \
    -DTBB_TEST=off \
    -DBUILD_SHARED_LIBS=$BUILD_SHARED_LIBS \
    ${TBB_TSAN}
  makej
  make install
  popd
  popd
  check_artifact_cleanup v${TBB_VERSION}.tar.gz oneTBB-${TBB_VERSION}
}

ABSEIL_VERSION=20260107.1

function install_abseil() {
  rm -rf abseil
  mkdir -p abseil
  pushd abseil
  download https://github.com/abseil/abseil-cpp/archive/$ABSEIL_VERSION.tar.gz
  tar xvf $ABSEIL_VERSION.tar.gz
  pushd abseil-cpp-$ABSEIL_VERSION
  mkdir build
  pushd build
  cmake \
      -DCMAKE_INSTALL_PREFIX=$PREFIX \
      -DABSL_BUILD_TESTING=off \
      -DABSL_USE_GOOGLETEST_HEAD=off \
      -DABSL_PROPAGATE_CXX_STD=on \
      ..
  make install
  popd
  popd
  popd
}

VULKAN_VERSION=1.3.275.0 # 12/22/23
# updating past this version is not possible at this time due to glslang changes
# @TODO update to Vulkan SDK 1.4.x and use slang instead of glslang

function install_vulkan() {
  rm -rf vulkan
  mkdir -p vulkan/${VULKAN_VERSION}
  pushd vulkan
  pushd ${VULKAN_VERSION}
  # copy the build script locally
  \cp ${SCRIPTS_DIR}/../ThirdParty/vulkan/vulkansdk-${VULKAN_VERSION} vulkansdk
  # build just what we need for this platform
  ./vulkansdk --maxjobs --skip-deps loader glslang spirvcross vul layers
  # also add these non-default glslang headers
  \cp source/glslang/glslang/Include/InfoSink.h ${ARCH}/include/glslang/Include
  \cp source/glslang/glslang/Include/intermediate.h ${ARCH}/include/glslang/Include
  \cp source/glslang/glslang/Include/Common.h ${ARCH}/include/glslang/Include
  \cp source/glslang/glslang/Include/arrays.h ${ARCH}/include/glslang/Include
  \cp source/glslang/glslang/Include/BaseTypes.h ${ARCH}/include/glslang/Include
  \cp source/glslang/glslang/Include/Types.h ${ARCH}/include/glslang/Include
  \cp source/glslang/glslang/Include/PoolAlloc.h ${ARCH}/include/glslang/Include
  \cp source/glslang/glslang/Include/SpirvIntrinsics.h ${ARCH}/include/glslang/Include
  \cp source/glslang/glslang/Include/ConstantUnion.h ${ARCH}/include/glslang/Include
  \cp source/glslang/glslang/MachineIndependent/iomapper.h ${ARCH}/include/glslang/MachineIndependent
  \cp source/glslang/glslang/MachineIndependent/gl_types.h ${ARCH}/include/glslang/MachineIndependent
  \cp source/glslang/glslang/MachineIndependent/LiveTraverser.h ${ARCH}/include/glslang/MachineIndependent
  \cp source/glslang/glslang/MachineIndependent/localintermediate.h ${ARCH}/include/glslang/MachineIndependent
  \cp source/glslang/glslang/MachineIndependent/reflection.h ${ARCH}/include/glslang/MachineIndependent
  \cp source/glslang/build/include/glslang/build_info.h ${ARCH}/include/glslang
  \cp source/glslang/SPIRV/disassemble.h ${ARCH}/include/glslang/SPIRV
  popd
  # install
  rsync -av ${VULKAN_VERSION}/${ARCH}/* ${PREFIX}
  popd
}

GLM_VERSION=0.9.9.8

function install_glm() {
  download https://github.com/g-truc/glm/archive/refs/tags/${GLM_VERSION}.tar.gz
  extract ${GLM_VERSION}.tar.gz
  mkdir -p $PREFIX/include
  mv glm-${GLM_VERSION}/glm $PREFIX/include/
}



function install_blosc() {
  BLOSC_VERSION=1.21.2
  BLOSC_DLOAD=blosc_v${BLOSC_VERSION}.tar.gz
  download https://github.com/Blosc/c-blosc/archive/v${BLOSC_VERSION}.tar.gz ${BLOSC_DLOAD}
  tar xvf ${BLOSC_DLOAD}
  BDIR="c-blosc-${BLOSC_VERSION}/build"
  rm -rf "${BDIR}"
  mkdir -p "${BDIR}"
  pushd "${BDIR}"
  cmake \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=${PREFIX} \
      -DBUILD_BENCHMARKS=off \
      -DBUILD_TESTS=off \
      -DPREFER_EXTERNAL_SNAPPY=off \
      -DPREFER_EXTERNAL_ZLIB=off \
      -DPREFER_EXTERNAL_ZSTD=off \
      ..
  make -j ${NPROC}
  make install
  popd
  check_artifact_cleanup ${BLOSC_DLOAD} $BDIR
}

oneDAL_VERSION=2024.1.0
function install_onedal() {
  download https://github.com/oneapi-src/oneDAL/archive/refs/tags/${oneDAL_VERSION}.tar.gz
  extract ${oneDAL_VERSION}.tar.gz
  pushd oneDAL-${oneDAL_VERSION}
  ./dev/download_micromkl.sh
  if [ "$LIBRARY_TYPE" == "static" ]; then
    # oneDAL makefile only detects libTBB built as shared lib, so we hack it to allow static built
    sed -Ei 's/libtbb.so(\.\d+)?/libtbb.a/g' makefile
    sed -Ei 's/libtbbmalloc.so(\.\d+)?/libtbbmalloc.a/g' makefile

    # do not release shared library targets on CentOS
    sed -i '/foreach t,$(releasetbb\.LIBS_Y)/d' makefile
    sed -i 's/$(core_y) \\//g' makefile
    sed -i 's/:= $(oneapi_y)/:= /g' makefile
    sed -i '/$(thr_$(i)_y))/d' makefile

    # oneDAL always builds static and shared versions of its libraries by default, so hack the
    # makefile again to remove shared library building (the preceding spaces matter here)
    sed -i 's/ $(WORKDIR\.lib)\/$(core_y)//g' makefile
    sed -i 's/ $(WORKDIR\.lib)\/$(thr_tbb_y)//g' makefile
  fi

  # oneDAL's makefile hardcodes its libTBB directory to /gcc4.8/, make it so it looks in the
  # root PREFIX (where we install TBB)
  sed -i 's/$(_IA)\/gcc4\.8//g' makefile

  # explicitly use python3
  # could install python-is-python3 module, but that's overkill if only this needs it
  sed -i 's/python/python3/g' makefile

  # fix issues with c++20
  # remove c++ standard override in cmake
  # remove unnecessary template params from contructor decls (gcc fix)
  patch -p1 < $SCRIPTS_DIR/patch-onedal-cpp20.patch

  # these exports will only be valid in the subshell that builds oneDAL
  (export TBBROOT=${PREFIX}; \
   export LD_LIBRARY_PATH="${PREFIX}/lib64:${PREFIX}/lib:${LD_LIBRARY_PATH}"; \
   export LIBRARY_PATH="${PREFIX}/lib64:${PREFIX}/lib:${LIBRARY_PATH}"; \
   export CPATH="${PREFIX}/include:${CPATH}"; \
   export PATH="${PREFIX}/bin:${PATH}"; \
   make -f makefile daal_c oneapi_c PLAT=lnx32e REQCPU="avx2 avx512" COMPILER=gnu -j ${NPROC})

  # remove deprecated compression methods as they generate DEPRECATED warnings/errors
  sed -i '/bzip2compression\.h/d' __release_lnx_gnu/daal/latest/include/daal.h
  sed -i '/zlibcompression\.h/d' __release_lnx_gnu/daal/latest/include/daal.h

  mkdir -p $PREFIX/include
  cp -r __release_lnx_gnu/daal/latest/include/* $PREFIX/include
  cp -r __release_lnx_gnu/daal/latest/lib/intel64/* $PREFIX/lib
  mkdir -p ${PREFIX}/lib/cmake/oneDAL
  cp __release_lnx_gnu/daal/latest/lib/cmake/oneDAL/*.cmake ${PREFIX}/lib/cmake/oneDAL/.
  popd
  check_artifact_cleanup ${oneDAL_VERSION}.tar.gz oneDAL-${oneDAL_VERSION}
}

MOLD_VERSION=1.10.1

function install_mold() {
  download https://github.com/rui314/mold/releases/download/v${MOLD_VERSION}/mold-${MOLD_VERSION}-${ARCH}-linux.tar.gz
  tar --strip-components=1 -xvf mold-${MOLD_VERSION}-${ARCH}-linux.tar.gz -C ${PREFIX}
}

BZIP2_VERSION=1.0.6
BZIP_DLOAD=bzip2-${BZIP2_VERSION}.tar.gz
function install_bzip2() {
  download https://sourceforge.net/projects/kanapi/files/sources/Packages/mirror/${BZIP_DLOAD}/download ${BZIP_DLOAD}
  extract ${BZIP_DLOAD}
  pushd bzip2-${BZIP2_VERSION}
  sed -i 's/O2 -g \$/O2 -g -fPIC \$/' Makefile
  makej
  make install PREFIX=$PREFIX
  popd
  check_artifact_cleanup bzip2-${BZIP2_VERSION}.tar.gz bzip2-${BZIP2_VERSION}
}

DOUBLE_CONVERSION_VERSION=3.1.5
function install_double_conversion() {

  download https://github.com/google/double-conversion/archive/v${DOUBLE_CONVERSION_VERSION}.tar.gz
  extract v${DOUBLE_CONVERSION_VERSION}.tar.gz
  mkdir -p double-conversion-${DOUBLE_CONVERSION_VERSION}/build
  pushd double-conversion-${DOUBLE_CONVERSION_VERSION}/build
  cmake -DCMAKE_CXX_FLAGS="-fPIC" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$PREFIX ..
  makej
  make install
  popd
  check_artifact_cleanup  v${DOUBLE_CONVERSION_VERSION}.tar.gz double-conversion-${DOUBLE_CONVERSION_VERSION}
}

ARCHIVE_VERSION=2.2.2
function install_archive(){
  download https://github.com/gflags/gflags/archive/v$ARCHIVE_VERSION.tar.gz
  extract v$ARCHIVE_VERSION.tar.gz
  mkdir -p gflags-$ARCHIVE_VERSION/build
  pushd gflags-$ARCHIVE_VERSION/build
  cmake -DCMAKE_CXX_FLAGS="-fPIC" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$PREFIX ..
  makej
  make install
  popd
  check_artifact_cleanup  v${ARCHIVE_VERSION}.tar.gz gflags-$ARCHIVE_VERSION
}

LZ4_VERSION=1.10.0 # required by Arrow 16
function install_lz4(){
  download https://github.com/lz4/lz4/archive/refs/tags/v$LZ4_VERSION.tar.gz
  extract v$LZ4_VERSION.tar.gz
  ( cd lz4-$LZ4_VERSION/build
    cmake cmake \
        -DBUILD_SHARED_LIBS="$BUILD_SHARED_LIBS" \
        -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
        -DCMAKE_C_FLAGS="$CFLAGS" \
        -DCMAKE_INSTALL_PREFIX="$PREFIX" \
        -DCMAKE_POSITION_INDEPENDENT_CODE="$CMAKE_POSITION_INDEPENDENT_CODE"
    cmake_build_and_install
  )
  check_artifact_cleanup v$LZ4_VERSION.tar.gz lz4-$LZ4_VERSION
}

function install_zstd() {
  ## note BUILD_SHARED_LIBS set in calling script
  if [ "$LIBRARY_TYPE" == "static" ]; then
    BUILD_STATIC_LIBS=on
  else
    BUILD_STATIC_LIBS=off
  fi
  download https://github.com/facebook/zstd/archive/refs/tags/v$ZSTD_VERSION.tar.gz
  extract v$ZSTD_VERSION.tar.gz
  mkdir zstd-$ZSTD_VERSION/build/cmake/build
  ( cd zstd-$ZSTD_VERSION/build/cmake/build
    cmake .. -DCMAKE_INSTALL_PREFIX=$PREFIX -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DZSTD_BUILD_PROGRAMS=OFF -DZSTD_BUILD_SHARED=$BUILD_SHARED_LIBS -DZSTD_BUILD_STATIC=$BUILD_STATIC_LIBS
    cmake_build_and_install
  )
  check_artifact_cleanup v$ZSTD_VERSION.tar.gz zstd-$ZSTD_VERSION
}

URIPARSER_VERSION=0.9.8
function install_uriparser() {
  NAME="uriparser-$URIPARSER_VERSION"
  download https://github.com/uriparser/uriparser/archive/refs/tags/$NAME.tar.gz
  extract $NAME.tar.gz
  mkdir uriparser-$NAME/build
  ( cd uriparser-$NAME/build
    cmake .. \
        -DBUILD_SHARED_LIBS="$BUILD_SHARED_LIBS" \
        -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
        -DCMAKE_CXX_FLAGS="$CXXFLAGS" \
        -DCMAKE_C_FLAGS="$CFLAGS" \
        -DCMAKE_INSTALL_PREFIX="$PREFIX" \
        -DCMAKE_POSITION_INDEPENDENT_CODE="$CMAKE_POSITION_INDEPENDENT_CODE" \
        -DURIPARSER_BUILD_DOCS=off \
        -DURIPARSER_BUILD_TESTS=off
    makej
    make install
  )
  check_artifact_cleanup $NAME.tar.gz uriparser-$NAME
}

function safe_mkdir() {
  if [ ! -e "$1" ] || { [ -d "$1" ] && [ -z "$(ls -A "$1")" ]; }; then
    sudo mkdir -p "$1"
    sudo chown $(id -u) "$1"
  else
    echo "Error: $1 must either not exist or be an empty directory."
    exit 1
  fi
}

function safe_symlink() {
  if [ ! -e "$2" ] || [ -L "$2" ]; then
    sudo ln -fnrs "$1" "$2"
    sudo chown $(id -u) "$2"
  else
    echo "Error: $2 must either not exist or be a symbolic link."
    exit 1
  fi
}

H3_VERSION=4.2.0

function install_h3() {
  download https://github.com/uber/h3/archive/refs/tags/v${H3_VERSION}.tar.gz
  extract v${H3_VERSION}.tar.gz
  pushd h3-${H3_VERSION}
  mkdir build
  pushd build
  cmake \
    -DCMAKE_INSTALL_PREFIX=${PREFIX} \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POSITION_INDEPENDENT_CODE=on \
    -DBUILD_BENCHMARKS=off \
    -DBUILD_FILTERS=off \
    -DBUILD_FUZZERS=off \
    -DBUILD_GENERATORS=off \
    -DBUILD_TESTING=off \
    -DENABLE_DOCS=off \
    -DENABLE_WARNINGS=off \
    ..
  cmake_build_and_install
  popd
  popd
  check_artifact_cleanup v${H3_VERSION}.tar.gz h3-${H3_VERSION}
}

CPR_VERSION=1.11.2

function install_cpr() {
  download https://github.com/libcpr/cpr/archive/refs/tags/${CPR_VERSION}.tar.gz cpr-${CPR_VERSION}.tar.gz
  extract cpr-${CPR_VERSION}.tar.gz
  pushd cpr-${CPR_VERSION}
  mkdir build
  pushd build
  cmake \
    -DCMAKE_INSTALL_PREFIX=${PREFIX} \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POSITION_INDEPENDENT_CODE=on \
    -DCPR_USE_SYSTEM_CURL=on \
    -DBUILD_SHARED_LIBS=${BUILD_SHARED_LIBS} \
    ..
  cmake_build_and_install
  popd
  popd
  check_artifact_cleanup cpr-${CPR_VERSION}.tar.gz cpr-${CPR_VERSION}
}

XML_SECURITY_C_VERSION=2.0.4
XML_TOOLING_VERSION=3.0.4
OPENSAML_VERSION=3.0.1

function install_opensaml() {
  # xml-security-c
  # TODO: Test newer xml-security-c-3.0.0.tar.gz at https://shibboleth.net/downloads/xml-security-c/3.0.0/xml-security-c-3.0.0.tar.gz
  download_make_install https://archive.apache.org/dist/santuario/c-library/xml-security-c-${XML_SECURITY_C_VERSION}.tar.gz xml-security-c-${XML_SECURITY_C_VERSION}.tar.gz "" "$CONFIGURE_OPTS --without-xalan"
  
  # xmltooling and opensaml
  # These projects have a dependency on log4shib and LOG4CPP, which we do not want to build or link against as they are GPL
  # So, we patch the source to remove those dependencies before building

  # xmltooling
  # Yes, the download subdirectory is still 3.0.1 even though the version is 3.0.4
  download https://shibboleth.net/downloads/c++-opensaml/3.0.1/xmltooling-${XML_TOOLING_VERSION}.tar.gz
  extract xmltooling-${XML_TOOLING_VERSION}.tar.gz
  pushd xmltooling-${XML_TOOLING_VERSION}
  rm -f doc/LOG4CPP.LICENSE
  rm -f configure
  rm -f doc/Makefile.in
  rm -f xmltooling/Makefile.in
  patch -s -p1 < ../xmltooling-${XML_TOOLING_VERSION}-remove-log4shib.patch
  autoreconf -f -i
  ./configure --prefix=$PREFIX ${CONFIGURE_OPTS}
  makej
  make_install
  popd
  check_artifact_cleanup xmltooling-${XML_TOOLING_VERSION}.tar.gz xmltooling-${XML_TOOLING_VERSION}

  # opensaml
  download https://shibboleth.net/downloads/c++-opensaml/${OPENSAML_VERSION}/opensaml-${OPENSAML_VERSION}.tar.gz
  extract opensaml-${OPENSAML_VERSION}.tar.gz
  pushd opensaml-${OPENSAML_VERSION}
  rm -f doc/LOG4CPP.LICENSE
  rm -f configure
  rm -f doc/Makefile.in
  patch -s -p1 < ../opensaml-${OPENSAML_VERSION}-remove-log4shib.patch
  autoreconf -f -i
  CXXFLAGS="-std=c++14" ./configure --prefix=$PREFIX ${CONFIGURE_OPTS}
  makej
  make_install
  popd
  check_artifact_cleanup opensaml-${OPENSAML_VERSION}.tar.gz opensaml-${OPENSAML_VERSION}
}

LIBPNG_VERSION=1.6.58 # latest 1.6 release as of 20260604, suggested pngcrush 1.7.88

function install_png() {
  download_make_install http://download.sourceforge.net/libpng/libpng-$LIBPNG_VERSION.tar.xz
}

# generate_mapd_deps_sh PREFIX
# Writes $PREFIX/mapd-deps.sh with the standard environment variable exports.
# Uses sudo tee when $PREFIX is not writable by the current user.
# On Rocky Linux, also defaults CC and CXX to the heavydb gcc/g++ when unset.
function generate_mapd_deps_sh() {
  local prefix="$1"
  local tee_cmd="tee"
  [ -w "$prefix" ] || tee_cmd="sudo tee"
  local java_home=""
  if command -v java &>/dev/null; then
    java_home=$(dirname "$(dirname "$(readlink -f "$(command -v java)")")")
  fi
  $tee_cmd "$prefix/mapd-deps.sh" > /dev/null <<EOF
HEAVY_PREFIX=$prefix

LD_LIBRARY_PATH=/usr/local/cuda/lib64:\${LD_LIBRARY_PATH:-}
LD_LIBRARY_PATH=\$HEAVY_PREFIX/lib:\$LD_LIBRARY_PATH
LD_LIBRARY_PATH=\$HEAVY_PREFIX/lib64:\$LD_LIBRARY_PATH

PATH=/usr/local/cuda/bin:\${PATH:-}
PATH=\$HEAVY_PREFIX/maven/bin:\$PATH
PATH=\$HEAVY_PREFIX/bin:\$PATH

VULKAN_SDK=\$HEAVY_PREFIX
VK_LAYER_PATH=\$HEAVY_PREFIX/share/vulkan/explicit_layer.d

CMAKE_PREFIX_PATH=\$HEAVY_PREFIX:\${CMAKE_PREFIX_PATH:-}

JAVA_HOME=$java_home

export LD_LIBRARY_PATH PATH VULKAN_SDK VK_LAYER_PATH CMAKE_PREFIX_PATH JAVA_HOME
EOF

  if [[ ${ID} == "rocky" ]]; then
    $tee_cmd -a "$prefix/mapd-deps.sh" > /dev/null <<EOF
CC=\${CC:-\$HEAVY_PREFIX/bin/gcc}
CXX=\${CXX:-\$HEAVY_PREFIX/bin/g++}
export CC CXX
EOF
  fi

  # Pin Vulkan and EGL to NVIDIA manifests when present on this host.
  # Evaluated live when mapd-deps.sh is sourced (not at generation time).
  if [ -w "$prefix" ]; then
    cp "$SCRIPTS_DIR/nvidia-graphics-env.sh" "$prefix/nvidia-graphics-env.sh"
    chmod +x "$prefix/nvidia-graphics-env.sh"
  else
    sudo cp "$SCRIPTS_DIR/nvidia-graphics-env.sh" "$prefix/nvidia-graphics-env.sh"
    sudo chmod +x "$prefix/nvidia-graphics-env.sh"
  fi
  $tee_cmd -a "$prefix/mapd-deps.sh" > /dev/null <<EOF
# NVIDIA Vulkan/EGL pinning — evaluated on this host when sourced.
# Avoids Mesa EGL / LLVM symbol collisions during Vulkan bootstrap.
if [[ -f "\$HEAVY_PREFIX/nvidia-graphics-env.sh" ]]; then
  source "\$HEAVY_PREFIX/nvidia-graphics-env.sh"
  export_nvidia_graphics_env || true
fi
EOF
}

# install_profile_entry PREFIX ENABLE
# Symlinks $PREFIX/mapd-deps.sh to /etc/profile.d/xx-mapd-deps.sh when
# ENABLE is "true"; in both cases prints sourcing instructions.
function install_profile_entry() {
  local prefix="$1"
  local enable="${2:-false}"
  local profpath=/etc/profile.d/xx-mapd-deps.sh
  echo
  if [ "$enable" = "true" ] ; then
    sudo ln -sf "$prefix/mapd-deps.sh" "$profpath"
    echo "Done. A file at $profpath has been created and will be run on startup"
    echo "Source this file or reboot to load vars in this shell"
  else
    echo "Done. Be sure to source the 'mapd-deps.sh' file to pick up the required environment variables:"
    echo "    source $prefix/mapd-deps.sh"
  fi
}

# compress_deps_tarball OS LIBRARY_TYPE ARCH SUFFIX TSAN NPROC PREFIX
# Creates and compresses the deps tarball from $PREFIX.
function compress_deps_tarball() {
  local os="$1" lib_type="$2" arch="$3" suffix="$4" tsan="$5" nproc="$6" prefix="$7"
  local tsan_tag=""
  [ "$tsan" = "true" ] && tsan_tag="-tsan"
  local filename="mapd-deps-${os}${tsan_tag}-${lib_type}-${arch}-${suffix}.tar"
  tar cvf "$filename" -C "$prefix" .
  xz -T"$nproc" "$filename"
}
