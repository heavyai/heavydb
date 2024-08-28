#!/usr/bin/env bash

set -e
set -x

# Parse inputs
TSAN=false
COMPRESS=false
NOCUDA=false
CACHE=
LIBRARY_TYPE=

# Establish number of cores to compile with
# Default to 8, Limit to 24
# Can be overridden with --nproc option
NPROC=$(nproc)
NPROC=${NPROC:-8}
if [ "${NPROC}" -gt "24" ]; then
  NPROC=24
fi

while (( $# )); do
  case "$1" in
    --compress)
      COMPRESS=true
      ;;
    --savespace)
      SAVE_SPACE=true
      ;;
    --tsan)
      TSAN=true
      ;;
    --nocuda)
      NOCUDA=true
      ;;
    --cache=*)
      CACHE="${1#*=}"
      ;;
    --static)
      LIBRARY_TYPE=static
      ;;
    --shared)
      LIBRARY_TYPE=shared
      ;;
    --nproc=*)
      NPROC="${1#*=}"
      ;;
    *)
      break
      ;;
  esac
  shift
done

# Validate LIBRARY_TYPE
if [ "$LIBRARY_TYPE" == "" ] ; then
  echo "ERROR - Library type must be specified (--static or --shared)"
  exit
fi

# Establish architecture
ARCH=$(uname -m)

if [[ -n $CACHE && ( ! -d $CACHE  ||  ! -w $CACHE )  ]]; then
  # To prevent possible mistakes CACHE must be a writable directory
  echo "Invalid cache argument [$CACHE] supplied. Ignoring."
  CACHE=
fi

echo "Building with ${NPROC} cores"

if [[ ! -x  "$(command -v sudo)" ]] ; then
  if [ "$EUID" -eq 0 ] ; then
    apt update -y
    apt install -y sudo
  else
    echo "ERROR - sudo not installed and not running as root"
    exit
  fi
fi

HTTP_DEPS="https://dependencies.mapd.com/thirdparty"

SUFFIX=${SUFFIX:=$(date +%Y%m%d)}
PREFIX=/usr/local/mapd-deps

CMAKE_BUILD_TYPE=Release

if [ "$LIBRARY_TYPE" == "static" ]; then
  ARROW_BOOST_USE_SHARED=off
  BUILD_SHARED_LIBS=off
  CFLAGS=-fPIC
  CMAKE_POSITION_INDEPENDENT_CODE=on
  CONFIGURE_OPTS="--enable-static --disable-shared"
  CXXFLAGS=-fPIC
else
  ARROW_BOOST_USE_SHARED=on
  BUILD_SHARED_LIBS=on
  CFLAGS=""
  CMAKE_POSITION_INDEPENDENT_CODE=off
  CONFIGURE_OPTS=""
  CXXFLAGS=""
fi

SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $SCRIPTS_DIR/common-functions.sh

# Establish distro
source /etc/os-release
if [ "$ID" == "ubuntu" ] ; then
  PACKAGER="apt -y"
  if [ "$VERSION_ID" != "24.04" ] && [ "$VERSION_ID" != "22.04" ]; then
    echo "Ubuntu 24.04 and 22.04 are the only Debian-based releases supported by this script"
    echo "If you are still using 20.04 or 23.10 then you need to upgrade!"
    exit 1
  fi
else
  echo "Only Ubuntu is supported by this script"
  exit 1
fi

safe_mkdir "$PREFIX"

# this should be based on the actual distro, but they're the same files.
DEBIAN_FRONTEND=noninteractive sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub

DEBIAN_FRONTEND=noninteractive sudo apt update

install_required_ubuntu_packages

DEBIAN_FRONTEND=noninteractive sudo apt install -y \
  gcc-11 \
  g++-11

# Set up gcc-11 as default gcc
DEBIAN_FRONTEND=noninteractive sudo update-alternatives \
  --install /usr/bin/gcc gcc /usr/bin/gcc-11 1100 \
  --slave /usr/bin/g++ g++ /usr/bin/g++-11

generate_deps_version_file

# Needed to find sqlite3, xmltooling, xml_security_c, and LLVM (for iwyu)
export PKG_CONFIG_PATH=$PREFIX/lib/pkgconfig:$PREFIX/lib64/pkgconfig:$PKG_CONFIG_PATH
export PATH=$PREFIX/bin:$PREFIX/include:$PATH
export LD_LIBRARY_PATH=$PREFIX/lib64:$PREFIX/lib:$LD_LIBRARY_PATH

# mold fast linker
install_mold

install_maven

install_openssl

if [ "$LIBRARY_TYPE" == "static" ]; then
  install_openldap2
fi

install_cmake

install_ninja

install_boost
export BOOST_ROOT=$PREFIX/include

install_memkind

VERS=3.3.2
CFLAGS="$CFLAGS" download_make_install ${HTTP_DEPS}/libarchive-$VERS.tar.gz "" "$CONFIGURE_OPTS --without-nettle"

install_uriparser

VERS=7.75.0
# https://curl.haxx.se/download/curl-$VERS.tar.xz
download_make_install ${HTTP_DEPS}/curl-$VERS.tar.xz "" "--disable-ldap --disable-ldaps"

# c-blosc
install_blosc

# Geo Support
install_gdal_and_pdal
install_geos

# llvm
# (see common-functions.sh)
install_llvm

# install AWS core and s3 sdk
install_awscpp

# thrift
install_thrift

VERS=3.52.16
CFLAGS="-fPIC" CXXFLAGS="-fPIC" download_make_install ${HTTP_DEPS}/libiodbc-${VERS}.tar.gz

# Include What You Use
install_iwyu

# bison
download_make_install ${HTTP_DEPS}/bisonpp-1.21-45.tar.gz bison++-1.21

# TBB
install_tbb

# OneDAL (Intel only)
if [ "$ARCH" == "x86_64" ] ; then
  install_onedal
fi

# jemalloc (ARM only)
if [ "$ARCH" == "aarch64" ] ; then
  install_jemalloc
fi

# Apache Arrow (see common-functions.sh)
install_arrow

# Go
install_go

# librdkafka
install_rdkafka

# abseil
install_abseil

# glslang (with spirv-tools)
VERS=11.6.0 # stable 8/25/21
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
VERS=2020-06-29 # latest from 6/29/20
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
install_vulkan

# GLM (GL Mathematics)
install_glm

# LZ4 required by rdkafka
install_lz4

# Rendering sandbox support
if [ "$LIBRARY_TYPE" != "static" ]; then
  install_glfw
  install_imgui
  install_implot
fi

# OpenSAML
download_make_install ${HTTP_DEPS}/xml-security-c-2.0.4.tar.gz "" "$CONFIGURE_OPTS --without-xalan"
download_make_install ${HTTP_DEPS}/xmltooling-3.0.4-nolog4shib.tar.gz "" "$CONFIGURE_OPTS"
CXXFLAGS="-std=c++14" download_make_install ${HTTP_DEPS}/opensaml-3.0.1-nolog4shib.tar.gz "" "$CONFIGURE_OPTS"

# Generate mapd-deps.sh
cat > $PREFIX/mapd-deps.sh <<EOF
HEAVY_PREFIX=$PREFIX

LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH
LD_LIBRARY_PATH=\$HEAVY_PREFIX/lib:\$LD_LIBRARY_PATH
LD_LIBRARY_PATH=\$HEAVY_PREFIX/lib64:\$LD_LIBRARY_PATH

PATH=/usr/local/cuda/bin:\$PATH
PATH=\$HEAVY_PREFIX/go/bin:\$PATH
PATH=\$HEAVY_PREFIX/maven/bin:\$PATH
PATH=\$HEAVY_PREFIX/bin:\$PATH

VULKAN_SDK=\$HEAVY_PREFIX
VK_LAYER_PATH=\$HEAVY_PREFIX/share/vulkan/explicit_layer.d

CMAKE_PREFIX_PATH=\$HEAVY_PREFIX:\$CMAKE_PREFIX_PATH

GOROOT=\$HEAVY_PREFIX/go

export LD_LIBRARY_PATH PATH VULKAN_SDK VK_LAYER_PATH CMAKE_PREFIX_PATH GOROOT
EOF

echo
echo "Done. Be sure to source the 'mapd-deps.sh' file to pick up the required environment variables:"
echo "    source $PREFIX/mapd-deps.sh"

if [ "$COMPRESS" = "true" ]; then
  OS=ubuntu${VERSION_ID}
  # we don't have 24.04 builds yet, so just use the 22.04 bundle
  if [ $VERSION_ID == "24.04" ]; then
    OS=ubuntu22.04
  fi
  TARBALL_TSAN=""
  if [ "$TSAN" = "true" ]; then
    TARBALL_TSAN="-tsan"
  fi
  FILENAME=mapd-deps-${OS}${TARBALL_TSAN}-${LIBRARY_TYPE}-${ARCH}-${SUFFIX}.tar
  tar cvf ${FILENAME} -C ${PREFIX} .
  xz -T${NPROC} ${FILENAME}
fi
