#!/usr/bin/env bash

set -e
set -x

# Parse inputs
TSAN=false
COMPRESS=false
NOCUDA=false
CACHE=

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
    *)
      break
      ;;
  esac
  shift
done

if [[ -n $CACHE && ( ! -d $CACHE  ||  ! -w $CACHE )  ]]; then
  # To prevent possible mistakes CACHE must be a writable directory
  echo "Invalid cache argument [$CACHE] supplied. Ignoring."
  CACHE=
fi

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

SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $SCRIPTS_DIR/common-functions.sh

# Establish distro
source /etc/os-release
if [ "$ID" == "ubuntu" ] ; then
  PACKAGER="apt -y"
  if [ "$VERSION_ID" != "23.10" ] && [ "$VERSION_ID" != "22.04" ] && [ "$VERSION_ID" != "20.04" ]; then
    echo "Ubuntu 23.10, 22.04, and 20.04 are the only debian-based releases supported by this script"
    exit 1
  fi
else
  echo "Only Ubuntu is supported by this script"
  exit 1
fi

sudo mkdir -p $PREFIX
sudo chown -R $(id -u) $PREFIX
# create a  txt file in $PREFIX

# this should be based on the actual distro and arch, but they're the same files.
DEBIAN_FRONTEND=noninteractive sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub

DEBIAN_FRONTEND=noninteractive sudo apt update

DEBIAN_FRONTEND=noninteractive sudo apt install -y \
    software-properties-common \
    build-essential \
    ccache \
    git \
    wget \
    curl \
    libevent-dev \
    default-jre \
    default-jre-headless \
    default-jdk \
    default-jdk-headless \
    libncurses5-dev \
    libldap2-dev \
    google-perftools \
    libdouble-conversion-dev \
    libevent-dev \
    libgflags-dev \
    libgoogle-perftools-dev \
    libiberty-dev \
    libjemalloc-dev \
    libglu1-mesa-dev \
    liblz4-dev \
    liblzma-dev \
    libbz2-dev \
    libarchive-dev \
    libedit-dev \
    uuid-dev \
    libsnappy-dev \
    zlib1g-dev \
    autoconf \
    autoconf-archive \
    automake \
    bison \
    flex \
    libpng-dev \
    rsync \
    unzip \
    jq \
    python3-dev \
    python3-yaml \
    swig \
    pkg-config \
    libxerces-c-dev \
    libtool \
    patchelf \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxi-dev \
    libegl-dev \
    binutils-dev \
    libnuma-dev

if [ "$VERSION_ID" == "20.04" ]; then
  # required for gcc-11 on Ubuntu < 22.04
  DEBIAN_FRONTEND=noninteractive sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
fi

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
install_mold_precompiled_x86_64

install_ninja

install_maven

install_openssl

install_cmake

install_boost
export BOOST_ROOT=$PREFIX/include

install_memkind

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
LLVM_BUILD_DYLIB=true
install_llvm

# install AWS core and s3 sdk
install_awscpp -j $(nproc)

# thrift
install_thrift

VERS=3.52.16
CFLAGS="-fPIC" CXXFLAGS="-fPIC" download_make_install ${HTTP_DEPS}/libiodbc-${VERS}.tar.gz

# fmt (must be installed before folly)
install_fmt

install_folly

# Include What You Use
install_iwyu

# bison
download_make_install ${HTTP_DEPS}/bisonpp-1.21-45.tar.gz bison++-1.21

# TBB
install_tbb

# OneDAL
install_onedal

# Apache Arrow (see common-functions.sh)
ARROW_BOOST_USE_SHARED="ON"
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

# Rendering sandbox support
# GLFW
VERS=3.3.6
download https://github.com/glfw/glfw/archive/refs/tags/${VERS}.tar.gz
extract ${VERS}.tar.gz
pushd glfw-${VERS}
mkdir -p build
pushd build
cmake \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_INSTALL_PREFIX=$PREFIX \
    -DBUILD_SHARED_LIBS=ON \
    -DGLFW_BUILD_EXAMPLES=OFF \
    -DGLFW_BUILD_TESTS=OFF \
    -DGLFW_BUILD_DOCS=OFF \
    ..
make -j $(nproc)
make install
popd #build
popd #glfw

# ImGui
VERS=1.89.1-docking
rm -rf imgui
mkdir -p imgui
pushd imgui
wget --continue ${HTTP_DEPS}/imgui.$VERS.tar.gz
tar xvf imgui.$VERS.tar.gz
mkdir -p $PREFIX/include
mkdir -p $PREFIX/include/imgui
rsync -av imgui.$VERS/* $PREFIX/include/imgui
popd #imgui

# ImPlot
VERS=0.14
rm -rf implot
mkdir -p implot
pushd implot
wget --continue ${HTTP_DEPS}/implot.$VERS.tar.gz
tar xvf implot.$VERS.tar.gz
# Patch #includes for imgui.h / imgui_internal.h
pushd implot.$VERS
patch -p0 < $SCRIPTS_DIR/implot-0.14_fix_imgui_includes.patch
popd
mkdir -p $PREFIX/include
mkdir -p $PREFIX/include/implot
rsync -av implot.$VERS/* $PREFIX/include/implot
popd #implot

# OpenSAML
download_make_install ${HTTP_DEPS}/xml-security-c-2.0.4.tar.gz "" "--without-xalan"
download_make_install ${HTTP_DEPS}/xmltooling-3.0.4-nolog4shib.tar.gz
CXXFLAGS="-std=c++14" download_make_install ${HTTP_DEPS}/opensaml-3.0.1-nolog4shib.tar.gz

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
VK_LAYER_PATH=\$HEAVY_PREFIX/etc/vulkan/explicit_layer.d

CMAKE_PREFIX_PATH=\$HEAVY_PREFIX:\$CMAKE_PREFIX_PATH

GOROOT=\$HEAVY_PREFIX/go

export LD_LIBRARY_PATH PATH VULKAN_SDK VK_LAYER_PATH CMAKE_PREFIX_PATH GOROOT
EOF

echo
echo "Done. Be sure to source the 'mapd-deps.sh' file to pick up the required environment variables:"
echo "    source $PREFIX/mapd-deps.sh"

if [ "$COMPRESS" = "true" ] ; then
    if [ "$TSAN" = "false" ]; then
      TARBALL_TSAN=""
    elif [ "$TSAN" = "true" ]; then
      TARBALL_TSAN="tsan-"
    fi
    tar acvf mapd-deps-ubuntu-${VERSION_ID}-${TARBALL_TSAN}${SUFFIX}.tar.xz -C ${PREFIX} .
fi
