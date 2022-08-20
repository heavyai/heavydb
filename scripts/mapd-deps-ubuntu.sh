#!/usr/bin/env bash

set -e
set -x

# Parse inputs
TSAN=false
COMPRESS=false
NOCUDA=false

while (( $# )); do
  case "$1" in
    --compress)
      COMPRESS=true
      ;;
    --tsan)
      TSAN=true
      ;;
    --nocuda)
      NOCUDA=true
      ;;
    *)
      break
      ;;
  esac
  shift
done

HTTP_DEPS="https://dependencies.mapd.com/thirdparty"

SUFFIX=${SUFFIX:=$(date +%Y%m%d)}
PREFIX=/usr/local/mapd-deps

SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $SCRIPTS_DIR/common-functions.sh

# Establish distro
source /etc/os-release
if [ "$ID" == "ubuntu" ] ; then
  PACKAGER="apt -y"
  if [ "$VERSION_ID" != "20.04" ] && [ "$VERSION_ID" != "19.10" ] && [ "$VERSION_ID" != "19.04" ] && [ "$VERSION_ID" != "18.04" ]; then
    echo "Ubuntu 20.04, 19.10, 19.04, and 18.04 are the only debian-based releases supported by this script"
    exit 1
  fi
else
  echo "Only Ubuntu is supported by this script"
  exit 1
fi

sudo mkdir -p $PREFIX
sudo chown -R $(id -u) $PREFIX

# this should be based on the actual distro and arch, but they're the same files.
DEBIAN_FRONTEND=noninteractive sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub

DEBIAN_FRONTEND=noninteractive sudo apt update

# required for gcc-9 on Ubuntu 18.04
if [ "$VERSION_ID" == "18.04" ]; then
  DEBIAN_FRONTEND=noninteractive sudo apt install -y software-properties-common
  DEBIAN_FRONTEND=noninteractive sudo add-apt-repository ppa:ubuntu-toolchain-r/test
fi

DEBIAN_FRONTEND=noninteractive sudo apt install -y \
    software-properties-common \
    build-essential \
    ccache \
    git \
    wget \
    curl \
    gcc-9 \
    g++-9 \
    libboost-all-dev \
    libgoogle-glog-dev \
    libssl-dev \
    libevent-dev \
    default-jre \
    default-jre-headless \
    default-jdk \
    default-jdk-headless \
    libncurses5-dev \
    libldap2-dev \
    binutils-dev \
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
    libcurl4-openssl-dev \
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
    python-dev \
    python-yaml \
    swig \
    pkg-config \
    libxerces-c-dev \
    libxmlsec1-dev \
    libtool \
    patchelf \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxi-dev

# Set up gcc-8 as default gcc
sudo update-alternatives \
  --install /usr/bin/gcc gcc /usr/bin/gcc-9 900 \
  --slave /usr/bin/g++ g++ /usr/bin/g++-9

# Needed to find sqlite3, xmltooling, and xml_security_c
export PKG_CONFIG_PATH=$PREFIX/lib/pkgconfig:$PREFIX/lib64/pkgconfig:$PKG_CONFIG_PATH
export PATH=$PREFIX/bin:$PATH

install_ninja

install_maven

install_cmake

install_memkind

# llvm
# (see common-functions.sh)
LLVM_BUILD_DYLIB=true
install_llvm

# c-blosc
install_blosc

# Geo Support
install_gdal
install_geos
install_pdal

# install AWS core and s3 sdk
install_awscpp -j $(nproc)

# thrift
install_thrift

VERS=3.52.15
CFLAGS="-fPIC" CXXFLAGS="-fPIC" download_make_install ${HTTP_DEPS}/libiodbc-${VERS}.tar.gz

install_folly

install_iwyu

download_make_install ${HTTP_DEPS}/bisonpp-1.21-45.tar.gz bison++-1.21

# TBB
install_tbb

# Apache Arrow (see common-functions.sh)
ARROW_BOOST_USE_SHARED="ON"
install_arrow

# Go
install_go

# librdkafka
install_rdkafka

# libuv
install_libuv

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

# GLM (GL Mathematics)
install_glm

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
VERS=1.87
rm -rf imgui
mkdir -p imgui
pushd imgui
download https://github.com/ocornut/imgui/archive/refs/tags/v${VERS}.tar.gz
extract v${VERS}.tar.gz
mkdir -p $PREFIX/include
mkdir -p $PREFIX/include/imgui
mv imgui-${VERS}/* $PREFIX/include/imgui
popd #imgui

# Vulkan
# Custom tarball which excludes the spir-v toolchain
VERS=1.2.198.1 # stable 12/3/21
rm -rf vulkan
mkdir -p vulkan
pushd vulkan
wget --continue ${HTTP_DEPS}/vulkansdk-linux-x86_64-no-spirv-$VERS.tar.gz -O vulkansdk-linux-x86_64-no-spirv-$VERS.tar.gz
tar xvf vulkansdk-linux-x86_64-no-spirv-$VERS.tar.gz
rsync -av $VERS/x86_64/* $PREFIX
popd # vulkan

# OpenSAML
download_make_install ${HTTP_DEPS}/xml-security-c-2.0.2.tar.gz "" "--without-xalan"
download_make_install ${HTTP_DEPS}/xmltooling-3.0.4-nolog4shib.tar.gz
download_make_install ${HTTP_DEPS}/opensaml-3.0.1-nolog4shib.tar.gz

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
