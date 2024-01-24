#!/bin/bash
#
# run this inside a suitably-old Ubuntu container
# docker-internal.mapd.com/cudagl:11.8.0-devel-ubuntu18.04 is a good one
#
# e.g
# docker run -it -v $REPO/scripts:/scripts docker-internal.mapd.com/cudagl:11.8.0-devel-ubuntu18.04 /scripts/build_vulkan_loader_dso.sh
#

VULKAN_VERSION=1.3.275.0
DSO_VERSION=1.3.275
BUILD_TYPE=Release
BUILD_DIR=/build
INSTALL_DIR=/scripts/vulkan_loader

# prepare
mkdir -p ${BUILD_DIR}
mkdir -p ${INSTALL_DIR}
cd ${BUILD_DIR}

# install basics
apt-get -y update
apt-get -y upgrade
# required for python 3.7 (required for 1.3.275)
sudo add-apt-repository ppa:deadsnakes/ppa
apt-get -y install git build-essential libx11-xcb-dev libxkbcommon-dev libwayland-dev libxrandr-dev wget python3.7 libssl-dev

# install or build cmake
BUILD_CMAKE=false
if [ "${BUILD_CMAKE}" = "true" ]; then
	# build locally
	CMAKE_VERSION=3.25.2
	wget https://dependencies.mapd.com/thirdparty/cmake-${CMAKE_VERSION}.tar.gz
	tar xzf cmake-${CMAKE_VERSION}.tar.gz
	pushd cmake-${CMAKE_VERSION}
	./configure
	make -j
	make install
	popd
else
	# install from KitWare APT repo
	apt-get -y remove --purge --auto-remove cmake
	apt-get -y install software-properties-common lsb-release
	wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
	apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main"
	apt-get -y update
	apt-get -y install cmake
fi

# build loader
wget https://github.com/KhronosGroup/Vulkan-Loader/archive/refs/tags/vulkan-sdk-${VULKAN_VERSION}.tar.gz
tar xzf vulkan-sdk-${VULKAN_VERSION}.tar.gz
pushd Vulkan-Loader-vulkan-sdk-${VULKAN_VERSION}
mkdir build
pushd build
cmake -DUPDATE_DEPS=ON -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DCMAKE_INSTALL_PREFIX=install ..
make -j
make install
chmod 755 install/lib/libvulkan.so.${DSO_VERSION}
cp -f install/lib/libvulkan.so.${DSO_VERSION} ${INSTALL_DIR}
popd
popd
