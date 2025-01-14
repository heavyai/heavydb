#!/usr/bin/env bash
#
########################################################################################
#
# Copyright 2025 HEAVY.AI, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
########################################################################################
#
# BUILD SCRIPT FOR libgeos DSOs FOR USE WITH HEAVY.AI HEAVYDB
# 
#   For Docker installations, build for Ubuntu 22.04
#   For bare-metal installations, build for native OS
#
########################################################################################
#
# DOCKER BUILD CONTAINER SET-UP
#
# Ubuntu (for HeavyDB Docker installation)
# Required version 22.04
#
#   docker pull ubuntu:22.04
#   docker run -it -v /home/$USER:/home/$USER ubuntu:22.04 /bin/bash
#   apt update
#   apt install build-essential cmake wget
#
# Red Hat (for RHEL/Rocky/Fedora bare-metal installation)
# Supported version 8.10
#
#   docker pull rockylinux/rockylinux:8.10
#   docker run -it -v /home/$USER:/home/$USER rockylinux/rockylinux:8.10 /bin/bash
#   dnf update
#   dnf group install "Development Tools"
#   dnf install cmake wget
#
########################################################################################
#
# BARE-METAL BUILD REQUIREMENTS
#
#   gcc & g++ (ideally v11.x but v8.5 on Red Hat should also work)
#   cmake
#   wget
#   xz
#
########################################################################################
#
# BUILD INSTRUCTIONS
#
# Go to a suitable location (which we will call $BUILD_LOCATION) and run this script.
#
# If it runs successfully, it will result in a file of the form:
#
#   heavydb-libgeos-$OS-$ARCH-$DATE.tar.xz
#
# e.g.
#
#   heavydb-libgeos-ubuntu22.04-x86_64-20250107.tar.xz
# 
########################################################################################
#
# INSTALLATION
#
# First install HeavyDB (container or bare-metal) per regular instructions, then:
#
#   cd /var/lib/heavyai
#   mkdir libgeos
#   cd libgeos
#   tar xvf $BUILD_LOCATION/heavydb-libgeos-$OS-$ARCH-$DATE.tar.xz
#
########################################################################################

set -e
set -x

NPROC=8
echo "Building with ${NPROC} cores"

HTTP_DEPS="https://dependencies.mapd.com/thirdparty"

source /etc/os-release

GEOS_VERSION=3.11.1

OS=${ID}${VERSION_ID}
ARCH=$(uname -m)
SUFFIX=${SUFFIX:=$(date +%Y%m%d)}

FILENAME=heavydb-libgeos-${OS}-${ARCH}-${SUFFIX}.tar

wget --continue ${HTTP_DEPS}/geos-${GEOS_VERSION}.tar.bz2
tar xvf geos-${GEOS_VERSION}.tar.bz2

pushd geos-${GEOS_VERSION}

mkdir build
mkdir install

pushd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_INSTALL_PREFIX=../install \
         -DBUILD_SHARED_LIBS=on \
         -DBUILD_GEOSOP=off \
         -DBUILD_TESTING=off
cmake --build . --parallel ${NPROC} && cmake --install .
popd

pushd install
if [ "${ID}" == "rocky" ] || [ "${ID}" == "rhel" ]; then
  pushd lib64
else
  pushd lib
fi
tar cvf ../../../${FILENAME} libgeos*
xz -T${NPROC} ../../../${FILENAME}
popd
popd

popd
rm -rf geos-${GEOS_VERSION}.tar.bz2 geos-${GEOS_VERSION}
