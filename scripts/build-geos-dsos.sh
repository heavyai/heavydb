#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
