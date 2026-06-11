#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# NOTE: This will perform a full system update via the command yay -Suy.
# See https://wiki.archlinux.org/title/System_maintenance#Partial_upgrades_are_unsupported
# for more information.

# Must be run from the scripts/ directory as the non-root user.
# Since we use an older version of Apache Arrow, automatic updates to arrow can be avoided by
# adding it to the uncommented IgnorePkg line in /etc/pacman.conf. Example:
# IgnorePkg   = arrow

set -e
set -x

cd "$( dirname "${BASH_SOURCE[0]}" )"

hash yay || { echo >&2 "yay is required but is not installed. Aborting."; exit 1; }

unset CMAKE_GENERATOR

# Install all normal dependencies
yay -Suy \
    aws-sdk-cpp \
    blosc \
    boost \
    c-ares \
    clang14 \
    cmake \
    compiler-rt14 \
    cpr \
    cuda \
    curl \
    double-conversion \
    doxygen \
    flex \
    fmt \
    gcc \
    gdal-libkml \
    geos \
    git \
    glm \
    glslang \
    go \
    hdf5 \
    intel-tbb \
    jdk-openjdk \
    libiodbc \
    librdkafka \
    llvm14 \
    lz4 \
    maven \
    minio \
    ninja \
    pdal \
    python-numpy \
    snappy \
    spirv-cross \
    thrift \
    vulkan-headers \
    vulkan-utility-libraries \
    vulkan-validation-layers \
    wget \
    zlib

# Install Arrow
( cd arch/arrow
  patch -p4 < heavyai.patch
  makepkg -cis
)

# Install oneDAL
( cd arch/onedal
  makepkg -cis
)

# Install Bison++
( cd arch/bison++
  makepkg -cis
)
