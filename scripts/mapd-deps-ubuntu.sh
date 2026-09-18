#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e
set -x

# Parse inputs
UPDATE_PACKAGES=false
UPDATE_OPTIONS=
COMPRESS=false
TSAN=false
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
    --update-packages)
      UPDATE_PACKAGES=true
      ;;
    --update-options=*)
      UPDATE_OPTIONS="${1#*=}"
      ;;
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

# update packages if requested
# pass options to allow dodging broken updates if necessary (e.g. --ignore=package1,package2)
if [ "${UPDATE_PACKAGES}" = "true" ]; then
  apt update -y ${UPDATE_OPTIONS}
fi

# install sudo if we don't have it
if [[ ! -x  "$(command -v sudo)" ]] ; then
  if [ "$EUID" -eq 0 ] ; then
    apt install -y sudo
  else
    echo "ERROR - sudo not installed and not running as root"
    exit
  fi
fi

SUFFIX=${SUFFIX:=$(date +%Y%m%d)}
PREFIX=/usr/local/mapd-deps

CMAKE_BUILD_TYPE=Release

if [ "$LIBRARY_TYPE" == "static" ]; then
  BUILD_SHARED_LIBS=off
  CFLAGS=-fPIC
  CMAKE_POSITION_INDEPENDENT_CODE=on
  CONFIGURE_OPTS="--enable-static --disable-shared"
  CXXFLAGS=-fPIC
else
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
  if [ "$VERSION_ID" != "22.04" ]; then
    echo "Ubuntu 22.04 is the only Debian-based release supported by this script"
    exit 1
  fi
else
  echo "Only Ubuntu is supported by this script"
  exit 1
fi

safe_mkdir "$PREFIX"

# this should be based on the actual distro, but they're the same files.
DEBIAN_FRONTEND=noninteractive sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub

install_required_ubuntu_packages

DEBIAN_FRONTEND=noninteractive sudo apt install -y \
  gcc-11 \
  g++-11

# Set up gcc-11 as default gcc
sudo update-alternatives \
  --install /usr/bin/gcc gcc /usr/bin/gcc-11 1100 \
  --slave /usr/bin/g++ g++ /usr/bin/g++-11
sudo update-alternatives --set gcc /usr/bin/gcc-11

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

install_xz

LIBARCHIVE_VERSION=3.8.7
CFLAGS="$CFLAGS" download_make_install https://libarchive.org/downloads/libarchive-${LIBARCHIVE_VERSION}.tar.gz libarchive-${LIBARCHIVE_VERSION}.tar.gz "" "$CONFIGURE_OPTS --without-nettle"

install_uriparser

VERS=8.20.0
C_DLOAD=curl-$VERS.tar.xz
download_make_install https://curl.se/download/${C_DLOAD} ${C_DLOAD} "" "--disable-ldap --disable-ldaps --with-openssl --without-libpsl --without-libidn2"

# cpr
install_cpr

# libpng
install_png

# c-blosc
install_blosc

# zstd required by GDAL and Arrow
install_zstd

# SQLite (catalog, PROJ, GDAL)
install_sqlite3

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
IO_DLOAD=libiodbc-${VERS}.tar.gz
CFLAGS="-fPIC" CXXFLAGS="-fPIC" download_make_install https://sourceforge.net/projects/iodbc/files/iodbc/3.52.16/${IO_DLOAD}/download ${IO_DLOAD}

# Include What You Use
install_iwyu

# TBB
install_tbb

# OneDAL (Intel only)
if [ "$ARCH" == "x86_64" ] ; then
  install_onedal
fi

# LZ4 required by rdkafka and Arrow
install_lz4

# Apache Arrow
install_arrow

# librdkafka
install_rdkafka

# abseil
install_abseil

# Vulkan
install_vulkan

# GLM (GL Mathematics)
install_glm

# OpenSAML
install_opensaml

# H3
install_h3

generate_mapd_deps_sh "$PREFIX"
install_profile_entry "$PREFIX"

if [ "$COMPRESS" = "true" ]; then
  OS=ubuntu${VERSION_ID}
  compress_deps_tarball "$OS" "$LIBRARY_TYPE" "$ARCH" "$SUFFIX" "$TSAN" "$NPROC" "$PREFIX"
fi
