#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e

DEBUG_FLAG=false

# Parse inputs
TSAN=false
COMPRESS=false
SAVE_SPACE=false
UPDATE_PACKAGES=false
UPDATE_OPTIONS=
CACHE=

BUILD_SHARED_LIBS=off
LIBRARY_TYPE=static

CFLAGS=-fPIC
CMAKE_POSITION_INDEPENDENT_CODE=on
CONFIGURE_OPTS="--enable-static --disable-shared"
CXXFLAGS=-fPIC

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
    --cache=*)
      CACHE="${1#*=}"
      ;;
    --static)
      ;;
    --shared)
      echo "ERROR - Only --static mode supported for Rocky"
      exit
      ;;
    --debug)
      DEBUG_FLAG=true
      ;;
    --nproc=*)
      NPROC="${1#*=}"
      ;;
    *)
      echo "Unexpected argument $1"
      exit
      ;;
  esac
  shift
done

[[ ${DEBUG_FLAG} == true ]] && set -x

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
  dnf update -y ${UPDATE_OPTIONS}
fi

# install sudo if we don't have it
if [[ ! -x  "$(command -v sudo)" ]] ; then
  if [ "$EUID" -eq 0 ] ; then
    dnf install -y sudo
  else
    echo "ERROR - sudo not installed and not running as root"
    exit
  fi
fi

SUFFIX=${SUFFIX:=$(date +%Y%m%d)}
PREFIX="/usr/local/mapd-deps"

SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source ${SCRIPTS_DIR}/common-functions.sh

## To help with running debugging and running
## individual components/functions create a .env.sh
## file.
[[ ${DEBUG_FLAG} == true ]]  && cat << __DEBUG_ENV > .rocky_env.sh
  source /etc/os-release
  source ${SCRIPTS_DIR}/common-functions.sh
  DEBUG_FLAG=${DEBUG_FLAG}
  ARCH=${ARCH}
  CACHE=${CACHE}
  SUFFIX=${SUFFIX}
  PREFIX=${PREFIX}
  NPROC=${NPROC}
  TSAN=${TSAN}
  SAVE_SPACE=${SAVE_SPACE}
  UPDATE_PACKAGES=${UPDATE_PACKAGES}
  COMPRESS=${COMPRESS}
  BUILD_SHARED_LIBS=${BUILD_SHARED_LIBS}
  LIBRARY_TYPE=${LIBRARY_TYPE}
  CFLAGS=${CFLAGS}
  CMAKE_POSITION_INDEPENDENT_CODE=${CMAKE_POSITION_INDEPENDENT_CODE}
  CONFIGURE_OPTS=${CONFIGURE_OPTS}
  CXXFLAGS=${CXXFLAGS}
__DEBUG_ENV

if [ ! -w $(dirname $PREFIX) ] ; then
    ## if we don't have write access to PREFIX
    ## use SUDO to set perms
    SUDO=sudo
fi
$SUDO mkdir -p $PREFIX
$SUDO chown -R $(id -u) $PREFIX
export PATH=$PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$PREFIX/lib64:$PREFIX/lib:$LD_LIBRARY_PATH

# Needed to find xmltooling and xml_security_c
export PKG_CONFIG_PATH=$PREFIX/lib/pkgconfig:$PREFIX/lib64/pkgconfig:$PKG_CONFIG_PATH
[[ ${DEBUG_FLAG} == true ]]  && cat << __DEBUG_ENV_PATH >> .rocky_env.sh
PATH=${PATH}
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}
PKG_CONFIG_PATH=${PKG_CONFIG_PATH}
__DEBUG_ENV_PATH

sudo dnf groupinstall -y "Development Tools"
dnf list installed | grep java-1.8.0-openjdk-headless >/dev/null; ST=$?
if [[ $ST -eq 0 ]] ; then
  sudo dnf remove -y java-1.8.0-openjdk-headless
fi
install_required_rockylinux_packages

generate_deps_version_file

# mold fast linker
install_mold

# gmp, mpfr, mpc, autoconf, automake
download_make_install https://ftp.gnu.org/gnu/gmp/gmp-6.1.2.tar.xz gmp-6.1.2.tar.xz "" "--enable-fat"
download_make_install https://www.mpfr.org/mpfr-3.1.5/mpfr-3.1.5.tar.xz mpfr-3.1.5.tar.xz "" "--with-gmp=$PREFIX"
download_make_install ftp://ftp.gnu.org/gnu/mpc/mpc-1.1.0.tar.gz mpc-1.1.0.tar.gz "" "--with-gmp=$PREFIX"

install_gcc_rocky

export CC=$PREFIX/bin/gcc
export CXX=$PREFIX/bin/g++

[[ ${DEBUG_FLAG} == true ]]  && cat << __DEBUG_ENV_GCC >> .rocky_env.sh
CC=${CC}
CXX=${CXX}
__DEBUG_ENV_GCC

install_openssl

install_cmake_rocky

install_ninja

install_maven

install_openldap2

LIBTOOL_VER=2.4.6
download_make_install ftp://ftp.gnu.org/gnu/libtool/libtool-${LIBTOOL_VER}.tar.gz

# requires libtool
install_libmd

# icu and bzip2 (needed for boost)
install_icu

install_bzip2

install_boost
# TODO(scb): BOOST_ROOT may no longer be necessary (it was only to build xerces-c *I think*)
export BOOST_ROOT=$PREFIX/include

[[ ${DEBUG_FLAG} == true ]]  && cat << __DEBUG_ENV_BOOST >> .rocky_env.sh
BOOST_ROOT=${BOOST_ROOT}
__DEBUG_ENV_BOOST

install_uriparser

install_xz

# libarchive
LIBARCHIVE_VERSION=3.8.7
CFLAGS="-fPIC" download_make_install https://libarchive.org/downloads/libarchive-${LIBARCHIVE_VERSION}.tar.gz libarchive-${LIBARCHIVE_VERSION}.tar.gz "" "--without-openssl --disable-shared"

# @TODO
# Why do we need these?
# Stock Rocky 8 is ncurses 6.1, readline 7.0
# Is it because we need static versions?
CFLAGS="-fPIC" download_make_install ftp://ftp.gnu.org/pub/gnu/ncurses/ncurses-6.4.tar.gz # "" "--build=powerpc64le-unknown-linux-gnu" 
CFLAGS="-fPIC" download_make_install ftp://ftp.gnu.org/gnu/readline/readline-7.0.tar.gz readline-7.0.tar.gz  "" "--disable-shared"

install_double_conversion

install_archive

GLOG_VERS=0.3.5
GLOG_DLOAD=v$GLOG_VERS.tar.gz
CXXFLAGS="-fPIC -std=c++11" download_make_install https://github.com/google/glog/archive/${GLOG_DLOAD} ${GLOG_DLOAD} glog-$GLOG_VERS "--enable-shared=no"

CURL_VERS=8.20.0
CURL_DLOAD=curl-$CURL_VERS.tar.xz
download_make_install https://curl.se/download/${CURL_DLOAD} ${CURL_DLOAD} ""  "--disable-ldap --disable-ldaps --with-openssl --without-libpsl --without-libidn2"

# cpr
install_cpr

# thrift
install_thrift

# librdkafka
install_rdkafka

# libpng
install_png

install_snappy
 
VERS=3.52.16
IO_DLOAD=libiodbc-${VERS}.tar.gz
CFLAGS="-fPIC" CXXFLAGS="-fPIC" download_make_install https://sourceforge.net/projects/iodbc/files/iodbc/3.52.16/${IO_DLOAD}/download ${IO_DLOAD}
# c-blosc
install_blosc

# zstd required by GDAL and Arrow
install_zstd

# SQLite (catalog, PROJ, GDAL)
install_sqlite3

# Geo Support
install_gdal_and_pdal
install_geos

CPPFLAGS="-I$PREFIX/include/ncurses" download_make_install http://thrysoee.dk/editline/libedit-20230828-3.1.tar.gz

# llvm
install_llvm 

install_iwyu 

# TBB
install_tbb

# OneDAL (Intel only)
if [ "$ARCH" == "x86_64" ] ; then
  install_onedal
fi

# install AWS core and s3 sdk
install_awscpp

# LZ4 required by rdkafka and Arrow
install_lz4

# Apache Arrow
install_arrow

# abseil
install_abseil

# Vulkan
install_vulkan

# GLM (GL Mathematics)
install_glm

# OpenSAML
install_xerces_c
install_opensaml

# H3
install_h3

generate_mapd_deps_sh "$PREFIX"

if [ "$COMPRESS" = "true" ] ; then
  compress_deps_tarball "rockylinux8" "static" "$ARCH" "$SUFFIX" "$TSAN" "$NPROC" "$PREFIX"
fi

install_profile_entry "$PREFIX" "$ENABLE"
source $PREFIX/mapd-deps.sh

echo "Finished!!!"
