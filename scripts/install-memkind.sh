#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
# Copyright 2019, Intel Corporation

#
# install-memkind.sh <OS> - installs memkind from sources; depends on
#		the system it uses proper installation paramaters
#

set -e

OS=$1

# v1.11.0, contains new libmemkind namespace
MEMKIND_VERSION=v1.11.0

GIT_REPO="https://github.com/memkind/memkind"

WORKDIR=$(pwd)

git clone ${GIT_REPO}
cd $WORKDIR/memkind
git checkout $MEMKIND_VERSION

export NDCTL_LIBRARY_VERSION=v70.1

./utils/docker/docker_install_ndctl.sh

# set OS-specific configure options
OS_SPECIFIC=""
case $(echo $OS | cut -d'-' -f1) in
	centos|opensuse)
		OS_SPECIFIC="--libdir=/usr/lib64"
		;;
esac

./autogen.sh
./configure --prefix=/usr $OS_SPECIFIC
make -j$(nproc)
make -j$(nproc) install

# cleanup
cd $WORKDIR
rm -r memkind
