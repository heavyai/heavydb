#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# Build the sqlite ODBC driver from source

set -x

mkdir /deps-build/
pushd /deps-build/

# Prevent apt-get asking for input
export DEBIAN_FRONTEND=noninteractive
# Implement workaround to prevent errors such as 
#  error: RPC failed; curl 56 GnuTLS recv error (-110): The TLS connection was non-properly terminated.
# See related issue: https://github.com/argoproj/argo-cd/issues/3994
apt-get update
apt-get install -y gnutls-bin
git config --global http.sslVerify false
git config --global http.postBuffer 1048576000

git clone https://github.com/sqlite/sqlite.git
pushd sqlite
CFLAGS=-fPIC ./configure --prefix=/opt/sqlite
autoconf
make
make install
popd

git clone https://github.com/softace/sqliteodbc
pushd sqliteodbc
LDFLAGS="$LDFLAGS -L/opt/sqlite/lib" ./configure --with-sqlite3=/opt/sqlite --prefix=/opt/sqliteodbc --with-odbc=/usr/local/mapd-deps/ --enable-static=no
mkdir -p /opt/sqliteodbc/lib
make
make install
popd

popd

rm -fr /deps-build/
