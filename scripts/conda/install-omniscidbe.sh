#!/usr/bin/env bash
set -xe
cmake --install build --component "DBE" --prefix ${PREFIX:-/usr/local}
