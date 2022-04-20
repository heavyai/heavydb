#!/usr/bin/env bash

set -ex
[ -z "$PREFIX" ] && PREFIX=${CONDA_PREFIX:-/usr/local}
export PREFIX=$(cd "$PREFIX"; pwd -P)

this_dir=$(dirname "${BASH_SOURCE[0]}")

bash $this_dir/build.sh
bash $this_dir/install-omniscidb-common.sh
cmake --install build --component "exe" --prefix $PREFIX
