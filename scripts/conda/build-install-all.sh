#!/usr/bin/env bash

set -ex
[ -z "$PREFIX" ] && export PREFIX=${CONDA_PREFIX:-/usr/local}
this_dir=$(dirname "${BASH_SOURCE[0]}")

bash $this_dir/build.sh
bash $this_dir/install-omniscidb-common.sh
cmake --install build --component "exe" --prefix $PREFIX
cmake --install build --component "DBE" --prefix $PREFIX
bash $this_dir/install-omniscidbe4py.sh
