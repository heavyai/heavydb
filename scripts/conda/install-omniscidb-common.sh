#!/usr/bin/env bash
set -xe
[ -z "$PREFIX" ] && export PREFIX=${CONDA_PREFIX:-/usr/local}
cmake --install build --component "include" --prefix $PREFIX/include/omnisci
cmake --install build --component "doc" --prefix $PREFIX/share/doc/omnisci
cmake --install build --component "data" --prefix $PREFIX/opt/omnisci
cmake --install build --component "thrift" --prefix $PREFIX/opt/omnisci
cmake --install build --component "QE" --prefix $PREFIX
cmake --install build --component "jar" --prefix $PREFIX
cmake --install build --component "Unspecified" --prefix $PREFIX/opt/omnisci
