#!/bin/bash

set -e
set -x

HTTP_DEPS="https://dependencies.mapd.com/thirdparty"

SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $SCRIPTS_DIR/../common-functions.sh

VERS=3.0.2
download ${HTTP_DEPS}/xmltooling-3.0.2.tar.gz
extract xmltooling-$VERS.tar.gz
pushd xmltooling-$VERS
patch -s -p1 < ../xmltooling.patch
autoreconf -f -i
popd

mv xmltooling-$VERS xmltooling-$VERS-nolog4shib
tar -zcvf xmltooling-$VERS-nolog4shib.tar.gz xmltooling-$VERS-nolog4shib

VERS=3.0.0
download ${HTTP_DEPS}/opensaml-3.0.0.tar.gz
extract opensaml-$VERS.tar.gz
pushd opensaml-$VERS
patch -s -p1 < ../opensaml.patch
autoreconf -f -i
popd

mv opensaml-$VERS opensaml-$VERS-nolog4shib
tar -zcvf opensaml-$VERS-nolog4shib.tar.gz opensaml-$VERS-nolog4shib
