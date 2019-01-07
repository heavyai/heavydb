#!/usr/bin/env bash

set -e
set -x

PREFIX=/usr/local/mapd-deps

SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $SCRIPTS_DIR/common-functions.sh

sudo mkdir -p $PREFIX
sudo chown -R $(id -u) $PREFIX

sudo apt update
sudo apt install -y \
    build-essential \
    ccache \
    cmake \
    cmake-curses-gui \
    git \
    wget \
    curl \
    clang \
    llvm \
    llvm-dev \
    clang-format \
    libgoogle-glog-dev \
    golang \
    libssl-dev \
    libevent-dev \
    default-jre \
    default-jre-headless \
    default-jdk \
    default-jdk-headless \
    maven \
    libncurses5-dev \
    libldap2-dev \
    binutils-dev \
    google-perftools \
    libdouble-conversion-dev \
    libevent-dev \
    libgdal-dev \
    libgflags-dev \
    libgoogle-perftools-dev \
    libiberty-dev \
    libjemalloc-dev \
    libglu1-mesa-dev \
    liblz4-dev \
    liblzma-dev \
    libbz2-dev \
    libarchive-dev \
    libcurl4-openssl-dev \
    uuid-dev \
    libsnappy-dev \
    zlib1g-dev \
    autoconf \
    autoconf-archive \
    automake \
    bison \
    flex-old \
    jq \
    python-yaml \
    libxmlsec1-dev

# Install gcc6
add-apt-repository ppa:ubuntu-toolchain-r/test -y
apt update && apt upgrade -y
apt install g++-6 -y
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-6 60 \
                    --slave /usr/bin/g++ g++ /usr/bin/g++-6
update-alternatives --config gcc

# Needed to find xmltooling and xml_security_c
export PKG_CONFIG_PATH=$PREFIX/lib/pkgconfig:$PREFIX/lib64/pkgconfig:$PKG_CONFIG_PATH

VERS=1_67_0
# http://downloads.sourceforge.net/project/boost/boost/${VERS//_/.}/boost_$VERS.tar.bz2
download https://internal-dependencies.mapd.com/thirdparty/boost_$VERS.tar.bz2
extract boost_$VERS.tar.bz2
pushd boost_$VERS
./bootstrap.sh --prefix=$PREFIX
./b2 cxxflags=-fPIC install --prefix=$PREFIX || true
popd
 # Needed for folly to find boost
export LIBRARY_PATH=$PREFIX/lib:$LIBRARY_PATH

# install AWS core and s3 sdk
install_awscpp -j $(nproc)

VERS=0.11.0
wget --continue http://apache.claz.org/thrift/$VERS/thrift-$VERS.tar.gz
tar xvf thrift-$VERS.tar.gz
pushd thrift-$VERS
CFLAGS="-fPIC" CXXFLAGS="-fPIC" JAVA_PREFIX=$PREFIX/lib ./configure \
    --with-lua=no \
    --with-python=no \
    --with-php=no \
    --with-ruby=no \
    --with-qt4=no \
    --with-qt5=no \
    --prefix=$PREFIX \
    --with-boost=$PREFIX
make -j $(nproc)
make install
popd

# c-blosc
VERS=1.14.4
wget --continue https://github.com/Blosc/c-blosc/archive/v$VERS.tar.gz
tar xvf v$VERS.tar.gz
BDIR="c-blosc-$VERS/build"
rm -rf "$BDIR"
mkdir -p "$BDIR"
pushd "$BDIR"
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=$PREFIX \
    -DBUILD_BENCHMARKS=off \
    -DBUILD_TESTS=off \
    -DPREFER_EXTERNAL_SNAPPY=off \
    -DPREFER_EXTERNAL_ZLIB=off \
    -DPREFER_EXTERNAL_ZSTD=off \
    ..
make -j $(nproc)
make install
popd

VERS=2018.05.07.00
wget --continue https://github.com/facebook/folly/archive/v$VERS.tar.gz
tar xvf v$VERS.tar.gz
pushd folly-$VERS/folly
/usr/bin/autoreconf -ivf
./configure --prefix=$PREFIX --with-boost=$PREFIX/
make -j $(nproc)
make install
popd

VERS=1.21-45
wget --continue https://github.com/jarro2783/bisonpp/archive/$VERS.tar.gz
tar xvf $VERS.tar.gz
pushd bisonpp-$VERS
./configure --prefix=$PREFIX
make -j $(nproc)
make install
popd

# Apache Arrow (see common-functions.sh)
ARROW_BOOST_USE_SHARED="ON"
install_arrow

VERS=3.0.2
wget --continue https://github.com/cginternals/glbinding/archive/v$VERS.tar.gz
tar xvf v$VERS.tar.gz
mkdir -p glbinding-$VERS/build
pushd glbinding-$VERS/build
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=$PREFIX \
    -DOPTION_BUILD_DOCS=OFF \
    -DOPTION_BUILD_EXAMPLES=OFF \
    -DOPTION_BUILD_TESTS=OFF \
    -DOPTION_BUILD_TOOLS=OFF \
    -DOPTION_BUILD_WITH_BOOST_THREAD=OFF \
    ..
make -j $(nproc)
make install
popd

# OpenSAML

#xml-security requires xerces >= 3.2
VERS=3.2.2
wget https://www-us.apache.org/dist//xerces/c/3/sources/xerces-c-$VERS.tar.gz
tar xzvf xerces-c-$VERS.tar.gz
pushd xerces-c-$VERS
./configure --prefix=$PREFIX
make -j $(nproc)
make install
popd

download_make_install ${HTTP_DEPS}/xml-security-c-2.0.0.tar.gz "" "--without-xalan"

#xmltooling and opensaml needs --with-boost
VERS=3.0.2-nolog4shib
wget --continue ${HTTP_DEPS}/xmltooling-$VERS.tar.gz
tar xvf xmltooling-$VERS.tar.gz
pushd xmltooling-$VERS
./configure --prefix=$PREFIX --with-boost=$PREFIX
make -j $(nproc)
make install
popd

VERS=3.0.0-nolog4shib
wget --continue ${HTTP_DEPS}/opensaml-$VERS.tar.gz
tar xvf opensaml-$VERS.tar.gz
pushd opensaml-$VERS
./configure --prefix=$PREFIX --with-boost=$PREFIX
make -j $(nproc)
make install
popd

cat > $PREFIX/mapd-deps.sh <<EOF
PREFIX=$PREFIX

LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH
LD_LIBRARY_PATH=\$PREFIX/lib:\$LD_LIBRARY_PATH
LD_LIBRARY_PATH=\$PREFIX/lib64:\$LD_LIBRARY_PATH

PATH=/usr/local/cuda/bin:\$PATH
PATH=\$PREFIX/bin:\$PATH

CMAKE_PREFIX_PATH=\$PREFIX:\$CMAKE_PREFIX_PATH

export LD_LIBRARY_PATH PATH CMAKE_PREFIX_PATH
EOF

echo
echo "Done. Be sure to source the 'mapd-deps.sh' file to pick up the required environment variables:"
echo "    source $PREFIX/mapd-deps.sh"
