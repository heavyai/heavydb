#!/bin/bash

set -e
#set -x

PREFIX=/usr/local/mapd-deps

# Establish distro
source /etc/os-release
if [ "$ID" == "ubuntu" ] ; then
  PACKAGER="apt -y"
  if [ "$VERSION_ID" != "20.04" ] && [ "$VERSION_ID" != "19.10" ] && [ "$VERSION_ID" != "19.04" ] && [ "$VERSION_ID" != "18.04" ]; then
    echo "Ubuntu 20.04, 19.10, 19.04, and 18.04 are the only debian-based releases supported by this script"
    exit 1
  fi
elif [ "$ID" == "centos" ] ; then
  MODPATH=/etc/modulefiles
  PACKAGER="yum -y"
  if [ "$VERSION_ID" != "7" ] ; then
    echo "CentOS 7 is the only fedora-based release supported by this script"
    exit 1
  fi
else
  echo "Only debian- and fedora-based OSs are supported by this script"
  exit 1
fi

# Parse inputs
FLAG=latest
ENABLE=false

while (( $# )); do
  case "$1" in
    --testing)
      FLAG=testing
      ;;
    --custom=*)
      FLAG="${1#*=}"
      ;;
    --enable)
      ENABLE=true
      ;;
    *)
      break
      ;;
  esac
  shift
done

# Distro-specific installations
if [ "$ID" == "ubuntu" ] ; then
  sudo $PACKAGER update

  sudo $PACKAGER install \
      software-properties-common \
      build-essential \
      ccache \
      git \
      wget \
      curl \
      libboost-all-dev \
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
      libgflags-dev \
      libgoogle-perftools-dev \
      libiberty-dev \
      libjemalloc-dev \
      liblz4-dev \
      liblzma-dev \
      libbz2-dev \
      libarchive-dev \
      libcurl4-openssl-dev \
      libedit-dev \
      uuid-dev \
      libsnappy-dev \
      zlib1g-dev \
      autoconf \
      autoconf-archive \
      automake \
      bison \
      flex-old \
      libpng-dev \
      rsync \
      unzip \
      jq \
      python-dev \
      python-yaml \
      libxerces-c-dev \
      swig

  # required for gcc-11 on Ubuntu < 22.04
  if [ "$VERSION_ID" == "20.04" ] || [ "$VERSION_ID" == "19.04" ] || [ "$VERSION_ID" == "18.04" ]; then
    DEBIAN_FRONTEND=noninteractive sudo add-apt-repository ppa:ubuntu-toolchain-r/test
  fi

  sudo $PACKAGER install \
      gcc-11 \
      g++-11

# Set up gcc-11 as default gcc
sudo update-alternatives \
  --install /usr/bin/gcc gcc /usr/bin/gcc-11 1100 \
  --slave /usr/bin/g++ g++ /usr/bin/g++-11
sudo update-alternatives --auto gcc

if [ "$VERSION_ID" == "19.04" ] || [ "$VERSION_ID" == "18.04" ] ; then
  sudo $PACKAGER install -y \
    libxerces-c-dev \
    libxmlsec1-dev \
    libegl1-mesa-dev
fi

if [ "$VERSION_ID" == "20.04" ] ; then
  sudo $PACKAGER install -y libegl-dev
fi

  sudo mkdir -p $PREFIX
  pushd $PREFIX
  sudo wget --continue https://dependencies.mapd.com/mapd-deps/mapd-deps-ubuntu-${VERSION_ID}-$FLAG.tar.xz
  sudo tar xvf mapd-deps-ubuntu-${VERSION_ID}-$FLAG.tar.xz
  sudo rm -f mapd-deps-ubuntu-${VERSION_ID}-$FLAG.tar.xz
  popd

  cat << EOF | sudo tee -a $PREFIX/mapd-deps.sh
HEAVY_PREFIX=$PREFIX

LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH
LD_LIBRARY_PATH=\$HEAVY_PREFIX/lib:\$LD_LIBRARY_PATH
LD_LIBRARY_PATH=\$HEAVY_PREFIX/lib64:\$LD_LIBRARY_PATH

PATH=/usr/local/cuda/bin:\$PATH
PATH=\$HEAVY_PREFIX/bin:\$PATH

VULKAN_SDK=\$HEAVY_PREFIX
VK_LAYER_PATH=\$HEAVY_PREFIX/etc/vulkan/explicit_layer.d

CMAKE_PREFIX_PATH=\$HEAVY_PREFIX:\$CMAKE_PREFIX_PATH

export LD_LIBRARY_PATH PATH VULKAN_SDK VK_LAYER_PATH CMAKE_PREFIX_PATH
EOF

  PROFPATH=/etc/profile.d/xx-mapd-deps.sh
  if [ "$ENABLE" = true ] ; then
    sudo ln -sf $PREFIX/mapd-deps.sh $PROFPATH
    echo "Done. A file at $PROFPATH has been created and will be run on startup"
    echo "Source this file or reboot to load vars in this shell"
  else
    echo "Done. Be sure to source the 'mapd-deps.sh' file to pick up the required environment variables:"
    echo "    source $PREFIX/mapd-deps.sh"
  fi
elif [ "$ID" == "centos" ] ; then
  sudo yum groupinstall -y "Development Tools"
  sudo yum install -y \
    zlib-devel \
    epel-release \
    which \
    libssh \
    openssl-devel \
    ncurses-devel \
    git \
    maven \
    java-1.8.0-openjdk-devel \
    java-1.8.0-openjdk-headless \
    gperftools \
    gperftools-devel \
    gperftools-libs \
    python-devel \
    wget \
    curl \
    python-yaml \
    libX11-devel \
    environment-modules \
    valgrind \
    openldap-devel \
    patchelf

  # Install packages from EPEL
  sudo yum install -y \
    cloc \
    jq \
    pxz

  if ! type module >/dev/null 2>&1 ; then
    sudo $PACKAGER install environment-modules
    source /etc/profile
  fi

  sudo mkdir -p $PREFIX
  pushd $PREFIX
  sudo wget --continue https://dependencies.mapd.com/mapd-deps/mapd-deps-$FLAG.tar.xz
  DIRNAME=$(tar tf mapd-deps-$FLAG.tar.xz | head -n 2 | tail -n 1 | xargs dirname)
  sudo tar xvf mapd-deps-$FLAG.tar.xz
  sudo rm -f mapd-deps-$FLAG.tar.xz
  MODFILE=$(readlink -e $(ls $DIRNAME/*modulefile | head -n 1))
  popd

  sudo mkdir -p $MODPATH/mapd-deps
  sudo ln -sf $MODFILE $MODPATH/mapd-deps/$DIRNAME

  if [ ! -e "$MODPATH/cuda" ]; then
    pushd $MODPATH
    sudo wget --continue https://dependencies.mapd.com/mapd-deps/cuda
    popd
  fi

  PROFPATH=/etc/profile.d/xx-mapd-deps.sh
  if [ "$ENABLE" = true ] ; then
    echo 'module load cuda mapd-deps' | sudo tee $PROFPATH
    echo "Done. A file at $PROFPATH has been created and will be run on startup"
    echo "Run 'source /etc/profile' or reboot to load mapd-deps and cuda module vars in this shell"
  else
    if [ ! -f "$PROFPATH" ] ; then
      echo '#module load cuda mapd-deps' | sudo tee $PROFPATH
    fi
    echo "Done. Be sure to load modules function and load the mapd-deps and cuda modules to load variables:"
    echo "    source /etc/profile.d/modules.sh"
    echo "    module load cuda mapd-deps"
  fi
  source $PROFPATH
fi
