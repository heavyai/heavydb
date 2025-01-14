#!/bin/bash

set -e
set -x

PREFIX=/usr/local/mapd-deps

# Establish distro
source /etc/os-release
if [ "$ID" == "ubuntu" ] ; then
  PACKAGER="apt -y"
  if [ "$VERSION_ID" != "24.04" ] && [ "$VERSION_ID" != "22.04" ]; then
    echo "Ubuntu 24.04 and 22.04 are the only Debian-based OSs supported by this script"
    echo "If you are still using 20.04 or 23.10 then you need to upgrade!"
    exit 1
  fi
elif [ "$ID" == "rocky" ] ; then
  MODPATH=/etc/modulefiles
  PACKAGER="dnf -y"
  if [ "${VERSION_ID:0:1}" != "8" ] ; then
    echo "Rocky Linux 8.x is the only RedHat-based OS supported by this script"
    exit 1
  fi
else
  echo "Only Debian- and RedHat-based OSs are supported by this script"
  exit 1
fi

# Parse inputs
UPDATE_PACKAGES=false
FLAG=latest
ENABLE=false
LIBRARY_TYPE=
TSAN=false

while (( $# )); do
  case "$1" in
    --update-packages)
      UPDATE_PACKAGES=true
      ;;
    --testing)
      FLAG=testing
      ;;
    --custom=*)
      FLAG="${1#*=}"
      ;;
    --enable)
      ENABLE=true
      ;;
    --static)
      LIBRARY_TYPE=static
      ;;
    --shared)
      LIBRARY_TYPE=shared
      ;;
    --tsan)
      TSAN=true
      ;;
    *)
      break
      ;;
  esac
  shift
done

# Validate LIBRARY_TYPE
if [ "$ID" == "ubuntu" ] &&  [ "$LIBRARY_TYPE" == "" ] ; then
  echo "ERROR - Library type must be specified for Ubuntu installs (--static or --shared)"
  exit
fi

# Establish architecture
ARCH=$(uname -m)

# Validate architecture
if [ "$ID" == "rocky" ] && [ "$ARCH" != "x86_64" ] ; then
  echo "ERROR - Only x86 builds supported on RedHat-based systems"
  exit
fi

SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Distro-specific installations
if [ "$ID" == "ubuntu" ] ; then
  sudo $PACKAGER update

  source $SCRIPTS_DIR/common-functions.sh

  update_container_packages
  
  install_required_ubuntu_packages

  sudo $PACKAGER install \
      gcc-11 \
      g++-11

  # Set up gcc-11 as default gcc
  sudo update-alternatives \
    --install /usr/bin/gcc gcc /usr/bin/gcc-11 1100 \
    --slave /usr/bin/g++ g++ /usr/bin/g++-11
  sudo update-alternatives --auto gcc

  sudo mkdir -p $PREFIX
  pushd $PREFIX

  OS=ubuntu${VERSION_ID}
  # always tag as 22.04 for now
  if [ $VERSION_ID == "24.04" ]; then
    OS=ubuntu22.04
  fi
  TARBALL_TSAN=""
  if [ "$TSAN" = "true" ]; then
    TARBALL_TSAN="-tsan"
  fi
  FILENAME=mapd-deps-${OS}${TARBALL_TSAN}-${LIBRARY_TYPE}-${ARCH}-${FLAG}.tar.xz
  sudo wget --continue https://dependencies.mapd.com/mapd-deps/${FILENAME}
  sudo tar xvf ${FILENAME}
  sudo rm -f ${FILENAME}
  popd

  # move validation layer JSON files from /etc to /share if needed (and remove then-empty vulkan subdir)
  # @TODO(simon) remove this once the bundle has been repackaged for both x86 and ARM
  if [ -d $PREFIX/etc/vulkan/explicit_layer.d ]; then
    sudo mv $PREFIX/etc/vulkan/explicit_layer.d -t $PREFIX/share/vulkan
    sudo rmdir $PREFIX/etc/vulkan
  fi

  cat << EOF | sudo tee $PREFIX/mapd-deps.sh
HEAVY_PREFIX=$PREFIX

LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH
LD_LIBRARY_PATH=\$HEAVY_PREFIX/lib:\$LD_LIBRARY_PATH
LD_LIBRARY_PATH=\$HEAVY_PREFIX/lib64:\$LD_LIBRARY_PATH

PATH=/usr/local/cuda/bin:\$PATH
PATH=\$HEAVY_PREFIX/go/bin:\$PATH
PATH=\$HEAVY_PREFIX/maven/bin:\$PATH
PATH=\$HEAVY_PREFIX/bin:\$PATH

VULKAN_SDK=\$HEAVY_PREFIX
VK_LAYER_PATH=\$HEAVY_PREFIX/share/vulkan/explicit_layer.d

CMAKE_PREFIX_PATH=\$HEAVY_PREFIX:\$CMAKE_PREFIX_PATH

GOROOT=\$HEAVY_PREFIX/go

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
elif [ "$ID" == "rocky" ] ; then
  sudo dnf groupinstall -y "Development Tools"
  
  source $SCRIPTS_DIR/common-functions-rockylinux.sh

  install_required_rockylinux_packages

  if ! type module >/dev/null 2>&1 ; then
    sudo $PACKAGER install environment-modules
    source /etc/profile
  fi

  sudo mkdir -p $PREFIX
  pushd $PREFIX
  OS=rockylinux8
  TARBALL_TSAN=""
  if [ "$TSAN" = "true" ]; then
    TARBALL_TSAN="-tsan"
  fi
  FILENAME=mapd-deps-${OS}${TARBALL_TSAN}-${LIBRARY_TYPE}-${ARCH}-${FLAG}.tar.xz
  sudo wget --continue https://dependencies.mapd.com/mapd-deps/${FILENAME}
  DIRNAME=$(tar tf ${FILENAME} | head -n 2 | tail -n 1 | xargs dirname)
  sudo tar xvf ${FILENAME}
  sudo rm -f ${FILENAME}
  MODFILE=$(readlink -e $(ls $DIRNAME/*modulefile | head -n 1))
  popd

  # move validation layer JSON files from /etc to /share if needed (and remove then-empty vulkan subdir)
  # @TODO(simon) remove this once the bundle has been repackaged for both x86 and ARM
  if [ -d $MODFILE/etc/vulkan/explicit_layer.d ]; then
    sudo mv $MODFILE/etc/vulkan/explicit_layer.d -t $MODFILE/share/vulkan
    sudo rmdir $MODFILE/etc/vulkan
  fi

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
