#!/usr/bin/env bash

set -e
set -x

CUDA_ROOT="/usr/local/cuda-7.0"
CUDA_URL="http://developer.download.nvidia.com/compute/cuda/7_0/Prod/local_installers/cuda_7.0.28_linux.run"

CUDA_RUN=$(basename ${CUDA_URL})

#tmpdir=$(mktemp -d cuda-XXXXX)
tmpdir="tmp-cuda"
mkdir -p $tmpdir
pushd $tmpdir

wget --continue $CUDA_URL
chmod +x $CUDA_RUN

# need to extract CUDA_RUN and run each part individually
# so that we detect failures
./$CUDA_RUN --extract=$PWD

# Driver install will fail if no CUDA-capable devices are available.
# Specifying no-kernel-module allows us to install libcuda, etc.
if ! cut -f2 /proc/bus/pci/devices | grep -q ^10de ; then
	DRIVER_EXTRA_OPTIONS="--no-kernel-module"
fi

sudo ./NVIDIA-*run \
	--silent \
	$DRIVER_EXTRA_OPTIONS

sudo ./cuda-linux64-rel-*run \
	-noprompt \
	-prefix=$CUDA_ROOT

cat << EOF
--------

CUDA was installed to: $CUDA_ROOT
Please add the following to your environment variables:

CUDA_TOOLKIT_ROOT_DIR=$CUDA_ROOT
PATH=\$CUDA_TOOLKIT_ROOT_DIR/bin:\$PATH
LD_LIBRARY_PATH=\$CUDA_TOOLKIT_ROOT_DIR/lib64:\$LD_LIBRARY_PATH
LD_LIBRARY_PATH=\$CUDA_TOOLKIT_ROOT_DIR/nvvm/lib64:\$LD_LIBRARY_PATH
export CUDA_TOOLKIT_ROOT_DIR PATH LD_LIBRARY_PATH

--------
EOF

popd
