#!/bin/bash

set -e
set -x

# Internal registry
IMAGE_NAME=docker-internal.mapd.com/cudagl

# Nvidia gitlab repository for cuda/cudagl container builder
NVIDIA_REPOSITORY_URL=https://gitlab.com/nvidia/container-images/cuda.git
# Local path for the gitlab repo
NVIDIA_REPOSITORY_PATH="not set"
# Keep repository after build completed
KEEP_NVIDIA_REPOSITORY=false

# Inputs to Nvidia build script
OS=ubuntu
OS_VERSION=22.04
CUDA_VERSION=12.2.2
ARCH=x86_64

# Push images to docker-internal.mapd.com once complete
PUSH_IMAGES=false

# Keep local images
# If push == false and keep_images == false, images are lost. Useful for testing these scripts
KEEP_IMAGES=false

SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Create a temp directory
TEMP_DIR=`mktemp -d -p "$SCRIPTS_DIR"`
# check if tmp dir was created
if [[ ! "$TEMP_DIR" || ! -d "$TEMP_DIR" ]]; then
  echo "Could not create temp dir"
  exit 1
fi

#
# cleanup
#

# Runs on script exit, deleting temp folder and Nvidia repository
function cleanup {      
  # Remove Nvidia repository
  if [[ ${KEEP_NVIDIA_REPOSITORY} != "true" ]]; then
    echo "Removing Nvidia repository"
    sudo rm -rf ${NVIDIA_REPOSITORY_PATH}
  fi

  # Remove TEMP_DIR
  # If repo-path was not set this will also delete the Nvidia repository
  # even if `--keep-repo` was passed as an arg
  rm -rf "$TEMP_DIR"
  echo "Deleted temp working directory $TEMP_DIR"
}

# register the cleanup function to be called on the EXIT signal
trap cleanup EXIT

#
# update_and_retag_image
#

# param 1 = image class ("runtime" or "devel")
# Apply package upgrades to images for security updates
# New images will be tagged using heavyai internal formatting
function update_and_retag_image() {
  # "runtime" or "devel"
  local IMAGE_CLASS="$1"

  printf -v DATE_TAG '%(%Y%m%d)T' -1
  ORIGINAL_IMAGE_NAME="${IMAGE_NAME}:${CUDA_VERSION}-${IMAGE_CLASS}-${OS}${OS_VERSION}"
  NEW_IMAGE_NAME="${IMAGE_NAME}/${OS}${OS_VERSION}-cuda${CUDA_VERSION}-${ARCH}-${IMAGE_CLASS}:${DATE_TAG}"

  # Generate package updater script
  cp ${SCRIPTS_DIR}/gen_cudagl_package_updater.sh ${TEMP_DIR}
  docker run --rm -t \
    -v $TEMP_DIR:/scripts \
    --workdir="/scripts" \
    -e USER=root \
    $ORIGINAL_IMAGE_NAME \
    bash -c "./gen_cudagl_package_updater.sh"

  # build using Dockerfile which will run generated cudagl_package_updater.sh
  # tag upgraded image using heavyai internal naming scheme
  cp $SCRIPTS_DIR/Dockerfile ${TEMP_DIR}
  pushd ${TEMP_DIR}
  sudo docker build \
    -f Dockerfile \
    -t ${NEW_IMAGE_NAME} \
    --build-arg BASE_IMAGE=${ORIGINAL_IMAGE_NAME} \
    .
  popd

  # Remove original image
  docker rmi "${ORIGINAL_IMAGE_NAME}"
  # Remove generated updater script
  sudo rm ${TEMP_DIR}/cudagl_package_updater.sh

  # Push new image to internal registry
  if [ ${PUSH_IMAGES} = "true" ]; then
    docker push ${NEW_IMAGE_NAME}
  fi

  # Remove local copy of generated image
  if [ ${KEEP_IMAGES} = "false" ]; then
    if [ ${PUSH_IMAGES} = "false" ]; then
      echo "WARNING: Removing unpushed local image copy!"
    fi
    docker rmi ${NEW_IMAGE_NAME}
  fi
}

#
# Main script
#

echo "${script_name} START"

while (( $# )); do
  case "$1" in
    --cuda-version=*)
      CUDA_VERSION="${1#*=]}"
      ;;
    --os=*)
      OS="${1#*=}"
      ;;
    --os-version=*)
      OS_VERSION="${1#*=}"
      ;;
    --arch=*)
      ARCH="${1#*=}"
      ;;
    --repo-path=*)
      NVIDIA_REPOSITORY_PATH="${1#*=}"
      ;;
    --keep-repo)
      KEEP_NVIDIA_REPOSITORY=true
      ;;
    --push)
      PUSH_IMAGES=true
      ;;
    --keep-images)
      KEEP_IMAGES=true
      ;;
    *)
      break
      ;;
  esac
  shift
done

if [[ ${NVIDIA_REPOSITORY_PATH} = "not set" ]]; then
  echo "Nvidia repository path not set, using temp directory"
  NVIDIA_REPOSITORY_PATH=${TEMP_DIR}
fi

echo "nvidia repository path: ${NVIDIA_REPOSITORY_PATH}"

# Ensure we have the latest nvidia repository
if [ ! -f ${NVIDIA_REPOSITORY_PATH}/build.sh ]; then
  echo "Can not find build.sh in ${NVIDIA_REPOSITORY_PATH}"
  echo "Cloning from ${NVIDIA_REPOSITORY_URL}"
  git clone ${NVIDIA_REPOSITORY_URL} ${NVIDIA_REPOSITORY_PATH}
else
  pushd ${NVIDIA_REPOSITORY_PATH}
  git fetch && git pull
  popd
fi

# Final sanity check for nvidia build script
if [ ! -f ${NVIDIA_REPOSITORY_PATH}/build.sh ]; then
  echo "ERROR: Can not find build.sh in ${NVIDIA_REPOSITORY_PATH}"
  exit 1
fi

# Apply patch to build.sh so it will patch the removal of i386 from the opengl sub-repository
cp ${SCRIPTS_DIR}/cudagl_remove_i386.patch ${NVIDIA_REPOSITORY_PATH}
patch -p1 -d ${NVIDIA_REPOSITORY_PATH} < $SCRIPTS_DIR/apply_cudagl_remove_i386_patch.patch

# Call nvidia script
pushd "${NVIDIA_REPOSITORY_PATH}"
# Run build.sh to generate the original Nvidia cudagl images
$(pwd)/build.sh "-d" "--image-name" "${IMAGE_NAME}" "--cuda-version" "${CUDA_VERSION}" "--os" "${OS}" "--os-version" "${OS_VERSION}" "--arch" "${ARCH}" "--cudagl"
popd

# Remove intermediate cudagl images
docker rmi $(docker images | grep build-intermediate | tr -s ' ' | cut -d ' ' -f 3)

# Change into the script dir
pushd ${SCRIPTS_DIR}

# Update packages and retag images
update_and_retag_image "runtime"
update_and_retag_image "devel"

popd

# Remove unused original cudagl base image
docker rmi "${IMAGE_NAME}:${CUDA_VERSION}-base-${OS}${OS_VERSION}"
