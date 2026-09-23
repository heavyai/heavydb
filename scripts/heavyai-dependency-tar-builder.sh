# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -x
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

function help_msg(){
  cat << _HELP 
  $0 --build_tmp_dir=<working directory for dependency build>
     --build_type=private_release|rc_release|full_release
     --build_container-image=<docker image used to build dependencies>
     --operating_system=rockylinux|ubuntu  ## used to find the appropriate build script
     	[--cpu_set=<cpu set used by the docker container. Default 0-10>
         --static|--shared  ## build either dependencies as share libraries (default) or static libraries
     	 --docker_memory_allocation=<memory allocated to the docker container. Default 64G>
         --cache_dir=<diretory to check for archives and to 'cache' downloaded files>
         --build_container_image=<docker container name>
         --tag_date=<tag used in tar file name>
         --tsan
         --help]
_HELP
  exit
}
while (( $# )); do
  case "$1" in
    --cpu_set=*)
      CPU_SET="${1#*=}"
      ;;
    --build_tmp_dir=*)
      BUILD_TMP_DIR="${1#*=}"
      ;;
    --cache_dir=*)
      PACKAGE_CACHE="${1#*=}"
      ;;
    --build_container_image=*)
      BUILD_CONTAINER_IMAGE="${1#*=}"
      ;;
    --build_container_name=*)
      BUILD_CONTAINER_NAME="${1#*=}"
      ;;
    --build_type=*)
      BUILD_TYPE="${1#*=}"
      ;;
    --operating_system=*)
      OPERATING_SYSTEM="${1#*=}"
      ;;
    --static)
      if [[ -n $LIBRARY_TYPE ]]; then "Error.  --static and --shared, mutually exclusive options have both been supplied"; exit -1; fi
      LIBRARY_TYPE="static"
      ;;
    --shared)
      if [[ -n $LIBRARY_TYPE ]]; then "Error.  --static and --shared, mutually exclusive options have both been supplied"; exit -1; fi
      LIBRARY_TYPE="shared"
      ;;
    --tag_date=*)
      TAG_DATE="${1#*=}"
      ;;
    --tsan)
      tsan="true"
      ;;
    --docker-memory-allocation)
      DOCKER_MEMORY_ALLOCATION="${1#*=}"
      ;;
    --help)
      help_msg;
      ;;
    *)
      echo "Unexpected argument $1"
      exit
      ;;
  esac
  shift
done
## test for manditory parameters
: ${BUILD_CONTAINER_IMAGE?"Error. --build_container_image is a required parameter, needed to set the docker image used to build the dependencies"}
: ${OPERATING_SYSTEM?"Error. --operating_sysem is a required parameter. Options are [rockylinux|ubuntu]"}
if [[ $OPERATING_SYSTEM != "rockylinux" && $OPERATING_SYSTEM != "ubuntu" ]] ; then
  echo "Error. An invalid options supplied with --operating_system.  Possibilites are [rockylinux|ubuntu]"
  exit -1
fi
## set defaults
: ${GIT_COMMIT:=$(git rev-parse --verify HEAD)}
: ${TAG_DATE:=$(date +%Y%m%d)}
: ${HASH_SHORT:=$(echo $GIT_COMMIT | cut -c1-9)}
: ${BUILD_CONTAINER_NAME:="deps_builder_$(</dev/urandom tr -dc 'A-Za-z' | head -c 15)_$TAG_DATE"}
: ${DOCKER_MEMORY_ALLOCATION="64G"}
: ${CPU_SET:="0-10"}
: ${LIBRARY_TYPE:="shared"}
BRANCH_NAME=$(git rev-parse --abbrev-ref HEAD)
if [[ -z $PACKAGE_CACHE  ]] ; then
  CACHE_OPTION=""
  CACHE_MOUNT=""
else
  CACHE_OPTION="--cache=/dep_cache"
  CACHE_MOUNT="-v $PACKAGE_CACHE:/dep_cache"
fi

if [ "$BUILD_TYPE" == "private_release" ] ; then
  SUFFIX="${TAG_DATE}-${HASH_SHORT}"
elif [ "${BUILD_TYPE}" == "full_release" ] ; then
  if [ "${BRANCH_NAME}" != "master" ] ; then
    echo "full_release can only be generated from master"
    exit 1
  fi
  SUFFIX="${TAG_DATE}"
elif [ "${BUILD_TYPE}" == "rc_release" ] ; then
  if [ "${BRANCH_NAME}" == "master" ] ; then
    echo "rc release can not be generated from master"
    exit 2
  fi
  SUFFIX=$(echo ${BRANCH_NAME} | tr '\/' '.')
else
  echo "invalid build type [$build_type] specified"
  exit 1
fi

LIBRARY_TYPE="--${LIBRARY_TYPE}"

if [ "$tsan" == "true" ] ; then
  TSAN_PARAM="--tsan"
fi

# Create temp dir and populate with build files
mkdir -p $BUILD_TMP_DIR
cp -r $SCRIPT_DIR $BUILD_TMP_DIR/

sudo docker pull $BUILD_CONTAINER_IMAGE

#
# Note we use two methods to pass run information to the docker container.
# Firstly via options on the command the docker container runs - 'docker_cmd'
# and secondly via environment varibles on the docker command itself (-e options)
# The value set in the environment, with the -e options are intended for use by the
# common-functions.sh script sourced by the 'main' mapd-deps-${OPERATING_SYSTEM}
# script.
#
if [[ $OPERATING_SYSTEM == "rockylinux" ]] ; then
  docker_cmd="dnf install sudo -y"
else
  docker_cmd='echo -e "#!/bin/sh\n\${@}" > /usr/sbin/sudo && chmod +x /usr/sbin/sudo'
fi
docker_cmd="${docker_cmd} && ./mapd-deps-'${OPERATING_SYSTEM}'.sh ${LIBRARY_TYPE} ${TSAN_PARAM} --update-packages --savespace --compress ${CACHE_OPTION}"

echo "Running [$docker_cmd] in $BUILD_CONTAINER_IMAGE"
BUILD_CONTAINER_IMAGE_ID=$(sudo docker images -q $BUILD_CONTAINER_IMAGE --no-trunc)
# Note - to log the container image name pass it 
# in as an environmemt.
sudo docker run --rm --runtime=nvidia \
  -v ${BUILD_TMP_DIR}:/build ${CACHE_MOUNT} \
  --workdir="/build/scripts" \
  -e USER=root \
  --memory=${DOCKER_MEMORY_ALLOCATION} \
  --cpuset-cpus=${CPU_SET} \
  -e SUFFIX=${SUFFIX} \
  -e BUILD_CONTAINER_IMAGE_ID=${BUILD_CONTAINER_IMAGE_ID} \
  -e BUILD_CONTAINER_IMAGE=${BUILD_CONTAINER_IMAGE} \
  -e BRANCH_NAME=${BRANCH_NAME} \
  -e GIT_COMMIT=${GIT_COMMIT} \
  --name $BUILD_CONTAINER_NAME \
  --network=host \
  ${BUILD_CONTAINER_IMAGE} \
  bash -c "$docker_cmd"

ls -ltr $BUILD_TMP_DIR
cp $BUILD_TMP_DIR/scripts/mapd-deps*xz .
echo "docker run complete"  
