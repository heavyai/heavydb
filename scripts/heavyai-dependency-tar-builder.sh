set -x
#
# This script is an enhanced copy of the code that was embedded
# in jenkins projects called from mapd-deps-pipeline such as
# mapd-deps-centos-7, mapd-deps-tsan-centos-7 and mapd-deps-ubuntu-18.04
# In creating the newer pipeline heavyai-deps-pipeline I (Jack) consolidated
# all the mapd-deps-* projects into a single project, add parameters and
# then re-factored it out of Jenkins into this file which is called 
# from Jenkins.


# The scripts primary purpose is to be called from the relevant
# jenkins project - heavyai-dependency-tar-build; however it 
# can be called directly from the command line, which can make
# testing the building of a dependency 'set' easier.
#


# default CPU_SET
CPU_SET="0-10"

while (( $# )); do
  case "$1" in
    --cpu_set=*)
      CPU_SET="${1#*=}"
      ;;
    --branch_name=*)
      BRANCH_NAME="${1#*=}"
      ;;
    --build_dir=*)
      BUILD_DIR="${1#*=}"
      ;;
    --build_tmp_dir=*)
      BUILD_TMP_DIR="${1#*=}"
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
    --git_commit=*)
      GIT_COMMIT="${1#*=}"
      ;;
    --operating_system=*)
      OPERATING_SYSTEM="${1#*=}"
      ;;
    --tag_date=*)
      TAG_DATE="${1#*=}"
      ;;
    --tsan)
      tsan="true"
      ;;
    *)
      break
      ;;
  esac
  shift
done

echo $BUILD_CONTAINER_NAME
TAG_DATE=${TAG_DATE:=$(date +%Y%m%d)}
HASH_SHORT=${HASH_SHORT:=$(echo $GIT_COMMIT | cut -c1-9)}
BRANCH_NAME=${BRANCH_NAME:=$(git rev-parse --abbrev-ref HEAD)}


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

if [ "$tsan" == "true" ] ; then
  TSAN_PARAM="--tsan"
fi

# Check that build dir exists and create if it does not
if [ ! -d $BUILD_DIR ] ; then
  mkdir $BUILD_DIR 
fi
# Create temp dir and populate with build files
mkdir $BUILD_TMP_DIR
cp -r . $BUILD_TMP_DIR/

echo "Pulling $BUILD_CONTAINER_IMAGE"

sudo docker pull $BUILD_CONTAINER_IMAGE


# strip verion number from os
OPERATING_SYSTEM=$(echo $OPERATING_SYSTEM | sed 's/[0-9,\.]*$//')
if [[ $OPERATING_SYSTEM == "centos" ]] ; then
 docker_cmd="yum install sudo -y && ./mapd-deps-${OPERATING_SYSTEM}.sh --savespace --compress $TSAN_PARAM --cache=/dep_cache"
else
  docker_cmd='echo -e "#!/bin/sh\n\${@}" > /usr/sbin/sudo && chmod +x /usr/sbin/sudo && ./mapd-deps-'${OPERATING_SYSTEM}'.sh --savespace --compress'
fi
PACKAGE_CACHE=/theHoard/export/dep_cache

echo "Running [$docker_cmd] in $BUILD_CONTAINER_IMAGE"

sudo docker run --rm --runtime=nvidia \
  -v $BUILD_TMP_DIR:/build \
  -v $PACKAGE_CACHE:/dep_cache \
  --workdir="/build/scripts" \
  -e USER=root \
  --memory=64G --cpuset-cpus=$CPU_SET \
  -e SUFFIX=${SUFFIX} \
  --name $BUILD_CONTAINER_NAME \
  $BUILD_CONTAINER_IMAGE \
  bash -c "$docker_cmd"

ls -ltr $BUILD_TMP_DIR
cp $BUILD_TMP_DIR/scripts/mapd-deps*xz .
echo "docker run complete"  
