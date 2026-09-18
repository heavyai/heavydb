#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

declare -A descs
declare -A vars

while (( $# )); do
  case "$1" in
    --non-interactive)
      NON_INTERACTIVE=true ;;
    *)
      break ;;
  esac
  shift
done

HEAVYAI_TMP=$(mktemp -d)

descs["HEAVYAI_PATH"]="HeavyDB install directory"
vars["HEAVYAI_PATH"]=${HEAVYAI_PATH:=$(dirname $(pwd))}

descs["HEAVYAI_BASE"]="HeavyDB base data and configuration storage directory"
vars["HEAVYAI_BASE"]=${HEAVYAI_BASE:="/var/lib/heavyai"}

descs["HEAVYAI_USER"]="user HeavyDB will be run as"
vars["HEAVYAI_USER"]=${HEAVYAI_USER:=$(id --user --name)}
descs["HEAVYAI_GROUP"]="group HeavyDB will be run as"
vars["HEAVYAI_GROUP"]=${HEAVYAI_GROUP:=$(id --group --name)}

if [ ! "$NON_INTERACTIVE" = true ]; then
  for v in HEAVYAI_PATH HEAVYAI_BASE HEAVYAI_USER HEAVYAI_GROUP ; do
    echo "$v: ${descs["$v"]}"
    read -p "[${vars[$v]}]: "
    if [ ! -z "$REPLY" ]; then
      vars[$v]=$REPLY
    fi
    echo
  done
fi

for v in HEAVYAI_PATH HEAVYAI_BASE HEAVYAI_USER HEAVYAI_GROUP ; do
  echo -e "$v:\t${vars[$v]}"
done

vars["HEAVYAI_STORAGE"]=${HEAVYAI_STORAGE:="${vars['HEAVYAI_BASE']}/storage"}
sudo mkdir -p "${vars['HEAVYAI_STORAGE']}"
sudo mkdir -p "${vars['HEAVYAI_BASE']}"

if [ ! -d "${vars['HEAVYAI_STORAGE']}/catalogs" ]; then
  sudo ${vars["HEAVYAI_PATH"]}/bin/initheavy ${vars['HEAVYAI_STORAGE']}
fi

sudo chown -R ${vars['HEAVYAI_USER']}:${vars['HEAVYAI_GROUP']} "${vars['HEAVYAI_STORAGE']}"
sudo chown -R ${vars['HEAVYAI_USER']}:${vars['HEAVYAI_GROUP']} "${vars['HEAVYAI_BASE']}"
# shellcheck source=nvidia-graphics-env.sh
source "${vars[HEAVYAI_PATH]}/scripts/nvidia-graphics-env.sh"
icd_path=
egl_vendor_path=
find_nvidia_vk_icd_path && icd_path="$NVIDIA_VK_ICD_PATH"
find_nvidia_egl_vendor_path && egl_vendor_path="$NVIDIA_EGL_VENDOR_PATH"

if [[ -z "$icd_path" ]]; then
  YELLOW='\033[1;33m'
  NORMAL='\033[0m'
  LBLUE='\033[1;34m'
  echo -e "${YELLOW}Warning: ${NORMAL}Cannot find the Nvidia Vulkan driver manifest file \"nvidia_icd.json\" in the expected system directories. As a result the backend renderer may not work.
  Please verify that the nvidia driver and vulkan loader are installed appropriately.
  See: ${LBLUE}https://docs.omnisci.com/troubleshooting/vulkan-graphics-api-beta#bare-metal-installs${NORMAL} for some installation and troubleshooting tips."
fi

for f in heavydb heavydb@ heavy_web_server heavy_web_server@ heavyiq ; do
  if [ -f $f.service.in ]; then
    if [[ "$f.service.in" == *"web_server"* ]]; then
      unset vulkan_envinment
    else
      if [[ -n $icd_path ]]; then
        vulkan_environment="\nEnvironment=\"VK_ICD_FILENAMES=$icd_path\""
        if [[ -n $egl_vendor_path ]]; then
          vulkan_environment="${vulkan_environment}\nEnvironment=\"__EGL_VENDOR_LIBRARY_FILENAMES=$egl_vendor_path\""
          vulkan_environment="${vulkan_environment}\nEnvironment=\"__GLX_VENDOR_LIBRARY_NAME=nvidia\""
        fi
      fi
    fi

    sed -e "s#@HEAVYAI_PATH@#${vars['HEAVYAI_PATH']}#g" \
        -e "s#@HEAVYAI_BASE@#${vars['HEAVYAI_BASE']}#g" \
        -e "s#@HEAVYAI_STORAGE@#${vars['HEAVYAI_STORAGE']}#g" \
        -e "s#@HEAVYAI_USER@#${vars['HEAVYAI_USER']}#g" \
        -e "s#@HEAVYAI_GROUP@#${vars['HEAVYAI_GROUP']}#g" \
        -e "s#\[Service\]#[Service]$vulkan_environment#" \
        $f.service.in > $HEAVYAI_TMP/$f.service
    sudo cp $HEAVYAI_TMP/$f.service /lib/systemd/system/
  fi
done

sed -e "s#@HEAVYAI_PATH@#${vars['HEAVYAI_PATH']}#g" \
    -e "s#@HEAVYAI_BASE@#${vars['HEAVYAI_BASE']}#g" \
    -e "s#@HEAVYAI_STORAGE@#${vars['HEAVYAI_STORAGE']}#g" \
    -e "s#@HEAVYAI_USER@#${vars['HEAVYAI_USER']}#g" \
    -e "s#@HEAVYAI_GROUP@#${vars['HEAVYAI_GROUP']}#g" \
    heavy.conf.in > $HEAVYAI_TMP/heavy.conf
sudo cp $HEAVYAI_TMP/heavy.conf ${vars['HEAVYAI_BASE']}
sudo chown ${vars['HEAVYAI_USER']}:${vars['HEAVYAI_GROUP']} "${vars['HEAVYAI_BASE']}/heavy.conf"

sudo systemctl daemon-reload
