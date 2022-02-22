#!/bin/bash

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

OMNISCI_TMP=$(mktemp -d)

descs["OMNISCI_PATH"]="OmniSci install directory"
vars["OMNISCI_PATH"]=${OMNISCI_PATH:=$(dirname $(pwd))}

descs["OMNISCI_STORAGE"]="OmniSci data and configuration storage directory"
vars["OMNISCI_STORAGE"]=${OMNISCI_STORAGE:="/var/lib/omnisci"}

descs["OMNISCI_USER"]="user OmniSci will be run as"
vars["OMNISCI_USER"]=${OMNISCI_USER:=$(id --user --name)}
descs["OMNISCI_GROUP"]="group OmniSci will be run as"
vars["OMNISCI_GROUP"]=${OMNISCI_GROUP:=$(id --group --name)}

if [ ! "$NON_INTERACTIVE" = true ]; then
  for v in OMNISCI_PATH OMNISCI_STORAGE OMNISCI_USER OMNISCI_GROUP ; do
    echo "$v: ${descs["$v"]}"
    read -p "[${vars[$v]}]: "
    if [ ! -z "$REPLY" ]; then
      vars[$v]=$REPLY
    fi
    echo
  done
fi

for v in OMNISCI_PATH OMNISCI_STORAGE OMNISCI_USER OMNISCI_GROUP ; do
  echo -e "$v:\t${vars[$v]}"
done

vars["OMNISCI_DATA"]=${OMNISCI_DATA:="${vars['OMNISCI_STORAGE']}/data"}
sudo mkdir -p "${vars['OMNISCI_DATA']}"
sudo mkdir -p "${vars['OMNISCI_STORAGE']}"
if [ -f omnisci-sds.conf.in ]; then
  sudo mkdir -p "${vars['OMNISCI_STORAGE']}/sds"
fi

if [ ! -d "${vars['OMNISCI_DATA']}/mapd_catalogs" ]; then
  sudo ${vars["OMNISCI_PATH"]}/bin/initdb ${vars['OMNISCI_DATA']}
fi

sudo chown -R ${vars['OMNISCI_USER']}:${vars['OMNISCI_GROUP']} "${vars['OMNISCI_DATA']}"
sudo chown -R ${vars['OMNISCI_USER']}:${vars['OMNISCI_GROUP']} "${vars['OMNISCI_STORAGE']}"

for i in "/etc/xdg" "/etc" "/usr/local/share" "/usr/share"; do
  if [ -f "$i/vulkan/icd.d/nvidia_icd.json" ]; then
    icd_path="$i/vulkan/icd.d/nvidia_icd.json"
    break
  fi
done

if [[ -z "$icd_path" ]]; then
  YELLOW='\033[1;33m'
  NORMAL='\033[0m'
  LBLUE='\033[1;34m'
  echo -e "${YELLOW}Warning: ${NORMAL}Cannot find the Nvidia Vulkan driver manifest file \"nvidia_icd.json\" in the expected system directories. As a result the backend renderer may not work. 
  Please verify that the nvidia driver and vulkan loader are installed appropriately. 
  See: ${LBLUE}https://docs.omnisci.com/troubleshooting/vulkan-graphics-api-beta#bare-metal-installs${NORMAL} for some installation and troubleshooting tips."
fi

for f in omnisci_server omnisci_server@ omnisci_sd_server omnisci_sd_server@ omnisci_web_server omnisci_web_server@ ; do
  if [ -f $f.service.in ]; then
    if [[ "$f.service.in" == *"web_server"* ]]; then
      unset vulkan_envinment
    else
      if [[ -n $icd_path ]]; then
        vulkan_environment="\nEnvironment=\"VK_ICD_FILENAMES=$icd_path\""
      fi
    fi
    
    sed -e "s#@OMNISCI_PATH@#${vars['OMNISCI_PATH']}#g" \
        -e "s#@OMNISCI_STORAGE@#${vars['OMNISCI_STORAGE']}#g" \
        -e "s#@OMNISCI_DATA@#${vars['OMNISCI_DATA']}#g" \
        -e "s#@OMNISCI_USER@#${vars['OMNISCI_USER']}#g" \
        -e "s#@OMNISCI_GROUP@#${vars['OMNISCI_GROUP']}#g" \
        -e "s#\[Service\]#[Service]$vulkan_environment#" \
        $f.service.in > $OMNISCI_TMP/$f.service
    sudo cp $OMNISCI_TMP/$f.service /lib/systemd/system/
  fi
done

sed -e "s#@OMNISCI_PATH@#${vars['OMNISCI_PATH']}#g" \
    -e "s#@OMNISCI_STORAGE@#${vars['OMNISCI_STORAGE']}#g" \
    -e "s#@OMNISCI_DATA@#${vars['OMNISCI_DATA']}#g" \
    -e "s#@OMNISCI_USER@#${vars['OMNISCI_USER']}#g" \
    -e "s#@OMNISCI_GROUP@#${vars['OMNISCI_GROUP']}#g" \
    omnisci.conf.in > $OMNISCI_TMP/omnisci.conf
if [ -f omnisci-sds.conf.in ]; then
  sed -e "s#@OMNISCI_PATH@#${vars['OMNISCI_PATH']}#g" \
      -e "s#@OMNISCI_STORAGE@#${vars['OMNISCI_STORAGE']}#g" \
      -e "s#@OMNISCI_DATA@#${vars['OMNISCI_DATA']}#g" \
      -e "s#@OMNISCI_USER@#${vars['OMNISCI_USER']}#g" \
      -e "s#@OMNISCI_GROUP@#${vars['OMNISCI_GROUP']}#g" \
      omnisci-sds.conf.in > $OMNISCI_TMP/omnisci-sds.conf
  sudo cp $OMNISCI_TMP/omnisci-sds.conf ${vars['OMNISCI_STORAGE']}
  sudo chown ${vars['OMNISCI_USER']}:${vars['OMNISCI_GROUP']} "${vars['OMNISCI_STORAGE']}/omnisci-sds.conf"
fi
sudo cp $OMNISCI_TMP/omnisci.conf ${vars['OMNISCI_STORAGE']}
sudo chown ${vars['OMNISCI_USER']}:${vars['OMNISCI_GROUP']} "${vars['OMNISCI_STORAGE']}/omnisci.conf"

sudo systemctl daemon-reload
