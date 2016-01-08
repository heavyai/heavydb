#!/bin/bash

declare -A descs
declare -A vars
descs["MAPD_PATH"]="MapD install directory"
vars["MAPD_PATH"]=${MAPD_PATH:=$(dirname $(pwd))}

descs["MAPD_STORAGE"]="MapD data and configuration storage directory"
vars["MAPD_STORAGE"]=${MAPD_STORAGE:="/var/lib/mapd"}
vars["MAPD_DATA"]=${MAPD_DATA:="${MAPD_STORAGE}/data"}

descs["MAPD_USER"]="user MapD will be run as"
vars["MAPD_USER"]=${MAPD_USER:=$(id --user --name)}
descs["MAPD_GROUP"]="group MapD will be run as"
vars["MAPD_GROUP"]=${MAPD_GROUP:=$(id --group --name)}

descs["CUDA_HOME"]="CUDA Toolkit install directory"
vars["CUDA_HOME"]=${CUDA_HOME:="/usr/local/cuda"}


for v in MAPD_PATH MAPD_STORAGE MAPD_USER MAPD_GROUP CUDA_HOME; do
  echo "$v: ${descs["$v"]}"
  read -p "[${vars[$v]}]: "
  if [ ! -z "$REPLY" ]; then
    vars[$v]=$REPLY
  fi
  echo
done

for v in MAPD_PATH MAPD_STORAGE MAPD_USER MAPD_GROUP CUDA_HOME; do
  echo -e "$v:\t${vars[$v]}"
done

sudo mkdir -p ${vars['MAPD_DATA']}
sudo mkdir -p ${vars['MAPD_STORAGE']}

for f in mapd_server mapd_server@ mapd_web_server mapd_web_server@ ; do
  sed -e "s#@MAPD_PATH@#${vars['MAPD_PATH']}#g" \
      -e "s#@MAPD_STORAGE@#${vars['MAPD_STORAGE']}#g" \
      -e "s#@MAPD_DATA@#${vars['MAPD_DATA']}#g" \
      -e "s#@MAPD_USER@#${vars['MAPD_USER']}#g" \
      -e "s#@MAPD_GROUP@#${vars['MAPD_GROUP']}#g" \
      -e "s#@CUDA_HOME@#${vars['CUDA_HOME']}#g" \
      $f.service.in > $f.service
  sudo cp $f.service /lib/systemd/system/
done

sed -e "s#@MAPD_PATH@#${vars['MAPD_PATH']}#g" \
    -e "s#@MAPD_STORAGE@#${vars['MAPD_STORAGE']}#g" \
    -e "s#@MAPD_DATA@#${vars['MAPD_DATA']}#g" \
    -e "s#@MAPD_USER@#${vars['MAPD_USER']}#g" \
    -e "s#@MAPD_GROUP@#${vars['MAPD_GROUP']}#g" \
    -e "s#@CUDA_HOME@#${vars['CUDA_HOME']}#g" \
    mapd.conf.in > mapd.conf
sudo cp mapd.conf ${vars['MAPD_STORAGE']}

systemctl daemon-reload
