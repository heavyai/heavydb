#!/bin/bash

declare -A descs
declare -A vars
descs["MAPD_PATH"]="MapD install directory"
vars["MAPD_PATH"]=${MAPD_PATH:=$(dirname $(pwd))}

descs["MAPD_STORAGE"]="MapD data and configuration storage directory"
vars["MAPD_STORAGE"]=${MAPD_STORAGE:="/var/lib/mapd"}

descs["MAPD_USER"]="user MapD will be run as"
vars["MAPD_USER"]=${MAPD_USER:=$(id --user --name)}
descs["MAPD_GROUP"]="group MapD will be run as"
vars["MAPD_GROUP"]=${MAPD_GROUP:=$(id --group --name)}

descs["MAPD_LIBJVM_DIR"]="directory containing Java's libjvm.so"
vars["MAPD_LIBJVM_DIR"]=${MAPD_LIBJVM_DIR:=$(bash ./find_libjvm_dir.sh)}

for v in MAPD_PATH MAPD_STORAGE MAPD_USER MAPD_GROUP MAPD_LIBJVM_DIR; do
  echo "$v: ${descs["$v"]}"
  read -p "[${vars[$v]}]: "
  if [ ! -z "$REPLY" ]; then
    vars[$v]=$REPLY
  fi
  echo
done

for v in MAPD_PATH MAPD_STORAGE MAPD_USER MAPD_GROUP MAPD_LIBJVM_DIR; do
  echo -e "$v:\t${vars[$v]}"
done

vars["MAPD_DATA"]=${MAPD_DATA:="${vars['MAPD_STORAGE']}/data"}
sudo mkdir -p "${vars['MAPD_DATA']}"
sudo mkdir -p "${vars['MAPD_STORAGE']}"
sudo chown -R ${vars['MAPD_USER']}:${vars['MAPD_GROUP']} "${vars['MAPD_DATA']}"
sudo chown -R ${vars['MAPD_USER']}:${vars['MAPD_GROUP']} "${vars['MAPD_STORAGE']}"

for f in mapd_server mapd_server@ mapd_web_server mapd_web_server@ ; do
  sed -e "s#@MAPD_PATH@#${vars['MAPD_PATH']}#g" \
      -e "s#@MAPD_STORAGE@#${vars['MAPD_STORAGE']}#g" \
      -e "s#@MAPD_DATA@#${vars['MAPD_DATA']}#g" \
      -e "s#@MAPD_USER@#${vars['MAPD_USER']}#g" \
      -e "s#@MAPD_GROUP@#${vars['MAPD_GROUP']}#g" \
      -e "s#@MAPD_LIBJVM_DIR@#${vars['MAPD_LIBJVM_DIR']}#g" \
      $f.service.in > $f.service
  sudo cp $f.service /lib/systemd/system/
done
if [ -f mapd_xorg.service ]; then
	sudo cp mapd_xorg.service /lib/systemd/system/
fi

sed -e "s#@MAPD_PATH@#${vars['MAPD_PATH']}#g" \
    -e "s#@MAPD_STORAGE@#${vars['MAPD_STORAGE']}#g" \
    -e "s#@MAPD_DATA@#${vars['MAPD_DATA']}#g" \
    -e "s#@MAPD_USER@#${vars['MAPD_USER']}#g" \
    -e "s#@MAPD_GROUP@#${vars['MAPD_GROUP']}#g" \
    -e "s#@MAPD_LIBJVM_DIR@#${vars['MAPD_LIBJVM_DIR']}#g" \
    mapd.conf.in > mapd.conf
sudo cp mapd.conf ${vars['MAPD_STORAGE']}
sudo chown ${vars['MAPD_USER']}:${vars['MAPD_GROUP']} "${vars['MAPD_STORAGE']}/mapd.conf"

sudo systemctl daemon-reload

if [ ! -d "${vars['MAPD_DATA']}/mapd_catalogs" ]; then
  echo
  echo "${vars['MAPD_DATA']}/mapd_catalogs does not exist. Please run:"
  echo "  ${vars["MAPD_PATH"]}/bin/initdb ${vars['MAPD_DATA']}"
fi
