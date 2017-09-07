#!/bin/bash

declare -A descs
declare -A vars

MAPD_TMP=$(mktemp -d)

descs["MAPD_PATH"]="MapD install directory"
vars["MAPD_PATH"]=${MAPD_PATH:=$(dirname $(pwd))}

descs["MAPD_STORAGE"]="MapD data and configuration storage directory"
vars["MAPD_STORAGE"]=${MAPD_STORAGE:="/var/lib/mapd"}

descs["MAPD_USER"]="user MapD will be run as"
vars["MAPD_USER"]=${MAPD_USER:=$(id --user --name)}
descs["MAPD_GROUP"]="group MapD will be run as"
vars["MAPD_GROUP"]=${MAPD_GROUP:=$(id --group --name)}

for v in MAPD_PATH MAPD_STORAGE MAPD_USER MAPD_GROUP ; do
  echo "$v: ${descs["$v"]}"
  read -p "[${vars[$v]}]: "
  if [ ! -z "$REPLY" ]; then
    vars[$v]=$REPLY
  fi
  echo
done

for v in MAPD_PATH MAPD_STORAGE MAPD_USER MAPD_GROUP ; do
  echo -e "$v:\t${vars[$v]}"
done

vars["MAPD_DATA"]=${MAPD_DATA:="${vars['MAPD_STORAGE']}/data"}
sudo mkdir -p "${vars['MAPD_DATA']}"
sudo mkdir -p "${vars['MAPD_STORAGE']}"
if [ -f mapd-sds.conf.in ]; then
  sudo mkdir -p "${vars['MAPD_STORAGE']}/sds"
fi

if [ ! -d "${vars['MAPD_DATA']}/mapd_catalogs" ]; then
  sudo ${vars["MAPD_PATH"]}/bin/initdb ${vars['MAPD_DATA']}
fi

sudo chown -R ${vars['MAPD_USER']}:${vars['MAPD_GROUP']} "${vars['MAPD_DATA']}"
sudo chown -R ${vars['MAPD_USER']}:${vars['MAPD_GROUP']} "${vars['MAPD_STORAGE']}"


for f in mapd_server mapd_server@ mapd_sd_server mapd_sd_server@ mapd_web_server mapd_web_server@ ; do
  if [ -f $f.service.in ]; then
    sed -e "s#@MAPD_PATH@#${vars['MAPD_PATH']}#g" \
        -e "s#@MAPD_STORAGE@#${vars['MAPD_STORAGE']}#g" \
        -e "s#@MAPD_DATA@#${vars['MAPD_DATA']}#g" \
        -e "s#@MAPD_USER@#${vars['MAPD_USER']}#g" \
        -e "s#@MAPD_GROUP@#${vars['MAPD_GROUP']}#g" \
        $f.service.in > $MAPD_TMP/$f.service
    sudo cp $MAPD_TMP/$f.service /lib/systemd/system/
  fi
done
if [ -f mapd_xorg.service ]; then
	sudo cp mapd_xorg.service /lib/systemd/system/
fi

sed -e "s#@MAPD_PATH@#${vars['MAPD_PATH']}#g" \
    -e "s#@MAPD_STORAGE@#${vars['MAPD_STORAGE']}#g" \
    -e "s#@MAPD_DATA@#${vars['MAPD_DATA']}#g" \
    -e "s#@MAPD_USER@#${vars['MAPD_USER']}#g" \
    -e "s#@MAPD_GROUP@#${vars['MAPD_GROUP']}#g" \
    mapd.conf.in > $MAPD_TMP/mapd.conf
if [ -f mapd-sds.conf.in ]; then
  sed -e "s#@MAPD_PATH@#${vars['MAPD_PATH']}#g" \
      -e "s#@MAPD_STORAGE@#${vars['MAPD_STORAGE']}#g" \
      -e "s#@MAPD_DATA@#${vars['MAPD_DATA']}#g" \
      -e "s#@MAPD_USER@#${vars['MAPD_USER']}#g" \
      -e "s#@MAPD_GROUP@#${vars['MAPD_GROUP']}#g" \
      mapd-sds.conf.in > $MAPD_TMP/mapd-sds.conf
  sudo cp $MAPD_TMP/mapd-sds.conf ${vars['MAPD_STORAGE']}
  sudo chown ${vars['MAPD_USER']}:${vars['MAPD_GROUP']} "${vars['MAPD_STORAGE']}/mapd-sds.conf"
fi
sudo cp $MAPD_TMP/mapd.conf ${vars['MAPD_STORAGE']}
sudo chown ${vars['MAPD_USER']}:${vars['MAPD_GROUP']} "${vars['MAPD_STORAGE']}/mapd.conf"

sudo systemctl daemon-reload
