#!/bin/bash

declare -A descs
declare -A vars

OMNISCI_TMP=$(mktemp -d)

descs["OMNISCI_PATH"]="OmniSci install directory"
vars["OMNISCI_PATH"]=${OMNISCI_PATH:=$(dirname $(pwd))}

descs["OMNISCI_STORAGE"]="OmniSci data and configuration storage directory"
vars["OMNISCI_STORAGE"]=${OMNISCI_STORAGE:="/var/lib/omnisci"}

descs["OMNISCI_USER"]="user OmniSci will be run as"
vars["OMNISCI_USER"]=${OMNISCI_USER:=$(id --user --name)}
descs["OMNISCI_GROUP"]="group OmniSci will be run as"
vars["OMNISCI_GROUP"]=${OMNISCI_GROUP:=$(id --group --name)}

for v in OMNISCI_PATH OMNISCI_STORAGE OMNISCI_USER OMNISCI_GROUP ; do
  echo "$v: ${descs["$v"]}"
  read -p "[${vars[$v]}]: "
  if [ ! -z "$REPLY" ]; then
    vars[$v]=$REPLY
  fi
  echo
done

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


for f in omnisci_server omnisci_server@ omnisci_sd_server omnisci_sd_server@ omnisci_web_server omnisci_web_server@ ; do
  if [ -f $f.service.in ]; then
    sed -e "s#@OMNISCI_PATH@#${vars['OMNISCI_PATH']}#g" \
        -e "s#@OMNISCI_STORAGE@#${vars['OMNISCI_STORAGE']}#g" \
        -e "s#@OMNISCI_DATA@#${vars['OMNISCI_DATA']}#g" \
        -e "s#@OMNISCI_USER@#${vars['OMNISCI_USER']}#g" \
        -e "s#@OMNISCI_GROUP@#${vars['OMNISCI_GROUP']}#g" \
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
