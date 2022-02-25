#!/bin/bash

## Version 0.1

set +x
if [ $# -eq 1 ]; then
   DATA_DIR=$1
else
   echo 'Invalid number of parameters. '$0' path_to_data_directory'
   exit 1
fi

one() {
   COLD=$1
   CNEW=$2
   DATA_DIR=$3
   CPATH=$4
   if [ -e "$DATA_DIR$CPATH/$COLD" ]; then
      mv $DATA_DIR$CPATH/$COLD $DATA_DIR$CPATH/$CNEW
   else
      echo "One - File not Found. " $DATA_DIR$CPATH/$COLD
   fi
}

symlink() {
   COLD=$1
   CNEW=$2
   DATA_DIR=$3
   CPATH=$4
   if [ -e "$DATA_DIR$CPATH/$COLD" ]; then
      ln -sf $DATA_DIR$CPATH/$COLD $DATA_DIR$CPATH/$CNEW
   else
      echo "Symlink - File not Found or exists. " $DATA_DIR$CPATH/$COLD
   fi
}

remove() {
   COLD=$1
   DATA_DIR=$2
   CPATH=$3
   if [ -e "$DATA_DIR$CPATH/$COLD" ] || [ -L "$DATA_DIR$CPATH/$COLD" ]; then
      rm $DATA_DIR$CPATH/$COLD
   else
      echo "Remove - File not Found. " $DATA_DIR$CPATH/$COLD
   fi
}

rename_data_files() {
  FILE_GLOB=$1
  for i in $DATA_DIR/$FILE_GLOB; do
    if [ -f $i ]; then
      NEW=${i/.mapd/.data}
      mv $i $NEW
    fi
  done
}

# main

IFS=","
while read -a RECORD ;
do
  CTYPE="${RECORD[0]}"
  if [[ ${CTYPE:1:1} == '#' ]]; then
    continue
  fi
  CPATH="${RECORD[1]}"
  COLD="${RECORD[2]}"
  CNEW="${RECORD[3]}"
  case $CTYPE in
    one)
      one $COLD $CNEW $DATA_DIR $CPATH
      ;;
    symlink)
      symlink $COLD $CNEW $DATA_DIR $CPATH
      ;;
    remove)
      remove $COLD $DATA_DIR $CPATH
      ;;
   esac
done <<EOF
############################################################
# This data file is used in rename.sh
#
#one,path to file with leading /,old filename, new filename
#symlink,path to folder with leading /,existing filename,symlink name
#remove,path to folder with leading /,filename to delete
#
############################################################
one,,mapd_catalogs,catalogs
one,,mapd_data,data
one,,mapd_log,log
one,,mapd_export,export
one,,mapd_import,import
symlink,,import,mapd_import
symlink,,export,mapd_export
symlink,,data,mapd_data
symlink,,log,mapd_log
symlink,,catalogs,mapd_catalogs
#
one,,omnisci.license,heavyai.license
one,/catalogs,omnisci_system_catalog,system_catalog
#
remove,,mapd_server_pid.lck
one,,omnisci_server_pid.lck,heavydb_pid.lk
#
one,,omnisci_disk_cache,disk_cache
one,,omnisci_key_store,key_store
one,/key_store,omnisci.pem,heavyai.pem
#
EOF

# Rename nested *.mapd data files
rename_data_files "data/table_*/*.mapd"
# Rename *.mapd disk cache data files
rename_data_files "disk_cache/*.mapd"
