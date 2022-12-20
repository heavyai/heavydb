#!/bin/bash

PATH=$1
OLD_FILE=$PATH/heavyai_git_hash.txt
NEW_FILE=$PATH/heavyai_git_hash_new.txt
if [ -f "$OLD_FILE" ]; then
  read -r OLD_HASH < $OLD_FILE
else
  OLD_HASH=""
fi
read -r NEW_HASH < $NEW_FILE
if [ "$OLD_HASH" != "$NEW_HASH" ]; then
  echo "$NEW_HASH" > $OLD_FILE
fi
