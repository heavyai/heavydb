#!/usr/bin/env bash

if hash java 2>/dev/null; then
  JAVA_PATH=$(dirname $(readlink -e $(which java)))
  JVM_DIR=$JAVA_PATH/../lib/amd64/server
fi

JVM_DIRS="
  /usr/lib/jvm/java-1.8.0-openjdk-amd64/jre/lib/amd64/server
  /usr/lib/jvm/java-1.7.0-openjdk-amd64/jre/lib/amd64/server
  /usr/lib/jvm/jre-1.8.0-openjdk/lib/amd64/server
  /usr/lib/jvm/jre-1.7.0-openjdk/lib/amd64/server
  /usr/lib/jvm/java-1.8.0-openjdk/jre/lib/amd64/server
  /usr/lib/jvm/java-1.7.0-openjdk/jre/lib/amd64/server
  /usr/lib/jvm/jre/lib/amd64/server
  /usr/lib/jvm/java/jre/lib/amd64/server
  /usr/lib/jvm/default-java/jre/lib/amd64/server
  $JVM_DIR
"


for i in $JVM_DIRS ; do
  if [ -e "$i/libjvm.so" ]; then
    LIBJVM_DIR="$i"
    break
  fi
done

echo "$LIBJVM_DIR"
