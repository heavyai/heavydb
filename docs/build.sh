#!/bin/bash

set -e
set -x

if [ "$1" = "--venv" ]; then
  VENV=true
  shift
fi
if ! hash sphinx-build 2>/dev/null; then
  VENV=true
fi

if [ ! -z "$1" ]; then
	VERSION="-D version=$1 -D release=$1"
fi

if [ ! -z "$2" ]; then
	DIST="$2"
else
	DIST="$PWD/generated-docs/"
fi

if [ "$VENV" = true ] ; then
	# Required so that Sphinx doesn't gen docs for the Python modules
	TMPVENV=$(mktemp -d /tmp/tmp-venv.$(date +%s).XXXXX)
	virtualenv $TMPVENV/.venv
	source $TMPVENV/.venv/bin/activate
fi

rm -rf "$DIST"
mkdir -p "$DIST"

TS=$(date +%s)
BUILD="$PWD/build"
mkdir -p $BUILD

if [ "$VENV" = true ] ; then
	pip install -r requirements.txt > $BUILD/log-$i-pip-$TS.txt  2>&1
fi

sphinx-build -Q -b dirhtml -d $BUILD/doctrees $VERSION . $BUILD/dirhtml > $BUILD/log-uber-$TS.txt 2>&1
cp -R $BUILD/dirhtml/* "$DIST"

for i in immerse-user-guide mapd-guide; do
	pushd $i
	if [ "$VENV" = true ] ; then
		pip install -r requirements.txt > $BUILD/log-$i-pip-$TS.txt  2>&1
	fi
	sphinx-build -Q -b latex -d $BUILD/doctrees $VERSION . $BUILD/latex > $BUILD/log-$i-$TS.txt 2>&1
	cp -R ../latex-macros/* $BUILD/latex
	make -C $BUILD/latex all-pdf >> $BUILD/log-$i-$TS.txt  2>&1
	cp $BUILD/latex/*pdf "$DIST"
	popd
done

if [ "$VENV" = true ] ; then
	rm -rf "$TMPVENV"
fi
