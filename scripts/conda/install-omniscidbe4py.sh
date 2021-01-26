#!/usr/bin/env bash
set -xe
cd build/Embedded
${PYTHON:-python} setup.py build_ext -g -f install
