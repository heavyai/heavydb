#!/bin/bash

if [ -f ../CMakeLists.txt ] ; then
echo $(cat ../CMakeLists.txt | grep -oP '(?<=set\(MAPD_VERSION_MAJOR ").*(?="\))').$(\
      cat ../CMakeLists.txt | grep -oP '(?<=set\(MAPD_VERSION_MINOR ").*(?="\))').$(\
      cat ../CMakeLists.txt | grep -oP '(?<=set\(MAPD_VERSION_PATCH ").*(?="\))')$(\
      cat ../CMakeLists.txt | grep -oP '(?<=set\(MAPD_VERSION_EXTRA ").*(?="\))')
fi
