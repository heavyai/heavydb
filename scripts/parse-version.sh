#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if [ -f ../CMakeLists.txt ] ; then
echo $(cat ../CMakeLists.txt | grep -oP '(?<=set\(MAPD_VERSION_MAJOR ").*(?="\))').$(\
      cat ../CMakeLists.txt | grep -oP '(?<=set\(MAPD_VERSION_MINOR ").*(?="\))').$(\
      cat ../CMakeLists.txt | grep -oP '(?<=set\(MAPD_VERSION_PATCH ").*(?="\))')$(\
      cat ../CMakeLists.txt | grep -oP '(?<=set\(MAPD_VERSION_EXTRA ").*(?="\))')
fi
