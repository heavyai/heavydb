#!/bin/sh
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if [ $# -ne 2 ]; then
  echo "Unexpected number of arguments. Expected 2 arguments in order:
        (1) path to downloaded dashboard
        (2) dashboard file name"
  exit 1
fi

head -2 $1 > DashboardContent/$2
tail -1 $1 | base64 -w 0 >> DashboardContent/$2
