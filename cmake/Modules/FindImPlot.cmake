# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

#.rst:
# FindImPlot
# --------
#
# Find the path to the ImPlot source tree
#
# Minimal search, checks CMAKE_PREFIX_PATH and IMPLOT_PATH env var
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# This module defines the following variables:
#
# ::
#
#   ImPlot_PATH - Path to the ImPlot source tree
#   ImPlot_FOUND - true if ImPlot has been found and can be used

find_path(ImPlot_PATH
  NAMES implot.h
  PATHS ENV CMAKE_PREFIX_PATH
  PATHS ENV IMPLOT_PATH
  PATH_SUFFIXES "implot"
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ImPlot REQUIRED_VARS ImPlot_PATH)
