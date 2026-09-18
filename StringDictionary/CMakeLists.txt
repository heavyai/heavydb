# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2015-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

add_library(StringDictionary
  StringDictionary.cpp
  StringDictionaryProxy.cpp
)
target_link_libraries(StringDictionary
  UtilsStandalone
  StringOps
  ${Boost_LIBRARIES}
  ${PROFILER_LIBS}
  ${TBB_LIBS}
)
