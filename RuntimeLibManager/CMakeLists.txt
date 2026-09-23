# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS -Wall -Wno-attributes")

add_library(RuntimeLibManager RuntimeLibManager.cpp RuntimeLibManager.h)

if(ENABLE_TORCH_TFS)
  target_include_directories(RuntimeLibManager PRIVATE ${TORCH_INCLUDE_DIRS})
endif()

target_link_libraries(RuntimeLibManager ${llvm_libs})