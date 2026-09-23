# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-attributes")

set(string_ops_source_files
    StringOpInfo.cpp
    StringOpInfo.h
    StringOps.cpp
    StringOps.h
)

add_library(StringOps ${string_ops_source_files})
target_link_libraries(StringOps ${CPR_LIBRARY} ${CURL_LIBRARIES})
