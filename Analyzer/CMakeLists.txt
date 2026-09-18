# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2015-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-attributes")

set(analyzer_source_files
    Analyzer.cpp
    Analyzer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../QueryEngine/DateTruncate.cpp
    RangeTableEntry.cpp
)

add_library(Analyzer ${analyzer_source_files})

target_link_libraries(Analyzer Catalog)
