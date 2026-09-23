# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2015-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-write-strings -Wno-unused-function -Wno-unused-label -Wno-sign-compare")

set(parser_source_files
    ParserNode.cpp
    ParserNode.h
    ParserWrapper.cpp
    ParserWrapper.h
)

add_library(Parser ${parser_source_files})

target_link_libraries(Parser QueryParsing QueryState ImportExport Shared Analyzer QueryEngine Catalog TableArchiver DataMgr LockMgr Geospatial)
include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})
