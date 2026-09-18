# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

set(THRIFT_HANDLER_SOURCES DBHandler.cpp RequestInfo.cpp TokenCompletionHints.cpp CommandLineOptions.cpp SystemValidator.cpp ForeignTableRefreshScheduler.cpp)
set(THRIFT_HANDLER_LIBS mapd_thrift Shared ThriftClient ${CMAKE_DL_LIBS})

if(ENABLE_RUNTIME_LIBS)
  list(APPEND THRIFT_HANDLER_LIBS RuntimeLibManager)
endif(ENABLE_RUNTIME_LIBS)

list(APPEND THRIFT_HANDLER_LIBS StringDictionary)
include_directories(${CMAKE_SOURCE_DIR})
list(APPEND THRIFT_HANDLER_SOURCES RenderHandler.cpp HAHandler.cpp)

add_library(QueryState QueryState.cpp)
add_dependencies(QueryState Calcite)

add_library(QueryParsing QueryParsing.cpp QueryAuth.cpp)
add_dependencies(QueryParsing Calcite)

add_library(token_completion_hints TokenCompletionHints.cpp)
target_link_libraries(token_completion_hints mapd_thrift)

add_library(thrift_handler ${THRIFT_HANDLER_SOURCES})
add_dependencies(thrift_handler Parser Calcite)
target_link_libraries(thrift_handler token_completion_hints QueryState QueryParsing UdfCompiler MigrationMgr ${THRIFT_HANDLER_LIBS})
