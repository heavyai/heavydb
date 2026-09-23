# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2015-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

find_package(Boost COMPONENTS program_options filesystem regex system REQUIRED QUIET)

include_directories(${CMAKE_SOURCE_DIR}/ThirdParty/linenoise)

add_executable(heavysql heavysql.cpp)
target_link_libraries(heavysql mapd_thrift Logger Shared ThriftClient linenoise ${Boost_LIBRARIES} ${CMAKE_DL_LIBS} ${PROFILER_LIBS} ${Thrift_LIBRARIES})

install(TARGETS heavysql DESTINATION bin COMPONENT "exe")
