# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

set(query_runner_source_files
    QueryRunner.cpp
    TestProcessSignalHandler.cpp
)

set(QR_LIBRARIES mapd_thrift Catalog QueryState Parser QueryEngine ImportExport Shared)

if (ENABLE_RUNTIME_LIBS)
  list(APPEND QR_LIBRARIES RuntimeLibManager)
endif()

add_library(QueryRunner ${query_runner_source_files})
target_link_libraries(QueryRunner ${QR_LIBRARIES} ${Boost_LIBRARIES})
