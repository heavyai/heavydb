# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

set(migration_mgr_source_files
    MigrationMgr.cpp
)

add_library(MigrationMgr ${migration_mgr_source_files})

target_link_libraries(MigrationMgr Shared QueryEngine ${Boost_LIBRARIES})

