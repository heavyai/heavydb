# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

set(lock_mgr_source_files
    LockMgr.cpp
)

add_library(LockMgr ${lock_mgr_source_files})

add_dependencies(LockMgr calciteserver_thrift thrift_gen)

target_link_libraries(LockMgr Shared ${Boost_THREAD_LIBRARY})
