# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2015-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

add_library(SqliteConnector SqliteConnector.cpp SqliteConnector.h)

target_link_libraries(SqliteConnector SQLite::SQLite3 ${Boost_THREAD_LIBRARY})
