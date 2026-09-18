# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2015-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

set(utils_standalone_source_files
    StringLike.cpp
    Regexp.cpp
    ChunkIter.cpp
)
set(utils_source_files
    ChunkAccessorTable.cpp
    DdlUtils.cpp
)

add_library(Utils ${utils_source_files})
add_library(UtilsStandalone ${utils_standalone_source_files})
add_dependencies(Utils Catalog)
target_link_libraries(UtilsStandalone ${Boost_LIBRARIES} Shared)
target_link_libraries(Utils UtilsStandalone ${Boost_LIBRARIES} Shared)
