# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

set(udf_compiler_source_files
  UdfCompiler.cpp)

add_library(UdfCompiler ${udf_compiler_source_files})

# Clang
find_package(Clang REQUIRED)
include_directories(SYSTEM ${CLANG_INCLUDE_DIRS})
add_definitions(${CLANG_DEFINITIONS})

# Clang headers may emit -Werror=nonnull in RelWithDebInfo builds.
if (ENABLE_WARNINGS_AS_ERRORS
    AND ${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU"
    AND "${CMAKE_BUILD_TYPE_LOWER}" STREQUAL "relwithdebinfo")
  set_source_files_properties(UdfCompiler.cpp PROPERTIES COMPILE_FLAGS -Wno-error=nonnull)
endif()

target_link_libraries(UdfCompiler Logger ${clang_libs})
