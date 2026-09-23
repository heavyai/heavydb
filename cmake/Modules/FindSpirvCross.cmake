# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

# FindSpirvCross
# --------
#
# Find the spirv-cross libraries (these are independent of spirv-tools)
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# This module defines the following variables:
#
#   SpirvCross_INCLUDE_DIR - include directories for spirv-cross
#   SpirvCross_LIBRARIES - libraries to link against
#   SpirvCross_FOUND - true if spirv-cross has been found and can be used

set(LibPaths
    /usr/lib
    /usr/local/lib)

find_library(SpirvCross_Core_LIBRARY
             NAMES
              spirv-cross-core
             HINTS
              ENV LD_LIBRARY_PATH
             PATHS ${LibPaths})

find_library(SpirvCross_C_LIBRARY
             NAMES
              spirv-cross-c
             HINTS
              ENV LD_LIBRARY_PATH
             PATHS ${LibPaths})

find_library(SpirvCross_Cpp_LIBRARY
             NAMES
              spirv-cross-cpp
             HINTS
              ENV LD_LIBRARY_PATH
             PATHS ${LibPaths})

find_library(SpirvCross_Reflect_LIBRARY
             NAMES
              spirv-cross-reflect
             HINTS
              ENV LD_LIBRARY_PATH
             PATHS ${LibPaths})

find_library(SpirvCross_Util_LIBRARY
             NAMES
              spirv-cross-util
             HINTS
              ENV LD_LIBRARY_PATH
             PATHS ${LibPaths})

find_library(SpirvCross_Glsl_LIBRARY
             NAMES
              spirv-cross-glsl
             HINTS
              ENV LD_LIBRARY_PATH
             PATHS ${LibPaths})

find_library(SpirvCross_Hlsl_LIBRARY
             NAMES
              spirv-cross-hlsl
             HINTS
              ENV LD_LIBRARY_PATH
             PATHS ${LibPaths})

find_library(SpirvCross_Msl_LIBRARY
             NAMES
              spirv-cross-msl
             HINTS
              ENV LD_LIBRARY_PATH
             PATHS ${LibPaths})

find_path(SpirvCross_INCLUDE_DIR
          NAMES spirv_cross.hpp
          HINTS ${SpirvCross_Core_LIBRARY}/../../include/spirv_cross
          PATHS
            /usr/include
            /usr/local/include
            /usr/local/homebrew/include
            /opt/local/include)

get_filename_component(SpirvCross_Core_LIBRARY_DIR ${SpirvCross_Core_LIBRARY} DIRECTORY)
get_filename_component(SpirvCross_C_LIBRARY_DIR ${SpirvCross_C_LIBRARY} DIRECTORY)
get_filename_component(SpirvCross_Cpp_LIBRARY_DIR ${SpirvCross_Cpp_LIBRARY} DIRECTORY)
get_filename_component(SpirvCross_Reflect_LIBRARY_DIR ${SpirvCross_Reflect_LIBRARY} DIRECTORY)
get_filename_component(SpirvCross_Util_LIBRARY_DIR ${SpirvCross_Util_LIBRARY} DIRECTORY)
get_filename_component(SpirvCross_Glsl_LIBRARY_DIR ${SpirvCross_Glsl_LIBRARY} DIRECTORY)
get_filename_component(SpirvCross_Hlsl_LIBRARY_DIR ${SpirvCross_Hlsl_LIBRARY} DIRECTORY)
get_filename_component(SpirvCross_Msl_LIBRARY_DIR ${SpirvCross_Msl_LIBRARY} DIRECTORY)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SpirvCross REQUIRED_VARS SpirvCross_Core_LIBRARY SpirvCross_INCLUDE_DIR
  SpirvCross_C_LIBRARY
  SpirvCross_Cpp_LIBRARY
  SpirvCross_Reflect_LIBRARY
  SpirvCross_Util_LIBRARY
  SpirvCross_Glsl_LIBRARY
  SpirvCross_Hlsl_LIBRARY
  SpirvCross_Msl_LIBRARY
)

if(SpirvCross_FOUND AND NOT TARGET spirv_cross::core)
  add_library(spirv_cross::core UNKNOWN IMPORTED)
  set_target_properties(spirv_cross::core PROPERTIES
    IMPORTED_LOCATION "${SpirvCross_Core_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${SpirvCross_INCLUDE_DIR}"
  )
  add_library(spirv_cross::c UNKNOWN IMPORTED)
  set_target_properties(spirv_cross::c PROPERTIES
    IMPORTED_LOCATION "${SpirvCross_C_LIBRARY}"
  )
  add_library(spirv_cross::cpp UNKNOWN IMPORTED)
  set_target_properties(spirv_cross::cpp PROPERTIES
    IMPORTED_LOCATION "${SpirvCross_Cpp_LIBRARY}"
  )
  add_library(spirv_cross::reflect UNKNOWN IMPORTED)
  set_target_properties(spirv_cross::reflect PROPERTIES
    IMPORTED_LOCATION "${SpirvCross_Reflect_LIBRARY}"
  )
  add_library(spirv_cross::util UNKNOWN IMPORTED)
  set_target_properties(spirv_cross::util PROPERTIES
    IMPORTED_LOCATION "${SpirvCross_Util_LIBRARY}"
  )
  add_library(spirv_cross::glsl UNKNOWN IMPORTED)
  set_target_properties(spirv_cross::glsl PROPERTIES
    IMPORTED_LOCATION "${SpirvCross_Glsl_LIBRARY}"
  )
  add_library(spirv_cross::hlsl UNKNOWN IMPORTED)
  set_target_properties(spirv_cross::hlsl PROPERTIES
    IMPORTED_LOCATION "${SpirvCross_Hlsl_LIBRARY}"
  )
  add_library(spirv_cross::msl UNKNOWN IMPORTED)
  set_target_properties(spirv_cross::msl PROPERTIES
    IMPORTED_LOCATION "${SpirvCross_Msl_LIBRARY}"
  )
endif()

# Currently unused libraries commented out here to spare the
# linker
set(SpirvCross_LIBRARIES
  # spirv_cross::c
  spirv_cross::cpp
  spirv_cross::reflect
  spirv_cross::util
  spirv_cross::glsl
  # spirv_cross::hlsl
  # spirv_cross::msl
  spirv_cross::core
)
