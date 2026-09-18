# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

# FindGlslang
# --------
#
# Find glslang and supporting Spirv libraries
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# This module defines the following variables:
#
#   Glslang_INCLUDE_DIR - include directories for glslang
#   Glslang_LIBRARIES - libraries to link against
#   Glslang_FOUND - true if Glslang and Spir-V have been found and can be used

find_path(Glslang_INCLUDE_DIR
          NAMES glslang/Public/ShaderLang.h
          HINTS ${Glslang_LIBRARY}/../../include
          PATHS
            /usr/include
            /usr/local/include
            /usr/local/homebrew/include
            /opt/local/include)

set(LibPaths
    /usr/lib
    /usr/local/lib)

find_library(Glslang_LIBRARY
             NAMES
              glslang
             HINTS
              ENV LD_LIBRARY_PATH
             PATHS ${LibPaths})

find_library(MachineIndependent_LIBRARY
            NAMES
              MachineIndependent
            HINTS
              ENV LD_LIBRARY_PATH
            PATHS ${LibPaths})

find_library(GenericCodeGen_LIBRARY
            NAMES
              GenericCodeGen
            HINTS
              ENV LD_LIBRARY_PATH
            PATHS ${LibPaths})

find_library(OSDependent_LIBRARY
             NAMES
              OSDependent
             HINTS
              ENV LD_LIBRARY_PATH
             PATHS ${LibPaths})

find_library(SPIRV_LIBRARY
             NAMES
              SPIRV
             HINTS
              ENV LD_LIBRARY_PATH
             PATHS ${LibPaths})

find_library(SPIRV-Tools_LIBRARY
             NAMES
              SPIRV-Tools
             HINTS
              ENV LD_LIBRARY_PATH
             PATHS ${LibPaths})

find_library(SPIRV-Tools-opt_LIBRARY
             NAMES
              SPIRV-Tools-opt
             HINTS
              ENV LD_LIBRARY_PATH
             PATHS ${LibPaths})

get_filename_component(Glslang_LIBRARY_DIR ${Glslang_LIBRARY} DIRECTORY)
get_filename_component(MachineIndependent_LIBRARY_DIR ${MachineIndependent_LIBRARY} DIRECTORY)
get_filename_component(GenericCodeGen_LIBRARY_DIR ${GenericCodeGen_LIBRARY} DIRECTORY)
get_filename_component(OSDependent_LIBRARY_DIR ${OSDependent_LIBRARY} DIRECTORY)
get_filename_component(SPIRV_LIBRARY_DIR ${SPIRV_LIBRARY} DIRECTORY)
get_filename_component(SPIRV-Tools_LIBRARY_DIR ${SPIRV-Tools_LIBRARY} DIRECTORY)
get_filename_component(SPIRV-Tools-opt_LIBRARY_DIR ${SPIRV-Tools-opt_LIBRARY} DIRECTORY)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Glslang REQUIRED_VARS
  Glslang_LIBRARY Glslang_INCLUDE_DIR
  MachineIndependent_LIBRARY
  GenericCodeGen_LIBRARY
  OSDependent_LIBRARY
  SPIRV_LIBRARY
  SPIRV-Tools_LIBRARY
  SPIRV-Tools-opt_LIBRARY
)

if(Glslang_FOUND AND NOT TARGET glslang::glslang)
  add_library(glslang::glslang UNKNOWN IMPORTED)
  set_target_properties(glslang::glslang PROPERTIES
    IMPORTED_LOCATION "${Glslang_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${Glslang_INCLUDE_DIR}"
  )

  add_library(glslang::MachineIndependent UNKNOWN IMPORTED)
  set_target_properties(glslang::MachineIndependent PROPERTIES
    IMPORTED_LOCATION "${MachineIndependent_LIBRARY}"
  )

  add_library(glslang::GenericCodeGen UNKNOWN IMPORTED)
  set_target_properties(glslang::GenericCodeGen PROPERTIES
    IMPORTED_LOCATION "${GenericCodeGen_LIBRARY}"
  )

  add_library(glslang::OSDependent UNKNOWN IMPORTED)
  set_target_properties(glslang::OSDependent PROPERTIES
    IMPORTED_LOCATION "${OSDependent_LIBRARY}"
  )

  add_library(glslang::SPIRV UNKNOWN IMPORTED)
  set_target_properties(glslang::SPIRV PROPERTIES
    IMPORTED_LOCATION "${SPIRV_LIBRARY}"
  )

  add_library(glslang::SPIRV-Tools UNKNOWN IMPORTED)
  set_target_properties(glslang::SPIRV-Tools PROPERTIES
    IMPORTED_LOCATION "${SPIRV-Tools_LIBRARY}"
  )

  add_library(glslang::SPIRV-Tools-opt UNKNOWN IMPORTED)
  set_target_properties(glslang::SPIRV-Tools-opt PROPERTIES
    IMPORTED_LOCATION "${SPIRV-Tools-opt_LIBRARY}"
  )
endif()

set(Glslang_LIBRARIES
  glslang::glslang
)

list(APPEND Glslang_LIBRARIES
  glslang::MachineIndependent
  glslang::GenericCodeGen
  glslang::OSDependent
  glslang::SPIRV
  glslang::SPIRV-Tools-opt
  glslang::SPIRV-Tools
)

mark_as_advanced(Glslang_INCLUDE_DIR Glslang_LIBRARY)
