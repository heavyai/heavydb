#.rst:
# FindGLBinding
# --------
#
# Find the OpenGL Extension Wrangler Library (glbinding)
#
# IMPORTED Targets
# ^^^^^^^^^^^^^^^^
#
# This module defines the :prop_tgt:`IMPORTED` target ``glbinding::glbinding``,
# if glbinding has been found.
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# This module defines the following variables:
#
# ::
#
#   glbinding_INCLUDE_DIRS - include directories for glbinding
#   glbinding_LIBRARIES - libraries to link against glbinding
#   glbinding_FOUND - true if glbinding has been found and can be used

#=============================================================================
# Copyright 2012 Benjamin Eikel
#
# Distributed under the OSI-approved BSD License (the "License");
# see accompanying file Copyright.txt for details.
#
# This software is distributed WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the License for more information.
#=============================================================================
# (To distribute this file outside of CMake, substitute the full
#  License text for the above reference.)

find_library(glbinding_LIBRARY
  NAMES glbinding
  HINTS ENV LD_LIBRARY_PATH
  PATH_SUFFIXES lib64)
get_filename_component(glbinding_LIB_DIR ${glbinding_LIBRARY} DIRECTORY)
get_filename_component(glbinding_INST_DIR ${glbinding_LIB_DIR} DIRECTORY)
find_path(glbinding_INCLUDE_DIR glbinding/glbinding.h HINTS ${glbinding_INST_DIR}/include)

find_library(glbinding-aux_LIBRARY
  NAMES glbinding-aux
  HINTS ENV LD_LIBRARY_PATH
  PATH_SUFFIXES lib64)
get_filename_component(glbinding-aux_LIB_DIR ${glbinding_LIBRARY} DIRECTORY)
get_filename_component(glbinding-aux_INST_DIR ${glbinding_LIB_DIR} DIRECTORY)
find_path(glbinding-aux_INCLUDE_DIR glbinding-aux/glbinding-aux_api.h HINTS ${glbinding_INST_DIR}/include)

set(glbinding_INCLUDE_DIRS ${glbinding_INCLUDE_DIR} ${glbinding-aux_INCLUDE_DIR})
set(glbinding_LIBRARIES ${glbinding_LIBRARY} ${glbinding-aux_LIBRARY})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(glbinding
                                  REQUIRED_VARS glbinding_INCLUDE_DIR glbinding_LIBRARY)

find_package_handle_standard_args(glbinding-aux
                                  REQUIRED_VARS glbinding-aux_INCLUDE_DIR glbinding-aux_LIBRARY)

if(glbinding_FOUND AND NOT TARGET glbinding::glbinding)
  add_library(glbinding::glbinding UNKNOWN IMPORTED)
  set_target_properties(glbinding::glbinding PROPERTIES
    IMPORTED_LOCATION "${glbinding_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${glbinding_INCLUDE_DIRS}")
endif()

if(glbinding-aux_FOUND AND NOT TARGET glbinding::glbinding-aux)
  add_library(glbinding::glbinding-aux UNKNOWN IMPORTED)
  set_target_properties(glbinding::glbinding-aux PROPERTIES
    IMPORTED_LOCATION "${glbinding-aux_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${glbinding-aux_INCLUDE_DIRS}")
endif()

mark_as_advanced(glbinding_INCLUDE_DIR glbinding_LIBRARY)
