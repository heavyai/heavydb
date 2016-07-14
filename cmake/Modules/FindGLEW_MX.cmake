#.rst:
# FindGLEW_MX
# --------
#
# Find the OpenGL Extension Wrangler Library (GLEW_MX)
#
# IMPORTED Targets
# ^^^^^^^^^^^^^^^^
#
# This module defines the :prop_tgt:`IMPORTED` target ``GLEW_MX::GLEW_MX``,
# if GLEW_MX has been found.
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# This module defines the following variables:
#
# ::
#
#   GLEW_MX_INCLUDE_DIRS - include directories for GLEW_MX
#   GLEW_MX_LIBRARIES - libraries to link against GLEW_MX
#   GLEW_MX_FOUND - true if GLEW_MX has been found and can be used

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

find_library(GLEW_MX_LIBRARY
  NAMES GLEWmx GLEWMX glewmx32 glewmx glewmx32s
  HINTS ENV LD_LIBRARY_PATH
  PATH_SUFFIXES lib64)
get_filename_component(GLEW_LIB_DIR ${GLEW_MX_LIBRARY} DIRECTORY)
get_filename_component(GLEW_INST_DIR ${GLEW_LIB_DIR} DIRECTORY)
find_path(GLEW_MX_INCLUDE_DIR GL/glew.h HINTS ${GLEW_INST_DIR}/include)

set(GLEW_MX_INCLUDE_DIRS ${GLEW_MX_INCLUDE_DIR})
set(GLEW_MX_LIBRARIES ${GLEW_MX_LIBRARY})
set(GLEW_MX_DEFINITIONS "-DGLEW_MX")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GLEW_MX
                                  REQUIRED_VARS GLEW_MX_INCLUDE_DIR GLEW_MX_LIBRARY)

if(GLEW_MX_FOUND AND NOT TARGET GLEW_MX::GLEW_MX)
  add_library(GLEW_MX::GLEW_MX UNKNOWN IMPORTED)
  set_target_properties(GLEW_MX::GLEW_MX PROPERTIES
    IMPORTED_LOCATION "${GLEW_MX_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${GLEW_MX_INCLUDE_DIRS}")
endif()

mark_as_advanced(GLEW_MX_INCLUDE_DIR GLEW_MX_LIBRARY)
