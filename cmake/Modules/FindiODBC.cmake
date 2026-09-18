# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

#.rst:
# FindiODBC.cmake
# -------------
#
# Find a iODBC installation.
#
# This module finds if iODBC is installed and selects a default
# configuration to use.
#
# find_package(iODBC ...)
#
#
# The following variables control which libraries are found::
#
#   iODBC_USE_STATIC_LIBS  - Set to ON to force use of static libraries.
#
# The following are set after the configuration is done:
#
# ::
#
#   iODBC_FOUND            - Set to TRUE if iODBC was found.
#   iODBC_INCLUDE_DIRS     - Include directories
#   iODBC_LIBRARIES        - Path to the iODBC libraries.
#   iODBCINST_LIBRARIES    - Path to the iODBC inst library.
#   iODBC_LIBRARY_DIRS     - compile time link directories
#
#
# Sample usage:
#
# ::
#
#    find_package(iODBC)
#    if(iODBC_FOUND)
#      target_link_libraries(<YourTarget> ${iODBC_LIBRARIES})
#    endif()

set(iODBC_FOUND "true") ## set true by default and set false for errors
if(iODBC_USE_STATIC_LIBS)
  message(DEBUG "FindiODBC.cmake attemtping to using STATIC libs")
  set(_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
  set(CMAKE_FIND_LIBRARY_SUFFIXES .lib .a ${CMAKE_FIND_LIBRARY_SUFFIXES})
endif()
find_library(iodbc_LIBRARY
  NAMES iodbc
  HINTS ENV LD_LIBRARY_PATH
  HINTS ENV DYLD_LIBRARY_PATH
  PATHS
  /usr/lib
  /usr/local/lib
  /usr/local/homebrew/lib
  /opt/local/lib)
if(NOT iodbc_LIBRARY)
  message(WARNING "FindiODBC.cmake iodbc_LIBRARY not found")
  set(iODBC_FOUND "false")
endif()

find_library(iodbcinst_LIBRARY
  NAMES iodbcinst
  HINTS ENV LD_LIBRARY_PATH
  HINTS ENV DYLD_LIBRARY_PATH
  PATHS
  /usr/lib
  /usr/local/lib
  /usr/local/homebrew/lib
  /opt/local/lib)
if(NOT iodbcinst_LIBRARY)
  message(WARNING "FindiODBC.cmake iodbcinst_LIBRARY not found")
  set(iODBC_FOUND "false")
endif()

get_filename_component(iodbc_LIBRARY_DIR ${iodbc_LIBRARY} DIRECTORY)

find_path(iodbc_INCLUDE_DIR
  NAMES sql.h
  HINTS ${iodbc_LIBRARY}/../../include
  PATHS
  /usr/include
  /usr/local/include
  /usr/local/homebrew/include
  /opt/local/include)
if(NOT iodbc_INCLUDE_DIR)
  message(WARNING "FindiODBC.cmake iodbc include dir not found")
  set(iODBC_FOUND "false")
endif()


# Set standard CMake FindPackage variables if found.
set(iODBC_LIBRARIES ${iodbc_LIBRARY} ${iodbcinst_LIBRARY})
set(iODBCINST_LIBRARY ${iodbcinst_LIBRARY})
set(iODBC_INCLUDE_DIRS ${iodbc_INCLUDE_DIR})
set(iODBC_LIBRARY_DIRS ${iodbc_LIBRARY_DIR})

## unset all of the variable, incase
## the find_package is called twice
## once with STATIC on and then without 
## STATIC
unset(iodbc_LIBRARY CACHE)
unset(iodbcinst_LIBRARY CACHE)
unset(iodbc_INCLUDE_DIR CACHE)
unset(iodbc_LIBRARY_DIR CACHE)


if(iODBC_USE_STATIC_LIBS)
  set(CMAKE_FIND_LIBRARY_SUFFIXES ${_CMAKE_FIND_LIBRARY_SUFFIXES})
endif()

include(FindPackageHandleStandardArgs)
#   iODBC_FOUND            - Set to TRUE if iODBC was found.
#   iODBC_INCLUDE_DIRS     - Include directories
#   iODBC_LIBRARIES        - Path to the iODBC libraries.
#   iODBCINST_LIBRARY    - Path to the iODBC inst library.
#   iODBC_LIBRARY_DIRS     - compile time link directories
find_package_handle_standard_args(iODBC REQUIRED_VARS iODBC_FOUND iODBC_INCLUDE_DIRS iODBC_LIBRARIES iODBCINST_LIBRARY iODBC_LIBRARY_DIRS)
