# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

#.rst:
# FindRdKafka.cmake
# -------------
#
# Find a RdKafka installation.
#
# This module finds if RdKafka is installed and selects a default
# configuration to use.
#
# find_package(RdKafka ...)
#
#
# The following variables control which libraries are found::
#
#   RdKafka_USE_STATIC_LIBS  - Set to ON to force use of static libraries.
#
# The following are set after the configuration is done:
#
# ::
#
#   RdKafka_FOUND            - Set to TRUE if RdKafka was found.
#   RdKafka_LIBRARIES        - Path to the RdKafka libraries.
#   RdKafka_LIBRARY_DIRS     - compile time link directories
#   RdKafka_INCLUDE_DIRS     - compile time include directories
#
#
# Sample usage:
#
# ::
#
#    find_package(RdKafka)
#    if(RdKafka_FOUND)
#      target_link_libraries(<YourTarget> ${RdKafka_LIBRARIES})
#    endif()

if(RdKafka_USE_STATIC_LIBS)
  set(_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
  set(CMAKE_FIND_LIBRARY_SUFFIXES .lib .a ${CMAKE_FIND_LIBRARY_SUFFIXES})
endif()


find_library(RdKafka_LIBRARY
  NAMES rdkafka
  HINTS
  ENV LD_LIBRARY_PATH
  ENV DYLD_LIBRARY_PATH
  PATHS
  /usr/lib
  /usr/local/lib
  /usr/local/homebrew/lib
  /opt/local/lib)

find_library(RdKafka++_LIBRARY
  NAMES rdkafka++
  HINTS
  ENV LD_LIBRARY_PATH
  ENV DYLD_LIBRARY_PATH
  PATHS
  /usr/lib
  /usr/local/lib
  /usr/local/homebrew/lib
  /opt/local/lib)

if(RdKafka_USE_STATIC_LIBS)
  set(CMAKE_FIND_LIBRARY_SUFFIXES ${_CMAKE_FIND_LIBRARY_SUFFIXES})
endif()

get_filename_component(RdKafka_LIBRARY_DIR ${RdKafka_LIBRARY} DIRECTORY)

find_library(lz4_LIBRARY lz4 REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(ZLIB REQUIRED)

# Set standard CMake FindPackage variables if found.
set(RdKafka_LIBRARIES
  ${RdKafka++_LIBRARY}
  ${RdKafka_LIBRARY}
  ${lz4_LIBRARY}
  ${OPENSSL_LIBRARIES}
  ${ZLIB_LIBRARIES}
  ${CMAKE_DL_LIBS})
# Static librdkafka may be built with libcurl (HTTP/OAuth support).
if(NOT CURL_FOUND)
  find_package(CURL QUIET)
endif()
if(CURL_FOUND)
  list(APPEND RdKafka_LIBRARIES ${CURL_LIBRARIES})
endif()
set(RdKafka_LIBRARY_DIRS ${RdKafka_LIBRARY_DIR})
set(RdKafka_INCLUDE_DIRS ${RdKafka_LIBRARY_DIR}/../include)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(RdKafka REQUIRED_VARS RdKafka_LIBRARY)
