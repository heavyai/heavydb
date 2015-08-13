#.rst:
# FindGlog.cmake
# -------------
#
# Find a Google glog installation.
#
# This module finds if Google glog is installed and selects a default
# configuration to use.
#
# find_package(Glog ...)
#
#
# The following variables control which libraries are found::
#
#   Glog_USE_STATIC_LIBS  - Set to ON to force use of static libraries.
#
# The following are set after the configuration is done:
#
# ::
#
#   Glog_FOUND            - Set to TRUE if Glog was found.
#   Glog_INCLUDE_DIRS     - Include directories
#   Glog_LIBRARIES        - Path to the Glog libraries.
#   Glog_LIBRARY_DIRS     - compile time link directories
#
#
# Sample usage:
#
# ::
#
#    find_package(Glog)
#    if(Glog_FOUND)
#      target_link_libraries(<YourTarget> ${Glog_LIBRARIES})
#    endif()

if(Glog_USE_STATIC_LIBS)
  set(_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
  set(CMAKE_FIND_LIBRARY_SUFFIXES .lib .a ${CMAKE_FIND_LIBRARY_SUFFIXES})
endif()

find_library(Glog_LIBRARY
  NAMES glog
  HINTS ENV LD_LIBRARY_PATH
  HINTS ENV DYLD_LIBRARY_PATH
  PATHS
  /usr/lib
  /usr/local/lib
  /usr/local/homebrew/lib
  /opt/local/lib)

find_path(Glog_INCLUDE_DIR
  NAMES glog/logging.h
  HINTS ${Glog_LIBRARY}/../../include
  PATHS
  /usr/include
  /usr/local/include
  /usr/local/homebrew/include
  /opt/local/include)

get_filename_component(Glog_LIBRARY_DIR ${Glog_LIBRARY} DIRECTORY)
# Set standard CMake FindPackage variables if found.
set(Glog_LIBRARIES ${Glog_LIBRARY})
set(Glog_INCLUDE_DIRS ${Glog_INCLUDE_DIR})
set(Glog_LIBRARY_DIRS ${Glog_LIBRARY_DIR})

find_package(Gflags)
if(GFLAGS_FOUND)
  list(APPEND Glog_LIBRARIES ${Gflags_LIBRARIES})
endif()

find_library(Unwind_LIBRARY NAMES unwind)
if(Unwind_LIBRARY)
  list(INSERT Glog_LIBRARIES 0 ${Unwind_LIBRARY})
endif()

if(Glog_USE_STATIC_LIBS)
  set(CMAKE_FIND_LIBRARY_SUFFIXES ${_CMAKE_FIND_LIBRARY_SUFFIXES})
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Glog REQUIRED_VARS Glog_LIBRARY Glog_INCLUDE_DIR)
