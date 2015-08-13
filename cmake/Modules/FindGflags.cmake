#.rst:
# FindGflags.cmake
# -------------
#
# Find a Google gflags installation.
#
# This module finds if Google gflags is installed and selects a default
# configuration to use.
#
# find_package(Gflags ...)
#
#
# The following variables control which libraries are found::
#
#   Gflags_USE_STATIC_LIBS  - Set to ON to force use of static libraries.
#
# The following are set after the configuration is done:
#
# ::
#
#   Gflags_FOUND            - Set to TRUE if Gflags was found.
#   Gflags_INCLUDE_DIRS     - Include directories
#   Gflags_LIBRARIES        - Path to the Gflags libraries.
#   Gflags_LIBRARY_DIRS     - compile time link directories
#
#
# Sample usage:
#
# ::
#
#    find_package(Gflags)
#    if(Gflags_FOUND)
#      target_link_libraries(<YourTarget> ${Gflags_LIBRARIES})
#    endif()

if(Gflags_USE_STATIC_LIBS)
  set(_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
  set(CMAKE_FIND_LIBRARY_SUFFIXES .lib .a ${CMAKE_FIND_LIBRARY_SUFFIXES})
endif()

find_library(Gflags_LIBRARY
  NAMES gflags
  HINTS ENV LD_LIBRARY_PATH
  HINTS ENV DYLD_LIBRARY_PATH
  PATHS
  /usr/lib
  /usr/local/lib
  /usr/local/homebrew/lib
  /opt/local/lib)

find_path(Gflags_INCLUDE_DIR
  NAMES gflags/gflags.h
  HINTS ${Gflags_LIBRARY}/../include
  PATHS
  /usr/include
  /usr/local/include
  /usr/local/homebrew/include
  /opt/local/include)

if(Gflags_USE_STATIC_LIBS)
  set(CMAKE_FIND_LIBRARY_SUFFIXES ${_CMAKE_FIND_LIBRARY_SUFFIXES})
endif()

get_filename_component(Gflags_LIBRARY_DIR ${Gflags_LIBRARY} DIRECTORY)
# Set standard CMake FindPackage variables if found.
set(Gflags_LIBRARIES ${Gflags_LIBRARY})
set(Gflags_INCLUDE_DIRS ${Gflags_INCLUDE_DIR})
set(Gflags_LIBRARY_DIRS ${Gflags_LIBRARY_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Gflags REQUIRED_VARS Gflags_LIBRARY)
