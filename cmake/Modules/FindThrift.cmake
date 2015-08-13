#.rst:
# FindThrift.cmake
# -------------
#
# Find a Thrift installation.
#
# This module finds if Thrift is installed and selects a default
# configuration to use.
#
# find_package(Thrift ...)
#
#
# The following variables control which libraries are found::
#
#   Thrift_USE_STATIC_LIBS  - Set to ON to force use of static libraries.
#
# The following are set after the configuration is done:
#
# ::
#
#   Thrift_FOUND            - Set to TRUE if Thrift was found.
#   Thrift_LIBRARIES        - Path to the Thrift libraries.
#   Thrift_LIBRARY_DIRS     - compile time link directories
#
#
# Sample usage:
#
# ::
#
#    find_package(Thrift)
#    if(Thrift_FOUND)
#      target_link_libraries(<YourTarget> ${Thrift_LIBRARIES})
#    endif()
#
# TODO(andrewseidl): Find thrift command and add macro for running thrift --gen.

if(Thrift_USE_STATIC_LIBS)
  set(_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
  set(CMAKE_FIND_LIBRARY_SUFFIXES .lib .a ${CMAKE_FIND_LIBRARY_SUFFIXES})
endif()


find_library(Thrift_LIBRARY
  NAMES thrift
  HINTS
  ENV LD_LIBRARY_PATH
  ENV DYLD_LIBRARY_PATH
  PATHS
  /usr/lib
  /usr/local/lib
  /usr/local/homebrew/lib
  /opt/local/lib)

if(Thrift_USE_STATIC_LIBS)
  set(CMAKE_FIND_LIBRARY_SUFFIXES ${_CMAKE_FIND_LIBRARY_SUFFIXES})
endif()

get_filename_component(Thrift_LIBRARY_DIR ${Thrift_LIBRARY} DIRECTORY)
# Set standard CMake FindPackage variables if found.
set(Thrift_LIBRARIES ${Thrift_LIBRARY})
set(Thrift_LIBRARY_DIRS ${Thrift_LIBRARY_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Thrift REQUIRED_VARS Thrift_LIBRARY)
