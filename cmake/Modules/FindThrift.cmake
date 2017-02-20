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
#   Thrift_EXECUTABLE       - Path to the Thrift executable.
#   Thrift_LIBRARY_DIRS     - compile time link directories
#   Thrift_INCLUDE_DIRS     - compile time include directories
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
# TODO(andrewseidl): Add macro for running thrift --gen.

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

get_filename_component(Thrift_LIBRARY_DIR ${Thrift_LIBRARY} DIRECTORY)

find_program(Thrift_EXECUTABLE
  NAMES thrift
  HINTS
  ENV PATH
  ${Thrift_LIBRARY_DIR}/../bin
  PATHS
  /usr/bin
  /usr/local/bin
  /usr/local/homebrew/bin
  /opt/local/bin)

execute_process(COMMAND ${Thrift_EXECUTABLE} --version
  OUTPUT_VARIABLE Thrift_version_output
  ERROR_VARIABLE Thrift_version_output
  RESULT_VARIABLE Thrift_version_result)
if(SWIG_version_result)
  message(SEND_ERROR "Failed to get Thrift version: ${Thrift_version_output}")
else()
  string(REGEX REPLACE "^[^0-9]+" ""
    Thrift_version_output "${Thrift_version_output}")
  string(STRIP "${Thrift_version_output}" Thrift_version_output)
  set(Thrift_VERSION ${Thrift_version_output} CACHE STRING "Thrift version" FORCE)
endif()

if(Thrift_USE_STATIC_LIBS)
  set(CMAKE_FIND_LIBRARY_SUFFIXES ${_CMAKE_FIND_LIBRARY_SUFFIXES})
endif()

# Set standard CMake FindPackage variables if found.
set(Thrift_LIBRARIES ${Thrift_LIBRARY})
set(Thrift_LIBRARY_DIRS ${Thrift_LIBRARY_DIR})
set(Thrift_INCLUDE_DIRS ${Thrift_LIBRARY_DIR}/../include)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Thrift REQUIRED_VARS Thrift_LIBRARY Thrift_VERSION)
