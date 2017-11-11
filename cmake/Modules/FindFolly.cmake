#.rst:
# FindFolly.cmake
# -------------
#
# Find a Folly installation.
#
# This module finds if Folly is installed and selects a default
# configuration to use.
#
# find_package(Folly ...)
#
#
# The following variables control which libraries are found::
#
#   Folly_USE_STATIC_LIBS  - Set to ON to force use of static libraries.
#
# The following are set after the configuration is done:
#
# ::
#
#   Folly_FOUND            - Set to TRUE if Folly was found.
#   Folly_LIBRARIES        - Path to the Folly libraries.
#   Folly_LIBRARY_DIRS     - compile time link directories
#   Folly_INCLUDE_DIRS     - compile time include directories
#
#
# Sample usage:
#
# ::
#
#    find_package(Folly)
#    if(Folly_FOUND)
#      target_link_libraries(<YourTarget> ${Folly_LIBRARIES})
#    endif()

if(Folly_USE_STATIC_LIBS)
  set(_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
  set(CMAKE_FIND_LIBRARY_SUFFIXES .lib .a ${CMAKE_FIND_LIBRARY_SUFFIXES})
endif()


find_library(Folly_LIBRARY
  NAMES folly
  HINTS
  ENV LD_LIBRARY_PATH
  ENV DYLD_LIBRARY_PATH
  PATHS
  /usr/lib
  /usr/local/lib
  /usr/local/homebrew/lib
  /opt/local/lib)

get_filename_component(Folly_LIBRARY_DIR ${Folly_LIBRARY} DIRECTORY)

find_library(Folly_DC_LIBRARY
  NAMES double-conversion
  HINTS
  ENV LD_LIBRARY_PATH
  ENV DYLD_LIBRARY_PATH
  PATHS
  /usr/lib
  /usr/local/lib
  /usr/local/homebrew/lib
  /opt/local/lib)

if(Folly_USE_STATIC_LIBS)
  set(CMAKE_FIND_LIBRARY_SUFFIXES ${_CMAKE_FIND_LIBRARY_SUFFIXES})
endif()

# Set standard CMake FindPackage variables if found.
set(Folly_LIBRARIES ${Folly_LIBRARY} ${Folly_DC_LIBRARY} ${CMAKE_DL_LIBS})
set(Folly_LIBRARY_DIRS ${Folly_LIBRARY_DIR})
set(Folly_INCLUDE_DIRS ${Folly_LIBRARY_DIR}/../include)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Folly REQUIRED_VARS Folly_LIBRARY Folly_DC_LIBRARY)
