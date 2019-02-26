#.rst:
# FindSnappy.cmake
# -------------
#
# Find a Snappy installation.
#
# This module finds if Snappy is installed and selects a default
# configuration to use.
#
# find_package(Snappy ...)
#
#
# The following variables control which libraries are found::
#
#   Snappy_USE_STATIC_LIBS  - Set to ON to force use of static libraries.
#
# The following are set after the configuration is done:
#
# ::
#
#   Snappy_FOUND            - Set to TRUE if Snappy was found.
#   Snappy_LIBRARIES        - Path to the Snappy libraries.
#   Snappy_LIBRARY_DIRS     - compile time link directories
#   Snappy_INCLUDE_DIRS     - compile time include directories
#
#
# Sample usage:
#
# ::
#
#    find_package(Snappy)
#    if(Snappy_FOUND)
#      target_link_libraries(<YourTarget> ${Snappy_LIBRARIES})
#    endif()

if(Snappy_USE_STATIC_LIBS)
  set(_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
  set(CMAKE_FIND_LIBRARY_SUFFIXES .lib .a ${CMAKE_FIND_LIBRARY_SUFFIXES})
endif()


find_library(Snappy_LIBRARY
  NAMES snappy
  HINTS
  ENV LD_LIBRARY_PATH
  ENV DYLD_LIBRARY_PATH
  PATHS
  /usr/lib
  /usr/local/lib
  /usr/local/homebrew/lib
  /opt/local/lib)

if(Snappy_USE_STATIC_LIBS)
  set(CMAKE_FIND_LIBRARY_SUFFIXES ${_CMAKE_FIND_LIBRARY_SUFFIXES})
endif()

get_filename_component(Snappy_LIBRARY_DIR ${Snappy_LIBRARY} DIRECTORY)

# Set standard CMake FindPackage variables if found.
set(Snappy_LIBRARIES ${Snappy_LIBRARY} ${CMAKE_DL_LIBS})
set(Snappy_LIBRARY_DIRS ${Snappy_LIBRARY_DIR})
set(Snappy_INCLUDE_DIRS ${Snappy_LIBRARY_DIR}/../include)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Snappy REQUIRED_VARS Snappy_LIBRARY)
