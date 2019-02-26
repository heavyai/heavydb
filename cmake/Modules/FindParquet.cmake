#.rst:
# FindParquet.cmake
# -------------
#
# Find a Parquet-cpp installation.
#
# This module finds if Parquet is installed and selects a default
# configuration to use.
#
# find_package(Parquet ...)
#
#
# The following variables control which libraries are found::
#
#   Parquet_USE_STATIC_LIBS  - Set to ON to force use of static libraries.
#
# The following are set after the configuration is done:
#
# ::
#
#   Parquet_FOUND            - Set to TRUE if Parquet was found.
#   Parquet_LIBRARIES        - Path to the Parquet libraries.
#   Parquet_LIBRARY_DIRS     - compile time link directories
#   Parquet_INCLUDE_DIRS     - compile time include directories
#
#
# Sample usage:
#
# ::
#
#    find_package(Parquet)
#    if(Parquet_FOUND)
#      target_link_libraries(<YourTarget> ${Parquet_LIBRARIES})
#    endif()

if(Parquet_USE_STATIC_LIBS)
  set(_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
  set(CMAKE_FIND_LIBRARY_SUFFIXES .lib .a ${CMAKE_FIND_LIBRARY_SUFFIXES})
endif()


find_library(Parquet_LIBRARY
  NAMES parquet
  HINTS
  ENV LD_LIBRARY_PATH
  ENV DYLD_LIBRARY_PATH
  PATHS
  /usr/lib
  /usr/local/lib
  /usr/local/homebrew/lib
  /opt/local/lib)

if(Parquet_USE_STATIC_LIBS)
  set(CMAKE_FIND_LIBRARY_SUFFIXES ${_CMAKE_FIND_LIBRARY_SUFFIXES})
endif()

get_filename_component(Parquet_LIBRARY_DIR ${Parquet_LIBRARY} DIRECTORY)

# Set standard CMake FindPackage variables if found.
set(Parquet_LIBRARIES ${Parquet_LIBRARY} ${CMAKE_DL_LIBS})
set(Parquet_LIBRARY_DIRS ${Parquet_LIBRARY_DIR})
set(Parquet_INCLUDE_DIRS ${Parquet_LIBRARY_DIR}/../include)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Parquet REQUIRED_VARS Parquet_LIBRARY)
