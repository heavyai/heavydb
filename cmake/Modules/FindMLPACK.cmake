#.rst:
# FindMLPACK
# -------------
#
# Find MLPACK
#
# Find the MLPACK C++ library
#
# Using MLPACK::
#
#   find_package(MLPACK REQUIRED)
#   include_directories(${MLPACK_INCLUDE_DIRS})
#   add_executable(foo foo.cc)
#   target_link_libraries(foo ${MLPACK_LIBRARIES})
#
# This module sets the following variables::
#
#   MLPACK_FOUND - set to true if the library is found
#   MLPACK_INCLUDE_DIRS - list of required include directories
#   MLPACK_LIBRARIES - list of libraries to be linked
#   MLPACK_VERSION_MAJOR - major version number
#   MLPACK_VERSION_MINOR - minor version number
#   MLPACK_VERSION_PATCH - patch version number
#   MLPACK_VERSION_STRING - version number as a string (ex: "1.0.4")

if(MLPACK_USE_STATIC_LIBS)
  set(_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
  set(CMAKE_FIND_LIBRARY_SUFFIXES .lib .a ${CMAKE_FIND_LIBRARY_SUFFIXES})
endif()

# UNIX paths are standard, no need to specify them.
find_library(MLPACK_LIBRARY
  NAMES mlpack
  HINTS
  ENV LD_LIBRARY_PATH
  ENV DYLD_LIBRARY_PATH
  PATHS
  /usr/lib
  /usr/lib/x86_64-linux-gnu
  /usr/local/lib
  /usr/local/homebrew/lib
  /opt/local/lib
)

get_filename_component(MLPACK_LIBRARY_DIR ${MLPACK_LIBRARY} DIRECTORY)

find_path(MLPACK_INCLUDE_DIR
  NAMES mlpack/core.hpp mlpack/prereqs.hpp
  HINTS
  ${MLPACK_LIBRARY_DIR}/../include
  ${MLPACK_LIBRARY_DIR}/../../include
  PATHS
  /include
  /usr/include
  /usr/local/include
  /usr/local/homebrew/include
  /opt/local/include
)

if(MLPACK_USE_STATIC_LIBS)
  set(CMAKE_FIND_LIBRARY_SUFFIXES ${_CMAKE_FIND_LIBRARY_SUFFIXES})
endif()

set(MLPACK_LIBRARIES ${MLPACK_LIBRARY})
set(MLPACK_INCLUDE_DIRS ${MLPACK_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(MLPACK
	REQUIRED_VARS MLPACK_LIBRARY MLPACK_INCLUDE_DIR
)

# Hide internal variables
mark_as_advanced(
  MLPACK_INCLUDE_DIR
  MLPACK_LIBRARY
)