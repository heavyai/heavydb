#.rst:
# FindCryptopp.cmake
# -------------
#
# Find a Cryptopp installation.
#
# This module finds if Cryptopp is installed and selects a default
# configuration to use.
#
# find_package(Cryptopp ...)
#
#
# The following variables control which libraries are found::
#
#   Cryptopp_USE_STATIC_LIBS  - Set to ON to force use of static libraries.
#
# The following are set after the configuration is done:
#
# ::
#
#   Cryptopp_FOUND            - Set to TRUE if Cryptopp was found.
#   Cryptopp_INCLUDE_DIRS     - Include directories
#   Cryptopp_LIBRARIES        - Path to the Cryptopp libraries.
#   Cryptopp_LIBRARY_DIRS     - compile time link directories
#
#
# Sample usage:
#
# ::
#
#    find_package(Cryptopp)
#    if(Cryptopp_FOUND)
#      target_link_libraries(<YourTarget> ${Cryptopp_LIBRARIES})
#    endif()

if(Cryptopp_USE_STATIC_LIBS)
  set(_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
  set(CMAKE_FIND_LIBRARY_SUFFIXES .lib .a ${CMAKE_FIND_LIBRARY_SUFFIXES})
endif()

find_library(Cryptopp_LIBRARY
  NAMES cryptopp
  HINTS ENV LD_LIBRARY_PATH
  HINTS ENV DYLD_LIBRARY_PATH
  PATHS
  /usr/lib
  /usr/local/lib
  /usr/local/homebrew/lib
  /opt/local/lib)

find_path(Cryptopp_INCLUDE_DIR
  NAMES cryptopp/cryptlib.h
  HINTS ${Cryptopp_LIBRARY}/../../include
  PATHS
  /usr/include
  /usr/local/include
  /usr/local/homebrew/include
  /opt/local/include)

get_filename_component(Cryptopp_LIBRARY_DIR ${Cryptopp_LIBRARY} DIRECTORY)
# Set standard CMake FindPackage variables if found.
set(Cryptopp_LIBRARIES ${Cryptopp_LIBRARY})
set(Cryptopp_INCLUDE_DIRS ${Cryptopp_INCLUDE_DIR})
set(Cryptopp_LIBRARY_DIRS ${Cryptopp_LIBRARY_DIR})

if(Cryptopp_USE_STATIC_LIBS)
  set(CMAKE_FIND_LIBRARY_SUFFIXES ${_CMAKE_FIND_LIBRARY_SUFFIXES})
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Cryptopp REQUIRED_VARS Cryptopp_LIBRARY Cryptopp_INCLUDE_DIR)
