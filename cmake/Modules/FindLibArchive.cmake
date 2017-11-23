#.rst:
# FindLibArchive.cmake
# -------------
#
# Find a LibArchive installation.
#
# This module finds if LibArchive is installed and selects a default
# configuration to use.
#
# find_package(LibArchive ...)
#
#
# The following variables control which libraries are found::
#
#   LibArchive_USE_STATIC_LIBS  - Set to ON to force use of static libraries.
#
# The following are set after the configuration is done:
#
# ::
#
#   LibArchive_FOUND            - Set to TRUE if LibArchive was found.
#   LibArchive_INCLUDE_DIRS     - Include directories
#   LibArchive_LIBRARIES        - Path to the LibArchive libraries.
#   LibArchive_LIBRARY_DIRS     - compile time link directories
#
#
# Sample usage:
#
# ::
#
#    find_package(LibArchive)
#    if(LibArchive_FOUND)
#      target_link_libraries(<YourTarget> ${LibArchive_LIBRARIES})
#    endif()


execute_process(COMMAND brew --prefix libarchive OUTPUT_VARIABLE PREFIX_LIBARCHIVE)
string(STRIP "${PREFIX_LIBARCHIVE}" PREFIX_LIBARCHIVE)

if(LibArchive_USE_STATIC_LIBS)
  set(_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
  set(CMAKE_FIND_LIBRARY_SUFFIXES .lib .a ${CMAKE_FIND_LIBRARY_SUFFIXES})
endif()

find_library(BZ2_LIBRARY
  NAMES bz2
  HINTS ENV LD_LIBRARY_PATH
  HINTS ENV DYLD_LIBRARY_PATH
  PATHS
  /usr/lib
  /usr/local/lib
  /usr/local/homebrew/lib
  /opt/local/lib)

find_library(LibArchive_LIBRARY
  NAMES archive
  HINTS ENV LD_LIBRARY_PATH
  HINTS ENV DYLD_LIBRARY_PATH
  HINTS /usr/local/opt/libarchive/lib
  HINTS ${PREFIX_LIBARCHIVE}/lib
  PATHS
  /usr/lib
  /usr/local/lib
  /usr/local/homebrew/lib
  /opt/local/lib)

find_library(LZMA_LIBRARY
  NAMES lzma
  HINTS ENV LD_LIBRARY_PATH
  HINTS ENV DYLD_LIBRARY_PATH
  PATHS
  /usr/lib
  /usr/local/lib
  /usr/local/homebrew/lib
  /opt/local/lib)

find_path(LibArchive_INCLUDE_DIR
  NAMES archive.h
  HINTS ${LibArchive_LIBRARY}/../include
  HINTS ${PREFIX_LIBARCHIVE}/include
  PATHS
  /usr/include
  /usr/local/include
  /usr/local/homebrew/include
  /usr/local/opt/libarchive/include
  /opt/local/include)

get_filename_component(LibArchive_LIBRARY_DIR ${LibArchive_LIBRARY} DIRECTORY)
# Set standard CMake FindPackage variables if found.
set(LibArchive_LIBRARIES ${LibArchive_LIBRARY} ${BZ2_LIBRARY} ${LZMA_LIBRARY})
set(LibArchive_INCLUDE_DIRS ${LibArchive_INCLUDE_DIR})
set(LibArchive_LIBRARY_DIRS ${LibArchive_LIBRARY_DIR})

if(LibArchive_USE_STATIC_LIBS)
  set(CMAKE_FIND_LIBRARY_SUFFIXES ${_CMAKE_FIND_LIBRARY_SUFFIXES})
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LibArchive REQUIRED_VARS LibArchive_LIBRARY LibArchive_INCLUDE_DIR BZ2_LIBRARY LZMA_LIBRARY)
