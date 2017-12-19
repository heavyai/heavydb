#.rst:
# FindLibAwsS3.cmake
# -------------
#
# Find a LibAwsS3 installation.
#
# This module finds if LibAwsS3 is installed and selects a default
# configuration to use.
#
# find_package(LibAwsS3 ...)
#
#
# The following variables control which libraries are found::
#
#   LibAwsS3_USE_STATIC_LIBS  - Set to ON to force use of static libraries.
#
# The following are set after the configuration is done:
#
# ::
#
#   LibAwsS3_FOUND            - Set to TRUE if LibAwsS3 was found.
#   LibAwsS3_INCLUDE_DIRS     - Include directories
#   LibAwsS3_LIBRARIES        - Path to the LibAwsS3 libraries.
#   LibAwsS3_LIBRARY_DIRS     - compile time link directories
#
#
# Sample usage:
#
# ::
#
#    find_package(LibAwsS3)
#    if(LibAwsS3_FOUND)
#      target_link_libraries(<YourTarget> ${LibAwsS3_LIBRARIES})
#    endif()

# kept here just in case mind is changed to use brew 'aws-sdk-cpp' pkg instead of self-build
execute_process(COMMAND brew --prefix aws-sdk-cpp OUTPUT_VARIABLE PREFIX_LIBAWSS3)
string(STRIP "${PREFIX_LIBAWSS3}" PREFIX_LIBAWSS3)

if(LibAwsS3_USE_STATIC_LIBS)
  set(_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
  set(CMAKE_FIND_LIBRARY_SUFFIXES .lib .a ${CMAKE_FIND_LIBRARY_SUFFIXES})
endif()

find_library(libAwsCore_LIBRARY
  NAMES aws-cpp-sdk-core
  HINTS ENV LD_LIBRARY_PATH
  HINTS ENV DYLD_LIBRARY_PATH
  HINTS ${PREFIX_LIBAWSS3}/lib
  PATHS
  /usr/lib
  /usr/local/lib
  /usr/local/homebrew/lib
  /opt/local/lib)

find_library(LibAwsS3_LIBRARY
  NAMES aws-cpp-sdk-s3
  HINTS ENV LD_LIBRARY_PATH
  HINTS ENV DYLD_LIBRARY_PATH
  HINTS ${PREFIX_LIBAWSS3}/lib
  PATHS
  /usr/lib
  /usr/local/lib
  /usr/local/homebrew/lib
  /opt/local/lib)

find_library(LibSsl_LIBRARY
  NAMES ssl
  HINTS ENV LD_LIBRARY_PATH
  HINTS ENV DYLD_LIBRARY_PATH
  PATHS
  /usr/lib
  /usr/local/lib
  /usr/local/homebrew/lib
  /opt/local/lib)

find_library(LibCrypto_LIBRARY
  NAMES crypto
  HINTS ENV LD_LIBRARY_PATH
  HINTS ENV DYLD_LIBRARY_PATH
  PATHS
  /usr/lib
  /usr/local/lib
  /usr/local/homebrew/lib
  /opt/local/lib)

find_library(LibCurl_LIBRARY
  NAMES curl
  HINTS ENV LD_LIBRARY_PATH
  HINTS ENV DYLD_LIBRARY_PATH
  PATHS
  /usr/lib
  /usr/local/lib
  /usr/local/homebrew/lib
  /opt/local/lib)

#message("libAwsCore_LIBRARY = ${libAwsCore_LIBRARY}")
#message("LibAwsS3_LIBRARY = ${LibAwsS3_LIBRARY}")
#message("LibCurl_LIBRARY = ${LibCurl_LIBRARY}")

get_filename_component(LibAwsS3_LIBRARY_DIR ${LibAwsS3_LIBRARY} DIRECTORY)
find_path(LibAwsS3_INCLUDE_DIR
  NAMES aws/core/Aws.h
  HINTS ${LibAwsS3_LIBRARY_DIR}/../include
  HINTS ${PREFIX_LIBAWSS3}/include
  PATHS
  /include
  /usr/include
  /usr/local/include
  /usr/local/homebrew/include
  /usr/local/opt/libarchive/include
  /opt/local/include
  )
#message("LibAwsS3_LIBRARY_DIR= ${LibAwsS3_LIBRARY_DIR}")
#message("LibAwsS3_INCLUDE_DIR= ${LibAwsS3_INCLUDE_DIR}")

# Set standard CMake FindPackage variables if found.
set(LibAwsS3_LIBRARIES ${LibAwsS3_LIBRARIES} ${LibAwsS3_LIBRARY} ${libAwsCore_LIBRARY} ${LibCurl_LIBRARY} ${LibSsl_LIBRARY} ${LibCrypto_LIBRARY})
set(LibAwsS3_INCLUDE_DIRS ${LibAwsS3_INCLUDE_DIR})
set(LibAwsS3_LIBRARY_DIRS ${LibAwsS3_LIBRARY_DIR})

if(LibAwsS3_USE_STATIC_LIBS)
  set(CMAKE_FIND_LIBRARY_SUFFIXES ${_CMAKE_FIND_LIBRARY_SUFFIXES})
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LibAwsS3 REQUIRED_VARS 
  LibAwsS3_INCLUDE_DIR
  LibAwsS3_LIBRARY
  libAwsCore_LIBRARY
  LibSsl_LIBRARY
  LibCrypto_LIBRARY
  LibCurl_LIBRARY
  )
