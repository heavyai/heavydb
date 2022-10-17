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

# Kept here just in case mind is changed to use brew 'aws-sdk-cpp' pkg instead of self-build
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

find_library(libAwsCCommon_LIBRARY
  NAMES aws-c-common
  HINTS ENV LD_LIBRARY_PATH
  HINTS ENV DYLD_LIBRARY_PATH
  HINTS ${PREFIX_LIBAWSS3}/lib
  PATHS
  /usr/lib
  /usr/local/lib
  /usr/local/homebrew/lib
  /opt/local/lib)

find_library(libAwsCEventStream_LIBRARY
  NAMES aws-c-event-stream
  HINTS ENV LD_LIBRARY_PATH
  HINTS ENV DYLD_LIBRARY_PATH
  HINTS ${PREFIX_LIBAWSS3}/lib
  PATHS
  /usr/lib
  /usr/local/lib
  /usr/local/homebrew/lib
  /opt/local/lib)

find_library(libAwsChecksums_LIBRARY
  NAMES aws-checksums
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

find_library(LibAwsSTS_LIBRARY
  NAMES aws-cpp-sdk-sts
  HINTS ENV LD_LIBRARY_PATH
  HINTS ENV DYLD_LIBRARY_PATH
  HINTS ${PREFIX_LIBAWSS3}/lib
  PATHS
  /usr/lib
  /usr/local/lib
  /usr/local/homebrew/lib
  /opt/local/lib)

find_library(LibAwsIM_LIBRARY
  NAMES aws-cpp-sdk-identity-management
  HINTS ENV LD_LIBRARY_PATH
  HINTS ENV DYLD_LIBRARY_PATH
  HINTS ${PREFIX_LIBAWSS3}/lib
  PATHS
  /usr/lib
  /usr/local/lib
  /usr/local/homebrew/lib
  /opt/local/lib)

find_library(LibAwsCI_LIBRARY
  NAMES aws-cpp-sdk-cognito-identity
  HINTS ENV LD_LIBRARY_PATH
  HINTS ENV DYLD_LIBRARY_PATH
  HINTS ${PREFIX_LIBAWSS3}/lib
  PATHS
  /usr/lib
  /usr/local/lib
  /usr/local/homebrew/lib
  /opt/local/lib)
#
# Moving to the new aws-sdk-cpp, the supplied cmake files
# added an -fPIC to the global compiler options, causing
# an issue with our inline macros.
# Keeping our original .cmake files and finding specific files
# get around this. However for the  newer version extra aws
# libraries are required
#
get_filename_component(Aws_LIBRARY_DIR ${LibAwsCI_LIBRARY} DIRECTORY)
include(${Aws_LIBRARY_DIR}/cmake/AWSSDK/AWSSDKConfigVersion.cmake)
message(STATUS "AWSSDK version ${PACKAGE_VERSION}")
#
# Extra  libraries needed for linking version 1.9.335
#

if("${PACKAGE_VERSION}" VERSION_EQUAL "1.9.335")
  find_library(libAwsCrt_LIBRARY
    NAMES aws-crt-cpp
    HINTS ENV LD_LIBRARY_PATH
    HINTS ENV DYLD_LIBRARY_PATH
    HINTS ${PREFIX_LIBAWSS3}/lib
    PATHS
    /usr/lib
    /usr/local/lib
    /usr/local/homebrew/lib
    /opt/local/lib)

  find_library(libAwsCIo_LIBRARY
    NAMES aws-c-io
    HINTS ENV LD_LIBRARY_PATH
    HINTS ENV DYLD_LIBRARY_PATH
    HINTS ${PREFIX_LIBAWSS3}/lib
    PATHS
    /usr/lib
    /usr/local/lib
    /usr/local/homebrew/lib
    /opt/local/lib)

  find_library(libAwsCAuth_LIBRARY
    NAMES aws-c-auth
    HINTS ENV LD_LIBRARY_PATH
    HINTS ENV DYLD_LIBRARY_PATH
    HINTS ${PREFIX_LIBAWSS3}/lib
    PATHS
    /usr/lib
    /usr/local/lib
    /usr/local/homebrew/lib
    /opt/local/lib)

  find_library(libAwsCHttp
    NAMES aws-c-http
    HINTS ENV LD_LIBRARY_PATH
    HINTS ENV DYLD_LIBRARY_PATH
    HINTS ${PREFIX_LIBAWSS3}/lib
    PATHS
    /usr/lib
    /usr/local/lib
    /usr/local/homebrew/lib
    /opt/local/lib)

  find_library(libAwsCSdkUtils
    NAMES aws-c-sdkutils
    HINTS ENV LD_LIBRARY_PATH
    HINTS ENV DYLD_LIBRARY_PATH
    HINTS ${PREFIX_LIBAWSS3}/lib
    PATHS
    /usr/lib
    /usr/local/lib
    /usr/local/homebrew/lib
    /opt/local/lib)

  find_library(libAwsCCal
    NAMES aws-c-cal
    HINTS ENV LD_LIBRARY_PATH
    HINTS ENV DYLD_LIBRARY_PATH
    HINTS ${PREFIX_LIBAWSS3}/lib
    PATHS
    /usr/lib
    /usr/local/lib
    /usr/local/homebrew/lib
    /opt/local/lib)

  find_library(LibS2N_LIBRARY
    NAMES s2n
    HINTS ENV LD_LIBRARY_PATH
    HINTS ENV DYLD_LIBRARY_PATH
    HINTS ${PREFIX_LIBAWSS3}/lib
    PATHS
    /usr/lib
    /usr/local/lib
    /usr/local/homebrew/lib
    /opt/local/lib)

  find_library(libAwsCompression
    NAMES aws-c-compression
    HINTS ENV LD_LIBRARY_PATH
    HINTS ENV DYLD_LIBRARY_PATH
    HINTS ${PREFIX_LIBAWSS3}/lib
    PATHS
    /usr/lib
    /usr/local/lib
    /usr/local/homebrew/lib
    /opt/local/lib)

  find_library(libAwsCMqtt
    NAMES aws-c-mqtt
    HINTS ENV LD_LIBRARY_PATH
    HINTS ENV DYLD_LIBRARY_PATH
    HINTS ${PREFIX_LIBAWSS3}/lib
    PATHS
    /usr/lib
    /usr/local/lib
    /usr/local/homebrew/lib
    /opt/local/lib)

  find_library(libAwsCS3
    NAMES aws-c-s3
    HINTS ENV LD_LIBRARY_PATH
    HINTS ENV DYLD_LIBRARY_PATH
    HINTS ${PREFIX_LIBAWSS3}/lib
    PATHS
    /usr/lib
    /usr/local/lib
    /usr/local/homebrew/lib
    /opt/local/lib)
endif()

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
# Set standard CMake FindPackage variables if found.

set(LibAwsS3_LIBRARIES ${LibAwsS3_LIBRARIES} ${LibAwsS3_LIBRARY} ${LibAwsIM_LIBRARY} ${LibAwsCI_LIBRARY} ${libAwsCore_LIBRARY} ${libAwsCEventStream_LIBRARY} ${libAwsCCommon_LIBRARY} ${libAwsChecksums_LIBRARY} ${LibAwsSTS_LIBRARY} ${libAwsCrt_LIBRARY} ${libAwsCAuth_LIBRARY} ${libAwsCIo_LIBRARY} ${LibS2N_LIBRARY} ${libAwsCSdkUtils} ${libAwsCCal} ${libAwsCHttp} ${libAwsCMqtt} ${libAwsCS3} ${libAwsCompression})
set(LibAwsS3_INCLUDE_DIRS ${LibAwsS3_INCLUDE_DIR})
set(LibAwsS3_LIBRARY_DIRS ${LibAwsS3_LIBRARY_DIR})

if(LibAwsS3_USE_STATIC_LIBS)
  set(CMAKE_FIND_LIBRARY_SUFFIXES ${_CMAKE_FIND_LIBRARY_SUFFIXES})
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LibAwsS3 REQUIRED_VARS
  LibAwsS3_INCLUDE_DIR
  LibAwsS3_LIBRARY
  LibAwsS3_LIBRARIES
  LibAwsSTS_LIBRARY
  LibAwsIM_LIBRARY
  LibAwsCI_LIBRARY
  libAwsCore_LIBRARY
  libAwsCCommon_LIBRARY
  libAwsCEventStream_LIBRARY
  libAwsChecksums_LIBRARY
  )
