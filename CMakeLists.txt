# fixme: this should be 3.3, but that breaks static ncurses
cmake_minimum_required(VERSION 3.2)

# force `Release` build type if left unspecified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

if("${CMAKE_VERSION}" VERSION_GREATER 3.11.999)
  cmake_policy(SET CMP0074 NEW)
endif()

find_program(CCACHE_EXE ccache)
if(CCACHE_EXE)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_EXE}")
endif()

option(ENABLE_IWYU "Enable include-what-you-use" OFF)
if(ENABLE_IWYU)
  find_program(IWYU_EXE include-what-you-use)
  if(IWYU_EXE)
    set(CMAKE_CXX_INCLUDE_WHAT_YOU_USE "${IWYU_EXE}")
  endif()
endif()

project(omnisci)

if(NOT EXISTS "${CMAKE_SOURCE_DIR}/Rendering")
  set(MAPD_EDITION "OS")
elseif(NOT DEFINED MAPD_EDITION)
  set(MAPD_EDITION "EE")
endif()
set(MAPD_EDITION "${MAPD_EDITION}" CACHE STRING "MapD edition" FORCE)
set_property(CACHE MAPD_EDITION PROPERTY STRINGS "EE" "CE" "OS")
add_definitions("-DMAPD_EDITION_${MAPD_EDITION}")
string(TOLOWER "${MAPD_EDITION}" MAPD_EDITION_LOWER)

set(MAPD_VERSION_MAJOR "4")
set(MAPD_VERSION_MINOR "5")
set(MAPD_VERSION_PATCH "0")
set(MAPD_VERSION_EXTRA "")
set(MAPD_VERSION_RAW "${MAPD_VERSION_MAJOR}.${MAPD_VERSION_MINOR}.${MAPD_VERSION_PATCH}${MAPD_VERSION_EXTRA}")
set(MAPD_IMMERSE_URL "http://builds.mapd.com/frontend/mapd2-dashboard-v2-137-release-prod.zip")
set(MAPD_DOCS_URL "http://docs.builds.mapd.com/v4.5.0/docs-v4.5.0.zip")
string(TIMESTAMP MAPD_BUILD_DATE "%Y%m%d")

if($ENV{BUILD_NUMBER})
  set(MAPD_BUILD_NUMBER "$ENV{BUILD_NUMBER}")
else()
  set(MAPD_BUILD_NUMBER "dev")
endif()
set(MAPD_VERSION "${MAPD_VERSION_RAW}-${MAPD_BUILD_NUMBER}")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")

option(PREFER_STATIC_LIBS "Prefer linking against static libraries" OFF)
if(PREFER_STATIC_LIBS)
  set(CMAKE_FIND_LIBRARY_SUFFIXES .lib .a ${CMAKE_FIND_LIBRARY_SUFFIXES})
  set(Boost_USE_STATIC_LIBS ON)
  set(OPENSSL_USE_STATIC_LIBS ON)
  set(Thrift_USE_STATIC_LIBS ON)
  if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libgcc -static-libstdc++")
  endif()

  set(CUDA_USE_STATIC_CUDA_RUNTIME ON)
endif()

option(ENABLE_JAVA_REMOTE_DEBUG "Enable Java Remote Debug" OFF )
if(ENABLE_JAVA_REMOTE_DEBUG)
  add_definitions("-DENABLE_JAVA_REMOTE_DEBUG")
endif()

option(ENABLE_VARLEN_UPDATE "Enable varlen updates" OFF )
if( ENABLE_VARLEN_UPDATE )
  add_definitions("-DENABLE_VARLEN_UPDATE")
endif()

option(ENABLE_CALCITE_UPDATE_PATH "Enable Calcite Update Path" ON)
if( ENABLE_CALCITE_UPDATE_PATH )
  add_definitions("-DCALCITE_UPDATE_ENABLED")
endif()

option(ENABLE_CALCITE_DELETE_PATH "Enable Calcite Delete Path" ON)
if( ENABLE_CALCITE_DELETE_PATH )
  add_definitions("-DCALCITE_DELETE_ENABLED")
endif()

option(ENABLE_CUDA "Enable CUDA support" ON)
if(ENABLE_CUDA)
  find_package(CUDA REQUIRED)
  include_directories(${CUDA_INCLUDE_DIRS})
  list(APPEND CUDA_LIBRARIES ${CUDA_CUDA_LIBRARY})
  add_definitions("-DHAVE_CUDA")

  set(MAPD_HOST_COMPILER "" CACHE STRING "Host compiler to use with nvcc")
  if(NOT "${MAPD_HOST_COMPILER}" STREQUAL "")
      set(MAPD_HOST_COMPILER_FLAG "-ccbin=${MAPD_HOST_COMPILER}")
  endif()
else()
  set(CUDA_LIBRARIES "")
  set(MAPD_PACKAGE_FLAGS "${MAPD_PACKAGE_FLAGS}-cpu")
endif()

option(ENABLE_CUDA_KERNEL_DEBUG "Enable debugging symbols for CUDA device Kernels" OFF)

option(ENABLE_JIT_DEBUG "Enable debugging symbols for the JIT" OFF)
if (ENABLE_JIT_DEBUG)
  add_definitions("-DWITH_JIT_DEBUG")
endif()

if(XCODE)
  if(ENABLE_CUDA)
    set(CMAKE_EXE_LINKER_FLAGS "-F/Library/Frameworks -framework CUDA")
  endif()
  add_definitions("-DXCODE")
endif()

# fixme: hack works for Homebrew, might not work for Conda
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(OPENSSL_ROOT_DIR "/usr/local/opt/openssl/")
endif()
find_package(OpenSSL REQUIRED)
include_directories(${OPENSSL_INCLUDE_DIR})

find_package(Thrift REQUIRED)
include_directories(${Thrift_INCLUDE_DIRS})
if("${Thrift_VERSION}" VERSION_LESS "0.11.0")
  add_definitions("-DHAVE_THRIFT_BOOST_SHAREDPTR")
else()
  add_definitions("-DHAVE_THRIFT_STD_SHAREDPTR")
endif()

find_package(Git)
find_package(Glog REQUIRED)
find_package(PNG REQUIRED)
find_package(ZLIB REQUIRED)
find_package(GDAL REQUIRED)
find_package(GDALExtra REQUIRED)
list(APPEND GDAL_LIBRARIES ${PNG_LIBRARIES} ${GDALExtra_LIBRARIES})
include_directories(${GDAL_INCLUDE_DIRS})

option(ENABLE_FOLLY "Use Folly" ON)
if(ENABLE_FOLLY)
  find_package(Folly REQUIRED)
  include_directories(${Folly_INCLUDE_DIRS})
  add_definitions("-DHAVE_FOLLY")
endif()

find_package(Curses REQUIRED)
include_directories(${CURSES_INCLUDE_DIRS})
if (CURSES_HAVE_NCURSES_CURSES_H AND NOT CURSES_HAVE_CURSES_H)
  include_directories(${CURSES_INCLUDE_DIRS}/ncurses/)
endif()

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux" AND (("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang"
    AND (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 3.6 OR CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 3.6))
    OR ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 4.9)))
  add_definitions("-DHAS_SHARED_MUTEX")
  find_package( Boost COMPONENTS filesystem program_options regex system REQUIRED QUIET )
else()
  find_package( Boost COMPONENTS filesystem program_options regex thread system REQUIRED QUIET )
  if(PREFER_STATIC_LIBS AND ${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    list(APPEND Boost_LIBRARIES rt)
    list(INSERT Boost_LIBRARIES 0 libboost-thread)
    list(INSERT Boost_LIBRARIES 0 pthread)
  endif()
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-unused-local-typedefs -D__GCC_HAVE_SYNC_COMPARE_AND_SWAP_1=1 -D__GCC_HAVE_SYNC_COMPARE_AND_SWAP_2=1 -D__GCC_HAVE_SYNC_COMPARE_AND_SWAP_4=1 -D__GCC_HAVE_SYNC_COMPARE_AND_SWAP_8=1 -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS")
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-register")
endif()
if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
endif()

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-deprecated-declarations")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
endif()

option(ENABLE_LLVM_DBG "switch to local debug version of llvm" OFF)
if(ENABLE_LLVM_DBG)
  # NOTE: specify LLVM_BIN_DIR to use your local llvm
  if(XCODE)
    set(LLVM_BIN_DIR ${CMAKE_SOURCE_DIR}/../llvm/build/xcode/Release/bin)
  elseif(APPLE)
    set(LLVM_BIN_DIR ${CMAKE_SOURCE_DIR}/../llvm/build/unix/bin)
  endif()
  set(llvm_config_cmd "${LLVM_BIN_DIR}/llvm-config")
else()
  set(llvm_config_cmd llvm-config)
endif()

# address and thread sanitizer
option(ENABLE_STANDALONE_CALCITE "Require standalone Calcite server" OFF)
option(ENABLE_ASAN "Enable address sanitizer" OFF)
option(ENABLE_TSAN "Enable thread sanitizer" OFF)
if(ENABLE_ASAN)
  set(SAN_FLAGS "-fsanitize=address -O1 -fno-omit-frame-pointer")
  add_definitions("-DWITH_DECODERS_BOUNDS_CHECKING")
elseif(ENABLE_TSAN)
  # Copy the config directory to the build dir for TSAN suppressions
  file(COPY config DESTINATION ${CMAKE_BINARY_DIR})

  set(SAN_FLAGS "-fsanitize=thread -fPIC -O1 -fno-omit-frame-pointer")
  # required for older GCC, see https://gcc.gnu.org/bugzilla/show_bug.cgi?id=64354
  add_definitions("-D__SANITIZE_THREAD__")
endif()
if(ENABLE_ASAN OR ENABLE_TSAN)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SAN_FLAGS}")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${SAN_FLAGS}")
  set(ENABLE_STANDALONE_CALCITE ON)
endif()

# Code coverage
option(ENABLE_CODE_COVERAGE "Enable compile time code coverage" OFF)
if(ENABLE_CODE_COVERAGE)
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(COVERAGE_FLAGS "-fprofile-instr-generate -fcoverage-mapping")
  else()
    message(FATAL_ERROR "Code coverage currently only supported with Clang compiler")
  endif()
  set(CMAKE_CXX_OUTPUT_EXTENSION_REPLACE ON)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COVERAGE_FLAGS}")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${COVERAGE_FLAGS}")
endif()

option(ENABLE_DECODERS_BOUNDS_CHECKING "Enable bounds checking for column decoding" OFF)

if(ENABLE_STANDALONE_CALCITE)
  add_definitions("-DSTANDALONE_CALCITE")
endif()

set(llvm_config_inc_arg "--includedir")
set(llvm_config_ld_arg "--ldflags")
set(llvm_config_lib_arg "--libs")
execute_process(COMMAND ${llvm_config_cmd} ${llvm_config_inc_arg}
    OUTPUT_VARIABLE LLVM_INC_FLAGS)
execute_process(COMMAND ${llvm_config_cmd} ${llvm_config_lib_arg}
    OUTPUT_VARIABLE LLVM_LIB_FLAGS)
execute_process(COMMAND ${llvm_config_cmd} ${llvm_config_ld_arg}
    OUTPUT_VARIABLE LLVM_LD_FLAGS)
if(ENABLE_LLVM_DBG)
  set(llvm_config_obj_arg "--obj-root")
  execute_process(COMMAND ${llvm_config_cmd} ${llvm_config_obj_arg}
    OUTPUT_VARIABLE LLVM_BUILD_DIR)
  string(REPLACE "\n" "" LLVM_BUILD_DIR "${LLVM_BUILD_DIR}")
  list(APPEND LLVM_INC_FLAGS "${LLVM_BUILD_DIR}/include")
endif()

string(REPLACE "\n" " " LLVM_LINKER_FLAGS "${LLVM_LIB_FLAGS} ${LLVM_LD_FLAGS}")
string(STRIP "${LLVM_LINKER_FLAGS}" LLVM_LINKER_FLAGS)
include_directories(${Boost_INCLUDE_DIR}
                    "/usr/local/include"
                    ${CMAKE_SOURCE_DIR}
                    ${CMAKE_SOURCE_DIR}/Parser
                    ${CMAKE_CURRENT_BINARY_DIR}
                    ${LLVM_INC_FLAGS}
                    "/usr/local/cuda/include/")

# EGL
include_directories(ThirdParty/egl)

# Google Test and Google Mock
if(NOT ${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
  add_definitions("-DGTEST_USE_OWN_TR1_TUPLE=0")
endif()
include_directories(ThirdParty/googletest)
add_subdirectory(ThirdParty/googletest)

# Arrow
find_package(Arrow REQUIRED)
add_definitions("-DARROW_NO_DEPRECATED_API")
if(HAVE_ARROW_STATIC_RECORDBATCH_CTOR)
  add_definitions("-DHAVE_ARROW_STATIC_RECORDBATCH_CTOR")
endif()
if(HAVE_ARROW_APPENDVALUES)
  add_definitions("-DHAVE_ARROW_APPENDVALUES")
else()
  message(WARNING "It appears that you have Arrow < 0.10.0. Please upgrade.")
endif()
include_directories(${Arrow_INCLUDE_DIRS})

# RapidJSON
include_directories(ThirdParty/rapidjson)

# Linenoise
include_directories(ThirdParty/linenoise)
add_subdirectory(ThirdParty/linenoise)

# SQLite
include_directories(ThirdParty/sqlite3)
add_subdirectory(ThirdParty/sqlite3)

#rdkafka
include_directories(ThirdParty/librdkafka/src-cpp)
add_subdirectory(ThirdParty/librdkafka)

# aws-sdk
option(ENABLE_AWS_S3 "Enable AWS S3 support" ON)
if(ENABLE_AWS_S3)
  find_package(LibAwsS3)
  if(NOT LibAwsS3_FOUND)
    set(ENABLE_AWS_S3 OFF CACHE BOOL "Enable AWS S3 support" FORCE)
    message(STATUS "AWS SDK not found. Disabling AWS S3 support.")
  else()
    add_definitions("-DHAVE_AWS_S3")
  endif()
endif()

# bcrypt
include_directories(ThirdParty/bcrypt)
add_subdirectory(ThirdParty/bcrypt)

# PicoSHA2
include_directories(ThirdParty/PicoSHA2)

if("${MAPD_EDITION_LOWER}" STREQUAL "ee")
# opensaml
  option(ENABLE_SAML "Enable SAML support" ON)
  if(ENABLE_SAML)
    find_package(OpenSaml)
    if(NOT OpenSaml_FOUND)
      set(ENABLE_SAML OFF CACHE BOOL "Enable SAML support" FORCE)
    else()
      add_definitions("-DHAVE_SAML")
    endif()
  endif()
endif()

add_custom_command(
    DEPENDS ${CMAKE_SOURCE_DIR}/mapd.thrift
    OUTPUT
        ${CMAKE_BINARY_DIR}/gen-cpp/MapD.cpp
        ${CMAKE_BINARY_DIR}/gen-cpp/MapD.h
        ${CMAKE_BINARY_DIR}/gen-cpp/mapd_constants.cpp
        ${CMAKE_BINARY_DIR}/gen-cpp/mapd_types.cpp
        ${CMAKE_BINARY_DIR}/gen-cpp/completion_hints_constants.cpp
        ${CMAKE_BINARY_DIR}/gen-cpp/completion_hints_types.cpp
    COMMAND ${Thrift_EXECUTABLE}
    ARGS -gen cpp -r -o ${CMAKE_BINARY_DIR} ${CMAKE_SOURCE_DIR}/mapd.thrift)
list(APPEND ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_BINARY_DIR}/gen-cpp/)

add_library(mapd_thrift
    ${CMAKE_BINARY_DIR}/gen-cpp/MapD.cpp
    ${CMAKE_BINARY_DIR}/gen-cpp/MapD.h
    ${CMAKE_BINARY_DIR}/gen-cpp/mapd_constants.cpp
    ${CMAKE_BINARY_DIR}/gen-cpp/mapd_types.cpp
    ${CMAKE_BINARY_DIR}/gen-cpp/completion_hints_constants.cpp
    ${CMAKE_BINARY_DIR}/gen-cpp/completion_hints_types.cpp)
target_compile_options(mapd_thrift PRIVATE -fPIC)
target_link_libraries(mapd_thrift ${Thrift_LIBRARIES})

if("${MAPD_EDITION_LOWER}" STREQUAL "ee")
  include_directories(Catalog/ee)
  include_directories(Distributed/ee)
else()
  include_directories(Catalog/os)
  include_directories(Distributed/os)
endif()

set(MAPD_RENDERING_LIBRARIES "")
if(NOT "${MAPD_EDITION_LOWER}" STREQUAL "os")
  option(ENABLE_RENDERING "Build backend renderer" OFF)
  if(ENABLE_RENDERING)
    # muparserx - currently render-only
    include_directories(ThirdParty/muparserx)
    add_subdirectory(ThirdParty/muparserx)

    add_definitions("-DHAVE_RENDERING")
    add_subdirectory(Rendering)
    add_subdirectory(QueryRenderer)

    include_directories(${RENDERING_INCLUDE_DIRS})
    set(MAPD_RENDERING_LIBRARIES QueryRenderer Rendering)

    set(MAPD_PACKAGE_FLAGS "${MAPD_PACKAGE_FLAGS}-render")
    if(${RENDERER_CONTEXT_TYPE} STREQUAL "GLX")
      set(MAPD_PACKAGE_FLAGS "${MAPD_PACKAGE_FLAGS}-glx")
    endif()
  endif()
else()
  set(ENABLE_RENDERING OFF CACHE BOOL "Build backend renderer" FORCE)
endif()

option(DISABLE_HIGH_PRECISION_TIMESTAMP "Disable parser support for creating milli, micro, and nanosecond timestamp columns" ON)
if(DISABLE_HIGH_PRECISION_TIMESTAMP)
  add_definitions("-DDISABLE_HIGH_PRECISION_TIMESTAMP")
endif()

option(ENABLE_PROFILER "Enable google perftools" OFF)
if(ENABLE_PROFILER)
  find_package(Gperftools REQUIRED COMPONENTS TCMALLOC PROFILER)
  set(PROFILER_LIBS ${Gperftools_TCMALLOC} ${Gperftools_PROFILER})
  add_definitions("-DHAVE_PROFILER")
else()
  set(PROFILER_LIBS "")
endif()

add_subdirectory(SqliteConnector)

add_subdirectory(StringDictionary)
add_subdirectory(Calcite)
get_target_property(CalciteThrift_BINARY_DIR calciteserver_thrift BINARY_DIR)
include_directories(${CalciteThrift_BINARY_DIR})

add_subdirectory(Distributed)

add_subdirectory(Catalog)
add_subdirectory(Parser)
add_subdirectory(Analyzer)
add_subdirectory(Planner)
add_subdirectory(Import)
add_subdirectory(QueryEngine)
add_subdirectory(DataMgr)
add_subdirectory(CudaMgr)
add_subdirectory(Fragmenter)
add_subdirectory(Chunk)
add_subdirectory(Shared)
add_subdirectory(Utils)
add_subdirectory(QueryRunner)
add_subdirectory(SQLFrontend)
add_subdirectory(ThriftHandler)

option(ENABLE_ODBC "Build ODBC driver" OFF)
if(ENABLE_ODBC)
  add_subdirectory(ODBC)
endif()

set(MAPD_LIBRARIES Shared Catalog SqliteConnector Parser Analyzer Planner CsvImport QueryRunner QueryEngine DataMgr Fragmenter Chunk)

if("${MAPD_EDITION_LOWER}" STREQUAL "ee")
  list(APPEND MAPD_LIBRARIES Distributed)
endif()

list(APPEND MAPD_LIBRARIES Calcite)

if(ENABLE_RENDERING)
  list(APPEND MAPD_LIBRARIES "${MAPD_RENDERING_LIBRARIES}")
endif()

list(APPEND MAPD_LIBRARIES ${Arrow_LIBRARIES})

if(ENABLE_FOLLY)
  list(APPEND MAPD_LIBRARIES ${Folly_LIBRARIES})
endif()

if(ENABLE_LICENSING_AWS)
  list(APPEND MAPD_LIBRARIES AWSMarketplace)
endif()

option(ENABLE_TESTS "Build unit tests" ON)
if (ENABLE_TESTS)
  enable_testing()
  add_subdirectory(Tests)
  add_subdirectory(SampleCode)
endif()

set(initdb_source_files initdb.cpp)
add_executable(initdb ${initdb_source_files})

set(omnisci_server_source_files MapDServer.cpp ${CMAKE_BINARY_DIR}/MapDRelease.h ${CMAKE_SOURCE_DIR}/Shared/ConfigResolve.h)

add_executable(omnisci_server ${omnisci_server_source_files})
set_target_properties(omnisci_server PROPERTIES COMPILE_DEFINITIONS "${TIME_LIMITED_DEFINITIONS}")

add_custom_command(
    DEPENDS ${CMAKE_SOURCE_DIR}/mapd.thrift
    OUTPUT
        ${CMAKE_SOURCE_DIR}/java/thrift/src/gen/com/mapd/thrift/server/MapD.java
        ${CMAKE_SOURCE_DIR}/java/thrift/src/gen/com/mapd/thrift/server/TRow.java
    COMMAND mkdir
    ARGS -p ${CMAKE_SOURCE_DIR}/java/thrift/src/gen
    COMMAND ${Thrift_EXECUTABLE}
    ARGS -gen java -r -out ${CMAKE_SOURCE_DIR}/java/thrift/src/gen/ ${CMAKE_SOURCE_DIR}/mapd.thrift)

add_custom_command(
    DEPENDS ${CMAKE_SOURCE_DIR}/java/thrift/calciteserver.thrift ${CMAKE_SOURCE_DIR}/completion_hints.thrift
    OUTPUT ${CMAKE_SOURCE_DIR}/java/thrift/src/gen/com/mapd/thrift/calciteserver/CalciteServer.java
    COMMAND mkdir
    ARGS -p ${CMAKE_SOURCE_DIR}/java/thrift/src/gen
    COMMAND ${Thrift_EXECUTABLE}
    ARGS -gen java -r -I ${CMAKE_SOURCE_DIR} -out ${CMAKE_SOURCE_DIR}/java/thrift/src/gen/ ${CMAKE_SOURCE_DIR}/java/thrift/calciteserver.thrift)
list(APPEND ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_SOURCE_DIR}/java/thrift/src/gen/)

execute_process(
  COMMAND ${GIT_EXECUTABLE} rev-parse --short=10 HEAD
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE MAPD_GIT_HASH
  OUTPUT_STRIP_TRAILING_WHITESPACE
  )
file(WRITE ${CMAKE_BINARY_DIR}/MAPD_GIT_HASH.txt "${MAPD_GIT_HASH}\n")
file(STRINGS ${CMAKE_BINARY_DIR}/MAPD_GIT_HASH.txt MAPD_GIT_HASH)
set(CPACK_PACKAGE_VERSION "${MAPD_VERSION_RAW}-${MAPD_BUILD_DATE}-${MAPD_GIT_HASH}")

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/Shared/release.h"
  "${CMAKE_BINARY_DIR}/MapDRelease.h"
  @ONLY
  )
list(APPEND ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_BINARY_DIR}/MAPD_GIT_HASH.txt)
list(APPEND ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_BINARY_DIR}/MapDRelease.h)
# required to force regen of MAPD_GIT_HASH.txt, MapDRelease.h
add_custom_target(rerun_cmake ALL
  COMMAND cmake .
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )
add_dependencies(omnisci_server rerun_cmake)

target_link_libraries(omnisci_server mapd_thrift thrift_handler ${MAPD_LIBRARIES} ${Boost_LIBRARIES} ${Glog_LIBRARIES} ${CMAKE_DL_LIBS} ${CUDA_LIBRARIES} ${LLVM_LINKER_FLAGS} ${PROFILER_LIBS} ${CURSES_LIBRARIES} ${ZLIB_LIBRARIES})

target_link_libraries(initdb mapd_thrift DataMgr ${MAPD_LIBRARIES} ${Boost_LIBRARIES} ${Glog_LIBRARIES} ${CMAKE_DL_LIBS} ${CUDA_LIBRARIES} ${LLVM_LINKER_FLAGS} ${CURSES_LIBRARIES} ${ZLIB_LIBRARIES})

macro(set_dpkg_arch arch_in arch_out)
  if("${arch_in}" STREQUAL "x86_64")
    set(${arch_out} "amd64")
  elseif("${arch_in}" STREQUAL "aarch64")
    set(${arch_out} "arm64")
  elseif("${arch_in}" STREQUAL "ppc64le")
    set(${arch_out} "ppc64el")
  else()
    set(${arch_out} "${arch_in}")
  endif()
endmacro()

# clang-tidy
find_program(JQ_EXECUTABLE NAMES jq)
if (NOT ${JQ_EXECUTABLE} STREQUAL "JQ_EXECUTABLE-NOTFOUND")
  file(WRITE ${CMAKE_BINARY_DIR}/jq.filter "map(select(.file | test(\".*/(build|ThirdParty)/.*\") | not))")
  add_custom_target(clang-tidy
    COMMAND mkdir -p clang-tidy
    COMMAND ${JQ_EXECUTABLE} -f jq.filter ${CMAKE_BINARY_DIR}/compile_commands.json > clang-tidy/compile_commands.json
    COMMAND cd clang-tidy && ${CMAKE_SOURCE_DIR}/ThirdParty/clang/run-clang-tidy.py -quiet -format -fix 2> /dev/null
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )
else()
  message(STATUS "jq not found, disabling clang-tidy target")
endif()

# doxygen
find_package(Doxygen)
if(DOXYGEN_FOUND)
  configure_file(
    ${CMAKE_SOURCE_DIR}/Doxyfile.in ${CMAKE_BINARY_DIR}/Doxyfile @ONLY)
  add_custom_target(doxygen ${DOXYGEN_EXECUTABLE} ${CMAKE_BINARY_DIR}/Doxyfile
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Generate API documentation with Doxygen"
    VERBATIM
    USES_TERMINAL
  )
  add_custom_target(doxygen_zip zip -r doxygen-${MAPD_GIT_HASH}.zip doxygen
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Generating .zip with API documentation from Doxygen"
    VERBATIM
    USES_TERMINAL
  )
  add_dependencies(doxygen_zip doxygen)
endif(DOXYGEN_FOUND)

# Packaging

string(TOLOWER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE_LOWER)
if(NOT "${CMAKE_BUILD_TYPE_LOWER}" STREQUAL "debug" AND NOT "${CMAKE_BUILD_TYPE_LOWER}" STREQUAL "relwithdebinfo")
  set(CPACK_STRIP_FILES ON)
else()
  set(MAPD_PACKAGE_FLAGS "${MAPD_PACKAGE_FLAGS}-debug")
endif()
set(CPACK_PACKAGE_VENDOR "OmniSci, Inc.")
set(CPACK_PACKAGE_CONTACT "support@omnisci.com")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "OmniSci Core Database")
set(CPACK_PROJECT_CONFIG_FILE ${CMAKE_SOURCE_DIR}/CMakePackaging.txt)
set(CPACK_DEBIAN_PACKAGE_DEPENDS "default-jre-headless | openjdk-8-jre-headless | java8-runtime-headless, bsdmainutils, curl | wget")
set(CPACK_RPM_PACKAGE_REQUIRES "java-headless, util-linux, curl")
set(CPACK_RPM_PACKAGE_AUTOREQ OFF)
set(CPACK_RPM_SPEC_MORE_DEFINE "%define __jar_repack %{nil}")
if("${MAPD_EDITION_LOWER}" STREQUAL "ee")
  set(CPACK_DEBIAN_PACKAGE_DEPENDS "${CPACK_DEBIAN_PACKAGE_DEPENDS}, libldap-2.4-2")
endif()
if(ENABLE_RENDERING)
  set(CPACK_RPM_PACKAGE_REQUIRES "${CPACK_RPM_PACKAGE_REQUIRES}, libX11, libXext")
  set(CPACK_DEBIAN_PACKAGE_DEPENDS "${CPACK_DEBIAN_PACKAGE_DEPENDS}, libx11-6, libxext6")
endif()

set_dpkg_arch(${CMAKE_SYSTEM_PROCESSOR} CPACK_DEBIAN_PACKAGE_ARCHITECTURE)

install(DIRECTORY ThirdParty/licenses DESTINATION "ThirdParty/")

# Frontend
option(MAPD_IMMERSE_DOWNLOAD "Download OmniSci Immerse for packaging" OFF)
set(MAPD_IMMERSE_URL ${MAPD_IMMERSE_URL} CACHE STRING "URL to bundled frontend")
if(MAPD_IMMERSE_DOWNLOAD)
  include(ExternalProject)
  externalproject_add(frontend
    URL ${MAPD_IMMERSE_URL}
    PREFIX external
    CONFIGURE_COMMAND ""
    UPDATE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    )
  externalproject_get_property(frontend source_dir)
  install(DIRECTORY ${source_dir}/ DESTINATION "frontend/" PATTERN .git EXCLUDE PATTERN node_modules EXCLUDE)
  add_custom_command(TARGET frontend COMMAND ${CMAKE_COMMAND} -E copy_directory ${source_dir} frontend)
  list(APPEND ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_BINARY_DIR}/frontend)
endif()

## Go web server
find_program(GO_EXECUTABLE NAMES go)
if(NOT GO_EXECUTABLE)
  message(FATAL_ERROR "go not found. Install Go(lang).")
endif()
file(GLOB_RECURSE GOLANG_SOURCES RELATIVE ${CMAKE_SOURCE_DIR} ThirdParty/go/src/mapd/vendor/**/*.go)
add_custom_command(
  OUTPUT ${CMAKE_BINARY_DIR}/bin/omnisci_web_server
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/ThirdParty/go/src/mapd/vendor/ ${CMAKE_BINARY_DIR}/go/src/
  COMMAND GOPATH=${CMAKE_BINARY_DIR}/go ${GO_EXECUTABLE} build -ldflags "-X main.version=${CPACK_PACKAGE_VERSION}" -o ${CMAKE_BINARY_DIR}/bin/omnisci_web_server ${CMAKE_SOURCE_DIR}/OmniSciWebServer.go
  DEPENDS OmniSciWebServer.go ${GOLANG_SOURCES}
  )
add_custom_target(omnisci_web_server ALL DEPENDS ${CMAKE_BINARY_DIR}/bin/omnisci_web_server)
install(PROGRAMS ${CMAKE_BINARY_DIR}/bin/omnisci_web_server DESTINATION bin)
list(APPEND ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_BINARY_DIR}/bin/omnisci_web_server)
list(APPEND ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_BINARY_DIR}/go)

add_subdirectory(ThirdParty/generate_cert)

# docs
option(MAPD_DOCS_DOWNLOAD "Download OmniSci documentation for packaging" OFF)
set(MAPD_DOCS_URL ${MAPD_DOCS_URL} CACHE STRING "URL to bundled docs")
if(MAPD_DOCS_DOWNLOAD)
  include(ExternalProject)
  externalproject_add(docs
    URL ${MAPD_DOCS_URL}
    PREFIX external
    CONFIGURE_COMMAND ""
    UPDATE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    )
  externalproject_get_property(docs source_dir)
  install(DIRECTORY ${source_dir}/ DESTINATION "docs/" PATTERN .git EXCLUDE)
  add_custom_command(TARGET docs COMMAND ${CMAKE_COMMAND} -E copy_directory ${source_dir} docs)
  list(APPEND ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_BINARY_DIR}/docs)
endif()


# systemd
if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
  if("${MAPD_EDITION_LOWER}" STREQUAL "ee")
    install(FILES systemd/omnisci_sd_server.service.in systemd/omnisci_sd_server@.service.in DESTINATION systemd)
    install(FILES systemd/omnisci-sds.conf.in DESTINATION systemd)
  endif()
  install(FILES systemd/omnisci_server.service.in systemd/omnisci_server@.service.in DESTINATION systemd)
  install(FILES systemd/omnisci_web_server.service.in systemd/omnisci_web_server@.service.in DESTINATION systemd)
  install(FILES systemd/omnisci.conf.in DESTINATION systemd)
  install(PROGRAMS systemd/install_omnisci_systemd.sh DESTINATION systemd)
endif()

add_custom_target(clean-all
  COMMAND ${CMAKE_BUILD_TOOL} clean
  )

## mvn process for java code
find_program(MVN_EXECUTABLE NAMES mvn)
if(NOT MVN_EXECUTABLE)
  message(FATAL_ERROR "mvn not found. Install Apache Maven.")
endif()
file(GLOB_RECURSE JAVA_POM RELATIVE ${CMAKE_SOURCE_DIR} java/**/pom.xml)
file(GLOB_RECURSE JAVA_SOURCES RELATIVE ${CMAKE_SOURCE_DIR} java/**/*.java)

set(OMNISCI_JAR_RELEASE_VERSION "${MAPD_VERSION_MAJOR}.${MAPD_VERSION_MINOR}.${MAPD_VERSION_PATCH}")
if("${MAPD_VERSION_EXTRA}" STREQUAL "dev")
  set (OMNISCI_JAR_RELEASE_VERSION "${OMNISCI_JAR_RELEASE_VERSION}-SNAPSHOT")
endif()

set (OMNISCI_JDBC_JAR "omnisci-jdbc-${OMNISCI_JAR_RELEASE_VERSION}.jar")

add_custom_command(
  OUTPUT
    ${CMAKE_BINARY_DIR}/bin/mapd-1.0-SNAPSHOT-jar-with-dependencies.jar
    ${CMAKE_BINARY_DIR}/bin/${OMNISCI_JDBC_JAR}
    ${CMAKE_BINARY_DIR}/bin/calcite-1.0-SNAPSHOT-jar-with-dependencies.jar
    COMMAND MVNPATH=${CMAKE_SOURCE_DIR}/java ${MVN_EXECUTABLE} -q clean install -Dthrift.version="${Thrift_VERSION}" -Domnisci.release.version="${OMNISCI_JAR_RELEASE_VERSION}"
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/java/mapd/target/mapd-1.0-SNAPSHOT-jar-with-dependencies.jar ${CMAKE_BINARY_DIR}/bin
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/java/omniscijdbc/target/${OMNISCI_JDBC_JAR} ${CMAKE_BINARY_DIR}/bin
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/java/calcite/target/calcite-1.0-SNAPSHOT-jar-with-dependencies.jar ${CMAKE_BINARY_DIR}/bin
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/java
  DEPENDS
    ${CMAKE_SOURCE_DIR}/java/thrift/src/gen/com/mapd/thrift/server/MapD.java
    ${CMAKE_SOURCE_DIR}/java/thrift/src/gen/com/mapd/thrift/calciteserver/CalciteServer.java
    ${CMAKE_SOURCE_DIR}/java/pom.xml
    ${CMAKE_SOURCE_DIR}/mapd.thrift
    ${JAVA_POM}
    ${JAVA_SOURCES}
  )
add_custom_target(mapd_java_components ALL DEPENDS
  ${CMAKE_BINARY_DIR}/bin/mapd-1.0-SNAPSHOT-jar-with-dependencies.jar
  ${CMAKE_BINARY_DIR}/bin/${OMNISCI_JDBC_JAR}
  ${CMAKE_BINARY_DIR}/bin/calcite-1.0-SNAPSHOT-jar-with-dependencies.jar)
add_custom_target(mapd_java_clean
  COMMAND MVNPATH=${CMAKE_SOURCE_DIR}/java ${MVN_EXECUTABLE} -q clean -Domnisci.release.version="${OMNISCI_JAR_RELEASE_VERSION}"
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/java
  )
add_dependencies(clean-all mapd_java_clean)
install(FILES ${CMAKE_SOURCE_DIR}/java/mapd/target/mapd-1.0-SNAPSHOT-jar-with-dependencies.jar DESTINATION bin)
install(FILES ${CMAKE_SOURCE_DIR}/java/omniscijdbc/target/${OMNISCI_JDBC_JAR} DESTINATION bin)
install(FILES ${CMAKE_SOURCE_DIR}/java/calcite/target/calcite-1.0-SNAPSHOT-jar-with-dependencies.jar DESTINATION bin)

set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES "${ADDITIONAL_MAKE_CLEAN_FILES}")

install(TARGETS initdb omnisci_server DESTINATION bin)
install(FILES ${CMAKE_BINARY_DIR}/MAPD_GIT_HASH.txt DESTINATION ".")
if(ENABLE_CUDA)
  install(FILES ${CMAKE_BINARY_DIR}/QueryEngine/cuda_mapd_rt.a DESTINATION QueryEngine)
endif()
install(FILES completion_hints.thrift DESTINATION ".")
install(FILES mapd.thrift DESTINATION ".")

if(NOT PREFER_STATIC_LIBS)
  install(FILES ${Boost_LIBRARIES} DESTINATION ThirdParty/lib)
endif()

if("${MAPD_EDITION_LOWER}" STREQUAL "ee")
  set(EULA_FILE "${CMAKE_SOURCE_DIR}/EULA-EE.txt")
else()
  set(EULA_FILE "${CMAKE_SOURCE_DIR}/EULA-CE.txt")
endif()

if("${MAPD_EDITION_LOWER}" STREQUAL "os")
  install(FILES LICENSE.md DESTINATION ".")
endif()

set(CPACK_RESOURCE_FILE_LICENSE "${EULA_FILE}")
install(FILES "${EULA_FILE}" DESTINATION ".")

install(PROGRAMS startomnisci DESTINATION ".")
install(PROGRAMS insert_sample_data DESTINATION ".")

if("${MAPD_EDITION_LOWER}" STREQUAL "ee")
  install(DIRECTORY docker/sds DESTINATION "docker")
endif()
install(FILES docker/Dockerfile docker/Dockerfile.mapd docker/README.md DESTINATION "docker")

execute_process(
  COMMAND ln -sf omnisql mapdql
  COMMAND ln -sf omnisci_server mapd_server
  COMMAND ln -sf omnisci_web_server mapd_web_server
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
install(FILES
  ${CMAKE_BINARY_DIR}/bin/mapdql
  ${CMAKE_BINARY_DIR}/bin/mapd_server
  ${CMAKE_BINARY_DIR}/bin/mapd_web_server
  DESTINATION "bin"
  )
execute_process(
  COMMAND ln -sf startomnisci startmapd
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )
install(FILES ${CMAKE_BINARY_DIR}/startmapd DESTINATION ".")

exec_program(uname ARGS -m OUTPUT_VARIABLE MAPD_HOST_SYSTEM_ARCHITECTURE) # does not account for cross-compiling or Windows
set(CPACK_PACKAGE_FILE_NAME "${CMAKE_PROJECT_NAME}-${MAPD_EDITION_LOWER}-${CPACK_PACKAGE_VERSION}-${CMAKE_SYSTEM_NAME}-${MAPD_HOST_SYSTEM_ARCHITECTURE}${MAPD_PACKAGE_FLAGS}")

set(CPACK_GENERATOR "STGZ")

include(CPack)
