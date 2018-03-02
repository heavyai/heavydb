cmake_minimum_required(VERSION 2.8)

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

project(mapd)

if(NOT EXISTS "${CMAKE_SOURCE_DIR}/Rendering")
  set(MAPD_EDITION "OS")
elseif(NOT DEFINED MAPD_EDITION)
  set(MAPD_EDITION "EE")
endif()
set(MAPD_EDITION "${MAPD_EDITION}" CACHE STRING "MapD edition" FORCE)
set_property(CACHE MAPD_EDITION PROPERTY STRINGS "EE" "CE" "OS")
add_definitions("-DMAPD_EDITION_${MAPD_EDITION}")
string(TOLOWER "${MAPD_EDITION}" MAPD_EDITION_LOWER)

set(MAPD_VERSION_MAJOR "3")
set(MAPD_VERSION_MINOR "6")
set(MAPD_VERSION_PATCH "1")
set(MAPD_VERSION_EXTRA "dev")
set(MAPD_VERSION_RAW "${MAPD_VERSION_MAJOR}.${MAPD_VERSION_MINOR}.${MAPD_VERSION_PATCH}${MAPD_VERSION_EXTRA}")
set(MAPD_IMMERSE_URL "http://builds.mapd.com/frontend/immerse-v2-latest-master-prod.zip")
set(MAPD_DOCS_URL "http://docs.builds.mapd.com/master/docs-master.zip")
string(TIMESTAMP MAPD_BUILD_DATE "%Y%m%d")

if($ENV{BUILD_NUMBER})
  set(MAPD_BUILD_NUMBER "$ENV{BUILD_NUMBER}")
else()
  set(MAPD_BUILD_NUMBER "dev")
endif()
set(MAPD_VERSION "${MAPD_VERSION_RAW}-${MAPD_BUILD_NUMBER}")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")

option(PREFER_STATIC_LIBS "Prefer linking against static libraries" OFF)
if(PREFER_STATIC_LIBS)
  set(CMAKE_FIND_LIBRARY_SUFFIXES .lib .a ${CMAKE_FIND_LIBRARY_SUFFIXES})
  set(Boost_USE_STATIC_LIBS ON)

  if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libgcc -static-libstdc++")
  endif()

  set(CUDA_USE_STATIC_CUDA_RUNTIME ON)
endif()

option(ENABLE_JAVA_REMOTE_DEBUG "Enable Java Remote Debug" OFF )
if(ENABLE_JAVA_REMOTE_DEBUG)
  add_definitions("-DENABLE_JAVA_REMOTE_DEBUG")
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

if(XCODE)
  if(ENABLE_CUDA)
    set(CMAKE_EXE_LINKER_FLAGS "-F/Library/Frameworks -framework CUDA")
  endif()
  add_definitions("-DXCODE")
endif()

find_package(Git)
find_package(Glog REQUIRED)
find_package(Thrift REQUIRED)
include_directories(${Thrift_INCLUDE_DIRS})
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
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1y -DHAS_SHARED_MUTEX")
  find_package( Boost COMPONENTS filesystem program_options regex system REQUIRED QUIET )
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
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
  set(SAN_FLAGS "-fsanitize=thread -fPIC -O1 -fno-omit-frame-pointer")
endif()
if(ENABLE_ASAN OR ENABLE_TSAN)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SAN_FLAGS}")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${SAN_FLAGS}")
  set(ENABLE_STANDALONE_CALCITE ON)
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
include_directories(${Arrow_INCLUDE_DIRS})

# RapidJSON
include_directories(ThirdParty/rapidjson)

# Poly2Tri
include_directories(ThirdParty/poly2tri)
add_subdirectory(ThirdParty/poly2tri)

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
target_link_libraries(mapd_thrift ${Thrift_LIBRARIES})

if("${MAPD_EDITION_LOWER}" STREQUAL "ee")
  include_directories(Distributed/ee)
else()
  include_directories(Distributed/os)
endif()

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

option(ENABLE_JOIN_EXEC "Enable RA vm to execute join node" ON)
if(ENABLE_JOIN_EXEC)
  add_definitions("-DENABLE_JOIN_EXEC")
  option(ENABLE_EQUIJOIN_FOLD "Enable multiple equi-join to be folded" ON)
  if(ENABLE_EQUIJOIN_FOLD)
    add_definitions("-DENABLE_EQUIJOIN_FOLD")
  endif()
endif()
option(ENABLE_MULTIFRAG_JOIN "Enable Join on Multi-frag table" ON)
if(ENABLE_MULTIFRAG_JOIN)
  add_definitions("-DENABLE_MULTIFRAG_JOIN")
endif()
option(ENABLE_ONE_TO_MANY_HASH_JOIN "Enable hash join on a column w/ duplicate values" ON)
if(ENABLE_ONE_TO_MANY_HASH_JOIN)
  add_definitions("-DENABLE_ONE_TO_MANY_HASH_JOIN")
endif()


option(ENABLE_COMPACTION "Enable Compaction and Overflow/Underflow Detection" OFF)
if(ENABLE_COMPACTION)
  add_definitions("-DENABLE_COMPACTION")
endif()

option(ENABLE_KEY_COMPACTION "Enable Key Compaction for Multi-column Baseline Hash" ON)
if(ENABLE_KEY_COMPACTION)
  add_definitions("-DENABLE_KEY_COMPACTION")
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

add_subdirectory(Distributed)

add_subdirectory(StringDictionary)
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
add_subdirectory(SQLFrontend)
add_subdirectory(ThriftHandler)

option(ENABLE_ODBC "Build ODBC driver" OFF)
if(ENABLE_ODBC)
  add_subdirectory(ODBC)
endif()

add_subdirectory(Calcite)

set(MAPD_LIBRARIES Shared Catalog SqliteConnector Parser Analyzer Planner
  QueryEngine DataMgr Fragmenter Chunk)

if("${MAPD_EDITION_LOWER}" STREQUAL "ee")
  list(APPEND MAPD_LIBRARIES Distributed)
endif()

if(ENABLE_FOLLY)
  list(APPEND MAPD_LIBRARIES ${Folly_LIBRARIES})
endif()

list(APPEND MAPD_LIBRARIES Calcite)

list(APPEND MAPD_LIBRARIES ${Arrow_LIBRARIES})

if(ENABLE_RENDERING)
  list(APPEND MAPD_LIBRARIES "${MAPD_RENDERING_LIBRARIES}")
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

set(mapd_server_source_files MapDServer.cpp ${CMAKE_BINARY_DIR}/MapDRelease.h ${CMAKE_SOURCE_DIR}/Shared/ConfigResolve.h)

add_executable(mapd_server ${mapd_server_source_files})
set_target_properties(mapd_server PROPERTIES COMPILE_DEFINITIONS "${TIME_LIMITED_DEFINITIONS}")

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
  COMMAND ${GIT_EXECUTABLE} log -1 --format=%h
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
add_dependencies(mapd_server rerun_cmake)

target_link_libraries(mapd_server mapd_thrift thrift_handler ${MAPD_LIBRARIES} ${Boost_LIBRARIES} ${Glog_LIBRARIES} ${CMAKE_DL_LIBS} ${CUDA_LIBRARIES} ${LLVM_LINKER_FLAGS} ${PROFILER_LIBS} ${CURSES_LIBRARIES} ${ZLIB_LIBRARIES} poly2tri)
if(ENABLE_FOLLY)
target_link_libraries(initdb Shared Catalog Fragmenter Chunk SqliteConnector DataMgr StringDictionary ${Boost_LIBRARIES} ${Folly_LIBRARIES} ${CMAKE_DL_LIBS})
else()
target_link_libraries(initdb Shared Catalog Fragmenter Chunk SqliteConnector DataMgr StringDictionary ${Boost_LIBRARIES} ${CMAKE_DL_LIBS})
endif()

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

# Packaging
set(CPACK_PACKAGE_VENDOR "MapD Technologies, Inc.")
set(CPACK_PACKAGE_CONTACT "support@mapd.com")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "MapD Core Database")
set(CPACK_PROJECT_CONFIG_FILE ${CMAKE_SOURCE_DIR}/CMakePackaging.txt)
set(CPACK_STRIP_FILES ON)
set(CPACK_DEBIAN_PACKAGE_DEPENDS "default-jre-headless | openjdk-8-jre-headless | java8-runtime-headless, bsdmainutils, curl | wget")
set(CPACK_RPM_PACKAGE_REQUIRES "java-headless, util-linux, curl")
set(CPACK_RPM_PACKAGE_AUTOREQ OFF)
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
option(MAPD_IMMERSE_DOWNLOAD "Download MapD Immerse for packaging" ON)
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
  OUTPUT ${CMAKE_BINARY_DIR}/bin/mapd_web_server
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/ThirdParty/go/src/mapd/vendor/ ${CMAKE_BINARY_DIR}/go/src/
  COMMAND GOPATH=${CMAKE_BINARY_DIR}/go ${GO_EXECUTABLE} build -ldflags "-X main.version=${CPACK_PACKAGE_VERSION}" -o ${CMAKE_BINARY_DIR}/bin/mapd_web_server ${CMAKE_SOURCE_DIR}/MapDWebServer.go
  DEPENDS MapDWebServer.go ${GOLANG_SOURCES}
  )
add_custom_target(mapd_web_server ALL DEPENDS ${CMAKE_BINARY_DIR}/bin/mapd_web_server)
install(PROGRAMS ${CMAKE_BINARY_DIR}/bin/mapd_web_server DESTINATION bin)
list(APPEND ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_BINARY_DIR}/bin/mapd_web_server)
list(APPEND ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_BINARY_DIR}/go)

add_subdirectory(ThirdParty/generate_cert)

# docs
option(MAPD_DOCS_DOWNLOAD "Download MapD documentation for packaging" OFF)
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
    install(FILES systemd/mapd_sd_server.service.in systemd/mapd_sd_server@.service.in DESTINATION systemd)
    install(FILES systemd/mapd-sds.conf.in DESTINATION systemd)
  endif()
  install(FILES systemd/mapd_web_server.service.in systemd/mapd_web_server@.service.in DESTINATION systemd)
  install(FILES systemd/mapd.conf.in DESTINATION systemd)
  install(PROGRAMS systemd/install_mapd_systemd.sh DESTINATION systemd)

  if(NOT ENABLE_RENDERING OR "${RENDERER_CONTEXT_TYPE}" STREQUAL "EGL")
    install(FILES systemd/mapd_server.service.in systemd/mapd_server@.service.in DESTINATION systemd)
  else()
    install(FILES systemd/mapd_server_render.service.in RENAME mapd_server.service.in DESTINATION systemd)
    install(FILES systemd/mapd_server_render@.service.in RENAME mapd_server@.service.in DESTINATION systemd)
    install(FILES systemd/mapd_xorg.service DESTINATION systemd)
  endif()
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
add_custom_command(
  OUTPUT
    ${CMAKE_BINARY_DIR}/bin/mapd-1.0-SNAPSHOT-jar-with-dependencies.jar
    ${CMAKE_BINARY_DIR}/bin/mapdjdbc-1.0-SNAPSHOT-jar-with-dependencies.jar
    ${CMAKE_BINARY_DIR}/bin/calcite-1.0-SNAPSHOT-jar-with-dependencies.jar
  COMMAND MVNPATH=${CMAKE_SOURCE_DIR}/java ${MVN_EXECUTABLE} -q install -Dthrift.version="${Thrift_VERSION}"
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/java/mapd/target/mapd-1.0-SNAPSHOT-jar-with-dependencies.jar ${CMAKE_BINARY_DIR}/bin
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/java/mapdjdbc/target/mapdjdbc-1.0-SNAPSHOT-jar-with-dependencies.jar ${CMAKE_BINARY_DIR}/bin
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
  ${CMAKE_BINARY_DIR}/bin/mapdjdbc-1.0-SNAPSHOT-jar-with-dependencies.jar
  ${CMAKE_BINARY_DIR}/bin/calcite-1.0-SNAPSHOT-jar-with-dependencies.jar)
add_custom_target(mapd_java_clean
  COMMAND MVNPATH=${CMAKE_SOURCE_DIR}/java ${MVN_EXECUTABLE} -q clean
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/java
  )
add_dependencies(clean-all mapd_java_clean)
install(FILES ${CMAKE_SOURCE_DIR}/java/mapd/target/mapd-1.0-SNAPSHOT-jar-with-dependencies.jar DESTINATION bin)
install(FILES ${CMAKE_SOURCE_DIR}/java/mapdjdbc/target/mapdjdbc-1.0-SNAPSHOT-jar-with-dependencies.jar DESTINATION bin)
install(FILES ${CMAKE_SOURCE_DIR}/java/calcite/target/calcite-1.0-SNAPSHOT-jar-with-dependencies.jar DESTINATION bin)

set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES "${ADDITIONAL_MAKE_CLEAN_FILES}")

install(TARGETS initdb mapd_server DESTINATION bin)
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

install(PROGRAMS startmapd DESTINATION ".")
install(PROGRAMS insert_sample_data DESTINATION ".")

install(DIRECTORY docker/ DESTINATION "docker")

exec_program(uname ARGS -m OUTPUT_VARIABLE MAPD_HOST_SYSTEM_ARCHITECTURE) # does not account for cross-compiling or Windows
set(CPACK_PACKAGE_FILE_NAME "${CMAKE_PROJECT_NAME}-${MAPD_EDITION_LOWER}-${CPACK_PACKAGE_VERSION}-${CMAKE_SYSTEM_NAME}-${MAPD_HOST_SYSTEM_ARCHITECTURE}${MAPD_PACKAGE_FLAGS}")

set(CPACK_GENERATOR "STGZ")

include(CPack)
