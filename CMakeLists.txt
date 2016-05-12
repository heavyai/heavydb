cmake_minimum_required(VERSION 2.8)

PROJECT(mapd2)

set(MAPD_VERSION_RAW "1.1.5dev")
set(MAPD_IMMERSE_URL "http://builds.mapd.com/frontend/mapd2-dashboard-v1-latest-release.zip")
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

# CUDA required for build, even when CUDA support is disabled
find_package(CUDA REQUIRED)
option(ENABLE_CUDA "Enable CUDA support" ON)
if(ENABLE_CUDA)
  list(APPEND CUDA_LIBRARIES ${CUDA_CUDA_LIBRARY})
  add_definitions("-DHAVE_CUDA")
else()
  set(CUDA_LIBRARIES "")
  set(MAPD_PACKAGE_FLAGS "${MAPD_PACKAGE_FLAGS}-nocuda")
endif()

if(XCODE)
if (ENABLE_CUDA)
set(CMAKE_EXE_LINKER_FLAGS "-F/Library/Frameworks -framework CUDA")
endif()
add_definitions("-DXCODE")
endif()

find_package(Git)
find_package(Glog REQUIRED)
find_package(Thrift REQUIRED)
include_directories(${Thrift_INCLUDE_DIRS})

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
if (${CMAKE_SYSTEM_NAME} STREQUAL "Linux" AND (("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang"
    AND (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 3.6 OR CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 3.6))
    OR ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 4.9)))
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1y -DHAS_SHARED_MUTEX")
  FIND_PACKAGE( Boost COMPONENTS filesystem program_options regex system REQUIRED )
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
  FIND_PACKAGE( Boost COMPONENTS filesystem program_options regex thread system REQUIRED )
  if(PREFER_STATIC_LIBS AND ${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    list(APPEND Boost_LIBRARIES rt)
    list(INSERT Boost_LIBRARIES 0 pthread)
  endif()
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-unused-local-typedefs -Wno-strict-aliasing -pthread -D__GCC_HAVE_SYNC_COMPARE_AND_SWAP_1=1 -D__GCC_HAVE_SYNC_COMPARE_AND_SWAP_2=1 -D__GCC_HAVE_SYNC_COMPARE_AND_SWAP_4=1 -D__GCC_HAVE_SYNC_COMPARE_AND_SWAP_8=1 -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS")
option(ENABLE_LLVM_DBG "switch to local debug version of llvm" OFF)
if (ENABLE_LLVM_DBG)
# NOTE: specify LLVM_BIN_DIR to use your local llvm
if (XCODE)
set(LLVM_BIN_DIR ${CMAKE_SOURCE_DIR}/../llvm/build/xcode/Release/bin)
elseif (APPLE)
set(LLVM_BIN_DIR ${CMAKE_SOURCE_DIR}/../llvm/build/unix/bin)
endif(XCODE)
set(llvm_config_cmd "${LLVM_BIN_DIR}/llvm-config")
else (ENABLE_LLVM_DBG)
set(llvm_config_cmd llvm-config)
endif (ENABLE_LLVM_DBG)

set(llvm_config_inc_arg "--includedir")
set(llvm_config_ld_arg "--ldflags")
set(llvm_config_lib_arg "--libs")
execute_process(COMMAND ${llvm_config_cmd} ${llvm_config_inc_arg}
    OUTPUT_VARIABLE LLVM_INC_FLAGS)
execute_process(COMMAND ${llvm_config_cmd} ${llvm_config_lib_arg}
    OUTPUT_VARIABLE LLVM_LIB_FLAGS)
execute_process(COMMAND ${llvm_config_cmd} ${llvm_config_ld_arg}
    OUTPUT_VARIABLE LLVM_LD_FLAGS)
if (ENABLE_LLVM_DBG)
set(llvm_config_obj_arg "--obj-root")
execute_process(COMMAND ${llvm_config_cmd} ${llvm_config_obj_arg}
    OUTPUT_VARIABLE LLVM_BUILD_DIR)
string(REPLACE "\n" "" LLVM_BUILD_DIR "${LLVM_BUILD_DIR}")
list(APPEND LLVM_INC_FLAGS "${LLVM_BUILD_DIR}/include")
endif (ENABLE_LLVM_DBG)

string(REPLACE "\n" " " LLVM_LINKER_FLAGS "${LLVM_LIB_FLAGS} ${LLVM_LD_FLAGS}")
string(STRIP "${LLVM_LINKER_FLAGS}" LLVM_LINKER_FLAGS)
include_directories(${Boost_INCLUDE_DIR}
                    "/usr/local/include"
                    ${CMAKE_SOURCE_DIR}
                    ${CMAKE_SOURCE_DIR}/Parser
                    ${CMAKE_CURRENT_BINARY_DIR}
                    ${LLVM_INC_FLAGS}
                    "/usr/local/cuda/include/")

# Google Test and Google Mock
include_directories(ThirdParty/googletest)
add_subdirectory(ThirdParty/googletest)

# RapidJSON
include_directories(ThirdParty/rapidjson)

# Poly2Tri
include_directories(ThirdParty/poly2tri)
add_subdirectory(ThirdParty/poly2tri)

# ShapeLib
include_directories(ThirdParty/shapelib)
add_subdirectory(ThirdParty/shapelib)



option(ENABLE_CALCITE "Build Calcite adapter" OFF)
if(ENABLE_CALCITE)
    add_definitions("-DHAVE_CALCITE")
    find_package(JNI REQUIRED)
    include_directories(${JNI_INCLUDE_DIRS})
    set(MAPD_PACKAGE_FLAGS "${MAPD_PACKAGE_FLAGS}-calcite")
endif()

option(ENABLE_RAVM "Enable RA vm" OFF)
if(ENABLE_RAVM)
    add_definitions("-DHAVE_RAVM")
endif()

option(ENABLE_OVERFLOW "Enable Overflow/Underflow Detection" OFF)
if(ENABLE_OVERFLOW)
    add_definitions("-DDETECT_OVERFLOW")
endif()

option(ENABLE_PROFILER "Enable google perftools" OFF)
if(ENABLE_PROFILER)
  find_package(Gperftools REQUIRED COMPONENTS TCMALLOC PROFILER)
  set(PROFILER_LIBS ${Gperftools_TCMALLOC} ${Gperftools_PROFILER} )
  add_definitions("-DHAVE_PROFILER")
else()
  set(PROFILER_LIBS "")
endif()

ADD_SUBDIRECTORY    (SqliteConnector)

# address sanitizer
option(ENABLE_ASAN "Enable address sanitizer" OFF)
if(ENABLE_ASAN)
  set(ASAN_FLAGS "-fsanitize=address -O1 -fno-omit-frame-pointer")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ASAN_FLAGS}")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${ASAN_FLAGS}")
  add_definitions("-DRUN_ASAN")
endif()

ADD_SUBDIRECTORY    (Catalog)
ADD_SUBDIRECTORY    (Parser)
ADD_SUBDIRECTORY    (Analyzer)
ADD_SUBDIRECTORY    (Planner)
ADD_SUBDIRECTORY    (Import)
ADD_SUBDIRECTORY    (StringDictionary)
ADD_SUBDIRECTORY    (QueryEngine)
ADD_SUBDIRECTORY    (DataMgr)
ADD_SUBDIRECTORY    (CudaMgr)
ADD_SUBDIRECTORY    (Fragmenter)
ADD_SUBDIRECTORY    (Chunk)
ADD_SUBDIRECTORY    (Shared)
ADD_SUBDIRECTORY    (Utils)
ADD_SUBDIRECTORY    (SQLFrontend)

option(ENABLE_ODBC "Build ODBC driver" OFF)
if(ENABLE_ODBC)
    ADD_SUBDIRECTORY    (ODBC)
endif()

if(ENABLE_CALCITE)
    ADD_SUBDIRECTORY    (Calcite)
endif()

set(MAPD_LIBRARIES Shared Catalog SqliteConnector Parser Analyzer Planner
    QueryEngine DataMgr Fragmenter Chunk)


if(ENABLE_CALCITE)
    list(APPEND MAPD_LIBRARIES Calcite)
endif()

if (ENABLE_RENDERING)
    list(APPEND MAPD_LIBRARIES "${MAPD_RENDERING_LIBRARIES}")
endif()

enable_testing()
ADD_SUBDIRECTORY    (Tests)
ADD_SUBDIRECTORY    (SampleCode)

set(mapd_source_files main.cpp ${CMAKE_BINARY_DIR}/MapDRelease.h)
add_executable(mapd ${mapd_source_files})

set(initdb_source_files initdb.cpp)
add_executable(initdb ${initdb_source_files})

set(mapd_server_source_files MapDServer.cpp ${CMAKE_BINARY_DIR}/gen-cpp/MapD.cpp ${CMAKE_BINARY_DIR}/gen-cpp/mapd_constants.cpp ${CMAKE_BINARY_DIR}/gen-cpp/mapd_types.cpp ${CMAKE_BINARY_DIR}/MapDRelease.h)

if(ENABLE_CALCITE)
    list(APPEND mapd_server_source_files ${CMAKE_BINARY_DIR}/gen-cpp/CalciteServer.cpp ${CMAKE_BINARY_DIR}/gen-cpp/calciteserver_constants.cpp ${CMAKE_BINARY_DIR}/gen-cpp/calciteserver_types.cpp)
endif()

add_executable(mapd_server ${mapd_server_source_files})
set_target_properties(mapd_server PROPERTIES COMPILE_DEFINITIONS "${TIME_LIMITED_DEFINITIONS}")

add_custom_command(
    DEPENDS ${CMAKE_SOURCE_DIR}/mapd.thrift
    OUTPUT ${CMAKE_BINARY_DIR}/gen-cpp/MapD.cpp
    COMMAND ${Thrift_EXECUTABLE}
    ARGS -gen cpp -o ${CMAKE_BINARY_DIR} ${CMAKE_SOURCE_DIR}/mapd.thrift)
list(APPEND ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_BINARY_DIR}/gen-cpp/)

if(ENABLE_CALCITE)
  add_custom_command(
      DEPENDS ${CMAKE_SOURCE_DIR}/mapd.thrift
      OUTPUT ${CMAKE_SOURCE_DIR}/java/src/gen/com/mapd/thrift/server/MapD.java ${CMAKE_SOURCE_DIR}/java/src/gen/com/mapd/thrift/server/TRow.java
      COMMAND mkdir
      ARGS -p ${CMAKE_SOURCE_DIR}/java/src/gen
      COMMAND ${Thrift_EXECUTABLE}
      ARGS -gen java -out ${CMAKE_SOURCE_DIR}/java/src/gen/ ${CMAKE_SOURCE_DIR}/mapd.thrift)

  add_custom_command(
      DEPENDS ${CMAKE_SOURCE_DIR}/java/calciteserver.thrift
      OUTPUT ${CMAKE_SOURCE_DIR}/java/src/gen/com/mapd/thrift/calciteserver/CalciteServer.java
      COMMAND mkdir
      ARGS -p ${CMAKE_SOURCE_DIR}/java/src/gen
      COMMAND ${Thrift_EXECUTABLE}
      ARGS -gen java -out ${CMAKE_SOURCE_DIR}/java/src/gen/ ${CMAKE_SOURCE_DIR}/java/calciteserver.thrift)
  list(APPEND ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_SOURCE_DIR}/java/src/gen/)
endif()

execute_process(
  COMMAND ${GIT_EXECUTABLE} log -1 --format=%h
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE MAPD_GIT_HASH
  OUTPUT_STRIP_TRAILING_WHITESPACE
  )
file(WRITE ${CMAKE_BINARY_DIR}/MAPD_GIT_HASH.txt "${MAPD_GIT_HASH}\n")
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
add_dependencies(mapd rerun_cmake)
add_dependencies(mapd_server rerun_cmake)

set_source_files_properties(${CMAKE_BINARY_DIR}/gen-cpp/MapD.cpp GENERATED)
set_source_files_properties(${CMAKE_BINARY_DIR}/gen-cpp/mapd_constants.cpp GENERATED)
set_source_files_properties(${CMAKE_BINARY_DIR}/gen-cpp/mapd_types.cpp GENERATED)

if(ENABLE_CALCITE)
    set_source_files_properties(${CMAKE_BINARY_DIR}/gen-cpp/CalciteServer.cpp GENERATED)
    set_source_files_properties(${CMAKE_BINARY_DIR}/gen-cpp/calciteserver_constants.cpp GENERATED)
    set_source_files_properties(${CMAKE_BINARY_DIR}/gen-cpp/calciteserver_types.cpp GENERATED)
endif()

MESSAGE( STATUS "Boost_LIBRARIES: " ${Boost_LIBRARIES} )

target_link_libraries (mapd ${MAPD_LIBRARIES} ${Boost_LIBRARIES} ${Glog_LIBRARIES} ${CMAKE_DL_LIBS} ${CUDA_LIBRARIES} ${LLVM_LINKER_FLAGS} "-lz -lncurses -lldap")
target_link_libraries (mapd_server ${MAPD_LIBRARIES} ${Boost_LIBRARIES} ${Glog_LIBRARIES} ${Thrift_LIBRARIES} ${CMAKE_DL_LIBS} ${CUDA_LIBRARIES} ${LLVM_LINKER_FLAGS} ${PROFILER_LIBS} poly2tri "-lz -lncurses -lldap")
target_link_libraries (initdb Shared Catalog Fragmenter Chunk SqliteConnector DataMgr StringDictionary ${Boost_LIBRARIES} ${CMAKE_DL_LIBS} "-lldap")

# Packaging
set(CPACK_PACKAGE_VENDOR "MapD Technologies, Inc.")

# Frontend
option(MAPD2_FRONTEND_DOWNLOAD "Download mapd2-frontend for packaging" OFF)
set(MAPD_IMMERSE_URL ${MAPD_IMMERSE_URL} CACHE STRING "URL to bundled frontend")
if(MAPD2_FRONTEND_DOWNLOAD)
  include(ExternalProject)
  ExternalProject_Add(frontend
    URL ${MAPD_IMMERSE_URL}
    PREFIX external
    CONFIGURE_COMMAND ""
    UPDATE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    )
  ExternalProject_Get_Property(frontend source_dir)
  install(DIRECTORY ${source_dir}/ DESTINATION "frontend/" PATTERN .git EXCLUDE PATTERN node_modules EXCLUDE)
  add_custom_command(TARGET frontend COMMAND ${CMAKE_COMMAND} -E copy_directory ${source_dir} frontend)
endif()

## Go web server
find_program(GO_EXECUTABLE NAMES go)
if(NOT GO_EXECUTABLE)
  message(FATAL_ERROR "go not found. Install Go(lang).")
endif()
# fake GO15VENDOREXPERIMENT since Go 1.5 is not available on all build platforms
add_custom_target(mapd_web_server ALL
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/ThirdParty/go/src/mapd/vendor/ ${CMAKE_BINARY_DIR}/go/src/
  COMMAND GOPATH=${CMAKE_BINARY_DIR}/go ${GO_EXECUTABLE} build -o ${CMAKE_BINARY_DIR}/bin/mapd_web_server ${CMAKE_SOURCE_DIR}/MapDWebServer.go
  SOURCES MapDWebServer.go
  )
install(PROGRAMS ${CMAKE_BINARY_DIR}/bin/mapd_web_server DESTINATION bin)
list(APPEND ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_BINARY_DIR}/bin/mapd_web_server)
list(APPEND ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_BINARY_DIR}/go)

add_subdirectory(ThirdParty/generate_cert)

# Docs
option(ENABLE_DOCS "Enable compilation of MapD docs" ON)
find_program(PANDOC_EXECUTABLE NAMES pandoc)
find_program(PDFLATEX_EXECUTABLE NAMES pdflatex)
configure_file(
  "${CMAKE_SOURCE_DIR}/docs/MapDGuide.in.md"
  "${CMAKE_BINARY_DIR}/docs/MapDGuide.md"
  @ONLY
  )
if(PANDOC_EXECUTABLE AND PDFLATEX_EXECUTABLE AND ENABLE_DOCS)
  add_custom_target(docs ALL
    COMMAND ${PANDOC_EXECUTABLE}
      --toc --standalone
      -V documentclass=report
      -V geometry:margin=1.5in
      -V graphics:on
      --template=mapd
      --data-dir=.
      --output=${CMAKE_BINARY_DIR}/MapDGuide.pdf
      ${CMAKE_BINARY_DIR}/docs/MapDGuide.md
    SOURCES docs/MapDGuide.md
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/docs
    )
  install(FILES ${CMAKE_BINARY_DIR}/MapDGuide.pdf DESTINATION docs)
  list(APPEND ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_BINARY_DIR}/MapDGuide.pdf)
elseif(ENABLE_DOCS)
  message(WARNING "pandoc or pdflatex not found. Skipping PDF docs.")
endif()
install(FILES ${CMAKE_BINARY_DIR}/docs/MapDGuide.md DESTINATION docs)
install(DIRECTORY docs/release-notes DESTINATION docs)

# systemd
if (${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
  install(FILES systemd/mapd_web_server.service.in systemd/mapd_web_server@.service.in DESTINATION systemd)
  install(FILES systemd/mapd.conf.in DESTINATION systemd)
  install(PROGRAMS systemd/install_mapd_systemd.sh DESTINATION systemd)

  if (NOT ENABLE_RENDERING)
    install(FILES systemd/mapd_server.service.in systemd/mapd_server@.service.in DESTINATION systemd)
  else()
    install(FILES systemd/mapd_server_render.service.in RENAME mapd_server.service.in DESTINATION systemd)
    install(FILES systemd/mapd_server_render@.service.in RENAME mapd_server@.service.in DESTINATION systemd)
    install(FILES systemd/mapd_xorg.service DESTINATION systemd)
  endif()
endif()

if(ENABLE_CALCITE)
  ## mvn process for java code
  find_program(MVN_EXECUTABLE NAMES mvn)
  if(NOT MVN_EXECUTABLE)
    message(FATAL_ERROR "mvn not found. Install Apache Maven.")
  endif()
  add_custom_target(mapd_java_components ALL
    COMMAND MVNPATH=${CMAKE_SOURCE_DIR}/java ${MVN_EXECUTABLE} -q install
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/java/target/mapd-1.0-SNAPSHOT-jar-with-dependencies.jar ${CMAKE_BINARY_DIR}/bin
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/java
    DEPENDS ${CMAKE_SOURCE_DIR}/java/src/gen/com/mapd/thrift/server/MapD.java  ${CMAKE_SOURCE_DIR}/java/src/gen/com/mapd/thrift/calciteserver/CalciteServer.java
    SOURCES ${CMAKE_SOURCE_DIR}/java/pom.xml ${CMAKE_SOURCE_DIR}/mapd.thrift
    )
  install(FILES ${CMAKE_SOURCE_DIR}/java/target/mapd-1.0-SNAPSHOT-jar-with-dependencies.jar DESTINATION bin)
endif()

set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES "${ADDITIONAL_MAKE_CLEAN_FILES}")

install(TARGETS initdb mapd_server DESTINATION bin)
install(FILES ${CMAKE_BINARY_DIR}/MAPD_GIT_HASH.txt DESTINATION ".")
if(ENABLE_CUDA)
  install(FILES ${CMAKE_BINARY_DIR}/QueryEngine/cuda_mapd_rt.a DESTINATION QueryEngine)
endif()
install(FILES mapd.thrift DESTINATION ".")

if(NOT PREFER_STATIC_LIBS)
  install(FILES ${Boost_LIBRARIES} DESTINATION thirdparty)
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/LICENSE.md")
  install(FILES LICENSE.md DESTINATION ".")
  set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE.md")
endif()

install(PROGRAMS startmapd DESTINATION ".")
install(PROGRAMS insert_sample_data DESTINATION ".")

exec_program(uname ARGS -m OUTPUT_VARIABLE MAPD_HOST_SYSTEM_ARCHITECTURE) # does not account for cross-compiling or Windows
file(STRINGS ${CMAKE_BINARY_DIR}/MAPD_GIT_HASH.txt MAPD_GIT_HASH)
set(CPACK_PACKAGE_VERSION "${MAPD_VERSION_RAW}-${MAPD_BUILD_DATE}-${MAPD_GIT_HASH}")
set(CPACK_PACKAGE_FILE_NAME "${CMAKE_PROJECT_NAME}-${CPACK_PACKAGE_VERSION}-${CMAKE_SYSTEM_NAME}-${MAPD_HOST_SYSTEM_ARCHITECTURE}${MAPD_PACKAGE_FLAGS}")

set(CPACK_GENERATOR "STGZ")

include(CPack)
