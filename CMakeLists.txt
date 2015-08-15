cmake_minimum_required(VERSION 2.8)

PROJECT(mapd2)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")

option(PREFER_STATIC_LIBS "Prefer linking against static libraries" OFF)
if(PREFER_STATIC_LIBS)
  set(CMAKE_FIND_LIBRARY_SUFFIXES .lib .a ${CMAKE_FIND_LIBRARY_SUFFIXES})
  set(Boost_USE_STATIC_LIBS ON)

  if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libgcc -static-libstdc++")
  endif()
endif()

find_package(Git)
find_package(Glog REQUIRED)
find_package(Thrift REQUIRED)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
if (${CMAKE_SYSTEM_NAME} STREQUAL "Linux" AND (("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang"
    AND (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 3.6 OR CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 3.6))
    OR ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 4.9)))
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1y -DHAS_SHARED_MUTEX")
  FIND_PACKAGE( Boost COMPONENTS filesystem program_options system REQUIRED )
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
  FIND_PACKAGE( Boost COMPONENTS filesystem program_options thread system REQUIRED )
  if(PREFER_STATIC_LIBS AND ${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    list(APPEND Boost_LIBRARIES rt)
    list(INSERT Boost_LIBRARIES 0 pthread)
  endif()
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pthread -D__GCC_HAVE_SYNC_COMPARE_AND_SWAP_1=1 -D__GCC_HAVE_SYNC_COMPARE_AND_SWAP_2=1 -D__GCC_HAVE_SYNC_COMPARE_AND_SWAP_4=1 -D__GCC_HAVE_SYNC_COMPARE_AND_SWAP_8=1 -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS -L/usr/local/cuda/lib -L/usr/local/cuda/lib64 -L/usr/local/cuda/nvvm/lib -L/usr/local/cuda/nvvm/lib64")
set(llvm_config_cmd "llvm-config")
set(llvm_config_inc_arg "--includedir")
set(llvm_config_ld_arg "--ldflags")
set(llvm_config_lib_arg "--libs")
execute_process(COMMAND ${llvm_config_cmd} ${llvm_config_inc_arg}
    OUTPUT_VARIABLE LLVM_INC_FLAGS)
execute_process(COMMAND ${llvm_config_cmd} ${llvm_config_ld_arg} ${llvm_config_lib_arg}
    OUTPUT_VARIABLE LLVM_LIB_FLAGS)
string(REPLACE "\n" "" LLVM_LINKER_FLAGS "${LLVM_LIB_FLAGS} ${LLVM_LD_FLAGS}")
include_directories(${Boost_INCLUDE_DIR} "/usr/local/include" ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/Parser ${CMAKE_CURRENT_BINARY_DIR} ${LLVM_INC_FLAGS} "/usr/local/cuda/nvvm/include/" "/usr/local/cuda/include/")

# Google Test and Google Mock
include_directories(ThirdParty/googletest)
add_subdirectory(ThirdParty/googletest)

ADD_SUBDIRECTORY    (SqliteConnector)
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

set(MAPD_LIBRARIES Shared Catalog SqliteConnector Parser Analyzer Planner
    QueryEngine DataMgr Fragmenter Chunk)

ADD_SUBDIRECTORY    (Tests)
ADD_SUBDIRECTORY    (SampleCode)

set(mapd_source_files main.cpp ${CMAKE_BINARY_DIR}/MapDRelease.h)
add_executable(mapd ${mapd_source_files})

set(initdb_source_files initdb.cpp)
add_executable(initdb ${initdb_source_files})

set(mapd_server_source_files MapDServer.cpp ${CMAKE_BINARY_DIR}/gen-cpp/MapD.cpp ${CMAKE_BINARY_DIR}/gen-cpp/mapd_constants.cpp ${CMAKE_BINARY_DIR}/gen-cpp/mapd_types.cpp ${CMAKE_BINARY_DIR}/MapDRelease.h)
set(mapd_http_server_source_files MapDHttpServer.cpp ${CMAKE_BINARY_DIR}/gen-cpp/MapD.cpp ${CMAKE_BINARY_DIR}/gen-cpp/mapd_constants.cpp ${CMAKE_BINARY_DIR}/gen-cpp/mapd_types.cpp ${CMAKE_BINARY_DIR}/MapDRelease.h)

add_executable(mapd_server ${mapd_server_source_files})
add_executable(mapd_http_server ${mapd_http_server_source_files})

add_custom_command(
    DEPENDS ${CMAKE_SOURCE_DIR}/mapd.thrift
    OUTPUT ${CMAKE_BINARY_DIR}/gen-cpp/MapD.cpp
    COMMAND thrift
    ARGS -gen cpp -o ${CMAKE_BINARY_DIR} ${CMAKE_SOURCE_DIR}/mapd.thrift)

execute_process(COMMAND ${CMAKE_SOURCE_DIR}/get_git_hash ${CMAKE_BINARY_DIR}/MAPD_GIT_HASH.txt)
add_custom_command(
    DEPENDS ${CMAKE_SOURCE_DIR}/Shared/release.h ${CMAKE_BINARY_DIR}/MAPD_GIT_HASH.txt
    OUTPUT ${CMAKE_BINARY_DIR}/MapDRelease.h
    COMMAND sed -e \"s/MAPD_GIT_HASH/`${GIT_EXECUTABLE} log -1 --pretty=format:%h`/\" ${CMAKE_SOURCE_DIR}/Shared/release.h > ${CMAKE_BINARY_DIR}/MapDRelease.h)
add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/MAPD_GIT_HASH.txt
  COMMAND ${CMAKE_SOURCE_DIR}/get_git_hash ${CMAKE_BINARY_DIR}/MAPD_GIT_HASH.txt
  DEPENDS githash)
add_custom_target(githash ALL
  COMMAND ${CMAKE_SOURCE_DIR}/get_git_hash ${CMAKE_BINARY_DIR}/MAPD_GIT_HASH.txt)

set_source_files_properties(${CMAKE_BINARY_DIR}/gen-cpp/MapD.cpp GENERATED)
set_source_files_properties(${CMAKE_BINARY_DIR}/gen-cpp/mapd_constants.cpp GENERATED)
set_source_files_properties(${CMAKE_BINARY_DIR}/gen-cpp/mapd_types.cpp GENERATED)

MESSAGE( STATUS "Boost_LIBRARIES: " ${Boost_LIBRARIES} )

target_link_libraries (mapd ${MAPD_LIBRARIES} ${Boost_LIBRARIES} ${Glog_LIBRARIES} ${CMAKE_DL_LIBS} "${LLVM_LINKER_FLAGS} -lnvvm -lcuda -lcudart -lz -lncurses")
target_link_libraries (mapd_server ${MAPD_LIBRARIES} ${Boost_LIBRARIES} ${Glog_LIBRARIES} ${Thrift_LIBRARIES} ${CMAKE_DL_LIBS} "${LLVM_LINKER_FLAGS} -lnvvm -lcuda -lcudart -lz -lncurses")
target_link_libraries (mapd_http_server ${MAPD_LIBRARIES} ${Boost_LIBRARIES} ${Glog_LIBRARIES} ${Thrift_LIBRARIES} ${CMAKE_DL_LIBS} "${LLVM_LINKER_FLAGS} -lnvvm -lcuda -lcudart -lz -lncurses")
target_link_libraries (initdb Shared Catalog Fragmenter Chunk SqliteConnector DataMgr StringDictionary ${Boost_LIBRARIES} ${CMAKE_DL_LIBS})

# Packaging
set(CPACK_PACKAGE_VENDOR "MapD Technologies, Inc.")

## Frontend
option(MAPD2_FRONTEND_DOWNLOAD "Download mapd2-frontend for packaging" ON)
if(MAPD2_FRONTEND_DOWNLOAD)
  include(ExternalProject)
  ExternalProject_Add(frontend
    GIT_REPOSITORY git@github.com:map-d/mapd2-frontend.git
    PREFIX external
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
    )
  ExternalProject_Get_Property(frontend source_dir)
  install(DIRECTORY ${source_dir} DESTINATION "." PATTERN .git EXCLUDE)
  add_custom_command(TARGET frontend COMMAND ${CMAKE_COMMAND} -E copy_directory ${source_dir} frontend)
endif()

## Go server
find_program(GO_EXECUTABLE NAMES go)
if(GO_EXECUTABLE)
  set(MAPD2_NODEJS_SERVER OFF)
  add_custom_target(mapd_web_server ALL
    COMMAND GOPATH=${CMAKE_BINARY_DIR}/go ${GO_EXECUTABLE} get github.com/gorilla/handlers github.com/namsral/flag
    COMMAND GOPATH=${CMAKE_BINARY_DIR}/go ${GO_EXECUTABLE} build -o ${CMAKE_BINARY_DIR}/bin/mapd_web_server ${CMAKE_SOURCE_DIR}/MapDWebServer.go
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    SOURCES MapDWebServer.go
    )
endif()

## Node server
option(MAPD2_NODEJS_SERVER "Download Node server for uploads and frontend" ON)
set(MAPD2_NODE_VERSION "0.12.7" CACHE STRING "Version of Node to download")
string(TOLOWER ${CMAKE_SYSTEM_NAME} MAPD2_NODE_SYSTEM_NAME)
if(MAPD2_NODEJS_SERVER AND NOT GO_EXECUTABLE)
  include(ExternalProject)
  ExternalProject_Add(upload_server
    URL "http://nodejs.org/dist/v${MAPD2_NODE_VERSION}/node-v${MAPD2_NODE_VERSION}-${MAPD2_NODE_SYSTEM_NAME}-x64.tar.gz" #TODO: architecture
    PREFIX external
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
    )
  ExternalProject_Get_Property(upload_server source_dir)
  install(DIRECTORY ${source_dir}/ DESTINATION upload_server/nodejs USE_SOURCE_PERMISSIONS)
  add_custom_command(TARGET upload_server COMMAND ${CMAKE_COMMAND} -E copy_directory ${source_dir} upload_server/nodejs)
  install(FILES ${CMAKE_SOURCE_DIR}/nodeUploadServer/index.js ${CMAKE_SOURCE_DIR}/nodeUploadServer/package.json DESTINATION upload_server)
  file(COPY ${CMAKE_SOURCE_DIR}/nodeUploadServer/index.js ${CMAKE_SOURCE_DIR}/nodeUploadServer/package.json DESTINATION upload_server)
  add_custom_command(TARGET upload_server COMMAND ${source_dir}/bin/npm install WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/upload_server)
  install(DIRECTORY ${CMAKE_BINARY_DIR}/upload_server/node_modules DESTINATION upload_server)
endif()

install(TARGETS initdb mapd mapd_server mapd_http_server DESTINATION bin)
install(PROGRAMS ${CMAKE_BINARY_DIR}/bin/mapd_web_server DESTINATION bin)
install(FILES ${CMAKE_BINARY_DIR}/MAPD_GIT_HASH.txt DESTINATION ".")
install(FILES ${CMAKE_BINARY_DIR}/QueryEngine/cuda_mapd_rt.a DESTINATION QueryEngine)
install(FILES mapd.thrift DESTINATION ".")
install(FILES MapDBetaGuide.md DESTINATION docs)

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
string(TIMESTAMP CURRENT_DATE "%Y%m%d")
file(STRINGS ${CMAKE_BINARY_DIR}/MAPD_GIT_HASH.txt MAPD_GIT_HASH)
set(CPACK_PACKAGE_VERSION "${CURRENT_DATE}-${MAPD_GIT_HASH}")
set(CPACK_PACKAGE_FILE_NAME "${CMAKE_PROJECT_NAME}-${CPACK_PACKAGE_VERSION}-${CMAKE_SYSTEM_NAME}-${MAPD_HOST_SYSTEM_ARCHITECTURE}${MAPD_PACKAGE_FLAGS}")

set(CPACK_GENERATOR "STGZ")

include(CPack)
