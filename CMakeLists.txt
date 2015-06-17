cmake_minimum_required(VERSION 2.8)

PROJECT			(mapd_exe)

find_package(Git)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -pthread ${CMAKE_CXX_FLAGS} -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS -L/usr/local/cuda/lib -L/usr/local/cuda/lib64 -L/usr/local/cuda/nvvm/lib -L/usr/local/cuda/nvvm/lib64")
set(Boost_USE_STATIC_LIBS ON)
FIND_PACKAGE( Boost COMPONENTS filesystem program_options thread system REQUIRED )
set(llvm_config_cmd "llvm-config")
set(llvm_config_inc_arg "--includedir")
set(llvm_config_ld_arg "--ldflags")
set(llvm_config_lib_arg "--libs")
execute_process(COMMAND ${llvm_config_cmd} ${llvm_config_inc_arg}
    OUTPUT_VARIABLE LLVM_INC_FLAGS)
execute_process(COMMAND ${llvm_config_cmd} ${llvm_config_ld_arg} ${llvm_config_lib_arg}
    OUTPUT_VARIABLE LLVM_LIB_FLAGS)
string(REPLACE "\n" "" LLVM_LINKER_FLAGS "${LLVM_LIB_FLAGS} ${LLVM_LD_FLAGS}")
include_directories(${Boost_INCLUDE_DIR} "/usr/local/include" ${CMAKE_SOURCE_DIR}/Parser ${CMAKE_CURRENT_BINARY_DIR} ${LLVM_INC_FLAGS} "/usr/local/cuda/nvvm/include/" "/usr/local/cuda/include/")

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

target_link_libraries (mapd ${MAPD_LIBRARIES} ${Boost_LIBRARIES} ${CMAKE_DL_LIBS} "${LLVM_LINKER_FLAGS} -L/usr/local/cuda/lib -lnvvm -lcuda   -lglog -lz -lncurses -ldl -pthread")
target_link_libraries (mapd_server ${MAPD_LIBRARIES} ${Boost_LIBRARIES} ${CMAKE_DL_LIBS} "${LLVM_LINKER_FLAGS} -L/usr/local/cuda/lib -lnvvm -lcuda   -lglog -lz -lncurses -ldl -lthrift -pthread")
target_link_libraries (mapd_http_server ${MAPD_LIBRARIES} ${Boost_LIBRARIES} ${CMAKE_DL_LIBS} "${LLVM_LINKER_FLAGS} -L/usr/local/cuda/lib -lnvvm -lcuda   -lglog -lz -lncurses -ldl -lthrift -pthread")
target_link_libraries (initdb Shared Catalog Fragmenter Chunk SqliteConnector DataMgr StringDictionary ${Boost_LIBRARIES} ${CMAKE_DL_LIBS} )
