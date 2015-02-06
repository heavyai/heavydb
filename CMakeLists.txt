cmake_minimum_required(VERSION 2.8)

PROJECT			(mapd_exe)

set(CMAKE_CXX_FLAGS "-g -std=c++11 -Wall -O3 -pthread ${CMAKE_CXX_FLAGS} -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS -L/usr/local/cuda/lib -L/usr/local/cuda/lib64 -L/usr/local/cuda/nvvm/lib -L/usr/local/cuda/nvvm/lib64")
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
include_directories(${Boost_INCLUDE_DIR} "/usr/local/include" ${CMAKE_CURRENT_SOURCE_DIR}/Parser ${CMAKE_CURRENT_BINARY_DIR} ${LLVM_INC_FLAGS} "/usr/local/cuda/nvvm/include/" "/usr/local/cuda/include/")
ADD_SUBDIRECTORY    (SqliteConnector)
ADD_SUBDIRECTORY    (Catalog)
ADD_SUBDIRECTORY    (Parser)
ADD_SUBDIRECTORY    (Analyzer)
ADD_SUBDIRECTORY    (Planner)
ADD_SUBDIRECTORY    (QueryEngine)
ADD_SUBDIRECTORY    (DataMgr)
ADD_SUBDIRECTORY    (CudaMgr)
ADD_SUBDIRECTORY    (Fragmenter)
ADD_SUBDIRECTORY    (Tests)

set(mapd_source_files main.cpp)
add_executable(mapd ${mapd_source_files})

set(initdb_source_files initdb.cpp)
add_executable(initdb ${initdb_source_files})

set(mapd_server_source_files MapDServer.cpp ${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/MapD.cpp ${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/mapd_constants.cpp ${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/mapd_types.cpp)
add_executable(mapd_server ${mapd_server_source_files})

add_custom_command(
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/mapd.thrift
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/MapD.cpp
    COMMAND thrift
    ARGS -gen cpp -o ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}/mapd.thrift)

set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/MapD.cpp GENERATED)
set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/mapd_constants.cpp GENERATED)
set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/mapd_types.cpp GENERATED)

MESSAGE( STATUS "Boost_LIBRARIES: " ${Boost_LIBRARIES} )

target_link_libraries (mapd Catalog SqliteConnector Parser Analyzer Planner QueryEngine DataMgr Fragmenter ${Boost_LIBRARIES} ${CMAKE_DL_LIBS} "${LLVM_LINKER_FLAGS} -L/usr/local/cuda/lib -lnvvm -lcuda   -lglog -lz -lncurses -ldl -pthread")
target_link_libraries (mapd_server Catalog SqliteConnector Parser Analyzer Planner QueryEngine DataMgr Fragmenter ${Boost_LIBRARIES} ${CMAKE_DL_LIBS} "${LLVM_LINKER_FLAGS} -L/usr/local/cuda/lib -lnvvm -lcuda   -lglog -lz -lncurses -ldl -lthrift -pthread")
target_link_libraries (initdb Catalog SqliteConnector DataMgr ${Boost_LIBRARIES} ${CMAKE_DL_LIBS} )
