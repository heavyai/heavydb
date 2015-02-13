cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_FLAGS "-D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS -std=c++11 -Wall -Wno-attributes -O3")
set(query_engine_source_files
    Codec.cpp
    Execute.cpp
    DeviceSideLaunch.cpp
    NvidiaKernel.cpp
    QueryTemplateGenerator.cpp
    RuntimeFunctions.cpp
    RuntimeFunctionsCodegenWithIncludes.cpp
    Codec.h
    Execute.h
    RuntimeFunctions.h
    NvidiaKernel.h
    QueryTemplateGenerator.h)
set(group_by_hash_test_files
    GroupByHashTest.cpp
    RuntimeFunctions.cpp
    RuntimeFunctions.h)

set(llvm_config_cmd "llvm-config")
set(llvm_config_inc_arg "--includedir")
set(llvm_config_ld_arg "--ldflags")
set(llvm_config_lib_arg "--libs")
execute_process(COMMAND ${llvm_config_cmd} ${llvm_config_inc_arg}
    OUTPUT_VARIABLE LLVM_INC_FLAGS)
execute_process(COMMAND ${llvm_config_cmd} ${llvm_config_ld_arg} ${llvm_config_lib_arg}
    OUTPUT_VARIABLE LLVM_LIB_FLAGS)
include_directories(.. ${LLVM_INC_FLAGS} "/usr/local/cuda/nvvm/include/" "/usr/local/cuda/include/" "/usr/local/cuda/nvvm/libnvvm-samples/common/include/")

add_custom_command(
    DEPENDS RuntimeFunctions.ll
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/RuntimeFunctionsCodegen.cpp
    COMMAND llc
    ARGS -march=cpp -cppgen=contents RuntimeFunctions.ll -o ${CMAKE_CURRENT_BINARY_DIR}/RuntimeFunctionsCodegen.cpp)

add_custom_command(
    DEPENDS RuntimeFunctionsCodegen.cpp RuntimeFunctionsCodegenIncludes.cpp RuntimeFunctionsCodegenEpilogue.cpp
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/RuntimeFunctionsCodegenWithIncludes.cpp
    COMMAND cat
    ARGS ${CMAKE_CURRENT_SOURCE_DIR}/RuntimeFunctionsCodegenIncludes.cpp ${CMAKE_CURRENT_BINARY_DIR}/RuntimeFunctionsCodegen.cpp ${CMAKE_CURRENT_SOURCE_DIR}/RuntimeFunctionsCodegenEpilogue.cpp > ${CMAKE_CURRENT_BINARY_DIR}/RuntimeFunctionsCodegenWithIncludes.cpp)

add_custom_command(
    DEPENDS RuntimeFunctions.cpp RuntimeFunctions.h
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/RuntimeFunctions.ll
    COMMAND clang++
    ARGS -std=c++11 -O3 -c -S -emit-llvm ${CMAKE_CURRENT_SOURCE_DIR}/RuntimeFunctions.cpp)

string(REPLACE "\n" "" LLVM_LINKER_FLAGS "${LLVM_LIB_FLAGS} ${LLVM_LD_FLAGS}")

add_library(QueryEngine ${query_engine_source_files} ${CMAKE_CURRENT_BINARY_DIR}/cuda_mapd_rt.a)

add_custom_command(
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/cuda_mapd_rt.o
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/cuda_mapd_rt.a
    COMMAND nvcc
    ARGS -arch sm_30 -lib ${CMAKE_CURRENT_BINARY_DIR}/cuda_mapd_rt.o -o ${CMAKE_CURRENT_BINARY_DIR}/cuda_mapd_rt.a)

add_custom_command(
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cuda_mapd_rt.cu
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/cuda_mapd_rt.o
    COMMAND nvcc
    ARGS --device-link -arch sm_30 -c ${CMAKE_CURRENT_SOURCE_DIR}/cuda_mapd_rt.cu)

add_executable(group_by_hash_test ${group_by_hash_test_files})
target_link_libraries(group_by_hash_test "-lgtest -pthread")
