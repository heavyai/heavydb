cmake_minimum_required(VERSION 2.8)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS -Wall -Wno-attributes -Wno-strict-aliasing")
set_source_files_properties(RuntimeFunctionsCodegenWithIncludes.cpp PROPERTIES COMPILE_FLAGS -O0)
set(query_engine_source_files
    ArrayOps.cpp
    CalciteAdapter.cpp
    CalciteDeserializerUtils.cpp
    Codec.cpp
    DateTimePlusRewrite.cpp
    Execute.cpp
    ExpressionRange.cpp
    ExpressionRewrite.cpp
    ExtensionFunctionsBinding.cpp
    ExtensionFunctionsWhitelist.cpp
    ExtensionFunctions.ast
    GpuMemUtils.cpp
    InPlaceSort.cpp
    GroupByAndAggregate.cpp
    InValuesBitmap.cpp
    InputMetadata.cpp
    IteratorTable.cpp
    MaxwellCodegenPatch.cpp
    NvidiaKernel.cpp
    OutputBufferInitialization.cpp
    QueryRewrite.cpp
    QueryTemplateGenerator.cpp
    RelAlgAbstractInterpreter.cpp
    RelAlgExecutionDescriptor.cpp
    RelAlgExecutor.cpp
    RelAlgTranslator.cpp
    RelAlgOptimizer.cpp
    ResultRows.cpp
    ResultSet.cpp
    ResultSetIteration.cpp
    ResultSetReduction.cpp
    RuntimeFunctions.cpp
    RuntimeFunctions.bc
    SpeculativeTopN.cpp
    StringFunctions.cpp
    RegexpFunctions.cpp
    JoinHashTable.cpp
    HashJoinRuntime.cpp
    Codec.h
    Execute.h
    NvidiaKernel.h
    QueryTemplateGenerator.h)

set(group_by_hash_test_files
    GroupByHashTest.cpp
    RuntimeFunctions.cpp
   )

execute_process(COMMAND ${llvm_config_cmd} "--includedir"
                OUTPUT_VARIABLE LLVM_INC_FLAGS)

include_directories(.. ${LLVM_INC_FLAGS} "/usr/local/cuda/nvvm/include/" "/usr/local/cuda/include/" "/usr/local/cuda/nvvm/libnvvm-samples/common/include/" ${Boost_INCLUDE_DIR})

if(ENABLE_CUDA)
    set(MAPD_DEFINITIONS "-DHAVE_CUDA")
endif()

if(ENABLE_LLVM_DBG)
set(CLANG_SDK_INC "-I/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.11.sdk/usr/include/")
set(CLANG_CRT_INC "-I/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/include/c++/v1/")
endif()

if (ENABLE_LLVM_DBG)
set(llvm_clangpp_cmd "${LLVM_BIN_DIR}/clang++")
else ()
set(llvm_clangpp_cmd clang++)
endif (ENABLE_LLVM_DBG)

add_custom_command(
    DEPENDS RuntimeFunctions.cpp RuntimeFunctions.h ${CMAKE_SOURCE_DIR}/Utils/StringLike.cpp GroupByRuntime.cpp
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/RuntimeFunctions.bc
    COMMAND ${llvm_clangpp_cmd}
    ARGS -std=c++11 -O3 -c -emit-llvm
    ${CLANG_SDK_INC}
    ${CLANG_CRT_INC}
    ${MAPD_DEFINITIONS}
    ${CMAKE_CURRENT_SOURCE_DIR}/RuntimeFunctions.cpp)

add_custom_command(
    DEPENDS ExtensionFunctions.hpp
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/ExtensionFunctions.ast
    COMMAND ${llvm_clangpp_cmd}
    ARGS -std=c++11 -fsyntax-only -Xclang -ast-dump -fno-diagnostics-color
    ${CMAKE_CURRENT_SOURCE_DIR}/ExtensionFunctions.hpp | grep FunctionDecl | grep ExtensionFunctions |
    sed -E "s/\\-FunctionDecl.*line:[0-9]+:[0-9]+//g" > ${CMAKE_CURRENT_BINARY_DIR}/ExtensionFunctions.ast)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/RuntimeFunctions.bc ${CMAKE_CURRENT_BINARY_DIR}/ExtensionFunctions.ast DESTINATION QueryEngine)

set(MAPD_HOST_COMPILER "" CACHE STRING "Host compiler to use with nvcc")
if(NOT "${MAPD_HOST_COMPILER}" STREQUAL "")
    set(MAPD_HOST_COMPILER_FLAG "-ccbin=${MAPD_HOST_COMPILER}")
endif()


if(ENABLE_CUDA)
    add_library(QueryEngine ${query_engine_source_files} ${CMAKE_CURRENT_BINARY_DIR}/cuda_mapd_rt.a ${CMAKE_CURRENT_BINARY_DIR}/InPlaceSortImpl.o ${CMAKE_CURRENT_BINARY_DIR}/GpuInitGroups.o ${CMAKE_CURRENT_BINARY_DIR}/HashJoinRuntimeGpu.o)
else()
    add_library(QueryEngine ${query_engine_source_files})
endif()

if(ENABLE_RENDERING)
    target_link_libraries(QueryEngine CsvImport Planner StringDictionary Utils QueryRenderer)
else()
    target_link_libraries(QueryEngine CsvImport Planner StringDictionary Utils)
endif()


add_custom_command(
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/cuda_mapd_rt.o
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/cuda_mapd_rt.a
    COMMAND nvcc
    ARGS
        ${MAPD_HOST_COMPILER_FLAG}
        -D__STDC_LIMIT_MACROS
        -D__STDC_CONSTANT_MACROS
        -D_FORCE_INLINES
        ${MAPD_DEFINITIONS}
        -arch sm_30
        -lib ${CMAKE_CURRENT_BINARY_DIR}/cuda_mapd_rt.o
        -o ${CMAKE_CURRENT_BINARY_DIR}/cuda_mapd_rt.a
    )

add_custom_command(
    DEPENDS cuda_mapd_rt.cu GpuInitGroups.cu GroupByRuntime.cpp DateTruncate.cpp ExtractFromTime.cpp ArrayOps.cpp StringFunctions.cpp RegexpFunctions.cpp ${CMAKE_SOURCE_DIR}/Utils/ChunkIter.cpp ${CMAKE_SOURCE_DIR}/Utils/StringLike.cpp ${CMAKE_SOURCE_DIR}/Utils/Regexp.cpp ${CMAKE_CURRENT_SOURCE_DIR}/ExtensionFunctions.hpp
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/cuda_mapd_rt.o
    COMMAND nvcc
    ARGS
        ${MAPD_HOST_COMPILER_FLAG}
        -D__STDC_LIMIT_MACROS
        -D__STDC_CONSTANT_MACROS
        -D_FORCE_INLINES
        ${MAPD_DEFINITIONS}
        --device-link
        -arch sm_30
        -c ${CMAKE_CURRENT_SOURCE_DIR}/cuda_mapd_rt.cu
    )

add_custom_command(
    DEPENDS InPlaceSortImpl.cu
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/InPlaceSortImpl.o
    COMMAND nvcc
    ARGS
        ${MAPD_HOST_COMPILER_FLAG}
        -O3
        -D_FORCE_INLINES
        ${MAPD_DEFINITIONS}
        -arch sm_30
        -c ${CMAKE_CURRENT_SOURCE_DIR}/InPlaceSortImpl.cu
    )

add_custom_command(
    DEPENDS GpuInitGroups.cu
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/GpuInitGroups.o
    COMMAND nvcc
    ARGS
        ${MAPD_HOST_COMPILER_FLAG}
        -DEXECUTOR_RT
        -D_FORCE_INLINES
        ${MAPD_DEFINITIONS}
        -O3
        -arch sm_30
        -c ${CMAKE_CURRENT_SOURCE_DIR}/GpuInitGroups.cu
    )

add_custom_command(
    DEPENDS HashJoinRuntimeGpu.cu HashJoinRuntime.cpp
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/HashJoinRuntimeGpu.o
    COMMAND nvcc
    ARGS
        ${MAPD_HOST_COMPILER_FLAG}
        -D_FORCE_INLINES
        ${MAPD_DEFINITIONS}
        -O3
        -arch sm_30
        -c ${CMAKE_CURRENT_SOURCE_DIR}/HashJoinRuntimeGpu.cu
    )

add_executable(group_by_hash_test ${group_by_hash_test_files})
target_link_libraries(group_by_hash_test gtest ${Glog_LIBRARIES} ${Boost_LIBRARIES})
