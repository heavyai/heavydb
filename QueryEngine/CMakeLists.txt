set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS -Wall -Wno-attributes")
set_source_files_properties(RuntimeFunctionsCodegenWithIncludes.cpp PROPERTIES COMPILE_FLAGS -O0)
set(query_engine_source_files
    AggregatedColRange.cpp
    ArithmeticIR.cpp
    ArrayIR.cpp
    ArrayOps.cpp
    ArrowResultSet.cpp
    ArrowUtil.cpp
    BaselineJoinHashTable.cpp
    CalciteAdapter.cpp
    CalciteDeserializerUtils.cpp
    CaseIR.cpp
    CastIR.cpp
    Codec.cpp
    ColumnarResults.cpp
    ColumnIR.cpp
    CompareIR.cpp
    ConstantIR.cpp
    DateTimeIR.cpp
    DateTimePlusRewrite.cpp
    EquiJoinCondition.cpp
    Execute.cpp
    ExecuteUpdate.cpp
    ExecutionDispatch.cpp
    ExpressionRange.cpp
    ExpressionRewrite.cpp
    ExtensionFunctionsBinding.cpp
    ExtensionFunctionsWhitelist.cpp
    ExtensionFunctions.ast
    ExtensionsIR.cpp
    GpuInterrupt.cpp
    GpuMemUtils.cpp
    InPlaceSort.cpp
    InValuesIR.cpp
    IRCodegen.cpp
    GroupByAndAggregate.cpp
    InValuesBitmap.cpp
    InputMetadata.cpp
    IteratorTable.cpp
    LegacyExecute.cpp
    LogicalIR.cpp
    LLVMFunctionAttributesUtil.cpp
    LLVMGlobalContext.cpp
    MaxwellCodegenPatch.cpp
    MurmurHash.cpp
    NativeCodegen.cpp
    NvidiaKernel.cpp
    OutputBufferInitialization.cpp
    QueryPhysicalInputsCollector.cpp
    QueryRewrite.cpp
    QueryTemplateGenerator.cpp
    RelAlgAbstractInterpreter.cpp
    RelLeftDeepInnerJoin.cpp
    RelAlgExecutionDescriptor.cpp
    RelAlgExecutor.cpp
    RelAlgTranslator.cpp
    RelAlgTranslatorGeo.cpp
    RelAlgOptimizer.cpp
    ResultRows.cpp
    ResultSet.cpp
    ResultSetIteration.cpp
    ResultSetReduction.cpp
    ResultSetConversion.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/LoopControlFlow/JoinLoop.cpp
    ResultSetSort.cpp
    RuntimeFunctions.cpp
    RuntimeFunctions.bc
    DynamicWatchdog.cpp
    SpeculativeTopN.cpp
    StreamingTopN.cpp
    StringDictionaryGenerations.cpp
    TableGenerations.cpp
    StringFunctions.cpp
    StringOpsIR.cpp
    RegexpFunctions.cpp
    JoinHashTable.cpp
    HashJoinRuntime.cpp
    
    Codec.h
    Execute.h
    NvidiaKernel.h
    QueryTemplateGenerator.h)

if(NOT "${MAPD_EDITION_LOWER}" STREQUAL "os")
  list(APPEND query_engine_source_files
    ${CMAKE_CURRENT_SOURCE_DIR}/Rendering/ee/ExecuteRenderInterface.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Rendering/ee/RelAlgExecutorRenderInterface.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Rendering/ee/RenderAllocator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Rendering/ee/RenderInfo.cpp)
else()
  list(APPEND query_engine_source_files
    ${CMAKE_CURRENT_SOURCE_DIR}/Rendering/os/ExecuteRenderInterface.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Rendering/os/RelAlgExecutorRenderInterface.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Rendering/os/RenderAllocator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Rendering/os/RenderInfo.cpp)
endif()

set(group_by_hash_test_files
    GroupByHashTest.cpp
    MurmurHash.cpp
    DynamicWatchdog.cpp
    RuntimeFunctions.cpp
)

execute_process(COMMAND ${llvm_config_cmd} "--includedir"
                OUTPUT_VARIABLE LLVM_INC_FLAGS)

if(ENABLE_CUDA)
  set(MAPD_DEFINITIONS "-DHAVE_CUDA")
endif()

if(ENABLE_MULTIFRAG_JOIN)
    list(APPEND MAPD_DEFINITIONS "-DENABLE_MULTIFRAG_JOIN")
endif()


if(ENABLE_DECODERS_BOUNDS_CHECKING)
  list(APPEND MAPD_DEFINITIONS "-DWITH_DECODERS_BOUNDS_CHECKING")
endif()

if(ENABLE_LLVM_DBG)
set(CLANG_SDK_INC "-I/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.11.sdk/usr/include/")
set(CLANG_CRT_INC "-I/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/include/c++/v1/")
endif()

if(ENABLE_LLVM_DBG)
  set(llvm_clangpp_cmd "${LLVM_BIN_DIR}/clang++")
else()
  set(llvm_clangpp_cmd clang++)
endif()

list(APPEND ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/)
include_directories(${CMAKE_CURRENT_BINARY_DIR})


add_custom_command(
    DEPENDS RuntimeFunctions.cpp RuntimeFunctions.h ${CMAKE_SOURCE_DIR}/Utils/StringLike.cpp GroupByRuntime.cpp TopKRuntime.cpp
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/RuntimeFunctions.bc
    COMMAND ${llvm_clangpp_cmd}
    ARGS -std=c++11 -O3 -c -emit-llvm
    ${CLANG_SDK_INC}
    ${CLANG_CRT_INC}
    ${MAPD_DEFINITIONS}
    ${CMAKE_CURRENT_SOURCE_DIR}/RuntimeFunctions.cpp)

add_custom_command(
    DEPENDS ExtensionFunctions.hpp ExtensionFunctionsGeo.hpp
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/ExtensionFunctions.ast
    COMMAND ${llvm_clangpp_cmd}
    ARGS -std=c++11 -fsyntax-only -Xclang -ast-dump -fno-diagnostics-color
    ${CMAKE_CURRENT_SOURCE_DIR}/ExtensionFunctions.hpp | grep FunctionDecl | grep ExtensionFunctions |
    sed -E "s/\\-FunctionDecl.*line:[0-9]+:[0-9]+//g" > ${CMAKE_CURRENT_BINARY_DIR}/ExtensionFunctions.ast)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/RuntimeFunctions.bc ${CMAKE_CURRENT_BINARY_DIR}/ExtensionFunctions.ast DESTINATION QueryEngine)

if(ENABLE_CUDA)
  add_library(QueryEngine ${query_engine_source_files} ${CMAKE_CURRENT_BINARY_DIR}/cuda_mapd_rt.a ${CMAKE_CURRENT_BINARY_DIR}/TopKSort.o ${CMAKE_CURRENT_BINARY_DIR}/InPlaceSortImpl.o ${CMAKE_CURRENT_BINARY_DIR}/ResultSetSortImpl.o ${CMAKE_CURRENT_BINARY_DIR}/GpuInitGroups.o ${CMAKE_CURRENT_BINARY_DIR}/HashJoinRuntimeGpu.o)
else()
  add_library(QueryEngine ${query_engine_source_files})
endif()

set(ARROW_LIBS ${Arrow_LIBRARIES})

target_link_libraries(QueryEngine Planner StringDictionary Utils ${ARROW_LIBS})

add_custom_command(
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/cuda_mapd_rt.o
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/cuda_mapd_rt.a
    COMMAND nvcc
    ARGS
        ${MAPD_HOST_COMPILER_FLAG}
        -Xcompiler -fPIC
        -D__STDC_LIMIT_MACROS
        -D__STDC_CONSTANT_MACROS
        -D_FORCE_INLINES
        ${MAPD_DEFINITIONS}
        -arch sm_30
        -lib ${CMAKE_CURRENT_BINARY_DIR}/cuda_mapd_rt.o
        -o ${CMAKE_CURRENT_BINARY_DIR}/cuda_mapd_rt.a
    )

add_custom_command(
    DEPENDS cuda_mapd_rt.cu GpuInitGroups.cu GroupByRuntime.cpp TopKRuntime.cpp DateTruncate.cpp DateAdd.cpp ExtractFromTime.cpp ArrayOps.cpp StringFunctions.cpp RegexpFunctions.cpp ${CMAKE_SOURCE_DIR}/Utils/ChunkIter.cpp ${CMAKE_SOURCE_DIR}/Utils/StringLike.cpp ${CMAKE_SOURCE_DIR}/Utils/Regexp.cpp ${CMAKE_CURRENT_SOURCE_DIR}/ExtensionFunctions.hpp ${CMAKE_CURRENT_SOURCE_DIR}/ExtensionFunctionsGeo.hpp
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/cuda_mapd_rt.o
    COMMAND nvcc
    ARGS
        ${MAPD_HOST_COMPILER_FLAG}
        -Xcompiler -fPIC
        -D__STDC_LIMIT_MACROS
        -D__STDC_CONSTANT_MACROS
        -D_FORCE_INLINES
        ${MAPD_DEFINITIONS}
        --device-link
        -arch sm_30
        -std=c++11
        -c ${CMAKE_CURRENT_SOURCE_DIR}/cuda_mapd_rt.cu
    )


add_custom_command(
    DEPENDS TopKSort.cu
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/TopKSort.o
    COMMAND nvcc
    ARGS
        ${MAPD_HOST_COMPILER_FLAG}
        -Xcompiler -fPIC
        -std=c++11
        -O3
        -D_FORCE_INLINES
        ${MAPD_DEFINITIONS}
        -arch sm_30
        -c ${CMAKE_CURRENT_SOURCE_DIR}/TopKSort.cu
    )

add_custom_command(
    DEPENDS InPlaceSortImpl.cu
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/InPlaceSortImpl.o
    COMMAND nvcc
    ARGS
        ${MAPD_HOST_COMPILER_FLAG}
        -Xcompiler -fPIC
        -O3
        -D_FORCE_INLINES
        ${MAPD_DEFINITIONS}
        -arch sm_30
        -std=c++11
        -c ${CMAKE_CURRENT_SOURCE_DIR}/InPlaceSortImpl.cu
    )

add_custom_command(
    DEPENDS ResultSetSortImpl.cu
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/ResultSetSortImpl.o
    COMMAND nvcc
    ARGS
        ${MAPD_HOST_COMPILER_FLAG}
        -Xcompiler -fPIC
        -std=c++11
        -O3
        -D_FORCE_INLINES
        ${MAPD_DEFINITIONS}
        -arch sm_30
        -c ${CMAKE_CURRENT_SOURCE_DIR}/ResultSetSortImpl.cu
    )

add_custom_command(
    DEPENDS GpuInitGroups.cu
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/GpuInitGroups.o
    COMMAND nvcc
    ARGS
        ${MAPD_HOST_COMPILER_FLAG}
        -Xcompiler -fPIC
        -D_FORCE_INLINES
        ${MAPD_DEFINITIONS}
        -O3
        -arch sm_30
        -std=c++11
        -c ${CMAKE_CURRENT_SOURCE_DIR}/GpuInitGroups.cu
    )

add_custom_command(
    DEPENDS HashJoinRuntimeGpu.cu HashJoinRuntime.cpp
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/HashJoinRuntimeGpu.o
    COMMAND nvcc
    ARGS
        ${MAPD_HOST_COMPILER_FLAG}
        -Xcompiler -fPIC
        -D_FORCE_INLINES
        ${MAPD_DEFINITIONS}
        -O3
        -arch sm_30
        -std=c++11
        -c ${CMAKE_CURRENT_SOURCE_DIR}/HashJoinRuntimeGpu.cu
    )

add_executable(group_by_hash_test ${group_by_hash_test_files})
target_link_libraries(group_by_hash_test gtest ${Glog_LIBRARIES} ${Boost_LIBRARIES})
