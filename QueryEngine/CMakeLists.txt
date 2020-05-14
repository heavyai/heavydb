set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS -Wall -Wno-attributes")
set_source_files_properties(RuntimeFunctionsCodegenWithIncludes.cpp PROPERTIES COMPILE_FLAGS -O0)
set(query_engine_source_files
    Allocators/CudaAllocator.cpp
    Allocators/ThrustAllocator.cpp
    AggregatedColRange.cpp
    ArithmeticIR.cpp
    ArrayIR.cpp
    ArrayOps.cpp
    ArrowResultSetConverter.cpp
    ArrowResultSet.cpp
    BaselineJoinHashTable.cpp
    CalciteAdapter.cpp
    CalciteDeserializerUtils.cpp
    CardinalityEstimator.cpp
    CaseIR.cpp
    CastIR.cpp
    CgenState.cpp
    Codec.cpp
    ColumnarResults.cpp
    ColumnFetcher.cpp
    ColumnIR.cpp
    CompareIR.cpp
    ConstantIR.cpp
    DateTimeIR.cpp
    DateTimePlusRewrite.cpp
    DateTimeTranslator.cpp
    DateTruncate.cpp
    Descriptors/ColSlotContext.cpp
    Descriptors/QueryCompilationDescriptor.cpp
    Descriptors/QueryFragmentDescriptor.cpp
    Descriptors/QueryMemoryDescriptor.cpp
    Descriptors/RelAlgExecutionDescriptor.cpp
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
    ExternalExecutor.cpp
    ExtractFromTime.cpp
    FromTableReordering.cpp
    GeoIR.cpp
    GpuInterrupt.cpp
    GpuMemUtils.cpp
    GpuSharedMemoryUtils.cpp
    InPlaceSort.cpp
    InValuesIR.cpp
    IRCodegen.cpp
    GroupByAndAggregate.cpp
    InValuesBitmap.cpp
    InputMetadata.cpp
    JoinFilterPushDown.cpp
    LogicalIR.cpp
    LLVMFunctionAttributesUtil.cpp
    LLVMGlobalContext.cpp
    MaxwellCodegenPatch.cpp
    MurmurHash.cpp
    NativeCodegen.cpp
    NvidiaKernel.cpp
    OutputBufferInitialization.cpp
    OverlapsJoinHashTable.cpp
    QueryPhysicalInputsCollector.cpp
    PlanState.cpp
    QueryRewrite.cpp
    QueryTemplateGenerator.cpp
    QueryExecutionContext.cpp
    QueryMemoryInitializer.cpp
    RelAlgDagBuilder.cpp
    RelLeftDeepInnerJoin.cpp
    RelAlgExecutor.cpp
    RelAlgTranslator.cpp
    RelAlgTranslatorGeo.cpp
    RelAlgOptimizer.cpp
    ResultSet.cpp
    ResultSetIteration.cpp
    ResultSetReduction.cpp
    ResultSetReductionCodegen.cpp
    ResultSetReductionInterpreter.cpp
    ResultSetReductionInterpreterStubs.cpp
    ResultSetReductionJIT.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/LoopControlFlow/JoinLoop.cpp
    ResultSetSort.cpp
    RuntimeFunctions.cpp
    RuntimeFunctions.bc
    DynamicWatchdog.cpp
    ScalarCodeGenerator.cpp
    SerializeToSql.cpp
    SpeculativeTopN.cpp
    StreamingTopN.cpp
    StringDictionaryGenerations.cpp
    TableFunctions/TableFunctionCompilationContext.cpp
    TableFunctions/TableFunctionExecutionContext.cpp
    TableFunctions/TableFunctionsFactory.cpp
    TableGenerations.cpp
    TableOptimizer.cpp
    TargetExprBuilder.cpp
    UDFCompiler.cpp
    StringFunctions.cpp
    StringOpsIR.cpp
    RegexpFunctions.cpp
    JoinHashTable.cpp
    JoinHashTableInterface.cpp
    HashJoinRuntime.cpp
    WindowContext.cpp
    WindowExpressionRewrite.cpp
    WindowFunctionIR.cpp

    Codec.h
    Execute.h
    NvidiaKernel.h
    QueryTemplateGenerator.h)

if(NOT "${MAPD_EDITION_LOWER}" STREQUAL "os")
  list(APPEND query_engine_source_files
    ${CMAKE_CURRENT_SOURCE_DIR}/Rendering/ee/RenderAllocator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Rendering/ee/RenderInfo.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Rendering/ee/RenderRelAlgUtils.cpp)
else()
  list(APPEND query_engine_source_files
    ${CMAKE_CURRENT_SOURCE_DIR}/Rendering/os/RenderAllocator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Rendering/os/RenderInfo.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Rendering/os/RenderRelAlgUtils.cpp)
endif()

set(group_by_hash_test_files
    GroupByHashTest.cpp
    MurmurHash.cpp
    DynamicWatchdog.cpp
    RuntimeFunctions.cpp
)

if(ENABLE_CUDA)
  set(MAPD_DEFINITIONS "-DHAVE_CUDA")
  set(NVCC_BUILD_TYPE_ARGS)
  string(TOUPPER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE_UPPERCASE)
  if(CMAKE_BUILD_TYPE_UPPERCASE MATCHES DEBUG)
    list(APPEND NVCC_BUILD_TYPE_ARGS -DTHRUST_DEBUG --debug)
    if (ENABLE_CUDA_KERNEL_DEBUG)
        list(APPEND NVCC_BUILD_TYPE_ARGS --device-debug)
    endif()
  else()
    list(APPEND NVCC_BUILD_TYPE_ARGS -O3)
  endif()
endif()

if(ENABLE_DECODERS_BOUNDS_CHECKING)
  list(APPEND MAPD_DEFINITIONS "-DWITH_DECODERS_BOUNDS_CHECKING")
endif()

if(NOT PREFER_STATIC_LIBS)
  list(APPEND MAPD_DEFINITIONS "-DBOOST_LOG_DYN_LINK")
endif()

if (SUPPRESS_NULL_LOGGER_DEPRECATION_WARNINGS)
  list(APPEND MAPD_DEFINITIONS "-DSUPPRESS_NULL_LOGGER_DEPRECATION_WARNINGS")
endif()

set(llvm_clangpp_cmd clang++)

list(APPEND ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/)
include_directories(${CMAKE_CURRENT_BINARY_DIR})


if(ENABLE_JIT_DEBUG AND NOT ENABLE_CUDA)
  set(RT_OPT_FLAGS -O0 -g)
else()
  set(RT_OPT_FLAGS -O3)
endif()

add_custom_command(
    DEPENDS RuntimeFunctions.cpp JoinHashTableQueryRuntime.cpp RuntimeFunctions.h ${CMAKE_SOURCE_DIR}/Utils/StringLike.cpp GroupByRuntime.cpp TopKRuntime.cpp
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/RuntimeFunctions.bc
    COMMAND ${llvm_clangpp_cmd}
    ARGS -std=c++14 ${RT_OPT_FLAGS} -c -emit-llvm
    ${CLANG_SDK_INC}
    ${CLANG_CRT_INC}
    ${MAPD_DEFINITIONS}
    ${CMAKE_CURRENT_SOURCE_DIR}/RuntimeFunctions.cpp)

add_custom_command(
    DEPENDS ExtensionFunctions.hpp ExtensionFunctionsGeo.hpp ExtensionFunctionsArray.hpp
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/ExtensionFunctions.ast
    COMMAND ${llvm_clangpp_cmd}
    ARGS -std=c++14 -fsyntax-only -Xclang -ast-dump -fno-diagnostics-color -Wno-return-type-c-linkage
    ${CMAKE_CURRENT_SOURCE_DIR}/ExtensionFunctions.hpp | grep FunctionDecl | grep ExtensionFunctions |
    sed -E "s/\\-FunctionDecl.*line:[0-9]+:[0-9]+//g" > ${CMAKE_CURRENT_BINARY_DIR}/ExtensionFunctions.ast)

add_custom_target(QueryEngineFunctionsTargets
    DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/RuntimeFunctions.bc
        ${CMAKE_CURRENT_BINARY_DIR}/ExtensionFunctions.ast
    )

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/OmniSciTypes.h DESTINATION ${CMAKE_BINARY_DIR}/QueryEngine/)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/OmniSciTypes.h ${CMAKE_CURRENT_BINARY_DIR}/RuntimeFunctions.bc ${CMAKE_CURRENT_BINARY_DIR}/ExtensionFunctions.ast DESTINATION QueryEngine)

if(ENABLE_CUDA)
  add_library(QueryEngine ${query_engine_source_files} ${CMAKE_CURRENT_BINARY_DIR}/TopKSort.o ${CMAKE_CURRENT_BINARY_DIR}/InPlaceSortImpl.o ${CMAKE_CURRENT_BINARY_DIR}/ResultSetSortImpl.o ${CMAKE_CURRENT_BINARY_DIR}/GpuInitGroups.o ${CMAKE_CURRENT_BINARY_DIR}/HashJoinRuntimeGpu.o)
  add_dependencies(QueryEngine QueryEngineFunctionsTargets QueryEngineCudaTargets)
else()
  add_library(QueryEngine ${query_engine_source_files})
  add_dependencies(QueryEngine QueryEngineFunctionsTargets)
endif()

set(QUERY_ENGINE_LIBS
  Chunk
  Analyzer
  StringDictionary
  Utils
  Shared
  sqlite3
  ${Arrow_LIBRARIES}
)

if(${LLVM_VERSION_MAJOR} EQUAL 10 AND NOT ${PREFER_STATIC_LIBS})
  # Arch no longer building Clang with static libs
  list(APPEND QUERY_ENGINE_LIBS clang-cpp)
else()
  list(APPEND QUERY_ENGINE_LIBS 
    clangFrontend
    clangSerialization
    clangDriver
    clangTooling
    clangParse
    clangSema
    clangAnalysis
    clangEdit
    clangAST
    clangLex
    clangBasic
    clangRewrite
    clangRewriteFrontend
  )
endif()

list(APPEND QUERY_ENGINE_LIBS ${LLVM_LINKER_FLAGS})

target_link_libraries(QueryEngine ${QUERY_ENGINE_LIBS})

add_custom_command(
    DEPENDS cuda_mapd_rt.cu JoinHashTableQueryRuntime.cpp GpuInitGroups.cu GroupByRuntime.cpp TopKRuntime.cpp DateTruncate.cpp DateAdd.cpp ExtractFromTime.cpp ArrayOps.cpp StringFunctions.cpp RegexpFunctions.cpp ${CMAKE_SOURCE_DIR}/Utils/ChunkIter.cpp ${CMAKE_SOURCE_DIR}/Utils/StringLike.cpp ${CMAKE_SOURCE_DIR}/Utils/Regexp.cpp ${CMAKE_CURRENT_SOURCE_DIR}/ExtensionFunctions.hpp ${CMAKE_CURRENT_SOURCE_DIR}/ExtensionFunctionsGeo.hpp ${CMAKE_CURRENT_SOURCE_DIR}/TableFunctions/TableFunctions.hpp
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/cuda_mapd_rt.fatbin
    COMMAND nvcc
    ARGS
        ${MAPD_HOST_COMPILER_FLAG}
        -Xcompiler -fPIC
        -D__STDC_LIMIT_MACROS
        -D__STDC_CONSTANT_MACROS
        -D_FORCE_INLINES
        -Xcompiler -Wno-return-type-c-linkage --expt-relaxed-constexpr
        ${MAPD_DEFINITIONS}
        -fatbin
        -rdc=true
        ${CUDA_COMPILATION_ARCH}
        -std=c++14
        ${NVCC_BUILD_TYPE_ARGS}
        -c ${CMAKE_CURRENT_SOURCE_DIR}/cuda_mapd_rt.cu
    )

add_custom_command(
    DEPENDS TopKSort.cu
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/TopKSort.o
    COMMAND nvcc
    ARGS
        ${MAPD_HOST_COMPILER_FLAG}
        -Xcompiler -fPIC
        -std=c++14
        -D_FORCE_INLINES
        ${MAPD_DEFINITIONS}
        ${CUDA_COMPILATION_ARCH}
        ${NVCC_BUILD_TYPE_ARGS}
        -c ${CMAKE_CURRENT_SOURCE_DIR}/TopKSort.cu
    )

add_custom_command(
    DEPENDS InPlaceSortImpl.cu
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/InPlaceSortImpl.o
    COMMAND nvcc
    ARGS
        ${MAPD_HOST_COMPILER_FLAG}
        -Xcompiler -fPIC
        -D_FORCE_INLINES
        ${MAPD_DEFINITIONS}
        ${CUDA_COMPILATION_ARCH}
        -std=c++14
        ${NVCC_BUILD_TYPE_ARGS}
        -c ${CMAKE_CURRENT_SOURCE_DIR}/InPlaceSortImpl.cu
    )

add_custom_command(
    DEPENDS ResultSetSortImpl.cu
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/ResultSetSortImpl.o
    COMMAND nvcc
    ARGS
        ${MAPD_HOST_COMPILER_FLAG}
        -Xcompiler -fPIC
        -std=c++14
        -D_FORCE_INLINES
        ${MAPD_DEFINITIONS}
        ${CUDA_COMPILATION_ARCH}
        ${NVCC_BUILD_TYPE_ARGS}
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
        ${CUDA_COMPILATION_ARCH}
        -std=c++14
        ${NVCC_BUILD_TYPE_ARGS}
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
        ${CUDA_COMPILATION_ARCH}
        -std=c++14
        ${NVCC_BUILD_TYPE_ARGS}
        -c ${CMAKE_CURRENT_SOURCE_DIR}/HashJoinRuntimeGpu.cu
    )

add_custom_target(QueryEngineCudaTargets
    DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/cuda_mapd_rt.fatbin
        ${CMAKE_CURRENT_BINARY_DIR}/TopKSort.o
        ${CMAKE_CURRENT_BINARY_DIR}/InPlaceSortImpl.o
        ${CMAKE_CURRENT_BINARY_DIR}/ResultSetSortImpl.o
        ${CMAKE_CURRENT_BINARY_DIR}/GpuInitGroups.o
        ${CMAKE_CURRENT_BINARY_DIR}/HashJoinRuntimeGpu.o
    )

add_executable(group_by_hash_test ${group_by_hash_test_files})
target_link_libraries(group_by_hash_test gtest Shared ${Boost_LIBRARIES} ${PROFILER_LIBS})
