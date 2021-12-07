set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR})

if(MSVC)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS -Wall")
else()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS -Wall -Wno-attributes")
endif()

set_source_files_properties(RuntimeFunctionsCodegenWithIncludes.cpp PROPERTIES COMPILE_FLAGS -O0)
set(query_engine_source_files
    AggregatedColRange.cpp
    ArithmeticIR.cpp
    ArrayIR.cpp
    ArrayOps.cpp
    ArrowResultSetConverter.cpp
    ArrowResultSet.cpp
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
    DeviceKernel.cpp
    EquiJoinCondition.cpp
    Execute.cpp
    ExecuteUpdate.cpp
    ExecutionKernel.cpp
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
    GeoOps.cpp
    GeoOperators/Codegen.cpp
    GroupByAndAggregate.cpp
    InValuesBitmap.cpp
    InputMetadata.cpp
    JoinFilterPushDown.cpp
    JoinHashTable/BaselineJoinHashTable.cpp
    JoinHashTable/HashJoin.cpp
    JoinHashTable/HashTable.cpp
    JoinHashTable/OverlapsJoinHashTable.cpp
    JoinHashTable/PerfectJoinHashTable.cpp
    JoinHashTable/Runtime/HashJoinRuntime.cpp
    JoinHashTable/RangeJoinHashTable.cpp
    LogicalIR.cpp
    LLVMFunctionAttributesUtil.cpp
    LLVMGlobalContext.cpp
    MaxwellCodegenPatch.cpp
    MurmurHash.cpp
    NativeCodegen.cpp
    NvidiaKernel.cpp
    OutputBufferInitialization.cpp
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
    ResultSetBuilder.cpp
    ResultSetIteration.cpp
    ResultSetReduction.cpp
    ResultSetReductionCodegen.cpp
    ResultSetReductionInterpreter.cpp
    ResultSetReductionInterpreterStubs.cpp
    ResultSetReductionJIT.cpp
    ResultSetStorage.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/TableFunctionsFactory_init.cpp
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
    TableFunctions/TableFunctionOps.cpp
    TableGenerations.cpp
    TableOptimizer.cpp
    TargetExprBuilder.cpp
    Utils/DiamondCodegen.cpp
    StringFunctions.cpp
    StringOpsIR.cpp
    RegexpFunctions.cpp
    Visitors/RelRexDagVisitor.cpp
    Visitors/RexSubQueryIdCollector.cpp
    WindowContext.cpp
    WindowExpressionRewrite.cpp
    WindowFunctionIR.cpp
    QueryPlanDagCache.cpp
    QueryPlanDagExtractor.cpp
    DataRecycler/HashtableRecycler.cpp
    DataRecycler/HashingSchemeRecycler.cpp
    DataRecycler/OverlapsTuningParamRecycler.cpp
    Visitors/QueryPlanDagChecker.cpp
    Visitors/SQLOperatorDetector.cpp

    Codec.h
    Execute.h
    NvidiaKernel.h
    QueryTemplateGenerator.h
    TableFunctions/TableFunctionManager.h)

set(query_engine_cuda_source_files
    TopKSort.cu
    InPlaceSortImpl.cu
    ResultSetSortImpl.cu
    GpuInitGroups.cu
    JoinHashTable/Runtime/HashJoinRuntimeGpu.cu)

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

#if(HAVE_PDAL)
#  list(APPEND query_engine_source_files
#  TableFunctions/SystemFunctions/Loaders/PdalLoader.cpp)
#endif()

set(group_by_hash_test_files
    GroupByHashTest.cpp
    MurmurHash.cpp
    DynamicWatchdog.cpp
    RuntimeFunctions.cpp
)

if(ENABLE_DECODERS_BOUNDS_CHECKING)
  list(APPEND MAPD_DEFINITIONS "-DWITH_DECODERS_BOUNDS_CHECKING")
endif()

if(NOT PREFER_STATIC_LIBS)
  list(APPEND MAPD_DEFINITIONS "-DBOOST_LOG_DYN_LINK")
endif()

if (SUPPRESS_NULL_LOGGER_DEPRECATION_WARNINGS)
  list(APPEND MAPD_DEFINITIONS "-DSUPPRESS_NULL_LOGGER_DEPRECATION_WARNINGS")
endif()

find_program(llvm_clangpp_cmd NAME clang++ HINTS ${LLVM_TOOLS_BINARY_DIR})

list(APPEND ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/)
include_directories(${CMAKE_CURRENT_BINARY_DIR})


if(ENABLE_JIT_DEBUG AND NOT ENABLE_CUDA)
  set(RT_OPT_FLAGS -O0 -g)
else()
  set(RT_OPT_FLAGS -O3)
endif()

if(ENABLE_CUDA)
  message(DEBUG "QueryEngine CUDA_COMPILATION_ARCH: ${CUDA_COMPILATION_ARCH}")
  set(MAPD_DEFINITIONS "-DHAVE_CUDA")
  set(NVCC_BUILD_TYPE_ARGS)
  set(CUDA_TABLE_FUNCTION_FILES ${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/TableFunctionsFactory_init_gpu.hpp)
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

set(TABLE_FUNCTION_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/TableFunctions/TableFunctions.hpp ${CMAKE_CURRENT_SOURCE_DIR}/TableFunctions/TableFunctionsTesting.hpp)

if (ENABLE_SYSTEM_TFS)
  list(APPEND TABLE_FUNCTION_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/TableFunctions/SystemFunctions/os/Shared/Utilities.h)
  list(APPEND TABLE_FUNCTION_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/TableFunctions/SystemFunctions/os/ExampleFunctions.h)
  list(APPEND query_engine_source_files ${CMAKE_CURRENT_SOURCE_DIR}/TableFunctions/SystemFunctions/os/ExampleFunctions.cpp)
  list(APPEND TABLE_FUNCTION_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/TableFunctions/SystemFunctions/os/GeoRaster.h)

  if (ENABLE_CUDA)
    list(APPEND CUDA_TABLE_FUNCTION_FILES ${CMAKE_CURRENT_SOURCE_DIR}/TableFunctions/SystemFunctions/os/ExampleFunctions.cpp)
  endif()

  if("${MAPD_EDITION_LOWER}" STREQUAL "ee")
    if(ENABLE_RF_PROP_TFS)
      list(APPEND TABLE_FUNCTION_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/TableFunctions/SystemFunctions/ee/RFProp.h)
    endif()
    if(ENABLE_POINT_CLOUD_TFS)
      list(APPEND TABLE_FUNCTION_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/TableFunctions/SystemFunctions/ee/PointCloud.h)
    endif()
  endif()

  if(ENABLE_MLPACK)
    list(APPEND TABLE_FUNCTION_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/TableFunctions/MLFunctions.hpp)
  endif()
endif()

if(ENABLE_CUDA)
  add_library(QueryEngine ${query_engine_source_files} ${query_engine_cuda_source_files})
  add_dependencies(QueryEngine QueryEngineFunctionsTargets QueryEngineTableFunctionsFactory_init QueryEngineCudaTargets)
else()
  add_library(QueryEngine ${query_engine_source_files})
  add_dependencies(QueryEngine QueryEngineFunctionsTargets QueryEngineTableFunctionsFactory_init)
endif()

if(ENABLE_CUDA)
  if("${CMAKE_CUDA_COMPILER_ID}" STREQUAL "NVIDIA")
    include(ProcessorCount)
    ProcessorCount(N)
    if (CMAKE_CUDA_COMPILER_VERSION GREATER_EQUAL 11.3 AND NOT N EQUAL 0)
      target_compile_options(QueryEngine PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
          --threads ${N};>)
      set(NVCC_THREADS --threads ${N})
    endif()
  endif()

  if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.18.0")
    message(STATUS "CMake 3.18+, Setting CUDA_ARCHITECTURES.")
    set_target_properties(QueryEngine PROPERTIES CUDA_ARCHITECTURES
        35-virtual
        50-virtual
        60-virtual
        70-virtual
        75-virtual)
  else()
    message(STATUS "CMake 3.17 or under, setting CUDA architecture flags manually.")
    target_compile_options(QueryEngine PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
      -gencode=arch=compute_35,code=compute_35;
      -gencode=arch=compute_50,code=compute_50;
      -gencode=arch=compute_60,code=compute_60;
      -gencode=arch=compute_70,code=compute_70;
      -gencode=arch=compute_75,code=compute_75;
      -Wno-deprecated-gpu-targets>)
  endif()

  target_compile_options(QueryEngine PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
      ${MAPD_HOST_COMPILER_FLAG};
      -Xcompiler -fPIC;
      -std=c++17;
      -D_FORCE_INLINES;
      ${MAPD_DEFINITIONS};
      ${NVCC_BUILD_TYPE_ARGS};>)
endif()

add_custom_command(
    DEPENDS RuntimeFunctions.h RuntimeFunctions.cpp GeoOpsRuntime.cpp DecodersImpl.h JoinHashTable/Runtime/JoinHashTableQueryRuntime.cpp ${CMAKE_SOURCE_DIR}/Utils/StringLike.cpp GroupByRuntime.cpp TopKRuntime.cpp ${CMAKE_SOURCE_DIR}/Geospatial/Utm.h
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/RuntimeFunctions.bc
    COMMAND ${llvm_clangpp_cmd}
    ARGS -std=c++17 ${RT_OPT_FLAGS} -c -emit-llvm
    ${CLANG_SDK_INC}
    ${CLANG_CRT_INC}
    ${MAPD_DEFINITIONS}
    -I ${CMAKE_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/RuntimeFunctions.cpp)

add_custom_command(
    DEPENDS GeosRuntime.cpp GeosRuntime.h ${CMAKE_SOURCE_DIR}/Geospatial/Compression.cpp
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/GeosRuntime.bc
    COMMAND ${llvm_clangpp_cmd}
    ARGS -std=c++17 ${RT_OPT_FLAGS} -c -emit-llvm
    ${CLANG_SDK_INC}
    ${CLANG_CRT_INC}
    -I ${Boost_INCLUDE_DIR}
    ${MAPD_DEFINITIONS}
    ${GEOS_RT_DEFINITIONS}
    -I ${CMAKE_SOURCE_DIR}
    -I ${GEOS_INCLUDE_DIR} -Qy
    ${CMAKE_CURRENT_SOURCE_DIR}/GeosRuntime.cpp)

find_package(PythonInterp REQUIRED)

add_custom_command(
    DEPENDS ExtensionFunctions.hpp ExtensionFunctionsGeo.hpp ExtensionFunctionsArray.hpp ExtensionFunctionsTesting.hpp
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/ExtensionFunctions.ast.raw
    COMMAND ${llvm_clangpp_cmd}
    ARGS -DNO_BOOST -std=c++17 -fsyntax-only -Xclang -ast-dump -fno-diagnostics-color -Wno-return-type-c-linkage
    -I ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/ExtensionFunctions.hpp > ${CMAKE_CURRENT_BINARY_DIR}/ExtensionFunctions.ast.raw)

add_custom_command(
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/ExtensionFunctions.ast.raw
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/ExtensionFunctions.ast
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/parse_ast.py ${CMAKE_CURRENT_BINARY_DIR}/ExtensionFunctions.ast.raw > ${CMAKE_CURRENT_BINARY_DIR}/ExtensionFunctions.ast)

add_custom_target(QueryEngineFunctionsTargets
    DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/RuntimeFunctions.bc
        ${CMAKE_CURRENT_BINARY_DIR}/ExtensionFunctions.ast
        ${CMAKE_CURRENT_BINARY_DIR}/GeosRuntime.bc
    )

add_custom_command(
    DEPENDS ${TABLE_FUNCTION_HEADERS}
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/TableFunctionsFactory_init.cpp ${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/TableFunctionsFactory_init_cpu.hpp ${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/TableFunctionsFactory_init_gpu.hpp
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_TableFunctionsFactory_init.py ${TABLE_FUNCTION_HEADERS} ${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/TableFunctionsFactory_init.cpp)

add_custom_target(QueryEngineTableFunctionsFactory_init
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/TableFunctionsFactory_init.cpp ${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/TableFunctionsFactory_init_cpu.hpp ${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/TableFunctionsFactory_init_gpu.hpp)

set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/TableFunctionsFactory_init.cpp GENERATED)
set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/TableFunctionsFactory_init_cpu.hpp GENERATED)
set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/TableFunctionsFactory_init_gpu.hpp GENERATED)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/OmniSciTypes.h DESTINATION ${CMAKE_BINARY_DIR}/QueryEngine/)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/OmniSciTypes.h ${CMAKE_CURRENT_BINARY_DIR}/RuntimeFunctions.bc ${CMAKE_CURRENT_BINARY_DIR}/GeosRuntime.bc ${CMAKE_CURRENT_BINARY_DIR}/ExtensionFunctions.ast DESTINATION QueryEngine COMPONENT "QE")

target_precompile_headers(QueryEngine PRIVATE
  [["QueryEngine/Execute.h"]]
)

set(QUERY_ENGINE_LIBS
  OSDependent
  Analyzer
  StringDictionary
  Utils
  Logger
  Shared
  sqlite3
  ${Arrow_LIBRARIES}
)

list(APPEND QUERY_ENGINE_LIBS ${llvm_libs} ${ZLIB_LIBRARIES})

target_link_libraries(QueryEngine ${QUERY_ENGINE_LIBS})

add_custom_command(
    DEPENDS cuda_mapd_rt.cu JoinHashTable/Runtime/JoinHashTableQueryRuntime.cpp GpuInitGroups.cu GroupByRuntime.cpp TopKRuntime.cpp DateTruncate.cpp DateAdd.cpp ExtractFromTime.cpp GeoOps.cpp StringFunctions.cpp RegexpFunctions.cpp ${CMAKE_SOURCE_DIR}/Utils/ChunkIter.cpp ${CMAKE_SOURCE_DIR}/Utils/StringLike.cpp ${CMAKE_SOURCE_DIR}/Utils/Regexp.cpp ${CMAKE_CURRENT_SOURCE_DIR}/ExtensionFunctions.hpp ${CMAKE_CURRENT_SOURCE_DIR}/ExtensionFunctionsGeo.hpp  ${CMAKE_CURRENT_SOURCE_DIR}/ExtensionFunctionsTesting.hpp ${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/TableFunctionsFactory_init_gpu.hpp ${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/TableFunctionsFactory_init.cpp
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/cuda_mapd_rt.fatbin
    COMMAND nvcc
    ARGS
        -I ${CMAKE_SOURCE_DIR}
        -I ${CMAKE_CURRENT_BINARY_DIR}
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
        -std=c++17
        ${NVCC_BUILD_TYPE_ARGS}
        ${NVCC_THREADS}
        -c ${CMAKE_CURRENT_SOURCE_DIR}/cuda_mapd_rt.cu
    )

# TODO (Breno): Find a way to build as regular library (with cuda_add_library())
# There's no simple way to compile a file with two different compilers in CMake.
# Since we already compile these as C++ source files in libQueryEngine, we need
# to recompile them explicitly with nvcc.
add_custom_command(
  DEPENDS ${CUDA_TABLE_FUNCTION_FILES}
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/CudaTableFunctions.a
  COMMAND ${CMAKE_CUDA_COMPILER}
  ARGS
      -I ${CMAKE_SOURCE_DIR}
      -I ${CMAKE_CURRENT_BINARY_DIR}
      ${MAPD_HOST_COMPILER_FLAG}
      -DNO_BOOST
      -Xcompiler -fPIC
      -Xcompiler -Wno-return-type-c-linkage --expt-relaxed-constexpr
      ${MAPD_DEFINITIONS}
      ${compDef}
      --lib
      -rdc=true
      ${CUDA_COMPILATION_ARCH}
      -std=c++17
      ${NVCC_BUILD_TYPE_ARGS}
      ${NVCC_THREADS}
      -x cu
      ${CUDA_TABLE_FUNCTION_FILES} -o ${CMAKE_CURRENT_BINARY_DIR}/CudaTableFunctions.a
  )

add_custom_target(QueryEngineCudaTargets
    DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/cuda_mapd_rt.fatbin
        ${CMAKE_CURRENT_BINARY_DIR}/CudaTableFunctions.a
    )
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/CudaTableFunctions.a DESTINATION QueryEngine COMPONENT "QE")

add_executable(group_by_hash_test ${group_by_hash_test_files})
target_link_libraries(group_by_hash_test gtest Logger Shared ${Boost_LIBRARIES} ${PROFILER_LIBS})
