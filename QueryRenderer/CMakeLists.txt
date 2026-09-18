# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2015-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

find_package(PNG REQUIRED)

set(QUERY_RENDERER_LIBS ${PNG_LIBRARIES} muparserx)
set(QUERY_RENDERER_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/ThirdParty/rapidjson)

if(ENABLE_RENDER_LOGGER)
  message(STATUS "Render logging enabled for QueryRenderer")
  add_compile_definitions("ENABLE_RENDER_LOGGER")
endif()

set(QUERY_RENDERER_SOURCES
  PngData.cpp
  QueryDataLayout.cpp
  QueryRenderManager.cpp
  Renderer.cpp
  QueryRendererContext.cpp
  AggregationContext.cpp
  VegaParser.cpp
  PerGpuData.cpp
  GlobalRenderContext.cpp
  JSONRefObject.cpp
  RenderCmdQueue.cpp
  RenderSession.cpp
  RenderSessionMgr.cpp
  ResourceTracking.cpp
  SlabAddressTable.cpp
  QueryBufferManager.cpp
  QueryBufferAllocator.cpp

  Events/RefEvent.cpp
  Events/RefEventCallbacksMap.cpp

  Cache/LineMgr.cpp
  Cache/PolyMgr.cpp
  Cache/RasterMeshMgr.cpp
  Cache/ResultCache.cpp
  Cache/RowIdHitTestOffsetData.cpp

  Data/Enums/DataFormatType.cpp

  Data/Parsers/RasterMeshFormatJson.cpp
  Data/Parsers/CrossSectionFormatJson.cpp
  Data/Parsers/SqlQueryLineFormatJson.cpp

  Data/BaseQueryDataTable.cpp
  Data/BaseEmbeddedDataTable.cpp
  Data/BaseRowDataTable.cpp
  Data/BaseLineDataTable.cpp
  Data/BasePolyDataTable.cpp

  Data/QueryDataTableSQL.cpp

  Data/QueryRowDataTable.cpp
  Data/EmbeddedRowDataTable.cpp
  Data/RowDataTableGpuResources.cpp

  Data/QueryPolyDataTable.cpp
  Data/EmbeddedPolyDataTable.cpp
  Data/PolyDataTableGpuResources.cpp

  Data/QueryLineDataTable.cpp
  Data/EmbeddedLineDataTable.cpp
  Data/LineDataTableGpuResources.cpp
  
  Data/QueryMeshDataTable.cpp
  Data/MeshDataTableGpuResources.cpp
  Data/QuerySourceDataTable.cpp
  Data/Utils.cpp
  Data/EmbeddedDataUtils.cpp
  Data/Types.cpp
  Data/QueryDataTableQueues.cpp

  Data/Transforms/Enums.cpp
  Data/Transforms/Utils.cpp
  Data/Transforms/BaseXform.cpp
  Data/Transforms/XformOp.cpp
  Data/Transforms/Aggregate/AggXform.cpp
  Data/Transforms/Aggregate/AggXformOp.cpp
  Data/Transforms/Aggregate/ValidateUtils.cpp
  Data/Transforms/Formula/FormulaXform.cpp
  Data/Transforms/Formula/FormulaXformOp.cpp
  Data/Transforms/Aggregate/Ops/AvgOp.cpp
  Data/Transforms/Aggregate/Ops/CountDistinctOp.cpp
  Data/Transforms/Aggregate/Ops/DistinctHistogramProps.cpp
  Data/Transforms/Aggregate/Ops/QuantileProps.cpp
  Data/Transforms/Aggregate/Ops/StdDevOp.cpp
  Data/Transforms/Aggregate/Ops/VarianceOp.cpp)

set (THRUST_OP_EXECUTOR_SOURCES
  Data/Transforms/Aggregate/Ops/CountOp.cpp
  Data/Transforms/Aggregate/Ops/DistinctOp.cpp
  Data/Transforms/Aggregate/Ops/MinMaxOp.cpp
  Data/Transforms/Aggregate/Ops/DistinctHistogramOp.cpp
  Data/Transforms/Aggregate/Ops/MissingOp.cpp
  Data/Transforms/Aggregate/Ops/QuantileOp.cpp
  Data/Transforms/Aggregate/Ops/SumOp.cpp
  Data/Transforms/Aggregate/Ops/SqDiffSumOp.cpp
  Data/Transforms/Aggregate/Ops/ValidOp.cpp
  Data/Transforms/Aggregate/Ops/TopBottomKOp.cpp
  Data/Transforms/Aggregate/OpExecuteUtils.cpp
  Data/Transforms/Aggregate/thrust/ThrustOpInterface.cpp)

list(APPEND QUERY_RENDERER_SOURCES ${THRUST_OP_EXECUTOR_SOURCES})

set (THRUST_OP_IMPL_SOURCES
    Data/Transforms/Aggregate/thrust/Ops/CountOp.cu
    Data/Transforms/Aggregate/thrust/Ops/ValidOp.cu
    Data/Transforms/Aggregate/thrust/Ops/MissingOp.cu
    Data/Transforms/Aggregate/thrust/Ops/MinOp.cu
    Data/Transforms/Aggregate/thrust/Ops/MaxOp.cu
    Data/Transforms/Aggregate/thrust/Ops/SumOp.cu
    Data/Transforms/Aggregate/thrust/Ops/SqDiffSumOp.cu
    Data/Transforms/Aggregate/thrust/Ops/DistinctOp.cu
    Data/Transforms/Aggregate/thrust/Ops/DistinctHistogramOp.cu
    Data/Transforms/Aggregate/thrust/Ops/QuantileOp.cu
    Data/Transforms/Aggregate/thrust/Ops/TopBottomKOp.cu)

list(APPEND QUERY_RENDERER_SOURCES ${THRUST_OP_IMPL_SOURCES})

if(ENABLE_CUDA)
  # NOTE: the following suppresses innocuous compile-time warnings such as this:
  #
  # In file included from /usr/local/cuda/include/thrust/system/cuda/detail/util.h:36,
  #                  from /usr/local/cuda/include/thrust/system/cuda/detail/internal/copy_cross_system.h:41,
  #                  from /usr/local/cuda/include/thrust/system/cuda/detail/copy.h:100,
  #                  from /usr/local/cuda/include/thrust/system/detail/adl/copy.h:42,
  #                  from /usr/local/cuda/include/thrust/detail/copy.inl:22,
  #                  from /usr/local/cuda/include/thrust/detail/copy.h:90,
  #                  from /usr/local/cuda/include/thrust/system/detail/sequential/merge.inl:19,
  #                  from /usr/local/cuda/include/thrust/system/detail/sequential/merge.h:78,
  #                  from /usr/local/cuda/include/thrust/system/cpp/detail/merge.h:22,
  #                  from /usr/local/cuda/include/thrust/system/cpp/execution_policy.h:51,
  #                  from /usr/local/cuda/include/thrust/execution_policy.h:32,
  #                  from ../DataMgr/Allocators/thrust/DataMgrAllocationPolicy.h:19,
  #                  from ../QueryRenderer/Utils/thrust/ThrustExecutionContext.h:7,
  #                  from ../QueryRenderer/Data/Transforms/Aggregate/thrust/ThrustOpExecutorInterface.h:7,
  #                  from ../QueryRenderer/Data/Transforms/Aggregate/thrust/ThrustOpExecutorImplFactory.h:5,
  #                  from ../QueryRenderer/Data/Transforms/Aggregate/thrust/ThrustOpExecutor.h:3,
  #                  from ../QueryRenderer/Data/Transforms/Aggregate/OpExecuteUtils.cpp:19:
  # /usr/local/cuda/include/cub/detail/device_synchronize.cuh:33: warning: ignoring #pragma nv_exec_check_disable  [-Wunknown-pragmas]
  #    33 | #pragma nv_exec_check_disable
  #       | 
  #
  set_source_files_properties(${THRUST_OP_EXECUTOR_SOURCES}
                              PROPERTIES COMPILE_OPTIONS "-Wno-unknown-pragmas")

  list(APPEND QUERY_RENDERER_SOURCES Data/Transforms/Aggregate/thrust/ThrustOpExecutorImplFactoryCuda.cu)
else()
  find_package(TBB REQUIRED)
  find_package(Thrust REQUIRED)

  list(APPEND QUERY_RENDERER_SOURCES Data/Transforms/Aggregate/thrust/ThrustOpExecutorImplFactoryTbb.cpp)

  # need to tell the compiler to recognize the .cu files as cpp files.
  set_source_files_properties(
    ${THRUST_OP_IMPL_SOURCES}
    PROPERTIES LANGUAGE CXX COMPILE_FLAGS "-x c++"
  )

  # Need to set the THRUST_DEVICE_SYSTEM to TBB for all thrust-related files
  set_source_files_properties(${THRUST_OP_EXECUTOR_SOURCES}
                              ${THRUST_OP_IMPL_SOURCES}
                              Data/Transforms/Aggregate/thrust/ThrustOpExecutorImplFactoryTbb.cpp
                              PROPERTIES COMPILE_DEFINITIONS "THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_TBB")

  list(APPEND QUERY_RENDERER_LIBS ${TBB_IMPORTED_TARGETS})
  list(APPEND QUERY_RENDERER_INCLUDE_DIRS ${THRUST_INCLUDE_DIR})
endif()

list(APPEND QUERY_RENDERER_SOURCES
  Marks/BaseMark.cpp
  Marks/MarkPerGpuData.cpp
  Marks/PointMark.cpp
  Marks/PolyMark.cpp
  Marks/LineMark.cpp
  Marks/LineUtils.cpp
  Marks/SymbolMark.cpp
  Marks/SymbolMark_Proc.cpp
  Marks/SymbolDefinitions.cpp
  Marks/WindBarbDefinitions.cpp
  Marks/WindBarbMark.cpp
  Marks/Mesh2dMark.cpp
  Marks/BaseRenderProperty.cpp
  Marks/RenderProperty.cpp
  Marks/RenderPropertyContainer.cpp
  Marks/RenderPropertyBufferState.cpp
  Marks/RenderPropertyBufferStateHandler.cpp
  Marks/CoordAttrInfo2d.cpp
  Marks/Utils.cpp
  Marks/Enums.cpp
  Marks/MarkProjectionShaderPolicy.cpp)

list(APPEND QUERY_RENDERER_SOURCES
  Rendering/QueryFramebuffer.cpp
  Rendering/QueryIdMapPixelBuffer.cpp
  Rendering/HitTestBuffers.cpp
  Rendering/QueryRenderSMAAPass.cpp
  Rendering/SeparateMultiSamplesPass.cpp
  Rendering/TextureTransferContext.cpp
  Rendering/MultiGpuCompositor.cpp
  Rendering/AccumRenderer.cpp)

if (ENABLE_CUDA)
  list(APPEND QUERY_RENDERER_SOURCES
    Rendering/Vulkan/VulkanTextureTransferContext.cpp
    Rendering/Vulkan/VulkanTextureTransferContextCudaUtils.cpp)
endif()

list(APPEND QUERY_RENDERER_SOURCES
  Scales/ScaleFactory.cpp
  Scales/BaseScale.cpp
  Scales/BaseScaleDomainRangeData.cpp
  Scales/ScaleDomainRangeData.cpp
  Scales/BaseScaleRef.cpp
  Scales/ScaleAccumState.cpp
  Scales/ScaleAccumRenderState.cpp
  Scales/Types.cpp
  Scales/Utils.cpp)

list(APPEND QUERY_RENDERER_SOURCES
  Projections/Projection.cpp
  Projections/ProjectionShader.cpp
  Projections/Utils.cpp)

list(APPEND QUERY_RENDERER_SOURCES
  Utils/AnyDataType.cpp
  Utils/RapidJSONUtils.cpp
  Utils/JSONRefErrorLogger.cpp
  Utils/StringUtils.cpp
  Utils/TimeUtils.cpp
  Utils/TypeUtils.cpp
  Utils/NumericUtils.cpp
  Utils/AnalyzerUtils.cpp
  Utils/thrust/ThrustExecutionContext.cpp)

list(APPEND QUERY_RENDERER_SOURCES
  Interop/Enums.cpp
  Interop/QueryBuffer.cpp
  Interop/InteropBufferMgr.cpp)

list(APPEND QUERY_RENDERER_SOURCES
  Interface/RenderSessionKey.cpp
  Interface/SqlSelectedTableInfo.cpp)

find_package(BLOSC)

find_package(GLM REQUIRED)

add_library(QueryRenderer ${QUERY_RENDERER_SOURCES})
target_include_directories(QueryRenderer PRIVATE ${QUERY_RENDERER_INCLUDE_DIRS} ${GFXDRIVER_INCLUDE_DIRS})
target_link_libraries(QueryRenderer ${QUERY_RENDERER_LIBS})
target_compile_definitions(QueryRenderer PRIVATE ${GFXDRIVER_DEFS})

if(HOTLOAD_SHADER_TEMPLATE_FILES)
  message(STATUS "Hotloading of shader template strings enabled")
  add_definitions(-DHOTLOAD_SHADER_TEMPLATE_FILES)
  add_definitions(-DQUERY_RENDERER_SOURCE_PATH=\"${CMAKE_CURRENT_SOURCE_DIR}/\")
else()
  include(GenerateShaderTemplates)

  list(APPEND ShaderTemplateFiles
    "${CMAKE_CURRENT_SOURCE_DIR}/ShaderManifest.json"
    "${CMAKE_SOURCE_DIR}/GfxDriver/Render/shaders/ShaderManifest.json")

  generateShaderTemplates(
    ${CMAKE_SOURCE_DIR}/scripts
    "${ShaderTemplateFiles}"
    ${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/ShaderTemplates
    QueryRendererShaderTemplates
    QueryRendererShaders)

  target_include_directories(QueryRenderer PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/gen-cpp/ShaderTemplates)
  target_link_libraries(QueryRenderer QueryRendererShaderTemplates)
endif()
