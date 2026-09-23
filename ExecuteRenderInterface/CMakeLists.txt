# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

set(EXECUTE_RENDER_INTERFACE_SOURCES)
set(EXECUTE_RENDER_INTERFACE_DEPS)
set(EXECUTE_RENDER_INTERFACE_LIBS)
set(EXECUTE_RENDER_INTERFACE_INCLUDE_DIRS)

list(APPEND EXECUTE_RENDER_INTERFACE_SOURCES
      RenderHandlerImpl.cpp
      RenderHandlerImplHitTest.cpp
      RenderQueryUtils/SQLSelectedTablesBuilder.cpp
      RenderQueryUtils/RenderQueryRunner.cpp
      RenderQueryUtils/RenderQueryResultProcessor.cpp
      RenderQueryUtils/ProcessResultsUtils.cpp
      RenderQueryUtils/ProcessRowResults.cpp
      RenderQueryUtils/ProcessGeoResults.cpp
      RenderQueryUtils/ProcessRasterMeshResults.cpp
      RenderQueryUtils/RenderRelAlgUtils.cpp
      RenderQueryUtils/RelScanTree.cpp)

list(APPEND EXECUTE_RENDER_INTERFACE_SOURCES
      RenderQueryUtils/HitTesting/ProjHitTestColSqlBuilder.cpp
      RenderQueryUtils/NonInsituQueryClassifier.cpp)

if(ENABLE_CUDA)
  list(APPEND EXECUTE_RENDER_INTERFACE_SOURCES
    RenderQueryUtils/thrust/ThrustGeoImplFactoryCuda.cu)
else()
  list(APPEND EXECUTE_RENDER_INTERFACE_SOURCES
    RenderQueryUtils/thrust/ThrustGeoImplFactoryTbb.cpp)
endif()

add_library(ExecuteRenderInterface ${EXECUTE_RENDER_INTERFACE_SOURCES})

if(ENABLE_RENDER_LOGGER)
  message(STATUS "Render logging enabled for ExecuteRenderInterface")
  target_compile_definitions(ExecuteRenderInterface PUBLIC "ENABLE_RENDER_LOGGER")
endif()

if(ENABLE_CUDA)
  target_compile_options(ExecuteRenderInterface PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
    -Xcudafe --diag_suppress=virtual_function_decl_hidden>)
else()
  find_package(TBB REQUIRED)
  find_package(Thrust REQUIRED)

  set_source_files_properties(
    RenderQueryUtils/thrust/ThrustGeoImplFactoryTbb.cpp
    PROPERTIES COMPILE_DEFINITIONS THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_TBB)

  list(APPEND EXECUTE_RENDER_INTERFACE_LIBS ${TBB_IMPORTED_TARGETS})
  list(APPEND EXECUTE_RENDER_INTERFACE_INCLUDE_DIRS ${THRUST_INCLUDE_DIR})
endif()

if(NOT ENABLE_CUDA)
  # Need to set the THRUST_DEVICE_SYSTEM to TBB
  set_source_files_properties(RenderQueryUtils/ProcessGeoResults.cpp
                              PROPERTIES COMPILE_DEFINITIONS "THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_TBB")
endif()

if(EXECUTE_RENDER_INTERFACE_DEPS)
  add_dependencies(ExecuteRenderInterface ${EXECUTE_RENDER_INTERFACE_DEPS})
endif()

target_include_directories(ExecuteRenderInterface PRIVATE ${EXECUTE_RENDER_INTERFACE_INCLUDE_DIRS})
target_link_libraries(ExecuteRenderInterface ${EXECUTE_RENDER_INTERFACE_LIBS})

# NOTE: needed the below calcite include to get access to TFilterPushDownInfo in "gen-cpp/calciteserver_types.h"
# TFilterPushDownInfo is required for RA parsing
add_dependencies(ExecuteRenderInterface Calcite)
