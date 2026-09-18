# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

set(VEGA_RENDER_TEST_SOURCE_FILES VegaRenderTest.cpp VegaRenderTestCpuSetup.cpp)

set(VEGA_RENDER_TEST_LIBS ${MAPD_LIBRARIES} ${LLVM_LINKER_FLAGS} ${Boost_LIBRARIES})
list(APPEND VEGA_RENDER_TEST_LIBS mapd_thrift)

if(ENABLE_CUDA)
  list(APPEND VEGA_RENDER_TEST_SOURCE_FILES VegaRenderTestCudaSetup.cu)
  if (SUPPRESS_NULL_LOGGER_DEPRECATION_WARNINGS)
    target_compile_options(VegaRenderTest PRIVATE -DSUPPRESS_NULL_LOGGER_DEPRECATION_WARNINGS)
  endif()
  list(APPEND VEGA_RENDER_TEST_LIBS GfxDriver ${MAPD_LIBRARIES} ${CUDA_LIBRARIES})
endif()

#add_executable(VegaRenderTest ${VEGA_RENDER_TEST_SOURCE_FILES})

#target_include_directories(VegaRenderTest PRIVATE ${GFXDRIVER_INCLUDE_DIRS})
#target_link_libraries(VegaRenderTest ${VEGA_RENDER_TEST_LIBS})
