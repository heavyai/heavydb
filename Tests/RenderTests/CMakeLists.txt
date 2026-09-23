# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_subdirectory(VegaRenderTest)

add_executable(ShaderCompilerTest ShaderCompiler/ShaderCompilerTest.cpp)
add_executable(GfxDriverTest
  GfxDriver/GfxDriverTest.cpp
  GfxDriver/GfxCommandTest.cpp
  GfxDriver/GfxRenderTest.cpp
  Utils/AttachmentUtils.cpp
  Utils/GoldenImage.cpp
  Utils/MeshUtils.cpp
  Utils/ShaderUtils.cpp)

add_executable(GfxInteropTest
  GfxInterop/GfxInteropTest.cpp
  Utils/AttachmentUtils.cpp
  Utils/GoldenImage.cpp
  Utils/ShaderUtils.cpp)

add_executable(GfxExperimentalTest
  GfxExperimental/GfxExperimentalTest.cpp
  Utils/AttachmentUtils.cpp
  Utils/GoldenImage.cpp
  Utils/ShaderUtils.cpp)

add_executable(PPLLTest
  PPLL/PPLLTest.cpp
  Utils/AttachmentUtils.cpp
  Utils/GoldenImage.cpp
  Utils/MeshUtils.cpp
  Utils/PPLLUtils.cpp
  Utils/ShaderUtils.cpp)

add_executable(RaytracingTest
  Raytracing/RaytracingTest.cpp
  Utils/AttachmentUtils.cpp
  Utils/GoldenImage.cpp
  Utils/MeshUtils.cpp
  Utils/ShaderUtils.cpp)

add_executable(QueryRendererTest
  QueryRenderer/QueryRendererTest.cpp
  Utils/GoldenImage.cpp
  Utils/MeshUtils.cpp
  Utils/ShaderUtils.cpp)

# Paths for data file loading
add_compile_definitions(RENDER_TESTS_PATH=\"${CMAKE_CURRENT_SOURCE_DIR}/\")
add_compile_definitions(GFX_DRIVER_PATH=\"${CMAKE_SOURCE_DIR}/GfxDriver/\")

target_include_directories(ShaderCompilerTest PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/../../QueryRenderer/gen-cpp/ShaderTemplates)
target_include_directories(QueryRendererTest PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/../../QueryRenderer/gen-cpp/ShaderTemplates)

target_link_libraries(ShaderCompilerTest gtest GfxDriver Shared QueryRendererShaderTemplates)
target_link_libraries(GfxDriverTest gtest GfxDriver Shared ${TBB_LIBS} ${CMAKE_DL_LIBS})

set(GFX_INTEROP_TEST_LIBS gtest GfxDriver GfxInterop Shared)
if (ENABLE_CUDA)
  list(APPEND GFX_INTEROP_TEST_LIBS ${CUDA_LIBRARIES})
endif()
target_link_libraries(GfxInteropTest ${GFX_INTEROP_TEST_LIBS} ${CMAKE_DL_LIBS})

target_link_libraries(GfxExperimentalTest gtest GfxDriver Shared ${CMAKE_DL_LIBS})
target_link_libraries(PPLLTest gtest GfxDriver Shared ${TBB_LIBS} ${CMAKE_DL_LIBS})
target_link_libraries(RaytracingTest gtest GfxDriver Shared ${TBB_LIBS} ${CMAKE_DL_LIBS})

set(QUERY_RENDER_TEST_LIBS mapd_thrift GfxDriver QueryRenderer QueryRendererShaderTemplates ${MAPD_LIBRARIES} ${CMAKE_DL_LIBS} ${CUDA_LIBRARIES} ${Boost_LIBRARIES} ${AwsS3_CURL_SUPPORT})
target_link_libraries(QueryRendererTest gtest ${QUERY_RENDER_TEST_LIBS})

set(TEST_ARGS "--gtest_output=xml:../../")
add_test(ShaderCompilerTest ShaderCompilerTest ${TEST_ARGS})
add_test(GfxDriverTest GfxDriverTest ${TEST_ARGS})
add_test(GfxInteropTest GfxInteropTest ${TEST_ARGS})
add_test(GfxExperimentalTest GfxExperimentalTest ${TEST_ARGS})
add_test(PPLLTest PPLLTest ${TEST_ARGS})
add_test(RaytracingTest RaytracingTest ${TEST_ARGS})
add_test(QueryRendererTest QueryRendererTest ${TEST_ARGS})

set(RENDER_SANITY_TESTS
  ShaderCompilerTest
  GfxDriverTest
  GfxInteropTest
  GfxExperimentalTest
  PPLLTest
  RaytracingTest
  QueryRendererTest
)

set_tests_properties(${RENDER_SANITY_TESTS} PROPERTIES LABELS "render_sanity")

add_custom_target(render_sanity_tests_build_only DEPENDS ${RENDER_SANITY_TESTS})

set(TEST_COMMAND GTEST_COLOR=1 ${CMAKE_CTEST_COMMAND} --verbose)

add_custom_target(render_sanity_tests
  COMMAND env ${TEST_COMMAND} --label-regex render_sanity
  DEPENDS ${RENDER_SANITY_TESTS}
  USES_TERMINAL)

if (ENABLE_TESTS)
  add_dependencies(sanity_tests render_sanity_tests)
endif()

add_custom_target(shader_compiler_tests
  COMMAND ${TEST_COMMAND} --tests-regex "\"(ShaderCompilerTest)\""
  DEPENDS ShaderCompilerTest
  USES_TERMINAL)

add_custom_target(gfx_driver_tests
  COMMAND ${TEST_COMMAND} --tests-regex "\"(GfxDriverTest)\""
  DEPENDS GfxDriverTest
  USES_TERMINAL)

add_custom_target(gfx_interop_tests
  COMMAND ${TEST_COMMAND} --tests-regex "\"(GfxInteropTest)\""
  DEPENDS GfxInteropTest
  USES_TERMINAL)

add_custom_target(gfx_experimental_tests
  COMMAND ${TEST_COMMAND} --tests-regex "\"(GfxExperimentalTest)\""
  DEPENDS GfxExperimentalTest
  USES_TERMINAL)

add_custom_target(ppll_tests
  COMMAND ${TEST_COMMAND} --tests-regex "\"(PPLLTest)\""
  DEPENDS PPLLTest
  USES_TERMINAL)

add_custom_target(raytracing_tests
  COMMAND ${TEST_COMMAND} --tests-regex "\"(RaytracingTest)\""
  DEPENDS RaytracingTest
  USES_TERMINAL)

find_package(GLFW3)

if (GLFW3_FOUND)
  find_package(GLM REQUIRED)

  add_executable(GfxWSITest
    GfxWSI/WSIAppBase.cpp
    GfxWSI/GfxWSITest.cpp
    Utils/AttachmentUtils.cpp)

  add_executable(SymbolSandbox
    GfxWSI/WSIAppBase.cpp
    GfxWSI/SymbolSandbox/SymbolSandbox.cpp
    GfxWSI/Background.cpp
    Utils/AttachmentUtils.cpp
    Utils/RenderPropertyUtils.cpp
    ${CMAKE_SOURCE_DIR}/QueryRenderer/Marks/SymbolDefinitions.cpp
    ${CMAKE_SOURCE_DIR}/QueryRenderer/Marks/Enums.cpp
    ${CMAKE_SOURCE_DIR}/QueryRenderer/Utils/StringUtils.cpp
    )

  add_executable(WindBarbSandbox
    GfxWSI/WSIAppBase.cpp
    GfxWSI/WindBarbSandbox/WindBarbSandbox.cpp
    GfxWSI/Background.cpp
    Utils/AttachmentUtils.cpp
    Utils/RenderPropertyUtils.cpp
    ${CMAKE_SOURCE_DIR}/QueryRenderer/Marks/WindBarbDefinitions.cpp
    ${CMAKE_SOURCE_DIR}/QueryRenderer/Marks/Enums.cpp
    ${CMAKE_SOURCE_DIR}/QueryRenderer/Utils/StringUtils.cpp
    )

  target_link_libraries(GfxWSITest GfxDriver Shared ${CMAKE_DL_LIBS})
  target_link_libraries(SymbolSandbox GfxDriver Shared ${CMAKE_DL_LIBS})
  target_link_libraries(WindBarbSandbox GfxDriver Shared ${CMAKE_DL_LIBS})

  add_custom_target(wsi_test
      COMMAND ${CMAKE_BINARY_DIR}/Tests/RenderTests/GfxWSITest
      DEPENDS GfxWSITest
      USES_TERMINAL)

  add_custom_target(symbol_sandbox
      COMMAND ${CMAKE_BINARY_DIR}/Tests/RenderTests/SymbolSandbox
      DEPENDS SymbolSandbox
      USES_TERMINAL)

  add_custom_target(windbarb_sandbox
      COMMAND ${CMAKE_BINARY_DIR}/Tests/RenderTests/WindBarbSandbox
      DEPENDS WindBarbSandbox
      USES_TERMINAL)

  set(RENDER_SANDBOXES
      GfxWSITest
      SymbolSandbox
      WindBarbSandbox)

  add_custom_target(render_sandboxes DEPENDS ${RENDER_SANDBOXES})

else()
  message(STATUS "GLFW3 not found - skipping WSI render tests")
endif()

# Symlink to build/QueryEngine to enable locating runtime files for native codegen and cuda (.bc and .fatbin)
add_custom_target(symlink_query_engine_target
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_BINARY_DIR}/../../QueryEngine ${CMAKE_CURRENT_BINARY_DIR}/../QueryEngine)

add_dependencies(QueryRendererTest symlink_query_engine_target)

add_custom_target(query_renderer_tests
    COMMAND ${TEST_COMMAND} --tests-regex "\"(QueryRendererTest)\""
    DEPENDS QueryRendererTest
    USES_TERMINAL)
