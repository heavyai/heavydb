# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

set(GFXDRIVER_TYPE "Vulkan" CACHE STRING "GfxDriver type")
set_property(CACHE GFXDRIVER_TYPE PROPERTY STRINGS "Vulkan")

if(ENABLE_RENDER_LOGGER)
  message(STATUS "Render logging enabled for GfxDriver")
  add_compile_definitions("ENABLE_RENDER_LOGGER")
endif()
if(ENABLE_SHADER_DEBUG_PRINTF)
  message(STATUS "Shader Debug printf extension enabled for GfxDriver")
  add_compile_definitions("ENABLE_SHADER_DEBUG_PRINTF")
endif()

set(GFXDRIVER_SOURCES
  DeviceContext.cpp
  DriverInstance.cpp
  Enums.cpp
  GfxContext.cpp
  RenderLogger.cpp
  TypeGLSL.cpp)

list(APPEND GFXDRIVER_SOURCES
  WSI/Enums.cpp
  WSI/WindowSystemIntegration.cpp
  WSI/Events.cpp
  WSI/Icons/HeavyAIIcon.cpp)

list(APPEND GFXDRIVER_SOURCES
  Resources/AttachmentManager.cpp
  Resources/Buffer.cpp
  Resources/BufferLayout.cpp
  Resources/BufferWrapper.cpp
  Resources/Enums.cpp
  Resources/Framebuffer.cpp
  Resources/IndexBuffer.cpp
  Resources/IndirectDrawBuffer.cpp
  Resources/HostVisibleBufferWrapper.cpp
  Resources/PixelBuffer2d.cpp
  Resources/RenderPass.cpp
  Resources/Resource.cpp
  Resources/ResourceManager.cpp
  Resources/ShaderBlockLayout.cpp
  Resources/Texture.cpp
  Resources/Types.cpp
  Resources/VertexBuffer.cpp)

list(APPEND GFXDRIVER_SOURCES
  Pipeline/Material.cpp
  Pipeline/Pipeline.cpp
  Pipeline/PipelineDescriptor.cpp
  Pipeline/PrimitiveAssembly.cpp
  Pipeline/BufferLayoutManager.cpp
  Pipeline/PushConstantRanges.cpp
  Pipeline/ShaderBindingTable.cpp)

list(APPEND GFXDRIVER_SOURCES
  Commands/MemoryArena.cpp
  Commands/CommandList.cpp
  Commands/CommandImpls.cpp
  Commands/QueryPool.cpp)

list(APPEND GFXDRIVER_SOURCES
  Colors/BaseColor4f.cpp
  Colors/ColorRGBA.cpp
  Colors/ColorHSL.cpp
  Colors/ColorLAB.cpp
  Colors/ColorHCL.cpp
  Colors/ColorUnion.cpp
  Colors/Utils.cpp
  Colors/GlmUtils.cpp
  Colors/Types.cpp)

list(APPEND GFXDRIVER_SOURCES
  Objects/TileBuilder.cpp)

list(APPEND GFXDRIVER_SOURCES
  ShaderCompiler/JSONSchemas.cpp
  ShaderCompiler/Library.cpp
  ShaderCompiler/ShaderManager.cpp
  ShaderCompiler/ShaderRedecorator.cpp
  ShaderCompiler/ShaderReflection.cpp
  ShaderCompiler/ShaderCache.cpp
  ShaderCompiler/Types.cpp
  ShaderCompiler/GlslStructBuilder.cpp
  ShaderCompiler/ShaderCompilerStats.cpp)

list(APPEND GFXDRIVER_SOURCES
  Render/PPLLResources.cpp
  Render/PPLLRender.cpp
  Render/Camera.cpp
  Render/LightManager.cpp
  Render/PBRMaterialManager.cpp
  Render/PBRScene.cpp
  Render/GeoCountResources.cpp)

list(APPEND GFXDRIVER_SOURCES
  RenderDoc/RenderDoc.cpp)

list(APPEND GFXDRIVER_SOURCES
  Utils/ImageUtils.cpp
  Utils/NoiseUtils.cpp
  Utils/MaterialUtils.cpp
  Utils/LoggingUtils.cpp)

set(GFXDRIVER_DEFS "" PARENT_SCOPE)
set(GFXDRIVER_INCLUDE_DIRS "")
set(GFXDRIVER_LIBS "")

find_package(Glslang REQUIRED)

# Extension enabling. Must match flags specified when building glslang
add_definitions(-DNV_EXTENSIONS -DAMD_EXTENSIONS)
list(APPEND GFXDRIVER_SOURCES
  ShaderCompiler/GlslangWrapper.cpp
  ShaderCompiler/TShaderIRUtils.cpp
  ShaderCompiler/ResourceLimits.cpp)
list(APPEND GFXDRIVER_LIBS ${Glslang_LIBRARIES})

find_package(SpirvCross REQUIRED)
list(APPEND GFXDRIVER_SOURCES
  ShaderCompiler/SpirvCrossUtils.cpp
  ShaderCompiler/SpirvArtifacts.cpp)

list(APPEND GFXDRIVER_LIBS ${SpirvCross_LIBRARIES})

find_package(absl REQUIRED)
list(APPEND GFXDRIVER_LIBS absl::strings)

add_compile_definitions(VK_INSTANCE_APP_VERS_MAJOR=${MAPD_VERSION_MAJOR})
add_compile_definitions(VK_INSTANCE_APP_VERS_MINOR=${MAPD_VERSION_MINOR})
add_compile_definitions(VK_INSTANCE_APP_VERS_PATCH=${MAPD_VERSION_PATCH})
list(APPEND GFXDRIVER_SOURCES
  Drivers/Vulkan/VulkanDriver.cpp
  Drivers/Vulkan/VulkanDebugUtils.cpp
  Drivers/Vulkan/VulkanPlatform.cpp
  Drivers/Vulkan/VulkanPlatformUtils.cpp
  Drivers/Vulkan/VulkanResult.cpp
  Drivers/Vulkan/VulkanPhysicalDevice.cpp
  Drivers/Vulkan/VulkanQueue.cpp
  Drivers/Vulkan/VulkanMemoryMgr.cpp
  Drivers/Vulkan/VulkanMemoryUtils.cpp
  Drivers/Vulkan/VulkanDeviceContext.cpp)

list(APPEND GFXDRIVER_SOURCES
  Drivers/Vulkan/Resources/ImageLayoutManager.cpp
  Drivers/Vulkan/Resources/Utils.cpp
  Drivers/Vulkan/Resources/VulkanBaseBuffer.cpp
  Drivers/Vulkan/Resources/VulkanHostVisibleBufferWrapper.cpp
  Drivers/Vulkan/Resources/VulkanPixelBuffer2d.cpp
  Drivers/Vulkan/Resources/VulkanRenderPass.cpp
  Drivers/Vulkan/Resources/VulkanResourceManager.cpp
  Drivers/Vulkan/Resources/VulkanShaderModule.cpp
  Drivers/Vulkan/Resources/VulkanTexture.cpp
  Drivers/Vulkan/Resources/VulkanFramebuffer.cpp
  Drivers/Vulkan/Resources/VulkanAccelerationStructure.cpp)

list(APPEND GFXDRIVER_SOURCES
  Drivers/Vulkan/Pipeline/VulkanPipelineManager.cpp
  Drivers/Vulkan/Pipeline/VulkanComputePipeline.cpp
  Drivers/Vulkan/Pipeline/VulkanGraphicsPipeline.cpp
  Drivers/Vulkan/Pipeline/VulkanRaytracingPipeline.cpp
  Drivers/Vulkan/Pipeline/VulkanMaterial.cpp
  Drivers/Vulkan/Pipeline/VulkanPrimitiveAssembly.cpp
  Drivers/Vulkan/Pipeline/VulkanShaderBindingTable.cpp
  Drivers/Vulkan/Pipeline/Utils.cpp)

list(APPEND GFXDRIVER_SOURCES
  Drivers/Vulkan/Commands/FenceManager.cpp
  Drivers/Vulkan/Commands/VulkanCommandBuffers.cpp
  Drivers/Vulkan/Commands/VulkanCommandExecutor.cpp
  Drivers/Vulkan/Commands/VulkanCommandExecutionContext.cpp
  Drivers/Vulkan/Commands/StagingContext.cpp
  Drivers/Vulkan/Commands/VulkanQueryPool.cpp)

list(APPEND GFXDRIVER_SOURCES
  Drivers/Vulkan/WSI/VulkanSwapchain.cpp
  Drivers/Vulkan/WSI/VulkanWSI.cpp)

find_package(GLFW3)

if(GLFW3_FOUND AND NOT PREFER_STATIC_LIBS)
  list(APPEND GFXDRIVER_LIBS ${GLFW3_LIBRARIES})
  list(APPEND GFXDRIVER_DEFS "-DHAVE_GLFW")
  list(APPEND GFXDRIVER_SOURCES
    Drivers/Vulkan/WSI/VulkanWSI_GLFW.cpp)

  # ImGui support (required if GLFW found)
  find_package(ImGui REQUIRED)
  add_compile_definitions(IMGUI_FONT_PATH=\"${ImGui_PATH}/misc/fonts/\")

  # Build as a separate library to allow warning suppression
  set(ImGuiSourceFiles
    ${ImGui_PATH}/imgui.cpp
    ${ImGui_PATH}/imgui_draw.cpp
    ${ImGui_PATH}/imgui_widgets.cpp
    ${ImGui_PATH}/imgui_tables.cpp
    ${ImGui_PATH}/imgui_demo.cpp
    ${ImGui_PATH}/backends/imgui_impl_vulkan.cpp
    ${ImGui_PATH}/backends/imgui_impl_glfw.cpp)

  add_library(GfxImGui ${ImGuiSourceFiles})
  target_include_directories(GfxImGui PRIVATE ${ImGui_PATH})

  if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    target_compile_options(GfxImGui PRIVATE -Wno-class-memaccess)
  endif()

  list(APPEND GFXDRIVER_INCLUDE_DIRS ${ImGui_PATH})
  list(APPEND GFXDRIVER_LIBS GfxImGui)

  # Add ImGuiBridge to WSI files
  list(APPEND GFXDRIVER_SOURCES
    Drivers/Vulkan/WSI/ImGuiBridge.cpp
    Drivers/Vulkan/WSI/ImGuiBridge_DefaultBackend.cpp
    Drivers/Vulkan/WSI/ImGuiBridge_CustomBackend.cpp)
endif()

# Vulkan required packages
find_package(Vulkan REQUIRED)
list(APPEND GFXDRIVER_INCLUDE_DIRS ${VULKAN_INCLUDE_DIR})
list(APPEND GFXDRIVER_LIBS ${VULKAN_LIBRARY})

# EGL is still required for Vulkan bootstrap on NVIDIA systems
# Explicitly add includes and libs for EGL only, rather than the
# entire OpenGL::EGL bundle, as the GLVND package on Rocky also 
# pulls in OpenGL, causing that to also be a run-time dependency
find_package(OpenGL REQUIRED COMPONENTS EGL)
list(APPEND GFXDRIVER_INCLUDE_DIRS ${OPENGL_EGL_INCLUDE_DIRS})
list(APPEND GFXDRIVER_LIBS ${OPENGL_egl_LIBRARY})

# only enable shader compiler artifacts during development (pre-release candidate)
if("${MAPD_VERSION_EXTRA}" STREQUAL "dev")
  list(APPEND GFXDRIVER_DEFS "-DSHADER_ARTIFACTS_ENABLED")
endif()

add_library(GfxDriver ${GFXDRIVER_SOURCES})
target_include_directories(GfxDriver PRIVATE ${GFXDRIVER_INCLUDE_DIRS})
target_link_libraries(GfxDriver ${GFXDRIVER_LIBS})
target_compile_definitions(GfxDriver PRIVATE ${GFXDRIVER_DEFS})

# expose the definitions to the parent
set(GFXDRIVER_INCLUDE_DIRS ${GFXDRIVER_INCLUDE_DIRS} PARENT_SCOPE)
set(GFXDRIVER_DEFS ${GFXDRIVER_DEFS} PARENT_SCOPE)
