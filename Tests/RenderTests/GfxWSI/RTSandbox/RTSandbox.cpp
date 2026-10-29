/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Tests/RenderTests/GfxWSI/RTSandbox/RTSandbox.h"

#include <iostream>

#include <tbb/parallel_for.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/mat4x4.hpp>

#include "GfxDriver/Pipeline/Material.h"
#include "GfxDriver/Render/PBRMaterialManager.h"
#include "Shared/measure.h"
#include "Tests/RenderTests/GfxWSI/RTSandbox/RTSandboxUI.h"

using namespace gfx;

static std::string general_help_text =
    R"general_help(Raytracing sandbox
Press 'u' to open the main UI and select the Help tab for interactive usage
)general_help";

constexpr auto kAccelInputBits = BufferUsageBits::kDeviceAddressBit |
                                 BufferUsageBits::kAccelerationStructureReadOnlyBit;

//
// RTSandbox
//
void RTSandbox::appPrintHelp(std::ostream& os) {
  os << general_help_text << "\n";
}

void RTSandbox::addOrModifyProgramOptions(WSIAppOptions& options) {
  // Override default options and reinit
#if DEMO_MODE
  options.window_width = 1920;
  options.window_height = 1080;
#else
  options.window_width = 2400;
  options.window_height = 1200;
#endif
  options.num_samples = 1;
  options.setOptions();

  show_resolution_in_title_ = false;
}

void RTSandbox::shaderLibraryInit(Library& library) {
  library.addFromManifestFile(
      "ShaderManifest.json",
      std::string(RENDER_TESTS_PATH) + "GfxWSI/RTSandbox/shaders/");
  library.addFromManifestFile("ShaderManifest.json",
                              std::string(GFX_DRIVER_PATH) + "Render/shaders/");
}

uint64_t generate_spheres_from_mesh(float sphere_radius,
                                    const Mesh& mesh,
                                    ResourceManager& resource_mgr,
                                    BufferWrapperUqPtr& procedurals_buffer,
                                    BufferWrapperUqPtr& bounds_buffer) {  // Procedurals
  struct Sphere {
    glm::vec3 center;
    float radius;
    Sphere() = default;
    Sphere(const glm::vec3& center, float radius) : center{center}, radius{radius} {}
  };
  std::vector<Sphere> procedurals;
  std::vector<AccelerationStructure::AABB> bounds;

  auto num_verts = mesh.numVertices();
  auto vert_stride = mesh.getNumAttrs();
  procedurals.resize(num_verts);
  bounds.resize(num_verts);

  tbb::parallel_for(
      tbb::blocked_range<uint32_t>(0, num_verts),
      [&](const tbb::blocked_range<uint32_t>& r) {
        const auto start_idx = r.begin();
        const auto end_idx = r.end();
        auto* vert_data = mesh.vertexData() + vert_stride * start_idx;
        for (uint32_t vi = start_idx; vi < end_idx; ++vi, vert_data += vert_stride) {
          procedurals[vi].center = glm::vec3{vert_data[0], vert_data[1], vert_data[2]};
          procedurals[vi].radius = sphere_radius;

          bounds[vi].min = glm::vec3(-sphere_radius) + procedurals[vi].center;
          bounds[vi].max = glm::vec3(sphere_radius) + procedurals[vi].center;
        }
      });

  auto procedurals_buffer_size = sizeof(Sphere) * procedurals.size();
  procedurals_buffer = resource_mgr.createBuffer(
      "procedurals",
      {BufferType::kUnspecified, procedurals_buffer_size, kAccelInputBits});

  auto bounds_buffer_size = bounds.size() * sizeof(AccelerationStructure::AABB);
  bounds_buffer = resource_mgr.createBuffer(
      "bounds", {BufferType::kUnspecified, bounds_buffer_size, kAccelInputBits});

  procedurals_buffer->updateSubData(procedurals.data(), procedurals_buffer_size, 0);
  bounds_buffer->updateSubData(bounds.data(), bounds_buffer_size, 0);

  return procedurals.size();
}

void RTSandbox::initObjects() {
  scene_stats_.reset();
  auto& resource_mgr = device_->getResourceManager();
  auto start_time = timer_start();

  auto& ui_state = ui_->getState();
  heightfield_mesh_ = build_heightfield(3.0f,
                                        3.0f,
                                        ui_state.hf_tesselation,
                                        ui_state.hf_tesselation,
                                        heightfield_callback_);

  timing_stats_.hf_mesh_build_time = timer_stop(start_time);
  scene_stats_.num_hf_vertices = heightfield_mesh_->numVertices();
  scene_stats_.num_hf_triangles = heightfield_mesh_->numTriangles();
  scene_stats_.total_vertices += scene_stats_.num_hf_vertices;
  scene_stats_.total_triangles += scene_stats_.num_hf_triangles;

  if (!ui_state.hf_as_spheres) {
    heightfield_resources_ =
        build_mesh_raytrace_data(resource_mgr, *heightfield_mesh_, "Heightfield");
  } else {
    auto start_time = timer_start();
    generate_spheres_from_mesh(
        2.0f / (float)ui_state.hf_tesselation * ui_state.hf_as_spheres_scale,
        *heightfield_mesh_,
        resource_mgr,
        procedurals_buffer_,
        procedural_aabb_buffer_);
    timing_stats_.hf_mesh_to_spheres_time = timer_stop(start_time);
  }

  sphere_mesh_ = build_uv_sphere(80, 80, true);
  sphere_resources_ = build_mesh_raytrace_data(resource_mgr, *sphere_mesh_, "Sphere");

  cube_mesh_ = build_cube(true);
  cube_resources_ = build_mesh_raytrace_data(resource_mgr, *cube_mesh_, "Cube");
}

void RTSandbox::initAccelerationStructures() {
  auto& resource_mgr = device_->getResourceManager();
  auto& ui_state = ui_->getState();

  // Create acceleration structure builder
  auto accel_builder = resource_mgr.createAccelerationStructureBuilder();

  // MeshObjectDescs
  std::vector<MeshObjectDesc> obj_descs;
  auto add_mesh = [&](const std::string& name, const Mesh& mesh, MeshRaytraceData& data) {
    accel_builder->clearBlasData();
    obj_descs.emplace_back(data.vbo_wrapper->getDeviceAddress(),
                           data.ibo_wrapper->getDeviceAddress(),
                           identity_matrix_4x4_address_,
                           identity_matrix_3x3_address_);

    accel_builder->addTriangleData({data.vbo_wrapper->getDeviceAddress(),
                                    mesh.numVertices(),
                                    data.layout->getNumBytesPerItem(),
                                    data.ibo_wrapper->getDeviceAddress(),
                                    IndexBufferDataType::kUnsigned32,
                                    mesh.numIndices(),
                                    0,  // transform
                                    mesh.numTriangles()});

    scene_stats_.total_vertices += mesh.numVertices();
    scene_stats_.total_triangles += mesh.numTriangles();

    return resource_mgr.createBottomLevelAccelerationStructure(name + " BLAS",
                                                               *accel_builder);
  };

  // Procedurals
  if (ui_state.hf_as_spheres) {
    auto start_time = timer_start();
    auto num_procedurals = generate_spheres_from_mesh(
        2.0f / (float)ui_state.hf_tesselation * ui_state.hf_as_spheres_scale,
        *heightfield_mesh_,
        resource_mgr,
        procedurals_buffer_,
        procedural_aabb_buffer_);
    timing_stats_.hf_mesh_to_spheres_time = timer_stop(start_time);

    accel_builder->clearBlasData();
    accel_builder->addAABBData({*procedural_aabb_buffer_,
                                (uint32_t)num_procedurals,
                                sizeof(AccelerationStructure::AABB)});
    scene_stats_.total_aabbs = num_procedurals;
    start_time = timer_start();
    procedural_blas_ = resource_mgr.createBottomLevelAccelerationStructure(
        "Procedural BLAS", *accel_builder);
    timing_stats_.hf_blas_build_time = timer_stop(start_time);
    obj_descs.emplace_back(0, 0, 0, 0);
  } else {
    timing_stats_.hf_mesh_to_spheres_time = 0;
    auto start_time = timer_start();
    heightfield_blas_ =
        add_mesh("Heightfield", *heightfield_mesh_, heightfield_resources_);
    timing_stats_.hf_blas_build_time = timer_stop(start_time);
  }
  sphere_blas_ = add_mesh("Sphere", *sphere_mesh_, sphere_resources_);
  cube_blas_ = add_mesh("Cube", *cube_mesh_, cube_resources_);

  //
  // TLAS
  //
  std::vector<uint32_t> material_indices;

  // Heightfield (ground plane)
  auto obj_tm = glm::mat4{1};
  if (ui_state.hf_as_spheres) {
    accel_builder->addInstanceData(obj_tm,
                                   procedural_blas_->getDeviceAddress(),
                                   0xff,
                                   0,
                                   kShadeProceduralHitGroupIndex);
  } else {
    accel_builder->addInstanceData(
        obj_tm, heightfield_blas_->getDeviceAddress(), 0xff, 0, 0);
  }
  material_indices.push_back(0);

  // Sphere mesh
  obj_tm = glm::translate(glm::mat4{1}, glm::vec3(-0.6f, -0.4f, 0.2f));
  obj_tm = glm::scale(obj_tm, glm::vec3{0.4f});
  obj_tm = glm::rotate(obj_tm, 0.0f, glm::vec3{0.0f, 1.0f, 0.0f});
  accel_builder->addInstanceData(
      glm::transpose(obj_tm), sphere_blas_->getDeviceAddress(), 0xff, 1, 0);
  material_indices.push_back(1);

  // Cube meshes
  obj_tm = glm::translate(glm::mat4{1}, glm::vec3{0.6f, -0.2f, 0.2f});
  obj_tm = glm::rotate(obj_tm, -0.6f, glm::vec3{0.0f, 1.0f, 0.0f});
  obj_tm = glm::scale(obj_tm, glm::vec3{0.1f});
  accel_builder->addInstanceData(
      glm::transpose(obj_tm), cube_blas_->getDeviceAddress(), 0xff, 2, 0);
  material_indices.push_back(2);

  obj_tm = glm::translate(glm::mat4{1}, glm::vec3{0.2f, -0.175f, -0.2f});
  obj_tm = glm::rotate(obj_tm, 0.5f, glm::vec3{0.0f, 1.0f, 0.0f});
  obj_tm = glm::scale(obj_tm, glm::vec3{0.15f});
  accel_builder->addInstanceData(
      glm::transpose(obj_tm), cube_blas_->getDeviceAddress(), 0xff, 2, 0);
  material_indices.push_back(3);

  // Create required instances buffer
  instances_buffer_ = resource_mgr.createBuffer("Instance data",
                                                {BufferType::kUnspecified,
                                                 accel_builder->getInstancesBufferSize(),
                                                 kAccelInputBits});

  // Top level acceleration structure
  tlas_ = resource_mgr.createTopLevelAccelerationStructure(
      "TLAS", *accel_builder, *instances_buffer_);

  // MeshObjDescs buffer
  obj_descs_buffer_ =
      resource_mgr.createBuffer("Obj info",
                                {BufferType::kUnspecified,
                                 sizeof(MeshObjectDesc) * obj_descs.size(),
                                 BufferUsageBits::kDeviceAddressBit});
  obj_descs_buffer_->updateSubData(
      obj_descs.data(), sizeof(MeshObjectDesc) * obj_descs.size(), 0);

  material_indices_buffer_ =
      resource_mgr.createBuffer("Material indices",
                                {BufferType::kUnspecified,
                                 sizeof(MeshObjectDesc) * obj_descs.size(),
                                 BufferUsageBits::kDeviceAddressBit});
  material_indices_buffer_->updateSubData(
      material_indices.data(), sizeof(uint32_t) * material_indices.size(), 0);
}

void RTSandbox::initMaterials() {
  auto& resource_mgr = device_->getResourceManager();
  auto const& shader_mgr = resource_mgr.getShaderManager();

  auto builders = shader_mgr.createBuilderVector({{"RTSandbox/RTSandbox.raygen"},
                                                  {"RTSandbox/RTSandbox.miss"},
                                                  {"RTSandbox/shadeTriangles.closest"},
                                                  {"Shading/rtShadowsSimple.miss"}});
  builders[0]->setExternalUniformBuffers({"PROJECTION_UBO_TYPE"});

  builders.push_back(shader_mgr.createBuilder("RTSandbox/sphere.isect"));
  builders.back()->setRaytracingHitGroupIndex(kShadeProceduralHitGroupIndex);
  builders.push_back(shader_mgr.createBuilder("RTSandbox/shadeProcedural.closest"));
  builders.back()->setRaytracingHitGroupIndex(kShadeProceduralHitGroupIndex);

  builders.push_back(shader_mgr.createBuilder("RTSandbox/showTriangles.closest"));
  builders.back()->setRaytracingHitGroupIndex(kShowTrianglesHitGroupIndex);

  auto caches = shader_mgr.createCacheVector(std::move(builders));

  material_ = device_->getResourceManager().createMaterial("RT Material", caches, true);

  scene_->bindBuffersToMaterial(*material_);

  // Update Material descriptors
  material_->setAccelerationStructureAttribute("topLevelAS", *tlas_);
  material_->setImageLoadStoreAttribute("outputImage",
                                        *render_targets_.textures[0].get());
  material_->bindExternalUniformBufferToBlock("PROJECTION_UBO_TYPE", *projection_ubo_);

  material_->updateDescriptorSets();
}

void RTSandbox::initPipelinesAndSBTs() {
  auto& resource_mgr = device_->getResourceManager();
  pipeline_ = resource_mgr.createRaytracingPipeline(
      "RT pipeline",
      *material_,
      {PushConstantRange{ShaderStageBits::kClosestHit | ShaderStageBits::kIntersection,
                         0u,
                         sizeof(PushConstants)}});
  pipeline_->create(2);

  sbt_ = resource_mgr.createShaderBindingTable(*pipeline_);
}

//
// App initialization
//
void RTSandbox::appInit() {
  //
  // Create render targets, framebuffer, and renderpasses
  //
  auto& resource_mgr = device_->getResourceManager();

  // attachment manager and textures
  auto color_format = wsi_->getWindowPixelFormat();
  render_targets_ = build_attachments(resource_mgr,
                                      window_width_,
                                      window_height_,
                                      {{color_format, Framebuffer::Attachment::kColor0}},
                                      "Framebuffer",
                                      num_samples_,
                                      ImageUsageBits::kStorageBit);

  // Renderpass and framebuffer
  render_pass_ = resource_mgr.createRenderPass("RenderPass",
                                               render_targets_.attachment_mgr.getLayout(),
                                               RenderPass::ClearBits::kNone,
                                               ImageLayout::kAttachment,
                                               ImageLayout::kAttachment);

  framebuffer_ = resource_mgr.createFramebuffer("Framebuffer",
                                                *render_pass_,
                                                render_targets_.attachment_mgr,
                                                window_width_,
                                                window_height_,
                                                num_samples_);

  auto& cmd_list = device_->getCommandList();
  for (auto& texture : render_targets_.textures) {
    cmd_list.clearTexture(*texture, ImageLayout::kAttachment);
  }
  cmd_list.flush("Init render targets");

  // Shared projection UBO
  projection_ubo_ = resource_mgr.createBuffer("Projection UBO",
                                              {BufferType::kUnspecified,
                                               Camera::getBufferDataSize(),
                                               BufferUsageBits::kUniformBufferBit,
                                               BufferAccessType::kHostVisible});

  // Identity matrix
  identity_matrix_buffer_ =
      resource_mgr.createBuffer("Identity Matrix",
                                {BufferType::kUnspecified,
                                 sizeof(glm::mat4) + sizeof(glm::mat3),
                                 BufferUsageBits::kDeviceAddressBit});

  glm::mat4 mat4{1};
  glm::mat3 mat3{1};
  identity_matrix_buffer_->updateSubData(&mat4, sizeof(glm::mat4), 0);
  identity_matrix_buffer_->updateSubData(&mat3, sizeof(glm::mat3), sizeof(glm::mat4));
  identity_matrix_4x4_address_ = identity_matrix_buffer_->getDeviceAddress();
  identity_matrix_3x3_address_ = identity_matrix_4x4_address_ + sizeof(glm::mat4);

  // Camera
  camera_ =
      std::make_unique<Camera>(Camera::Projection::kPerspective, Camera::Type::kLookAt);
  camera_->setPerspectiveParams(kDefaultFOV, 1.0f, 0.1f, 512.0f);
  camera_->setPosition({0.1f, -0.4f, 1.8f});
  camera_->setTarget(glm::vec3{0.0f});

  //
  // Init world
  //

  // PBR Scene
  scene_ = std::make_unique<PBRScene>(resource_mgr);

  // Lights
  scene_->addPointLight(
      glm::vec3{-20, -30, 20}, glm::vec3{0.8f, 0.75f, 0.7f}, 1.0f, false);
  scene_->addPointLight(glm::vec3{5, -2, 1}, glm::vec3{0.3, 0.5, 0.7}, 1.0f, false);

  scene_->updateLightDataBuffer();

  // PBRMaterial properties
  scene_->addCookTorranceMaterial(PBRMaterialDescriptor::ColorSource::kConstant,
                                  glm::vec4{glm::vec3{0.8f}, 1.0f},
                                  1.0f,
                                  0.25f,
                                  0.085f);

  scene_->addCookTorranceMaterial(PBRMaterialDescriptor::ColorSource::kConstant,
                                  glm::vec4{0.8f, 0.5f, 0.4f, 1.0f},
                                  1.0f,
                                  0.3f,
                                  0.6f);

  scene_->addCookTorranceMaterial(PBRMaterialDescriptor::ColorSource::kConstant,
                                  glm::vec4{0.2f, 0.7f, 0.3f, 1.0f},
                                  1.0f,
                                  0.04f,
                                  0.4f);

  scene_->addCookTorranceMaterial(PBRMaterialDescriptor::ColorSource::kConstant,
                                  glm::vec4{0.5f, 0.7f, 0.2f, 1.0f},
                                  1.0f,
                                  0.025f,
                                  0.5f);

  scene_->updateMaterialDataBuffer();

  //
  // Init UI
  //

  imgui_bridge_->initBackendAndResources(
      framebuffer_->getAttachmentManager().getLayout());

  wsi_->registerEventHandler([this](const WSIEvent& event) { handleWSIEvent(event); });

  ui_ = std::make_unique<RTSandboxUI>(*this,
                                      *device_,
                                      *wsi_,
                                      *imgui_bridge_,
                                      *camera_,
                                      *scene_,
                                      heightfield_callback_,
                                      timing_stats_,
                                      scene_stats_);

  initObjects();
  initAccelerationStructures();
  initMaterials();
  initPipelinesAndSBTs();

  ready_to_render_ = true;
}

void RTSandbox::updateUniforms() {
  auto& ui_state = ui_->getState();
  // Shared material properties
  scene_->updateMaterialDataBuffer();
  // General uniforms
  material_->setUniformAttribute("maxRayDepth", ui_state.max_ray_depth);
  material_->setUniformAttribute("numAASamples", ui_state.num_aa_samples);
  material_->setUniformAttribute("viewportOffset", ui_state.viewport_offset);
  material_->setUniformAttribute("doGlossyReflections", ui_state.do_glossy_reflections);
  material_->setUniformAttribute("numHitGroups", kNumHitGroups);
  if (ui_state.do_show_triangles) {
    material_->setUniformAttribute("hitGroupIndex", kShowTrianglesHitGroupIndex);
  } else {
    material_->setUniformAttribute("hitGroupIndex", kShadeTrianglesHitGroupIndex);
  }

  material_->setUniformAttribute("gradientStart", ui_state.bg_gradient_colors[0]);
  material_->setUniformAttribute("gradientEnd", ui_state.bg_gradient_colors[1]);
}

void RTSandbox::updateMatrices() {
  camera_->updateBufferData(*projection_ubo_);
}

//
// render
//

void RTSandbox::render() {
  if (!ready_to_render_) {
    return;
  }

  auto& ui_state = ui_->getState();

  // Update uniforms
  if (ui_state.uniforms_changed) {
    updateUniforms();
    ui_state.uniforms_changed = false;
  }
  if (ui_state.matrices_changed) {
    updateMatrices();
    ui_state.matrices_changed = false;
  }

  auto& cmd_list = device_->getCommandList();

  cmd_list.clearTexture(*render_targets_.textures[0].get(), ImageLayout::kGeneral)
      .flush("clear");

  //
  // Render scene
  //
  auto clock_start = timer_start();
  PushConstants push_constants{
      obj_descs_buffer_->getDeviceAddress(),
      material_indices_buffer_->getDeviceAddress(),
      ui_state.hf_as_spheres ? procedurals_buffer_->getDeviceAddress() : 0};

  cmd_list
      .setPushConstants(*pipeline_,
                        "push constants",
                        ShaderStageBits::kClosestHit | ShaderStageBits::kIntersection,
                        &push_constants,
                        sizeof(PushConstants))
      .traceRays(*pipeline_, *sbt_, ui_state.viewport_width, ui_state.viewport_height, 1)
      .flush("trace");

  // Add the time to the running total
  auto time = timer_stop_microseconds(clock_start);
  timing_stats_.total_time_ms += static_cast<double>(time) / 1000.0;
  timing_stats_.num_renders++;

  //
  // Draw UI
  //
  cmd_list
      .imageMemoryBarrier(*render_targets_.textures[0].get(),
                          ImageMemoryBarrierType::kImageLayout,
                          ImageLayout::kAttachment)
      .flush("layout");

  ui_->draw(*framebuffer_);

  // Display image
  wsi_->copyAndPresentTexture(*render_targets_.textures[0]);

  //
  // Process UI state for next frame
  //
  if (ui_state.do_animation) {
    // Orbit camera around scene
    glm::mat4 m = glm::rotate(0.01f, glm::vec3(0, 1, 0));
    camera_->setPosition(glm::vec4(camera_->getPosition(), 1.0f) * m);
    ui_state.matrices_changed = true;
  }

  maybeResetTime();

  if (ui_state.objects_changed) {
    // Rebuild geometry and acceleration structures
    destroyAccelerationStructures();
    destroyObjects();
    initObjects();
    initAccelerationStructures();
    material_->setAccelerationStructureAttribute("topLevelAS", *tlas_);
    material_->updateDescriptorSets();
    ui_state.objects_changed = false;
  }
}

//
// Resource cleanup
//

void RTSandbox::destroyAccelerationStructures() {
  auto& resource_mgr = device_->getResourceManager();

  if (heightfield_blas_) {
    resource_mgr.destroyAccelerationStructure(std::move(heightfield_blas_));
  }
  if (sphere_blas_) {
    resource_mgr.destroyAccelerationStructure(std::move(sphere_blas_));
  }
  if (cube_blas_) {
    resource_mgr.destroyAccelerationStructure(std::move(cube_blas_));
  }
  if (procedural_blas_) {
    resource_mgr.destroyAccelerationStructure(std::move(procedural_blas_));
  }
  if (tlas_) {
    resource_mgr.destroyAccelerationStructure(std::move(tlas_));
  }
}

void RTSandbox::destroyObjects() {
  auto& resource_mgr = device_->getResourceManager();

  if (instances_buffer_) {
    resource_mgr.destroyBuffer(std::move(instances_buffer_));
  }

  heightfield_resources_.destroyResources(resource_mgr);
  sphere_resources_.destroyResources(resource_mgr);
  cube_resources_.destroyResources(resource_mgr);

  if (obj_descs_buffer_) {
    resource_mgr.destroyBuffer(std::move(obj_descs_buffer_));
  }
  if (material_indices_buffer_) {
    resource_mgr.destroyBuffer(std::move(material_indices_buffer_));
  }
  if (procedurals_buffer_) {
    resource_mgr.destroyBuffer(std::move(procedurals_buffer_));
  }
  if (procedural_aabb_buffer_) {
    resource_mgr.destroyBuffer(std::move(procedural_aabb_buffer_));
  }
}

void RTSandbox::appShutdown() {
  auto& resource_mgr = device_->getResourceManager();

  destroyPipelinesAndMaterials();
  destroyAccelerationStructures();
  destroyObjects();

  if (render_pass_) {
    resource_mgr.destroyRenderPass(std::move(render_pass_));
  }

  if (identity_matrix_buffer_) {
    resource_mgr.destroyBuffer(std::move(identity_matrix_buffer_));
  }

  if (projection_ubo_) {
    resource_mgr.destroyBuffer(std::move(projection_ubo_));
  }

  scene_->destroyResources();
}

void RTSandbox::destroyPipelinesAndMaterials() {
  auto& resource_mgr = device_->getResourceManager();
  if (pipeline_) {
    resource_mgr.destroyPipeline(std::move(pipeline_));
  }

  material_ = nullptr;
}

//
// Reload shaders
//
// Will catch exceptions due to shader errors and stop rendering,
// allowing shaders to be fixed and reloaded again
//
void RTSandbox::reloadShaders() {
  LOG(INFO) << "Reloading shaders";
  ready_to_render_ = false;

  auto destroy_resources = [this]() { destroyPipelinesAndMaterials(); };

  destroy_resources();

  try {
    auto library = std::make_unique<Library>();
    shaderLibraryInit(*library);
    const_cast<ShaderManager&>(gfx_context_->getShaderManager())
        .replaceLibrary(std::move(library));

    initMaterials();
    initPipelinesAndSBTs();
  } catch (std::exception& e) {
    // cleanup anything that did complete
    destroy_resources();
    std::cout << "Error reloading shaders" << std::endl;
    return;
  }
  auto& ui_state = ui_->getState();
  ui_state.uniforms_changed = true;
  ui_state.matrices_changed = true;
  render_time_dirty_ = true;

  wsi_->notifyEvent(WSIWindowResizeEvent(window_width_, window_height_));
  ready_to_render_ = true;
}

//
// Reset time stats if necessary
//
void RTSandbox::maybeResetTime() {
  // Reset timing stats if anything changed that would invalidate current totals
  auto& ui_state = ui_->getState();
  if (ui_state.uniforms_changed || ui_state.matrices_changed ||
      ui_state.objects_changed || render_time_dirty_) {
    timing_stats_.num_renders = 0;
    timing_stats_.total_time_ms = 0.0;
  }
  render_time_dirty_ = false;
}

void RTSandbox::handleWSIEvent(const WSIEvent& event) {
  if (event.type() == WSIEvent::Type::kWindowResize) {
    if (material_) {
      material_->setImageLoadStoreAttribute("outputImage",
                                            *render_targets_.textures[0].get());
      material_->updateDescriptorSets();
    }
  }
}

DECLARE_WSI_APP_MAIN(RTSandbox)
