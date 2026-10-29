/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "Tests/RenderTests/GfxWSI/WSIAppBase.h"

#include "GfxDriver/Pipeline/Pipeline.h"
#include "GfxDriver/Render/Camera.h"
#include "GfxDriver/Render/PBRScene.h"
#include "Tests/RenderTests/GfxWSI/RTSandbox/HeightfieldCallback.h"
#include "Tests/RenderTests/Utils/MeshUtils.h"

// Demo mode adjusts resolution and edit panel size for screen sharing
#define DEMO_MODE false

#if DEMO_MODE
static constexpr int kEditPanelWidth = 360;
#else
static constexpr int kEditPanelWidth = 640;
#endif

enum HitGroups {
  kShadeTrianglesHitGroupIndex,
  kShadeProceduralHitGroupIndex,
  kShowTrianglesHitGroupIndex,
  kNumHitGroups
};

// Defaults
static constexpr float kDefaultFOV = 75.0f;

// Timing
struct TimingStats {
  // FPS
  uint64_t num_renders{0};
  double total_time_ms{0.0};

  // Heightfield timing
  uint64_t hf_mesh_build_time{0};
  uint64_t hf_blas_build_time{0};
  uint64_t hf_mesh_to_spheres_time{0};
};

struct SceneStats {
  uint64_t total_vertices{0};
  uint64_t total_triangles{0};
  uint64_t total_aabbs{0};
  uint32_t num_hf_vertices{0};
  uint32_t num_hf_triangles{0};

  void reset() {
    total_vertices = 0;
    total_triangles = 0;
    total_aabbs = 0;
    num_hf_vertices = 0;
    num_hf_triangles = 0;
  }
};

class RTSandboxUI;

class RTSandbox : public gfx::WSIAppBase {
 public:
  ~RTSandbox() override = default;

  // WSIAppBase
  std::string_view getAppName() const override { return "RTSandbox"; }
  void appPrintHelp(std::ostream& os) override;
  void addOrModifyProgramOptions(gfx::WSIAppOptions& options) override;
  void shaderLibraryInit(gfx::Library& library) override;
  void appInit() override;
  void render() override;
  void appShutdown() override;

  // own functions
  void reloadShaders();

 private:
  struct PushConstants {
    gfx::DeviceAddress obj_descs;
    gfx::DeviceAddress colors;
    gfx::DeviceAddress spheres;
  };

  // RenderPasses
  gfx::resource_ptr<gfx::RenderPass> render_pass_;

  // Scene
  std::unique_ptr<gfx::PBRScene> scene_;

  //
  // Materials and Pipelines
  //
  gfx::MaterialUqPtr material_;
  gfx::resource_ptr<gfx::RaytracingPipeline> pipeline_;
  gfx::ShaderBindingTableUqPtr sbt_;
  gfx::BufferWrapperUqPtr obj_descs_buffer_;
  gfx::BufferWrapperUqPtr material_indices_buffer_;

  // Camera
  std::unique_ptr<gfx::Camera> camera_;
  gfx::BufferWrapperUqPtr projection_ubo_;

  // Meshes
  std::unique_ptr<gfx::Mesh> heightfield_mesh_;
  std::unique_ptr<gfx::Mesh> sphere_mesh_;
  std::unique_ptr<gfx::Mesh> cube_mesh_;
  gfx::MeshRaytraceData heightfield_resources_;
  gfx::MeshRaytraceData sphere_resources_;
  gfx::MeshRaytraceData cube_resources_;

  // Identity matrix buffer
  gfx::BufferWrapperUqPtr identity_matrix_buffer_;
  gfx::DeviceAddress identity_matrix_4x4_address_;
  gfx::DeviceAddress identity_matrix_3x3_address_;

  // Object data stuff
  std::vector<glm::mat3x4> blas_transforms_;  // row major (vulkan)
  std::vector<glm::mat4> model_transforms_;   // column major (glsl)
  std::vector<glm::mat3> normal_transforms_;

  // Procedurals
  gfx::BufferWrapperUqPtr procedurals_buffer_;      // sphere definitions for isect shader
  gfx::BufferWrapperUqPtr procedural_aabb_buffer_;  // AABB definitions for BLAS

  // Acceleration structures
  gfx::resource_ptr<gfx::AccelerationStructure> heightfield_blas_;
  gfx::resource_ptr<gfx::AccelerationStructure> sphere_blas_;
  gfx::resource_ptr<gfx::AccelerationStructure> cube_blas_;
  gfx::resource_ptr<gfx::AccelerationStructure> procedural_blas_;

  gfx::resource_ptr<gfx::AccelerationStructure> tlas_;
  gfx::BufferWrapperUqPtr instances_buffer_;

  // UI
  std::unique_ptr<RTSandboxUI> ui_;

  //
  // Internal state
  //
  bool ready_to_render_{false};

  // Heightfield
  HeightfieldCallback heightfield_callback_;

  // Dirty flags
  bool render_time_dirty_{true};

  // Stats
  TimingStats timing_stats_;
  SceneStats scene_stats_;

  void maybeResetTime();

  void initObjects();
  void initMaterials();
  void initPipelinesAndSBTs();
  void initAccelerationStructures();

  void updateUniforms();
  void updateMatrices();

  void handleWSIEvent(const gfx::WSIEvent& event);

  void destroyPipelinesAndMaterials();
  void destroyObjects();
  void destroyAccelerationStructures();
};
