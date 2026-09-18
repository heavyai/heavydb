/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "Tests/RenderTests/GfxWSI/WSIAppBase.h"

#include <array>

#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include "GfxDriver/Pipeline/Material.h"
#include "GfxDriver/Pipeline/Pipeline.h"
#include "GfxDriver/ShaderCompiler/Library.h"
#include "QueryRenderer/Marks/Enums.h"
#include "Tests/RenderTests/GfxWSI/Background.h"

//
// SymbolSandbox procedural symbol rendering sandbox app
//
// Future improvements:
// - Interactive multisampling settings (will require full app re-init)
// - Use VPmatrix to remove VBO rebuild requirement when resizing the window
//   Requires fixing pivot offsets (eg Wedge) to respect the VPmatrix
// - Random angle support. Require separate vertex buffer for geometry shader mode
//   since the non-pass through vertex shader doesn't handle the attributes
// - Program options for initial state
// - Profile run mode. Run for specified frame count and report avg times
// - Width / height / pivot controls
// - Fill / stroke toggles (requires material rebuild for blend function)
// - Angle unit selection
// - Editor: Vertex highlighting when hovering over sliders
// - Editor: Symbol flags
// - Editor: Add and remove vertices

class SymbolSandbox : public gfx::WSIAppBase {
 public:
  ~SymbolSandbox() override = default;

  std::string_view getAppName() const override { return "SymbolSandbox"; }
  void appPrintHelp(std::ostream& os) override;
  void addOrModifyProgramOptions(gfx::WSIAppOptions& options) override;
  void shaderLibraryInit(gfx::Library& library) override;
  void appInit() override;
  void render() override;
  void appShutdown() override;

  void handleWSIEvent(const gfx::WSIEvent& event);

 private:
  // RenderPasses
  gfx::resource_ptr<gfx::RenderPass> render_pass_clear_;
  gfx::resource_ptr<gfx::RenderPass> render_pass_;

  // Materials and Pipelines
  std::unique_ptr<gfx::Material> material_points_;
  std::unique_ptr<gfx::Material> material_geom_;
  gfx::resource_ptr<gfx::GraphicsPipeline> pipeline_points_;
  gfx::resource_ptr<gfx::GraphicsPipeline> pipeline_geom_;

  // Symbol vertex buffer distributions
  enum Distribution { kSingle, kGrid, kRandom };
  static constexpr int kNumDistributions = 3;

  gfx::InterleavedBufferLayoutShPtr buffer_layout_;
  struct DistributionData {
    std::string name;
    uint32_t num_symbols = 0;
    gfx::BufferWrapperUqPtr vbo;
    gfx::PrimitiveAssemblyUqPtr primitive_assembly;
    gfx::PrimitiveAssemblyUqPtr primitive_assembly_geom;
  };
  std::array<DistributionData, kNumDistributions> distribution_data_;

  // Background resources
  gfx::resource_ptr<gfx::GraphicsPipeline> pipeline_background_;
  std::unique_ptr<gfx::Material> material_background_;

  //
  // internal state
  //

  // Dirty flags
  bool uniforms_dirty_{true};
  bool view_uniforms_dirty_{true};
  bool single_distribution_dirty_{false};
  bool grid_distribution_dirty_{false};
  bool random_distribution_dirty_{false};
  bool render_time_dirty_{false};

  // Distribution properties
  Distribution distribution_{Distribution::kGrid};
  float eccentricity_{0.0f};
  int random_draw_count_{10000};
  float min_size_{2.0f};
  float max_size_{30.0f};

  QueryRenderer::SymbolShapeType single_symbol_type_{
      QueryRenderer::SymbolShapeType::kAirplane};
  bool use_geometry_shader_{false};
  float angle_degrees_{0.0f};

  // Symbol render properties
  float opacity_{1.0f};
  glm::vec4 fill_color_{0, 0.7f, 0.7f, 1};
  float fill_opacity_{0.8f};
  glm::vec4 stroke_color_{1};
  float stroke_opacity_{1.0f};
  float stroke_width_{3.0f};

  // Timing
  uint64_t num_renders_{0};
  double total_time_ms_{0.0};
  bool show_time_overlay_{true};

  // UI and Editor
  bool show_ui_{false};
  bool show_editor_{false};
  bool symbol_defs_dirty_{true};

  // Background
  std::unique_ptr<Background> background_;

  void maybeResetTime();
  void drawUI();
  void drawSymbolEditor();

  void initVertexBuffers();
  void initMaterialsAndPipelines();
  void destroyMaterialsAndPipelines();
  void reloadShaders();

  // SymbolVertex struct
  // Used to populate vertex buffers
  // Must match buffer_layout_
  struct SymbolVertex {
    float x;
    float y;
    float width;
    float height;
    uint32_t shape;
  };

  void createDistributionBuffer(DistributionData& data,
                                const std::vector<SymbolVertex>& vertices,
                                const std::string& name);
  void updateSingleDistribution();
  void updateGridDistribution();
  void updateRandomDistribution();

  void updateSymbolDefinitionUniforms();
  void updateUniforms();

  void handleWindowResizeEvent(const gfx::WSIWindowResizeEvent& event);
  void handleKeyboardEvent(const gfx::WSIKeyboardEvent& event);
};
