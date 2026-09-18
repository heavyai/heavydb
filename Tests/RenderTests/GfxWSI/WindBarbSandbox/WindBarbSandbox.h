/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "Tests/RenderTests/GfxWSI/WSIAppBase.h"

#include <array>

#include <glm/vec2.hpp>
#include <glm/vec4.hpp>

#include "GfxDriver/Pipeline/Material.h"
#include "GfxDriver/Pipeline/Pipeline.h"
#include "GfxDriver/ShaderCompiler/Library.h"
#include "Tests/RenderTests/GfxWSI/Background.h"

//
// WindBarb rendering sandbox app
//
// 3 modes: single barb, grid showing all but 3 pennants, and vector field
// Modes 1 and 2 share Material/Pipeline setup, testing direction as a uniform
// Mode 3 (field) add direction and color properties to the vertex data

struct DistributionData {
  std::string name;
  uint32_t num_barbs = 0;
  gfx::InterleavedBufferLayoutShPtr buffer_layout;
  gfx::BufferWrapperUqPtr vbo;
  gfx::PrimitiveAssemblyUqPtr primitive_assembly;
};

class WindBarbSandbox : public gfx::WSIAppBase {
 public:
  ~WindBarbSandbox() override = default;

  std::string_view getAppName() const override { return "Wind Barbs Sandbox"; }
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
  std::unique_ptr<gfx::Material> material_;
  gfx::resource_ptr<gfx::GraphicsPipeline> pipeline_;
  std::unique_ptr<gfx::Material> field_material_;
  gfx::resource_ptr<gfx::GraphicsPipeline> field_pipeline_;

  // Barb vertex buffer distributions
  enum Distribution { kSingle, kGrid, kField };
  static constexpr int kNumDistributions = 3;

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
  bool field_distribution_dirty_{false};
  bool render_time_dirty_{false};

  // Distribution properties
  Distribution distribution_{Distribution::kGrid};
  uint32_t single_barb_speed_{145};  // 2 pennants and max barbs
  int32_t field_density_{30};
  int32_t field_draw_count_{900};  // density squared
  float field_min_speed_{0.0f};
  float field_max_speed_{145.0f};
  float field_barb_size_{60.0f};
  float field_noise_size_{1.5f};
  glm::vec2 field_noise_center_{0.0f, 0.0f};
  bool field_use_auto_color_{false};

  float direction_{0.0f};
  bool do_quantize_direction_{false};

  // Barb render properties
  bool show_billboards_{false};
  float anchor_scale_{0.5f};
  float opacity_{1.0f};
  glm::vec4 fill_color_{0, 0, 0, 1};
  glm::vec4 stroke_color_{0, 0, 0, 1};
  float stroke_width_{2.0f};

  // Timing
  uint64_t num_renders_{0};
  double total_time_ms_{0.0};
  bool show_time_overlay_{false};

  // UI
  bool show_ui_{false};
  bool is_capturing_mouse_{false};
  glm::vec2 last_cursor_;

  // Background
  std::unique_ptr<Background> background_;

  void maybeResetTime();
  void drawUI();

  void initVertexBuffers();
  void initMaterialsAndPipelines();
  void destroyMaterialsAndPipelines();
  void reloadShaders();

  void updateSingleDistribution();
  void updateGridDistribution();
  void updateFieldDistribution();

  void updateBarbDefinitionUniforms(gfx::Material& material);
  void updateUniforms();

  void handleWindowResizeEvent(const gfx::WSIWindowResizeEvent& event);
  void handleMouseButtonEvent(const gfx::WSIMouseButtonEvent& event);
  void handleCursorEvent(const gfx::WSIMouseCursorEvent& event);
  void handleKeyboardEvent(const gfx::WSIKeyboardEvent& event);
  void handleScrollEvent(const gfx::WSIScrollEvent& event);
};
