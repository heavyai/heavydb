/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <imgui/imgui.h>
#include <glm/vec2.hpp>

#include "GfxDriver/Pipeline/Material.h"
#include "GfxDriver/Resources/ResourcePtr.h"
#include "GfxDriver/Resources/Types.h"
#include "GfxDriver/Types.h"

namespace gfx {

// Custom backend for ImGui using GfxDriver API
// EXPERIMENTAL - DOES NOT RENDER CORRECTLY
class ImGuiBridge_CustomBackend {
 public:
  ImGuiBridge_CustomBackend(const DeviceContext& device);

  void init(const RenderPass& render_pass, const RasterSampleCount num_samples);
  void shutdown();

  void newFrame() {}
  void draw(ImDrawData* draw_data, RenderPass& render_pass, Framebuffer& framebuffer);

 private:
  const DeviceContext& device_;

  uint32_t num_verts_;
  uint32_t num_indices_;

  std::unique_ptr<HostVisibleBufferWrapper> vertex_buffer_;
  std::unique_ptr<HostVisibleBufferWrapper> index_buffer_;
  void* vertex_buffer_mapped_;
  void* index_buffer_mapped_;

  std::unique_ptr<Material> material_;
  std::unique_ptr<PrimitiveAssembly> primitive_assembly_;
  resource_ptr<GraphicsPipeline> pipeline_;
  resource_ptr<Texture> font_texture_atlas_;

  struct {
    glm::vec2 scale;
    glm::vec2 translate;
  } PushConstants;

  void createFontAtlas();
  void updateDrawBuffers(const ImDrawData* draw_data);
};

}  // namespace gfx
