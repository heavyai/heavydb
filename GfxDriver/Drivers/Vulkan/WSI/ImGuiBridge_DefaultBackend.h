/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <imgui/imgui.h>
#include <vulkan/vulkan.h>

#include "GfxDriver/Resources/Types.h"
#include "GfxDriver/Types.h"

namespace gfx {

// Default backend for ImGuiBridge
// Wraps ImGui's Vulkan backend
// Does not allow for resource tracking
class ImGuiBridge_DefaultBackend {
 public:
  explicit ImGuiBridge_DefaultBackend(const DeviceContext& device,
                                      const WindowSystemIntegration& wsi);

  void init(const RenderPass& render_pass, const uint32_t num_samples);
  void shutdown();

  void newFrame();
  void draw(ImDrawData* draw_data, RenderPass& render_pass, Framebuffer& framebuffer);

 private:
  bool is_initialized_;
  const DeviceContext& device_;
  const WindowSystemIntegration& wsi_;
  // ImGui requires the application to pass a descriptor pool
  // Only a single combined image sampler is required
  VkDescriptorPool vk_descriptor_pool_;
};

}  // namespace gfx
