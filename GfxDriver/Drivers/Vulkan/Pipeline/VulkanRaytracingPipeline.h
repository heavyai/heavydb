/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "GfxDriver/Pipeline/Pipeline.h"

#include <vulkan/vulkan.h>

#include "GfxDriver/Drivers/Vulkan/Pipeline/VulkanPipelineManager.h"

namespace gfx {

//
// VulkanRaytracingPipeline
//
// Compiles and packages all the shaders that may be invoked by a traceRays command
// (primary and secondary rays)
// ShaderGroups can be retrieved after create() has been called
class VulkanRaytracingPipeline : public RaytracingPipeline {
 public:
  explicit VulkanRaytracingPipeline(const DeviceContext& device_ctx,
                                    std::string_view resource_tracking_string,
                                    const Material& material,
                                    const PushConstantRanges& push_constant_ranges);
  ~VulkanRaytracingPipeline() override;

  ResourceHandle getPipelineHandle(const uint32_t specialization_id) const override;
  ResourceHandle getLayout() const override;

  void create(uint32_t max_ray_recursion_depth) override;

  const std::vector<ShaderGroup>& getShaderGroups() const override;

 private:
  std::unique_ptr<VulkanPipelineManager> pipeline_mgr_;
  VkPipelineLayout pipeline_layout_;
  std::vector<VkRayTracingShaderGroupCreateInfoKHR> vk_shader_groups_;
  std::vector<ShaderGroup> shader_groups_;

  void cleanupResourceBase() override;
  void makeEmpty() override;
};

}  // namespace gfx
