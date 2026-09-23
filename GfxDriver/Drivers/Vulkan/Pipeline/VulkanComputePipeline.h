/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "GfxDriver/Pipeline/Pipeline.h"

#include <vector>

#include <vulkan/vulkan.h>

#include "GfxDriver/Drivers/Vulkan/Pipeline/VulkanPipelineManager.h"

namespace gfx {

class VulkanComputePipeline : public ComputePipeline {
 public:
  explicit VulkanComputePipeline(
      const DeviceContext& device_ctx,
      std::string_view resource_tracking_string,
      const Material& material,
      const std::vector<SpecializationMapEntry>& specializations,
      const PushConstantRanges& push_constant_ranges);
  ~VulkanComputePipeline() override;

  ResourceHandle getPipelineHandle(const uint32_t specialization_id) const override;
  ResourceHandle getLayout() const override;

  void create() override;
  void createSpecialization(const uint32_t specialiation_id,
                            const void* specialization_data,
                            const uint64_t data_size) override;

 private:
  std::unique_ptr<VulkanPipelineManager> pipeline_mgr_;
  VkPipelineLayout pipeline_layout_;

  void cleanupResourceBase() override;
  void makeEmpty() override;
};

}  // namespace gfx
