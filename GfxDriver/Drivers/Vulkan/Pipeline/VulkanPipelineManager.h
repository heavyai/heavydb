/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <map>
#include <vector>

#include <vulkan/vulkan.h>

#include "GfxDriver/Drivers/Vulkan/VulkanDeviceContext.h"
#include "GfxDriver/Pipeline/PushConstantRanges.h"

namespace gfx {

//
// VulkanPipelineManager class
//
// Helper class to manage pipeline layout and specializations for any Pipeline type
class VulkanPipelineManager {
 public:
  // Constructor
  // `specializations` vector declares all specialization constants in the pipeline
  // and will be translated and cached in specialization_map_
  // `push_constant_ranges` declares the push constants to add to the pipeline
  // which are translated and cached in push_constants_
  explicit VulkanPipelineManager(
      const DeviceContext& device_ctx,
      std::string_view name_base,
      const std::vector<SpecializationMapEntry>& specializations,
      const PushConstantRanges& push_constant_ranges);

  // Create a VkPipelineLayout for the given VkDescriptorSetLayout
  VkPipelineLayout createPipelineLayout(VkDescriptorSetLayout descriptor_set_layout);

  // Get the Vulkan handle for the VkSpecializationInfo struct to pass to
  // VkPipelineShaderStageCreateInfo during pipeline creation
  VkSpecializationInfo getSpecializationInfo(const void* specialization_data,
                                             const uint64_t data_size);
  // Add a specialized pipeline to pipeline_map_
  void insertPipeline(const uint32_t specialization_id, VkPipeline pipeline);

  // Get a pipeline from the map
  VkPipeline getPipelineHandle(const uint32_t specialization_id) const;

  // Destroy pipelines
  void destroyPipeline(const uint32_t specialization_id);
  void destroyPipelines();

 private:
  const VulkanDeviceContext& vk_device_;
  std::string name_base_;
  std::map<uint32_t, VkPipeline> pipeline_map_;
  std::vector<VkSpecializationMapEntry> specialization_map_;
  uint64_t specialization_data_size_;
  std::vector<VkPushConstantRange> push_constants_;
};

}  // namespace gfx
