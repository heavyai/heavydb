/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Drivers/Vulkan/Pipeline/VulkanPipelineManager.h"

#include "GfxDriver/Drivers/Vulkan/Pipeline/Utils.h"
#include "GfxDriver/Drivers/Vulkan/VulkanResult.h"

namespace gfx {

VulkanPipelineManager::VulkanPipelineManager(
    const DeviceContext& device_ctx,
    const std::string_view name_base,
    const std::vector<SpecializationMapEntry>& specializations,
    const PushConstantRanges& push_constant_ranges)
    : vk_device_{static_cast<const VulkanDeviceContext&>(device_ctx)}
    , name_base_{name_base}
    , specialization_data_size_{0ull} {
  if (!specializations.empty()) {
    uint64_t max_offset_plus_size = 0ull;
    for (auto const& spec : specializations) {
      // translate specializations to their Vulkan counterparts
      VkSpecializationMapEntry new_entry = {};
      new_entry.constantID = spec.constant_id;
      new_entry.offset = spec.offset;
      new_entry.size = spec.size;
      specialization_map_.push_back(new_entry);

      // track the maximum offset + size for validating data size during specialization
      uint64_t offset_plus_size = spec.offset + spec.size;
      if (offset_plus_size > max_offset_plus_size) {
        max_offset_plus_size = offset_plus_size;
      }
    }
    specialization_data_size_ = max_offset_plus_size;
  }

  push_constants_ = push_constant_ranges_to_vk_push_constant_ranges(push_constant_ranges);
}

VkPipelineLayout VulkanPipelineManager::createPipelineLayout(
    VkDescriptorSetLayout descriptor_set_layout) {
  VkPipelineLayoutCreateInfo pipeline_layout_ci = {};
  pipeline_layout_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipeline_layout_ci.setLayoutCount = 1;

  // get descriptor set layout
  auto vk_descriptor_set_layout = descriptor_set_layout;
  pipeline_layout_ci.pSetLayouts = &vk_descriptor_set_layout;

  // push constants
  pipeline_layout_ci.pushConstantRangeCount = push_constants_.size();
  pipeline_layout_ci.pPushConstantRanges = push_constants_.data();

  VkPipelineLayout pipeline_layout{VK_NULL_HANDLE};
  CHECK_VKRESULT(vkCreatePipelineLayout(
                     vk_device_.getHandle(), &pipeline_layout_ci, NULL, &pipeline_layout),
                 "creating PipelineLayout");

  vk_device_.nameVulkanObject(
      VK_OBJECT_TYPE_PIPELINE_LAYOUT, pipeline_layout, name_base_);

  return pipeline_layout;
}

VkPipeline VulkanPipelineManager::getPipelineHandle(
    const uint32_t specialization_id) const {
  if (!pipeline_map_.empty()) {
    try {
      return pipeline_map_.at(specialization_id);
    } catch (std::out_of_range& err) {
      THROW_RUNTIME_EX("Pipeline specialization " + std::to_string(specialization_id) +
                       " out of range");
    }
  }
  return VK_NULL_HANDLE;
}

VkSpecializationInfo VulkanPipelineManager::getSpecializationInfo(
    const void* specialization_data,
    const uint64_t data_size) {
  VkSpecializationInfo spec_info = {};
  CHECK(!specialization_map_.empty());
  CHECK_EQ(data_size, specialization_data_size_);
  spec_info.pMapEntries = specialization_map_.data();
  spec_info.mapEntryCount = specialization_map_.size();
  spec_info.pData = specialization_data;
  spec_info.dataSize = data_size;
  return spec_info;
}

void VulkanPipelineManager::insertPipeline(const uint32_t specialization_id,
                                           VkPipeline new_pipeline) {
  auto [itr, did_insert] =
      pipeline_map_.insert_or_assign(specialization_id, new_pipeline);

  // Ensure it was added, otherwise destroy it and throw
  if (!did_insert || itr == pipeline_map_.end()) {
    vkDestroyPipeline(vk_device_.getHandle(), new_pipeline, nullptr);
    THROW_RUNTIME_EX("Failed to insert pipeline into pipeline map");
  }

  vk_device_.nameVulkanObject(VK_OBJECT_TYPE_PIPELINE,
                              new_pipeline,
                              name_base_ + "(" + std::to_string(specialization_id) + ")");
}

void VulkanPipelineManager::destroyPipeline(const uint32_t specialization_id) {
  if (pipeline_map_.count(specialization_id) > 0 &&
      pipeline_map_[specialization_id] != VK_NULL_HANDLE) {
    vkDestroyPipeline(vk_device_.getHandle(), pipeline_map_[specialization_id], nullptr);
    pipeline_map_[specialization_id] = VK_NULL_HANDLE;
  }
}

void VulkanPipelineManager::destroyPipelines() {
  // destroy existing pipeline
  for (auto& map_entry : pipeline_map_) {
    if (map_entry.second != VK_NULL_HANDLE) {
      vkDestroyPipeline(vk_device_.getHandle(), map_entry.second, nullptr);
      map_entry.second = VK_NULL_HANDLE;
    }
  }
  pipeline_map_.clear();
}

}  // namespace gfx
