/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Drivers/Vulkan/Pipeline/VulkanComputePipeline.h"

#include "GfxDriver/Drivers/Vulkan/Pipeline/VulkanMaterial.h"
#include "GfxDriver/Drivers/Vulkan/VulkanDeviceContext.h"
#include "GfxDriver/Drivers/Vulkan/VulkanResult.h"
#include "Logger/Logger.h"

namespace gfx {

VulkanComputePipeline::VulkanComputePipeline(
    const DeviceContext& device_ctx,
    std::string_view resource_tracking_string,
    const Material& material,
    const std::vector<SpecializationMapEntry>& specializations,
    const PushConstantRanges& push_constant_ranges)
    : ComputePipeline(device_ctx, resource_tracking_string, material)
    , pipeline_mgr_{std::make_unique<VulkanPipelineManager>(device_ctx,
                                                            resource_tracking_string,
                                                            specializations,
                                                            push_constant_ranges)}
    , pipeline_layout_{VK_NULL_HANDLE} {}

VulkanComputePipeline::~VulkanComputePipeline() {
  cleanupResource();
}

ResourceHandle VulkanComputePipeline::getPipelineHandle(
    const uint32_t specialization_id) const {
  return reinterpret_cast<ResourceHandle>(
      pipeline_mgr_->getPipelineHandle(specialization_id));
}

ResourceHandle VulkanComputePipeline::getLayout() const {
  return reinterpret_cast<ResourceHandle>(pipeline_layout_);
}

void VulkanComputePipeline::cleanupResourceBase() {
  pipeline_mgr_->destroyPipelines();

  if (pipeline_layout_ != VK_NULL_HANDLE) {
    vkDestroyPipelineLayout(
        static_cast<const VulkanDeviceContext&>(getDeviceContext()).getHandle(),
        pipeline_layout_,
        nullptr);
    pipeline_layout_ = VK_NULL_HANDLE;
  }
}

void VulkanComputePipeline::makeEmpty() {}

void VulkanComputePipeline::create() {
  // Destroy all pipelines and the layout
  cleanupResourceBase();

  // Create new default pipeline and add to map
  createSpecialization(0u, nullptr, 0u);
}

void VulkanComputePipeline::createSpecialization(const uint32_t specialization_id,
                                                 const void* specialization_data,
                                                 const uint64_t data_size) {
  // destroy existing pipeline if present
  pipeline_mgr_->destroyPipeline(specialization_id);

  // get material
  auto const& vk_material = static_cast<const VulkanMaterial&>(material_);

  // get the compute shader
  auto const vk_shader_module_comp = vk_material.getShaderModule(ShaderStage::kCompute);
  CHECK(vk_shader_module_comp) << "Material does not have a Compute Shader";

  // create pipeline layout
  if (pipeline_layout_ == VK_NULL_HANDLE) {
    pipeline_layout_ =
        pipeline_mgr_->createPipelineLayout(vk_material.getDescriptorSetLayout());
  }

  // create shader stage
  VkPipelineShaderStageCreateInfo shader_stage_ci = {};

  shader_stage_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  shader_stage_ci.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  shader_stage_ci.module = vk_shader_module_comp;
  shader_stage_ci.pName = vk_material.getEntryPoint(ShaderStage::kCompute);

  // handle specialization constants
  VkSpecializationInfo spec_info = {};
  if (specialization_data) {
    spec_info = pipeline_mgr_->getSpecializationInfo(specialization_data, data_size);
    shader_stage_ci.pSpecializationInfo = &spec_info;
  }

  // define pipeline
  VkComputePipelineCreateInfo pipeline_ci = {};
  pipeline_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipeline_ci.stage = shader_stage_ci;
  pipeline_ci.layout = pipeline_layout_;

  // create pipeline
  VkPipeline new_pipeline = VK_NULL_HANDLE;
  CHECK_VKRESULT(
      vkCreateComputePipelines(
          static_cast<const VulkanDeviceContext&>(getDeviceContext()).getHandle(),
          VK_NULL_HANDLE,
          1,
          &pipeline_ci,
          nullptr,
          &new_pipeline),
      "creating Compute Pipeline");

  // Add it to the map
  pipeline_mgr_->insertPipeline(specialization_id, new_pipeline);

  setUsable();
}

}  // namespace gfx
