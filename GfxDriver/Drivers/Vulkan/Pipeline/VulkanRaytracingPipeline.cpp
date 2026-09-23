/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Drivers/Vulkan/Pipeline/VulkanRaytracingPipeline.h"

#include "GfxDriver/Drivers/Vulkan/Pipeline/VulkanMaterial.h"
#include "GfxDriver/Drivers/Vulkan/Resources/Utils.h"
#include "GfxDriver/Drivers/Vulkan/Resources/VulkanShaderModule.h"
#include "GfxDriver/Drivers/Vulkan/VulkanDeviceContext.h"
#include "GfxDriver/Drivers/Vulkan/VulkanResult.h"

namespace gfx {

VulkanRaytracingPipeline::VulkanRaytracingPipeline(
    const DeviceContext& device_ctx,
    std::string_view resource_tracking_string,
    const Material& material,
    const PushConstantRanges& push_constant_ranges)
    : RaytracingPipeline(device_ctx, resource_tracking_string, material)
    , pipeline_mgr_{std::make_unique<VulkanPipelineManager>(
          device_ctx,
          resource_tracking_string,
          std::vector<SpecializationMapEntry>{},
          push_constant_ranges)}
    , pipeline_layout_{VK_NULL_HANDLE} {}

VulkanRaytracingPipeline::~VulkanRaytracingPipeline() {
  cleanupResource();
}

ResourceHandle VulkanRaytracingPipeline::getPipelineHandle(
    const uint32_t specialization_id) const {
  return reinterpret_cast<ResourceHandle>(
      pipeline_mgr_->getPipelineHandle(specialization_id));
}

ResourceHandle VulkanRaytracingPipeline::getLayout() const {
  return reinterpret_cast<ResourceHandle>(pipeline_layout_);
}

void VulkanRaytracingPipeline::cleanupResourceBase() {
  pipeline_mgr_->destroyPipelines();
  if (pipeline_layout_ != VK_NULL_HANDLE) {
    vkDestroyPipelineLayout(
        static_cast<const VulkanDeviceContext&>(getDeviceContext()).getHandle(),
        pipeline_layout_,
        nullptr);
    pipeline_layout_ = VK_NULL_HANDLE;
  }
}

void VulkanRaytracingPipeline::makeEmpty() {}

void VulkanRaytracingPipeline::create(uint32_t max_ray_recursion_depth) {
  const VulkanDeviceContext& vk_device =
      static_cast<const VulkanDeviceContext&>(getDeviceContext());
  auto const& vk_material = static_cast<const VulkanMaterial&>(material_);
  VkResult result = VK_SUCCESS;

  // Destroy all pipelines and the layout
  cleanupResourceBase();

  // create pipeline layout
  if (pipeline_layout_ == VK_NULL_HANDLE) {
    pipeline_layout_ =
        pipeline_mgr_->createPipelineLayout(vk_material.getDescriptorSetLayout());
  }

  //
  // Iterate ShaderModules and build ShaderGroup infos
  //
  // ShaderBindingTables (SBTs) are tables of shader 'handles' stored in a buffer
  // The traceRays command requires a SBT for each general ray interaction:
  // raygen (root shader), miss, hit (any, closest, intersection), and callable
  // The RaytracingPipeline maps VkShaderModule handles within the pipeline
  // to ShaderGroups, each of which stores an index into the shader modules for each
  // of the parts of ray traversal: general (generation / miss), any hit and closest hit
  // processing, and intersection processing for AABB data
  // Hit and intersection shaders for a raytrace command must be stored together in
  // the same ShaderGroup (a hit-group)
  // Note that shaders emitting rays can specify offsets into the SBT and alter
  // the miss shader index to facilitate having different behaviors per ray "type".
  auto const& shader_modules = vk_material.getShaderModuleMap();
  bool found_raygen_shader = false;

  auto init_vk_shader_group = [&]() -> auto{
    VkRayTracingShaderGroupCreateInfoKHR group{};
    group.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    group.generalShader = VK_SHADER_UNUSED_KHR;
    group.closestHitShader = VK_SHADER_UNUSED_KHR;
    group.anyHitShader = VK_SHADER_UNUSED_KHR;
    group.intersectionShader = VK_SHADER_UNUSED_KHR;
    return group;
  };

  auto init_shader_stage_ci = [&](ShaderStage stage,
                                  const VulkanShaderModule& module) -> auto{
    VkPipelineShaderStageCreateInfo shader_stage_ci = {};
    shader_stage_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shader_stage_ci.stage = shader_stage_to_vk_shader_stage_flag(stage);
    shader_stage_ci.module = reinterpret_cast<VkShaderModule>(module.getResourceHandle());
    shader_stage_ci.pName = module.getEntryPoint();

    return shader_stage_ci;
  };

  // Find general and callable stages and sift out hit groups
  // map of hit group indices to shader info
  std::multimap<uint32_t, VulkanShaderModule*> hit_group_map;
  // set of hit group indices
  std::set<uint32_t> hit_group_indices;
  std::vector<VkPipelineShaderStageCreateInfo> shader_stage_cis;

  for (auto const& [shader_stage, shader_module] : shader_modules) {
    CHECK(shader_module);
    switch (shader_stage) {
      //
      // Create General groups
      //
      case ShaderStage::kRayGen:
        found_raygen_shader = true;
        [[fallthrough]];
      case ShaderStage::kMiss:
      case ShaderStage::kCallable: {
        // Add ShaderGroup
        CHECK_EQ(shader_module->getRaytracingHitGroupIndex(), 0u)
            << "hit group index set on stage '" << to_string(shader_stage)
            << "' which is not a valid hit group stage";
        auto group = init_vk_shader_group();
        group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
        group.generalShader = shader_stage_cis.size();
        vk_shader_groups_.push_back(group);
        shader_groups_.emplace_back(ShaderGroup(ShaderGroup::Type::kGeneral,
                                                shader_stage_to_sbt_entry(shader_stage)));
        // Add ShaderStage
        auto shader_stage_ci = init_shader_stage_ci(shader_stage, *shader_module);
        shader_stage_cis.push_back(shader_stage_ci);
      } break;

      //
      // Collect hit-groups
      //
      case ShaderStage::kClosestHit:
      case ShaderStage::kAnyHit:
      case ShaderStage::kIntersection: {
        auto index = shader_module->getRaytracingHitGroupIndex();
        hit_group_map.insert({index, shader_module.get()});
        hit_group_indices.insert(index);
        continue;  // skip Vulkan shader group creation for hit-group shaders
      }

      default:
        CHECK(false) << "Unsupported shader stage: " << to_string(shader_stage);
    }
  }
  // Ensure we at least have a raygen shader
  // TODO(scb): we may want to enforce presence of at least 1 other stage
  CHECK(found_raygen_shader) << "A raygen shader is required";

  // Build hit groups
  for (auto const hit_group_index : hit_group_indices) {
    auto group = init_vk_shader_group();

    // iterate over shader modules that share hit_group_index
    auto const& [begin, end] = hit_group_map.equal_range(hit_group_index);
    CHECK(begin != hit_group_map.end());

    for (auto itr = begin; itr != end; ++itr) {
      auto const* shader_module = itr->second;
      CHECK(shader_module);
      auto shader_stage = shader_module->getShaderStage();
      uint32_t stage_index = shader_stage_cis.size();
      switch (shader_stage) {
        case ShaderStage::kClosestHit:
          CHECK(group.closestHitShader == VK_SHADER_UNUSED_KHR);
          group.closestHitShader = stage_index;
          break;
        case ShaderStage::kAnyHit:
          CHECK(group.anyHitShader == VK_SHADER_UNUSED_KHR);
          group.anyHitShader = stage_index;
          break;
        case ShaderStage::kIntersection:
          CHECK(group.intersectionShader == VK_SHADER_UNUSED_KHR);
          group.intersectionShader = stage_index;
          break;
        default:
          CHECK(false) << "Hit ShaderGroup contains unsupported shader stage: "
                       << to_string(shader_stage);
      }
      // Add ShaderStage
      shader_stage_cis.push_back(init_shader_stage_ci(shader_stage, *shader_module));
    }

    // Add ShaderGroup
    group.type = group.intersectionShader == VK_SHADER_UNUSED_KHR
                     ? VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR
                     : VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR;
    shader_groups_.emplace_back(
        ShaderGroup(group.intersectionShader == VK_SHADER_UNUSED_KHR
                        ? ShaderGroup::Type::kTriangleHit
                        : ShaderGroup::Type::kProceduralHit,
                    ShaderBindingTable::Entry::kHit));
    vk_shader_groups_.push_back(group);
  }

  // define pipeline
  VkRayTracingPipelineCreateInfoKHR pipeline_ci{};
  pipeline_ci.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
  pipeline_ci.stageCount = static_cast<uint32_t>(shader_stage_cis.size());
  pipeline_ci.pStages = shader_stage_cis.data();
  pipeline_ci.groupCount = static_cast<uint32_t>(vk_shader_groups_.size());
  pipeline_ci.pGroups = vk_shader_groups_.data();
  pipeline_ci.maxPipelineRayRecursionDepth = max_ray_recursion_depth;
  pipeline_ci.layout = pipeline_layout_;

  // create pipeline
  auto const& device_funcs = vk_device.getFunctions();
  VkPipeline new_pipeline = VK_NULL_HANDLE;
  result = device_funcs.vkCreateRayTracingPipelinesKHR(vk_device.getHandle(),
                                                       VK_NULL_HANDLE,
                                                       VK_NULL_HANDLE,
                                                       1,
                                                       &pipeline_ci,
                                                       nullptr,
                                                       &new_pipeline);

  if (result != VK_SUCCESS) {
    vkDestroyPipelineLayout(vk_device.getHandle(), pipeline_layout_, nullptr);
    pipeline_layout_ = VK_NULL_HANDLE;
    CHECK_VKRESULT(result, "creating Raytracing Pipeline");
  }

  // Add it to the map
  pipeline_mgr_->insertPipeline(0u, new_pipeline);

  // name it
  vk_device.nameVulkanObject(
      VK_OBJECT_TYPE_PIPELINE, new_pipeline, getTrackingData().origin);

  // Mark usable (required for correct destruction)
  setUsable();

  // Get shader group handles, which ShaderBindingTable uses to retrieve the
  // required DeviceAddresses for passing to the traceRays command
  auto const handle_size = vk_device.getLimits().shader_group_handle_size;
  uint32_t num_groups = shader_groups_.size();

  std::vector<uint8_t> handles(num_groups * handle_size);
  CHECK_VKRESULT(device_funcs.vkGetRayTracingShaderGroupHandlesKHR(vk_device.getHandle(),
                                                                   new_pipeline,
                                                                   0,
                                                                   num_groups,
                                                                   handles.size(),
                                                                   handles.data()),
                 "getting shader group handles");

  // Copy the handles into the local ShaderGroup structs
  for (uint32_t i = 0; i < num_groups; ++i) {
    auto& data = shader_groups_[i].handle_data;
    data.resize(handle_size);
    std::memcpy(data.data(), handles.data() + i * handle_size, handle_size);
  }
}

const std::vector<RaytracingPipeline::ShaderGroup>&
VulkanRaytracingPipeline::getShaderGroups() const {
  CHECK(isUsable()) << "create must be called before accessing ShaderGroups";
  return shader_groups_;
}

}  // namespace gfx
