/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Drivers/Vulkan/Resources/VulkanShaderModule.h"

#include "GfxDriver/Drivers/Vulkan/VulkanDeviceContext.h"
#include "GfxDriver/Drivers/Vulkan/VulkanResult.h"

namespace gfx {

VulkanShaderModule::VulkanShaderModule(const DeviceContext& device_ctx,
                                       std::string_view resource_tracking_string)
    : Resource(device_ctx, resource_tracking_string, ResourceType::kShaderModule)
    , shader_module_{VK_NULL_HANDLE} {}

VulkanShaderModule::~VulkanShaderModule() {
  cleanupResource();
}
void VulkanShaderModule::initResource(const ShaderCacheShPtr& cache) {
  cache_ = cache;

  auto const& spirv = cache->getSpirv();
  CHECK(!spirv.empty());

  VkShaderModuleCreateInfo shader_module_ci = {};
  shader_module_ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  shader_module_ci.codeSize = spirv.size() << 2;
  shader_module_ci.pCode = reinterpret_cast<const uint32_t*>(spirv.data());

  const VulkanDeviceContext& vk_device =
      static_cast<const VulkanDeviceContext&>(getDeviceContext());

  VkResult result = vkCreateShaderModule(
      vk_device.getHandle(), &shader_module_ci, nullptr, &shader_module_);
  CHECK_VKRESULT(result, "creating shader module from SPIR-V");

  // name it
  vk_device.nameVulkanObject(
      VK_OBJECT_TYPE_SHADER_MODULE, shader_module_, getTrackingData().origin);

  setUsable();
}

void VulkanShaderModule::cloneTo(VulkanShaderModule& dest) const {
  dest.initResource(cache_);
}

void VulkanShaderModule::cleanupResourceBase() {
  const VulkanDeviceContext& vk_device =
      static_cast<const VulkanDeviceContext&>(getDeviceContext());

  if (shader_module_ != VK_NULL_HANDLE) {
    vkDestroyShaderModule(vk_device.getHandle(), shader_module_, nullptr);
    shader_module_ = VK_NULL_HANDLE;
  }

  makeEmpty();
}

void VulkanShaderModule::makeEmpty() {
  cache_ = nullptr;
}

ShaderStage VulkanShaderModule::getShaderStage() const {
  CHECK(cache_);
  return cache_->getShaderStage();
}

const char* VulkanShaderModule::getEntryPoint() const {
  CHECK(cache_);
  return cache_->getEntryPoint().c_str();
}

uint32_t VulkanShaderModule::getRaytracingHitGroupIndex() const {
  CHECK(cache_);
  return cache_->getRaytracingHitGroupIndex();
}

}  // namespace gfx
