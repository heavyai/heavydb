/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <vulkan/vulkan.h>

#include "GfxDriver/Drivers/Vulkan/Pipeline/VulkanMaterial.h"
#include "GfxDriver/Drivers/Vulkan/Resources/VulkanResourceManager.h"
#include "GfxDriver/Resources/Resource.h"

namespace gfx {

class VulkanShaderModule : public Resource {
 public:
  explicit VulkanShaderModule(const DeviceContext& device_ctx,
                              std::string_view resource_tracking_string);
  ~VulkanShaderModule() override;

  ResourceHandle getResourceHandle() const override {
    return ResourceHandle(shader_module_);
  }

  ShaderStage getShaderStage() const;
  const char* getEntryPoint() const;
  uint32_t getRaytracingHitGroupIndex() const;

 private:
  ShaderCacheShPtr cache_;
  VkShaderModule shader_module_;

  void initResource(const ShaderCacheShPtr& cache);
  void cloneTo(VulkanShaderModule& dest) const;
  void cleanupResourceBase() override;
  void makeEmpty() override;

  friend class ::gfx::VulkanMaterial;
  friend class ::gfx::VulkanResourceManager;
};

}  // namespace gfx
