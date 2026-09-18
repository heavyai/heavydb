/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "GfxDriver/Pipeline/Material.h"

#include <vulkan/vulkan.h>

#include <map>
#include <vector>

namespace gfx {

class VulkanShaderModule;

class VulkanMaterial : public Material {
 public:
  using ShaderModuleMap = std::multimap<ShaderStage, resource_ptr<VulkanShaderModule>>;

  explicit VulkanMaterial(const DeviceContext& device_ctx,
                          std::string_view resource_tracking_string,
                          ShaderCacheShPtrVector& shader_caches,
                          bool allow_duplicate_shader_stages);
  explicit VulkanMaterial(const DeviceContext& device_ctx,
                          const Material& material_to_clone);
  VulkanMaterial() = delete;
  ~VulkanMaterial() override;

  //
  // Material
  //
  void updateDescriptorSets() override;

  void setSamplerAttribute(std::string_view attr_name,
                           const Texture& texture,
                           const uint32_t view_id = 0) override;
  void setSamplerArrayAttribute(
      std::string_view attr_name,
      const std::vector<resource_ptr<Texture>>& textures) override;
  void setImageLoadStoreAttribute(std::string_view attr_name,
                                  const Texture& texture,
                                  const uint32_t view_id = 0) override;
  void setImageLoadStoreArrayAttribute(
      std::string_view attr_name,
      const std::vector<resource_ptr<Texture>>& textures) override;
  void bindShaderStorageBufferToBlock(std::string_view block_name,
                                      const BufferWrapper& ssbo,
                                      std::optional<uint64_t> offset,
                                      std::optional<uint64_t> range) override;
  void bindExternalUniformBufferToBlock(std::string_view block_name,
                                        const BufferWrapper& ubo) override;
  void setAccelerationStructureAttribute(
      std::string_view attr_name,
      const AccelerationStructure& accel_structure) override;

  //
  // VulkanMaterial
  //

  VkShaderModule getShaderModule(ShaderStage shader_stage) const;
  const ShaderModuleMap& getShaderModuleMap() const;
  const char* getEntryPoint(ShaderStage shader_stage) const;
  bool hasFragmentShaderOutputLocation(int location) const;
  VkDescriptorSetLayout getDescriptorSetLayout() const;

  const std::vector<VkWriteDescriptorSet>& getDescriptorSetWriters() const;

  VkDescriptorSet getDescriptorSet() const;

  // call once to add descriptor writes for UBOs
  void bindLocalUniformBuffers();

 private:
  VkDescriptorSetLayout descriptor_set_layout_;
  VkDescriptorPool descriptor_pool_;
  VkDescriptorSet descriptor_set_;

  ShaderModuleMap shader_stage_modules_;

  struct DescriptorWriterInfo {
    VkWriteDescriptorSet* writer = nullptr;
    VkDescriptorBufferInfo buffer_info;
    std::vector<VkDescriptorImageInfo> image_infos;
    VkAccelerationStructureKHR accel_structure_handle;
    VkWriteDescriptorSetAccelerationStructureKHR accel_structure_info;
  };

  // Map binding to WriterInfo
  std::vector<DescriptorWriterInfo> writer_infos_;

  // Vector of handles to pass to Vulkan
  std::vector<VkWriteDescriptorSet> descriptor_set_writers_;
#ifndef NDEBUG
  std::set<std::string_view> external_uniform_buffers_bound_;
#endif

  bool descriptors_dirty_;

  void createShaderModules();
  void destroyShaderModules();

  const VulkanShaderModule& getShaderModuleForStage(ShaderStage shader_stage) const;

  // layout, pool, and set
  void createDescriptorResources();
  void destroyDescriptorResources();

  DescriptorWriterInfo& getWriterInfoForBinding(uint32_t binding);
  void setBufferInfoForBuffer(const BufferWrapper& buffer,
                              uint32_t binding,
                              std::optional<uint64_t> offset,
                              std::optional<uint64_t> range);
  void setImageInfoForImage(VkImageView image_view, VkSampler sampler, uint32_t binding);
  void setImageInfoForTextureVector(const std::vector<resource_ptr<Texture>>& textures,
                                    uint32_t binding);
  void setAccelerationStructureInfo(const AccelerationStructure& accel_structure,
                                    uint32_t binding);
};

}  // namespace gfx
