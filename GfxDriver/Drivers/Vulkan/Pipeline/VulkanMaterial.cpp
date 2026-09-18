/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Drivers/Vulkan/Pipeline/VulkanMaterial.h"

#include <algorithm>
#include <iostream>

#include "GfxDriver/Drivers/Vulkan/Pipeline/Utils.h"
#include "GfxDriver/Drivers/Vulkan/Resources/Utils.h"
#include "GfxDriver/Drivers/Vulkan/Resources/VulkanResourceManager.h"
#include "GfxDriver/Drivers/Vulkan/Resources/VulkanShaderModule.h"
#include "GfxDriver/Drivers/Vulkan/Resources/VulkanTexture.h"
#include "GfxDriver/Drivers/Vulkan/VulkanDeviceContext.h"
#include "GfxDriver/Drivers/Vulkan/VulkanResult.h"
#include "GfxDriver/Resources/AccelerationStructure.h"
#include "Shared/DebugOutputStream.h"

#define DEBUG_PRINT_WRITER_INFO false
#define DESCRIPTOR_DEBUG_PRINT() DEBUG_OUTPUT_STREAM(DEBUG_PRINT_WRITER_INFO, std::cout)

namespace gfx {

VulkanMaterial::VulkanMaterial(const DeviceContext& device_ctx,
                               std::string_view resource_tracking_string,
                               ShaderCacheShPtrVector& caches,
                               bool allow_duplicate_shader_stages)
    : Material(device_ctx,
               resource_tracking_string,
               caches,
               allow_duplicate_shader_stages)
    , descriptor_set_layout_{VK_NULL_HANDLE}
    , descriptor_pool_{VK_NULL_HANDLE}
    , descriptor_set_{VK_NULL_HANDLE} {
  createShaderModules();
  createDescriptorResources();
}

VulkanMaterial::VulkanMaterial(const DeviceContext& device_ctx,
                               const Material& material_to_clone)
    : Material(device_ctx, material_to_clone)
    , descriptor_set_layout_{VK_NULL_HANDLE}
    , descriptor_pool_{VK_NULL_HANDLE}
    , descriptor_set_{VK_NULL_HANDLE}
    , descriptors_dirty_{true} {
  createShaderModules();
  createDescriptorResources();
}

VulkanMaterial::~VulkanMaterial() {
  destroyDescriptorResources();
  destroyShaderModules();
}

void VulkanMaterial::createShaderModules() {
  auto& vk_resource_mgr =
      static_cast<VulkanResourceManager&>(device_ctx_.getResourceManager());

  for (auto& cache : shader_caches_) {
    auto shader_stage = cache->getShaderStage();
    if (!allow_duplicate_shader_stages_) {
      CHECK(shader_stage_modules_.find(shader_stage) == shader_stage_modules_.end())
          << "Found duplicate ShaderStage \'" << to_string(shader_stage)
          << "\' in Material \'" << resource_tracking_string_ << "\'";
    }
    auto shader_module_resource_tracking_string =
        resource_tracking_string_ + " (" + to_string(shader_stage) + ")";
    shader_stage_modules_.insert(std::pair<ShaderStage, resource_ptr<VulkanShaderModule>>{
        shader_stage,
        vk_resource_mgr.createShaderModule(shader_module_resource_tracking_string,
                                           cache)});
  }
}

void VulkanMaterial::destroyShaderModules() {
  auto& vk_rsrc_mgr =
      static_cast<VulkanResourceManager&>(device_ctx_.getResourceManager());

  for (auto& shader_stage_and_module : shader_stage_modules_) {
    auto& shader_module = shader_stage_and_module.second;
    vk_rsrc_mgr.destroyShaderModule(std::move(shader_module));
  }
  shader_stage_modules_.clear();
}

void VulkanMaterial::createDescriptorResources() {
  const VulkanDeviceContext& vk_device =
      static_cast<const VulkanDeviceContext&>(getDeviceContext());

  VkResult result{VK_SUCCESS};

  //
  // build descriptor set layout
  //
  std::vector<VkDescriptorSetLayoutBinding> layout_bindings;

  enum DescriptorPools {
    kUniformBuffer,
    kStorageBuffer,
    kCombinedImageSampler,
    kStorageImage,
    kAccelerationStructure,
    kCOUNT
  };

  std::array<VkDescriptorPoolSize, DescriptorPools::kCOUNT> pool_sizes{
      {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0},
       {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 0},
       {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 0},
       {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 0},
       {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 0}}};

  uint32_t max_binding = 0u;
  DESCRIPTOR_DEBUG_PRINT() << "Adding DescriptorSetLayoutBindings for '"
                           << resource_tracking_string_ << "': " << std::endl;

  for (auto const& cache : shader_caches_) {
    DESCRIPTOR_DEBUG_PRINT() << "cache: " << cache->getLibraryItemFilename() << std::endl;
    auto const& shader_stage = cache->getShaderStage();
    auto const& reflection = cache->getReflection();

    auto stage_flag = shader_stage_to_vk_shader_stage_flag(shader_stage);

    // get names
    auto const uniform_buffer_names = reflection.getAllUniformBufferNames();
    auto const shader_storage_buffer_names = reflection.getAllShaderStorageBufferNames();
    auto const sampler_names = reflection.getAllSamplerNames();
    auto const storage_image_names = reflection.getAllStorageImageNames();
    auto const acceleration_structure_names =
        reflection.getAllAccelerationStructureNames();

    // helper lambda
    // Returns true if a new descriptor is added otherwise false
    auto add_layout_binding_for = [&](std::string_view name,
                                      uint32_t binding,
                                      uint32_t array_size,
                                      VkDescriptorType descriptor_type) -> bool {
      // Check if we already have the binding
      auto predicate = [&](auto const& vkbinding) {
        return vkbinding.binding == binding;
      };
      auto itr = std::find_if(layout_bindings.begin(), layout_bindings.end(), predicate);
      if (itr != layout_bindings.end()) {
        // Binding in use, ensure identical type and array_size
        CHECK_EQ(itr->descriptorType, descriptor_type);
        CHECK_EQ(itr->descriptorCount, array_size);
        // Add our stage mask to the existing binding
        itr->stageFlags |= stage_flag;
        return false;
      } else {
        // Add a new binding
        VkDescriptorSetLayoutBinding layout_binding = {};
        layout_binding.binding = binding;
        layout_binding.descriptorType = descriptor_type;
        layout_binding.descriptorCount = array_size;
        layout_binding.stageFlags = stage_flag;
        layout_bindings.push_back(layout_binding);
        max_binding = std::max(max_binding, binding);
        DESCRIPTOR_DEBUG_PRINT() << "binding: " << binding << "  '" << name << "' ["
                                 << descriptor_type << "]" << std::endl;
        return true;
      }
    };

    // do uniform buffers
    for (auto const& name : uniform_buffer_names) {
      auto binding = reflection.getUniformBufferBinding(name);
      if (add_layout_binding_for(name, binding, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)) {
        pool_sizes[DescriptorPools::kUniformBuffer].descriptorCount++;
      }
    }

    // do shader storage buffers
    for (auto const& name : shader_storage_buffer_names) {
      auto binding = reflection.getShaderStorageBufferBinding(name);
      if (add_layout_binding_for(name, binding, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)) {
        pool_sizes[DescriptorPools::kStorageBuffer].descriptorCount++;
      }
    }

    // do samplers
    for (auto const& name : sampler_names) {
      auto binding = reflection.getSamplerBinding(name);
      auto array_size = reflection.getSamplerArraySize(name);
      if (add_layout_binding_for(
              name, binding, array_size, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)) {
        pool_sizes[DescriptorPools::kCombinedImageSampler].descriptorCount++;
      }
    }

    // do storage images
    for (auto const& name : storage_image_names) {
      auto binding = reflection.getStorageImageBinding(name);
      auto array_size = reflection.getStorageImageArraySize(name);
      if (add_layout_binding_for(
              name, binding, array_size, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)) {
        pool_sizes[DescriptorPools::kStorageImage].descriptorCount++;
      }
    }

    // do acceleration structures
    for (auto const& name : acceleration_structure_names) {
      auto binding = reflection.getAccelerationStructureBinding(name);
      auto array_size = reflection.getAccelerationStructureSize(name);
      if (add_layout_binding_for(
              name, binding, array_size, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR)) {
        pool_sizes[kAccelerationStructure].descriptorCount++;
      }
    }
  }

  // build create info
  VkDescriptorSetLayoutCreateInfo layout_ci = {};
  layout_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layout_ci.bindingCount = static_cast<uint32_t>(layout_bindings.size());
  layout_ci.pBindings = layout_bindings.data();

  // create the layout
  result = vkCreateDescriptorSetLayout(
      vk_device.getHandle(), &layout_ci, nullptr, &descriptor_set_layout_);
  CHECK_VKRESULT(result, "creating descriptor set layout");

  // name it
  vk_device.nameVulkanObject(VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT,
                             descriptor_set_layout_,
                             resource_tracking_string_);

  //
  // descriptor pool
  //
  std::vector<VkDescriptorPoolSize> used_pool_sizes;
  for (auto const& pool_size : pool_sizes) {
    if (pool_size.descriptorCount > 0) {
      used_pool_sizes.emplace_back(pool_size);
    }
  }

  if (used_pool_sizes.size()) {
    VkDescriptorPoolCreateInfo pool_ci = {};
    pool_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_ci.poolSizeCount = static_cast<uint32_t>(used_pool_sizes.size());
    pool_ci.pPoolSizes = used_pool_sizes.data();
    pool_ci.maxSets = 1;

    result = vkCreateDescriptorPool(
        vk_device.getHandle(), &pool_ci, nullptr, &descriptor_pool_);
    CHECK_OOM_VKRESULT(result,
                       "creating descriptor pool",
                       resource_tracking_string_,
                       0,
                       vk_device,
                       std::nullopt);

    // name it
    vk_device.nameVulkanObject(
        VK_OBJECT_TYPE_DESCRIPTOR_POOL, descriptor_pool_, resource_tracking_string_);

    //
    // descriptor set
    //

    VkDescriptorSetAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = descriptor_pool_;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &descriptor_set_layout_;

    result =
        vkAllocateDescriptorSets(vk_device.getHandle(), &alloc_info, &descriptor_set_);
    CHECK_VKRESULT(result, "creating descriptor set");

    vk_device.nameVulkanObject(
        VK_OBJECT_TYPE_DESCRIPTOR_SET, descriptor_set_, resource_tracking_string_);

    // initialize WriteInfos
    writer_infos_.resize(max_binding + 1);
    descriptor_set_writers_.clear();
    // Binding indices *may* be sparse if an external UBO binding
    // has a high index (we avoid this but...)
    // Reserve enough space for all bindings to ensure pointers stay
    // valid, but use push_back on VkWriteDescriptorSet structs to
    // ensure only used bindings are written
    descriptor_set_writers_.reserve(max_binding + 1);
    for (auto const& layout_binding : layout_bindings) {
      auto& writer_info = writer_infos_[layout_binding.binding];
      VkWriteDescriptorSet writer = {};
      writer.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writer.dstSet = descriptor_set_;
      writer.dstArrayElement = 0;  // always zero because we only have one set
      writer.dstBinding = layout_binding.binding;
      writer.descriptorType = layout_binding.descriptorType;
      writer.descriptorCount = layout_binding.descriptorCount;

      // Init image info or buffer info depending on resource type
      if (writer.descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER ||
          writer.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) {
        writer_info.image_infos.resize(writer.descriptorCount);
        for (auto& image_info : writer_info.image_infos) {
          image_info.imageLayout =
              writer.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                  ? VK_IMAGE_LAYOUT_GENERAL
                  : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
          image_info.imageView = VK_NULL_HANDLE;
          image_info.sampler = VK_NULL_HANDLE;
        }
        writer.pImageInfo = writer_info.image_infos.data();
      }

      if (writer.descriptorType == VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR) {
        writer_info.accel_structure_info.sType =
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
        writer_info.accel_structure_info.accelerationStructureCount = 0;
        writer_info.accel_structure_info.pAccelerationStructures = nullptr;
      }

      // Clear buffer info (not strictly necessary for image resources, but safer)
      auto& buffer_info = writer_info.buffer_info;
      buffer_info.buffer = VK_NULL_HANDLE;
      buffer_info.range = 0u;
      buffer_info.offset = 0u;
      writer.pBufferInfo = &buffer_info;

      descriptor_set_writers_.push_back(writer);
      writer_info.writer = &descriptor_set_writers_.back();
    }
  }

  // Update writers for local uniform buffers
  // Local UBOs are immutable so this is safe to do once
  bindLocalUniformBuffers();

  descriptors_dirty_ = true;
}

void VulkanMaterial::destroyDescriptorResources() {
  const VulkanDeviceContext& vk_device =
      static_cast<const VulkanDeviceContext&>(getDeviceContext());

  // forget descriptor set
  descriptor_set_ = VK_NULL_HANDLE;

  // destroy pool
  if (descriptor_pool_) {
    vkDestroyDescriptorPool(vk_device.getHandle(), descriptor_pool_, nullptr);
    descriptor_pool_ = VK_NULL_HANDLE;
  }

  // destroy layout
  if (descriptor_set_layout_) {
    vkDestroyDescriptorSetLayout(vk_device.getHandle(), descriptor_set_layout_, nullptr);
    descriptor_set_layout_ = VK_NULL_HANDLE;
  }
}

void VulkanMaterial::setSamplerAttribute(std::string_view attr_name,
                                         const Texture& texture,
                                         const uint32_t view_id) {
  auto const [binding, array_size] = getSamplerBindingAndArraySize(attr_name);
  CHECK_GE(binding, 0);
  CHECK_EQ(array_size, 1);
  auto const& vulkan_texture = static_cast<const VulkanTexture&>(texture);
  setImageInfoForImage(
      reinterpret_cast<VkImageView>(vulkan_texture.getViewHandle(view_id)),
      vulkan_texture.getSampler(),
      binding);
}

void VulkanMaterial::setSamplerArrayAttribute(
    std::string_view attr_name,
    const std::vector<resource_ptr<Texture>>& textures) {
  auto const [binding, array_size] = getSamplerBindingAndArraySize(attr_name);
  CHECK_GE(binding, 0);
  CHECK_EQ(array_size, static_cast<int>(textures.size()));
  setImageInfoForTextureVector(textures, binding);
}

void VulkanMaterial::setImageLoadStoreAttribute(std::string_view attr_name,
                                                const Texture& texture,
                                                const uint32_t view_id) {
  auto const [binding, array_size] = getStorageImageBindingAndArraySize(attr_name);
  CHECK_GE(binding, 0);
  CHECK_EQ(array_size, 1);
  auto const& vulkan_texture = static_cast<const VulkanTexture&>(texture);
  setImageInfoForImage(
      reinterpret_cast<VkImageView>(vulkan_texture.getViewHandle(view_id)),
      VK_NULL_HANDLE,
      binding);
}

void VulkanMaterial::setImageLoadStoreArrayAttribute(
    std::string_view attr_name,
    const std::vector<resource_ptr<Texture>>& textures) {
  auto const [binding, array_size] = getStorageImageBindingAndArraySize(attr_name);
  CHECK_GE(binding, 0);
  CHECK_EQ(array_size, static_cast<int>(textures.size()));
  setImageInfoForTextureVector(textures, binding);
}

void VulkanMaterial::bindShaderStorageBufferToBlock(std::string_view block_name,
                                                    const BufferWrapper& ssbo,
                                                    std::optional<uint64_t> offset,
                                                    std::optional<uint64_t> range) {
  CHECK(any_bits_set(ssbo.getUsageBits() & BufferUsageBits::kStorageBufferBit));
  for (auto const& cache : shader_caches_) {
    auto const& reflection = cache->getReflection();
    auto binding = reflection.getShaderStorageBufferBinding(block_name);
    if (binding >= 0) {
      setBufferInfoForBuffer(ssbo, binding, offset, range);
      return;
    }
  }
  CHECK(false) << "Shader Storage Block '" << block_name << "' does not exist in shader '"
               << resource_tracking_string_ << "'.";
}

void VulkanMaterial::setAccelerationStructureAttribute(
    std::string_view attr_name,
    const AccelerationStructure& accel_structure) {
  bool found = false;
  for (auto const& cache : shader_caches_) {
    auto const& reflection = cache->getReflection();
    auto binding = reflection.getAccelerationStructureBinding(attr_name);
    if (binding >= 0) {
      setAccelerationStructureInfo(accel_structure, binding);
      found = true;
    }
  }
  CHECK(found) << "Acceleration Structure '" << attr_name
               << "' does not exist in Material '" << resource_tracking_string_ << "'.";
}

void VulkanMaterial::bindExternalUniformBufferToBlock(std::string_view block_name,
                                                      const BufferWrapper& ubo) {
  CHECK(any_bits_set(ubo.getUsageBits() & BufferUsageBits::kUniformBufferBit));
  for (auto const& cache : shader_caches_) {
    auto const& external_uniform_buffer_names = cache->getExternalUniformBufferNames();
    if (external_uniform_buffer_names.find(block_name) !=
        external_uniform_buffer_names.end()) {
      auto const& reflection = cache->getReflection();
      auto binding = reflection.getUniformBufferBinding(block_name);
      if (binding >= 0) {
        setBufferInfoForBuffer(ubo, binding, std::nullopt, std::nullopt);
#ifndef NDEBUG
        external_uniform_buffers_bound_.insert(block_name);
#endif
        return;
      }
    }
  }
  CHECK(false) << "Uniform Block '" << block_name << "' does not exist in shader '"
               << resource_tracking_string_ << "' or is not declared as external.";
}

const VulkanShaderModule& VulkanMaterial::getShaderModuleForStage(
    ShaderStage shader_stage) const {
  CHECK_EQ(shader_stage_modules_.count(shader_stage), 1U)
      << "Multiple ShaderModules found for ShaderStage `" << to_string(shader_stage)
      << "' in Material '" << resource_tracking_string_ << "'";
  auto itr = shader_stage_modules_.find(shader_stage);
  CHECK(itr != shader_stage_modules_.end());
  return *itr->second;
}

VkShaderModule VulkanMaterial::getShaderModule(ShaderStage shader_stage) const {
  auto const& shader_module = getShaderModuleForStage(shader_stage);
  return reinterpret_cast<VkShaderModule>(shader_module.getResourceHandle());
}

const VulkanMaterial::ShaderModuleMap& VulkanMaterial::getShaderModuleMap() const {
  return shader_stage_modules_;
}

const char* VulkanMaterial::getEntryPoint(ShaderStage shader_stage) const {
  auto const& shader_module = getShaderModuleForStage(shader_stage);
  CHECK(shader_module.cache_);
  return shader_module.cache_->getEntryPoint().c_str();
}

bool VulkanMaterial::hasFragmentShaderOutputLocation(int location) const {
  auto const itr = shader_stage_modules_.find(ShaderStage::kFragment);
  if (itr != shader_stage_modules_.end()) {
    CHECK(itr->second->cache_);
    return itr->second->cache_->getReflection().hasFragmentShaderOutputLocation(location);
  }
  return false;
}

VkDescriptorSetLayout VulkanMaterial::getDescriptorSetLayout() const {
  CHECK(descriptor_set_layout_);
  return descriptor_set_layout_;
}

const std::vector<VkWriteDescriptorSet>& VulkanMaterial::getDescriptorSetWriters() const {
  return descriptor_set_writers_;
}

void VulkanMaterial::updateDescriptorSets() {
  if (descriptors_dirty_) {
#ifndef NDEBUG
    for (auto const& name : external_uniform_buffer_names_) {
      CHECK(external_uniform_buffers_bound_.find(name) !=
            external_uniform_buffers_bound_.end())
          << "External uniform buffer '" << name << "' is not bound for material '"
          << resource_tracking_string_ << "'";
    }
#endif
    const VulkanDeviceContext& vk_device =
        static_cast<const VulkanDeviceContext&>(getDeviceContext());
    vkUpdateDescriptorSets(vk_device.getHandle(),
                           descriptor_set_writers_.size(),
                           descriptor_set_writers_.data(),
                           0,
                           nullptr);
    descriptors_dirty_ = false;
  }
}

VkDescriptorSet VulkanMaterial::getDescriptorSet() const {
  return descriptor_set_;
}

void VulkanMaterial::bindLocalUniformBuffers() {
  for (auto const& [binding, ubo] : local_uniform_buffers_) {
    setBufferInfoForBuffer(*ubo, binding, std::nullopt, std::nullopt);
  }
}

VulkanMaterial::DescriptorWriterInfo& VulkanMaterial::getWriterInfoForBinding(
    uint32_t binding) {
  CHECK_LT(binding, writer_infos_.size());
  auto& writer_info = writer_infos_[binding];
  CHECK(writer_info.writer);
  CHECK_EQ(writer_info.writer->dstBinding, binding);
  return writer_info;
}

void VulkanMaterial::setBufferInfoForBuffer(const BufferWrapper& buffer,
                                            uint32_t binding,
                                            std::optional<uint64_t> offset,
                                            std::optional<uint64_t> range) {
  auto& writer_info = getWriterInfoForBinding(binding);
  CHECK(writer_info.writer);
  auto const descriptor_type = writer_info.writer->descriptorType;
  auto const additional_offset = offset ? *offset : 0ULL;
  auto const offset_to_use = buffer.getAllocationOffsetBytes() + additional_offset;
  auto const range_to_use = range ? *range : buffer.getNumBytes();

  auto validate_offset_and_range = [&](const std::string& buffer_type,
                                       uint64_t device_limit) {
    auto const buffer_size = buffer.getNumBytes();
    if (offset_to_use + range_to_use > buffer_size) {
      auto const msg =
          "Attempt to bind " + buffer_type + " '" + buffer.getTrackingDataNameOnly() +
          "' to Material '" + resource_tracking_string_ +
          "' failed, offset+range exceeds buffer size (offset = " +
          std::to_string(offset_to_use) + ", range = " + std::to_string(range_to_use) +
          ", buffer size = " + std::to_string(buffer_size) + ")";
      THROW_RUNTIME_EX(msg);
    }
    if (range_to_use > device_limit) {
      auto const msg = "Attempt to bind " + buffer_type + " '" +
                       buffer.getTrackingDataNameOnly() + "' to Material '" +
                       resource_tracking_string_ +
                       "' failed, range exceeds device limit (range = " +
                       std::to_string(range_to_use) +
                       ", device limit = " + std::to_string(device_limit) + ")";
      THROW_RUNTIME_EX(msg);
    }
  };

  if (descriptor_type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER) {
    validate_offset_and_range("UBO", device_ctx_.getLimits().max_uniform_buffer_size);
  } else if (descriptor_type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
    validate_offset_and_range("SSBO",
                              device_ctx_.getLimits().max_shader_storage_buffer_size);
  }

  auto& buffer_info = writer_info.buffer_info;
  buffer_info.buffer = reinterpret_cast<VkBuffer>(buffer.getResourceHandle());
  buffer_info.offset = offset_to_use;
  buffer_info.range = range_to_use;

  descriptors_dirty_ = true;
}

void VulkanMaterial::setImageInfoForImage(VkImageView image_view,
                                          VkSampler sampler,
                                          uint32_t binding) {
  auto& image_info = getWriterInfoForBinding(binding).image_infos[0];

  image_info.imageView = image_view;
  image_info.sampler = sampler;

  descriptors_dirty_ = true;
}

void VulkanMaterial::setImageInfoForTextureVector(
    const std::vector<resource_ptr<Texture>>& textures,
    uint32_t binding) {
  auto& writer_info = getWriterInfoForBinding(binding);
  CHECK_EQ(writer_info.image_infos.size(), textures.size());

  for (size_t i = 0; i < textures.size(); i++) {
    auto const& vulkan_texture = static_cast<const VulkanTexture&>(*textures[i]);
    auto& image_info = writer_info.image_infos[i];
    image_info.imageView = vulkan_texture.getImageView();
    image_info.sampler = vulkan_texture.getSampler();
  }

  descriptors_dirty_ = true;
}

void VulkanMaterial::setAccelerationStructureInfo(
    const AccelerationStructure& accel_structure,
    uint32_t binding) {
  auto& writer_info = getWriterInfoForBinding(binding);
  auto& accel_structure_info = writer_info.accel_structure_info;
  accel_structure_info.accelerationStructureCount = 1;
  writer_info.accel_structure_handle =
      reinterpret_cast<VkAccelerationStructureKHR>(accel_structure.getResourceHandle());
  accel_structure_info.pAccelerationStructures = &writer_info.accel_structure_handle;
  writer_info.writer->pNext = &accel_structure_info;

  descriptors_dirty_ = true;
}

}  // namespace gfx
