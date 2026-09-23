/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <optional>
#include <vector>

#include "GfxDriver/Resources/ResourcePtr.h"
#include "GfxDriver/ShaderCompiler/ShaderManager.h"
#include "GfxDriver/ShaderCompiler/ShaderReflection.h"

namespace gfx {

class Material {
 public:
  explicit Material(const DeviceContext& device_ctx,
                    std::string_view resource_tracking_string,
                    ShaderCacheShPtrVector& caches,
                    bool allow_duplicate_shader_stages);
  explicit Material(const DeviceContext& device_ctx, const Material& material_to_clone);
  Material() = delete;
  virtual ~Material();

  const DeviceContext& getDeviceContext() const { return device_ctx_; }

  // Vulkan only
  virtual void updateDescriptorSets() {}

  // TODO(croot): this doesn't work if called like so:
  // shader.setUniformAttribute("attrName", 1+2);
  // Need to find a way to deal with this
  template <typename T>
  void setUniformAttribute(std::string_view attr_name, const T attr_value) {
    auto const& ai = getUniformBufferAttrInfo(attr_name);
    setUniformBufferAttribute(ai, (void*)&attr_value, sizeof(T));
  }

  template <typename T>
  void setUniformAttribute(std::string_view attr_name, const std::vector<T>& attr_value) {
    auto const& ai = getUniformBufferAttrInfo(attr_name);
    // arrays must be expanded to minimum stride of 16
    for (uint32_t i = 0; i < attr_value.size(); i++) {
      setUniformBufferAttribute(ai, (void*)&attr_value[i], sizeof(T), i);
    }
  }

  template <typename T, size_t N>
  void setUniformAttribute(std::string_view attr_name,
                           const std::array<T, N>& attr_value) {
    auto const& ai = getUniformBufferAttrInfo(attr_name);
    setUniformBufferAttribute(ai, (void*)&attr_value[0], sizeof(T) * N);
  }
  // note specialization for <float, 6> below

  virtual void setSamplerAttribute(std::string_view attr_name,
                                   const Texture& texture,
                                   const uint32_t view_id = 0) = 0;
  virtual void setSamplerArrayAttribute(
      std::string_view attr_name,
      const std::vector<resource_ptr<Texture>>& texture_arrays) = 0;
  virtual void setImageLoadStoreAttribute(std::string_view attr_name,
                                          const Texture& texture,
                                          const uint32_t view_id = 0) = 0;
  virtual void setImageLoadStoreArrayAttribute(
      std::string_view attr_name,
      const std::vector<resource_ptr<Texture>>& textures) = 0;
  virtual void bindShaderStorageBufferToBlock(
      std::string_view block_name,
      const BufferWrapper& ssbo,
      std::optional<uint64_t> offset = std::nullopt,
      std::optional<uint64_t> range = std::nullopt) = 0;
  virtual void bindExternalUniformBufferToBlock(std::string_view block_name,
                                                const BufferWrapper& ubo) = 0;
  virtual void setAccelerationStructureAttribute(
      std::string_view attr_name,
      const AccelerationStructure& accel_structure) = 0;

  bool hasUniformAttribute(std::string_view attr_name) const;
  bool hasVertexAttribute(std::string_view attr_name) const;
  uint32_t getVertexAttributeLocation(std::string_view attr_name) const;

  void setViewportAttributes(uint32_t x, uint32_t y, uint32_t width, uint32_t height);

 protected:
  const DeviceContext& device_ctx_;
  std::string resource_tracking_string_;
  std::unordered_map<int, BufferWrapperUqPtr> local_uniform_buffers_;
  ShaderCacheShPtrVector shader_caches_;
  std::set<std::string_view> external_uniform_buffer_names_;
  bool allow_duplicate_shader_stages_;

  void setUniformBufferAttribute(const ShaderReflection::ItemInfo& ai,
                                 void* attr_value_bytes,
                                 uint32_t attr_size,
                                 uint32_t vector_index = 0);

  void shutdown();

  void validateShaderCaches() const;
  void createLocalUniformBuffers();

  const ShaderReflection::ItemInfo& getUniformBufferAttrInfo(
      std::string_view uniform_buffer_name) const;
  const ShaderReflection::ItemInfo& getShaderStorageBufferAttrInfo(
      std::string_view shader_storage_buffer_name) const;
  std::pair<int, int> getSamplerBindingAndArraySize(std::string_view sampler_name) const;
  std::pair<int, int> getStorageImageBindingAndArraySize(
      std::string_view storage_image_name) const;
};

};  // namespace gfx
