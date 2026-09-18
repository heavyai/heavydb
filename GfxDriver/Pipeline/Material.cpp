/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Pipeline/Material.h"

#include <unordered_set>

#include "GfxDriver/DeviceContext.h"
#include "GfxDriver/Resources/ResourceManager.h"
#include "Shared/misc.h"

namespace gfx {

Material::Material(const DeviceContext& device_ctx,
                   std::string_view resource_tracking_string,
                   ShaderCacheShPtrVector& shader_caches,
                   bool allow_duplicate_shader_stages)
    : device_ctx_{device_ctx}
    , resource_tracking_string_{resource_tracking_string}
    , shader_caches_{shader_caches}
    , allow_duplicate_shader_stages_{allow_duplicate_shader_stages} {
  validateShaderCaches();
  createLocalUniformBuffers();
}

Material::Material(const DeviceContext& device_ctx, const Material& material_to_clone)
    : device_ctx_{device_ctx}
    , resource_tracking_string_{material_to_clone.resource_tracking_string_}
    , shader_caches_{material_to_clone.shader_caches_}
    , external_uniform_buffer_names_{material_to_clone.external_uniform_buffer_names_}
    , allow_duplicate_shader_stages_{material_to_clone.allow_duplicate_shader_stages_} {
  createLocalUniformBuffers();
}

Material::~Material() {
  shutdown();
}

void Material::shutdown() {
  // get resource manager
  auto& rsrc_mgr = device_ctx_.getResourceManager();

  // destroy local uniform buffers
  for (auto& local_uniform_buffer : local_uniform_buffers_) {
    if (local_uniform_buffer.second) {
      rsrc_mgr.destroyBuffer(std::move(local_uniform_buffer.second));
    }
  }
  local_uniform_buffers_.clear();
  external_uniform_buffer_names_.clear();

  // forget caches
  shader_caches_.clear();
}

void Material::validateShaderCaches() const {
  // Ensure there are no invalid ShaderStage combinations
  std::unordered_set<ShaderStage> stages;
  for (auto const& cache : shader_caches_) {
    stages.insert(cache->getShaderStage());
  }

  // Standard graphics pipeline Material
  if (stages.count(ShaderStage::kVertex)) {
    RUNTIME_EX_ASSERT(stages.count(ShaderStage::kCompute) +
                              stages.count(ShaderStage::kMesh) +
                              stages.count(ShaderStage::kTask) ==
                          0,
                      "Invalid ShaderStage combination in Material");
  }
  // Compute Material
  if (stages.count(ShaderStage::kCompute)) {
    RUNTIME_EX_ASSERT(
        stages.count(ShaderStage::kGeometry) + stages.count(ShaderStage::kFragment) +
                stages.count(ShaderStage::kTessControl) +
                stages.count(ShaderStage::kTessEval) + stages.count(ShaderStage::kMesh) +
                stages.count(ShaderStage::kTask) ==
            0,
        "Invalid ShaderStage combination in Material");
  }
  // Mesh / Task Material
  if (stages.count(ShaderStage::kMesh)) {
    RUNTIME_EX_ASSERT(stages.count(ShaderStage::kGeometry) +
                              stages.count(ShaderStage::kTessControl) +
                              stages.count(ShaderStage::kTessEval) ==
                          0,
                      "Invalid ShaderStage combination in Material");
  }
  // Raytracing Material
  if (stages.count(ShaderStage::kRayGen)) {
    RUNTIME_EX_ASSERT(
        stages.count(ShaderStage::kCompute) + stages.count(ShaderStage::kGeometry) +
                stages.count(ShaderStage::kFragment) +
                stages.count(ShaderStage::kTessControl) +
                stages.count(ShaderStage::kTessEval) + stages.count(ShaderStage::kMesh) +
                stages.count(ShaderStage::kTask) ==
            0,
        "Invalid ShaderStage combination in Material");
  }
}

void Material::createLocalUniformBuffers() {
  auto& rsrc_mgr = device_ctx_.getResourceManager();

  std::unordered_map<int, std::string> ubo_bindings_and_names;

  auto const& ubo_alignment = getDeviceContext().getLimits().uniform_buffer_alignment;

  // for each cache
  for (auto const& cache : shader_caches_) {
    auto const& reflection = cache->getReflection();
    // get the UBO names in this reflection
    auto const uniform_buffer_names = reflection.getAllUniformBufferNames();
    auto external_uniform_buffer_names_that_must_exist =
        cache->getExternalUniformBufferNames();
    for (auto const& uniform_buffer_name : uniform_buffer_names) {
      // skip if external
      if (external_uniform_buffer_names_that_must_exist.find(uniform_buffer_name) !=
          external_uniform_buffer_names_that_must_exist.end()) {
        external_uniform_buffer_names_that_must_exist.erase(uniform_buffer_name);
        external_uniform_buffer_names_.insert(uniform_buffer_name);
        continue;
      }
      // get binding and size for this UBO by name and validate
      auto binding = reflection.getUniformBufferBinding(uniform_buffer_name);
      auto block_size = reflection.getUniformBufferBlockSize(uniform_buffer_name);
      CHECK(binding >= 0 && block_size > 0)
          << "Failed to create local UBOs for '" << resource_tracking_string_ << "': "
          << "Invalid binding/size for UBO '" << uniform_buffer_name << "'";

      // if we already have this binding and name, we don't need to create another one
      auto const [itr, create] =
          ubo_bindings_and_names.try_emplace(binding, uniform_buffer_name);
      CHECK(create || itr->second == uniform_buffer_name)
          << "Failed to create local UBOs for '" << resource_tracking_string_ << "': '"
          << itr->second << "' and '" << uniform_buffer_name << "' at binding "
          << binding;

      // create buffer if needed
      if (create) {
        // round up the required size to the UBO alignment for the device
        // otherwise the UBO creation will throw a warning
        auto const required_buffer_size =
            ((static_cast<uint64_t>(block_size) + ubo_alignment - 1) / ubo_alignment) *
            ubo_alignment;

        // create the buffer
        auto buffer_resource_tracking_string =
            resource_tracking_string_ + " UBO(" + std::to_string(binding) + ")";
        local_uniform_buffers_[binding] =
            rsrc_mgr.createBuffer(buffer_resource_tracking_string,
                                  {BufferType::kUnspecified,
                                   required_buffer_size,
                                   BufferUsageBits::kUniformBufferBit,
                                   BufferAccessType::kHostVisible});
      }
    }
    // ensure we found all the expected external uniform buffers
    CHECK_EQ(external_uniform_buffer_names_that_must_exist.size(), 0u)
        << "Failed to find the following external uniform buffers in shader cache(s) for "
           "material '"
        << resource_tracking_string_
        << "': " << shared::printContainer(external_uniform_buffer_names_that_must_exist);
  }
}

void Material::setUniformBufferAttribute(const ShaderReflection::ItemInfo& ai,
                                         void* attr_value_bytes,
                                         uint32_t attr_size,
                                         uint32_t vector_index) {
  // look up the UBO by binding
  auto const& it = local_uniform_buffers_.find(ai.binding_or_location);
  CHECK(it != local_uniform_buffers_.end());
  auto const& pubo = it->second;

  // compute any vector element offset (0 for non-vectors)
  uint32_t vector_element_offset = vector_index * attr_size;

  // validate offset and size within attr
  CHECK_LE(vector_element_offset + attr_size,
           static_cast<uint32_t>(ai.block_or_array_size));

  // compute overall offset
  uint32_t overall_offset = ai.offset + vector_element_offset;

  // validate offset and size within buffer
  CHECK_LE(overall_offset + attr_size, static_cast<uint32_t>(pubo->getNumBytes()));

  // write in the data
  pubo->updateSubData(attr_value_bytes, attr_size, overall_offset);
}

void Material::setViewportAttributes(uint32_t x,
                                     uint32_t y,
                                     uint32_t width,
                                     uint32_t height) {
  setUniformAttribute<int>("viewport.x", x);
  setUniformAttribute<int>("viewport.y", y);
  setUniformAttribute<int>("viewport.width", width);
  setUniformAttribute<int>("viewport.height", height);
}

bool Material::hasUniformAttribute(std::string_view attr_name) const {
  for (auto const& cache : shader_caches_) {
    auto const& reflection = cache->getReflection();
    if (reflection.hasUniformBufferAttr(attr_name)) {
      return true;
    }
  }
  return false;
}

bool Material::hasVertexAttribute(std::string_view attr_name) const {
  for (auto const& cache : shader_caches_) {
    auto const& reflection = cache->getReflection();
    if (reflection.hasVertexAttr(attr_name)) {
      return true;
    }
  }
  return false;
}

uint32_t Material::getVertexAttributeLocation(std::string_view attr_name) const {
  for (auto const& cache : shader_caches_) {
    auto const& reflection = cache->getReflection();
    if (reflection.hasVertexAttr(attr_name)) {
      return static_cast<uint32_t>(reflection.getVertexAttrLocation(attr_name));
    }
  }
  THROW_RUNTIME_EX(
      "Attribute '" + std::string(attr_name) +
      "' does not exist in shader. Cannot get attribute location in Material '" +
      resource_tracking_string_ + "'");
}

const ShaderReflection::ItemInfo& Material::getUniformBufferAttrInfo(
    std::string_view uniform_buffer_name) const {
  for (auto const& cache : shader_caches_) {
    auto const& reflection = cache->getReflection();
    auto const& ai = reflection.getUniformBufferAttrItemInfo(uniform_buffer_name);
    if (ai.binding_or_location >= 0) {
      return ai;
    }
  }
  THROW_RUNTIME_EX("Failed to find Uniform Buffer AttrInfo for '" +
                   std::string(uniform_buffer_name) + "' in Material '" +
                   resource_tracking_string_ + "'");
}

const ShaderReflection::ItemInfo& Material::getShaderStorageBufferAttrInfo(
    std::string_view shader_storage_buffer_name) const {
  for (auto const& cache : shader_caches_) {
    auto const& reflection = cache->getReflection();
    auto const& ai =
        reflection.getShaderStorageBufferAttrItemInfo(shader_storage_buffer_name);
    if (ai.binding_or_location >= 0) {
      return ai;
    }
  }
  THROW_RUNTIME_EX("Failed to find Shader Storage Buffer AttrInfo for '" +
                   std::string(shader_storage_buffer_name) + "' in Material '" +
                   resource_tracking_string_ + "'");
}

std::pair<int, int> Material::getSamplerBindingAndArraySize(
    std::string_view sampler_name) const {
  for (auto const& cache : shader_caches_) {
    auto const& reflection = cache->getReflection();
    auto binding = reflection.getSamplerBinding(sampler_name);
    if (binding >= 0) {
      auto array_size = reflection.getSamplerArraySize(sampler_name);
      return {binding, array_size};
    }
  }
  THROW_RUNTIME_EX("Failed to find Sampler AttrInfo for '" + std::string(sampler_name) +
                   "' in Material '" + resource_tracking_string_ + "'");
}

std::pair<int, int> Material::getStorageImageBindingAndArraySize(
    std::string_view storage_image_name) const {
  for (auto const& cache : shader_caches_) {
    auto const& reflection = cache->getReflection();
    auto binding = reflection.getStorageImageBinding(storage_image_name);
    if (binding >= 0) {
      auto array_size = reflection.getStorageImageArraySize(storage_image_name);
      return {binding, array_size};
    }
  }
  THROW_RUNTIME_EX("Failed to find StorageImage AttrInfo for '" +
                   std::string(storage_image_name) + "' in Material '" +
                   resource_tracking_string_ + "'");
}

}  // namespace gfx
