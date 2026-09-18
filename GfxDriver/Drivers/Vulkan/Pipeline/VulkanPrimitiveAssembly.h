/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "GfxDriver/Pipeline/PrimitiveAssembly.h"

#include <vulkan/vulkan.h>

namespace gfx {

class VulkanPrimitiveAssembly : public PrimitiveAssembly {
 public:
  explicit VulkanPrimitiveAssembly(const DeviceContext& device_ctx,
                                   std::string_view resource_tracking_string,
                                   const PrimitiveTopology topology,
                                   const Material& material,
                                   const PrimitiveAssemblyAttrInfo& attr_info,
                                   const IndexBuffer* ibo);
  explicit VulkanPrimitiveAssembly(const DeviceContext& device_ctx,
                                   std::string_view resource_tracking_string,
                                   const PrimitiveTopology topology,
                                   const Material& material,
                                   const PrimitiveAssemblyAttrInfo& instanced_attr_info,
                                   const PrimitiveAssemblyAttrInfo& instances_attr_info,
                                   const uint32_t num_instances_per_attr,
                                   const IndexBuffer* instanced_ibo);
  VulkanPrimitiveAssembly() = delete;
  ~VulkanPrimitiveAssembly() override = default;

  uint32_t numVertices() const override { return num_vertices_; }
  uint64_t getVertexBufferOffsetBytes() const override { return vertex_buffer_offset_; }
  uint32_t numIndices() const override { return num_indices_; }
  uint32_t numInstances() const override { return num_instances_; }

  bool isDirty() const override;
  void markDirty() override;

  const VkPipelineVertexInputStateCreateInfo* getPipelineVertexInputStateCI() const {
    return &vertex_input_state_ci_;
  }

  const VkPipelineInputAssemblyStateCreateInfo* getPipelineInputAssemblyStateCI() const {
    return &input_assembly_state_ci_;
  };

 protected:
  void init(const Material& material,
            const PrimitiveAssemblyAttrInfo& instanced_attr_info,
            const PrimitiveAssemblyAttrInfo& instances_attr_info,
            const IndexBuffer* ibo);

 private:
  VkPipelineVertexInputStateCreateInfo vertex_input_state_ci_;
  VkPipelineInputAssemblyStateCreateInfo input_assembly_state_ci_;
  std::vector<VkVertexInputAttributeDescription> attribute_descriptions_;
  std::vector<VkVertexInputBindingDescription> binding_descriptions_;

  uint32_t num_vertices_;
  uint64_t vertex_buffer_offset_;
  uint32_t num_indices_;
  uint32_t num_instances_;

  bool is_dirty_;
};

}  // namespace gfx
