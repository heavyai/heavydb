/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Drivers/Vulkan/Pipeline/VulkanPrimitiveAssembly.h"

#include "GfxDriver/Pipeline/Material.h"
#include "GfxDriver/Resources/IndexBuffer.h"
#include "GfxDriver/Resources/VertexBuffer.h"

namespace gfx {

namespace {

VkFormat attr_type_to_vk_format(BaseTypeGLSL* attr_type) {
  CHECK(attr_type);
  switch (attr_type->baseType()) {
    case BufferAttrType::kDouble:
      switch (attr_type->numComponents()) {
        case 1:
          return VK_FORMAT_R64_SFLOAT;
        case 2:
          return VK_FORMAT_R64G64_SFLOAT;
        case 3:
          return VK_FORMAT_R64G64B64_SFLOAT;
        case 4:
          return VK_FORMAT_R64G64B64A64_SFLOAT;
        default:
          CHECK(false);
      }
      break;
    case BufferAttrType::kInt64:
      switch (attr_type->numComponents()) {
        case 1:
          return VK_FORMAT_R64_SINT;
        case 2:
          return VK_FORMAT_R64G64_SINT;
        case 3:
          return VK_FORMAT_R64G64B64_SINT;
        case 4:
          return VK_FORMAT_R64G64B64A64_SINT;
        default:
          CHECK(false);
      }
      break;
    case BufferAttrType::kUint64:
      switch (attr_type->numComponents()) {
        case 1:
          return VK_FORMAT_R64_UINT;
        case 2:
          return VK_FORMAT_R64G64_UINT;
        case 3:
          return VK_FORMAT_R64G64B64_UINT;
        case 4:
          return VK_FORMAT_R64G64B64A64_UINT;
        default:
          CHECK(false);
      }
      break;
    case BufferAttrType::kInt:
      switch (attr_type->numComponents()) {
        case 1:
          return VK_FORMAT_R32_SINT;
        case 2:
          return VK_FORMAT_R32G32_SINT;
        case 3:
          return VK_FORMAT_R32G32B32_SINT;
        case 4:
          return VK_FORMAT_R32G32B32A32_SINT;
        default:
          CHECK(false);
      }
      break;
    case BufferAttrType::kUint:
      switch (attr_type->numComponents()) {
        case 1:
          return VK_FORMAT_R32_UINT;
        case 2:
          return VK_FORMAT_R32G32_UINT;
        case 3:
          return VK_FORMAT_R32G32B32_UINT;
        case 4:
          return VK_FORMAT_R32G32B32A32_UINT;
        default:
          CHECK(false);
      }
      break;
    case BufferAttrType::kFloat:
      switch (attr_type->numComponents()) {
        case 1:
          return VK_FORMAT_R32_SFLOAT;
        case 2:
          return VK_FORMAT_R32G32_SFLOAT;
        case 3:
          return VK_FORMAT_R32G32B32_SFLOAT;
        case 4:
          return VK_FORMAT_R32G32B32A32_SFLOAT;
        default:
          CHECK(false);
      }
      break;
    default:
      CHECK(false);
  }
  return VK_FORMAT_UNDEFINED;
}

VkPrimitiveTopology primitive_topology_to_vk_primitive_topology(
    PrimitiveTopology topology) {
  switch (topology) {
    case PrimitiveTopology::kPointList:
      return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
    case PrimitiveTopology::kLineList:
      return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
    case PrimitiveTopology::kLineStrip:
      return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
    case PrimitiveTopology::kTriangleList:
      return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    case PrimitiveTopology::kTriangleStrip:
      return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
    case PrimitiveTopology::kTriangleFan:
      return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN;
    case PrimitiveTopology::kLineListAdjacency:
      return VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY;
    case PrimitiveTopology::kLineStripAdjacency:
      return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY;
    case PrimitiveTopology::kTriangleListAdjacency:
      return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY;
    case PrimitiveTopology::kTriangleStripAdjacency:
      return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY;
  }
  UNREACHABLE();
  return VK_PRIMITIVE_TOPOLOGY_MAX_ENUM;
}

}  // namespace

VulkanPrimitiveAssembly::VulkanPrimitiveAssembly(
    const DeviceContext& device_ctx,
    std::string_view resource_tracking_string,
    const PrimitiveTopology topology,
    const Material& material,
    const PrimitiveAssemblyAttrInfo& attr_info,
    const IndexBuffer* ibo)
    : PrimitiveAssembly(topology, attr_info, ibo)
    , num_vertices_{0}
    , vertex_buffer_offset_{0}
    , num_indices_{0}
    , num_instances_{1}
    , is_dirty_{false} {
  // init with just vertices data
  init(material, attr_info, PrimitiveAssemblyAttrInfo(), ibo);

  // make this PA dependent on the given VBO and IBO
  addDependency(attr_info.vbo_and_layout.vertex_buffer->getPrimitiveAssemblyDependency());
  if (ibo) {
    addDependency(ibo->getPrimitiveAssemblyDependency());
  }
}

VulkanPrimitiveAssembly::VulkanPrimitiveAssembly(
    const DeviceContext& device_ctx,
    std::string_view resource_tracking_string,
    const PrimitiveTopology topology,
    const Material& material,
    const PrimitiveAssemblyAttrInfo& instanced_attr_info,
    const PrimitiveAssemblyAttrInfo& instances_attr_info,
    const uint32_t num_instances_per_attr,
    const IndexBuffer* instanced_ibo)
    : PrimitiveAssembly(topology)
    , num_vertices_{0}
    , vertex_buffer_offset_{0}
    , num_indices_{0}
    , num_instances_{1}
    , is_dirty_{false} {
  // init with both vertices and instances data
  init(material, instanced_attr_info, instances_attr_info, instanced_ibo);

  // make this PA dependent on the given VBOs and IBO
  addDependency(
      instanced_attr_info.vbo_and_layout.vertex_buffer->getPrimitiveAssemblyDependency());
  addDependency(
      instances_attr_info.vbo_and_layout.vertex_buffer->getPrimitiveAssemblyDependency());
  if (instanced_ibo) {
    addDependency(instanced_ibo->getPrimitiveAssemblyDependency());
  }
}

void VulkanPrimitiveAssembly::init(const Material& material,
                                   const PrimitiveAssemblyAttrInfo& instanced_attr_info,
                                   const PrimitiveAssemblyAttrInfo& instances_attr_info,
                                   const IndexBuffer* ibo) {
  // reset
  attribute_descriptions_.clear();
  binding_descriptions_.clear();

  // capture the index count
  num_indices_ = ibo ? ibo->numItems() : 0U;

  auto process_vbo = [this, &material](const PrimitiveAssemblyAttrInfo& attr_info,
                                       uint32_t& num_items,
                                       const uint32_t binding,
                                       const VkVertexInputRate vertex_input_rate) {
    // unpack and validate the VBO
    CHECK(attr_info.vbo_and_layout.vertex_buffer);
    auto const* vbo =
        static_cast<const VertexBuffer*>(attr_info.vbo_and_layout.vertex_buffer);
    CHECK(vbo->getType() == BufferType::kVertexBuffer);

    // unpack the layout
    // if one is not passed in, just use the first one the LM knows about
    // this will be the case in embedded-data mode
    // @TODO support sequential layout
    auto layout = attr_info.vbo_and_layout.buffer_layout;
    CHECK(vbo->hasLayout());
    if (layout == nullptr) {
      layout = vbo->getLayoutManager()->getBufferLayoutAtIndex(0);
    }
    CHECK(layout);
    CHECK(layout->getLayoutType() == BufferLayoutType::kInterleaved);

    // capture the count
    num_items = vbo->getLayoutManager()->numItems(layout);

    // get the layout details
    auto layout_data = vbo->getLayoutManager()->getBufferLayoutDataToUse(
        layout, "Cannot bind vertex buffer to shader. ");

    // get vertex buffer offset
    vertex_buffer_offset_ = layout_data.offset_bytes;

    // prepare to capture stride
    uint32_t the_stride = 0U;

    // lambda for the layout to call for each attribute
    auto bind_attribute = [this, &the_stride, binding](gfx::BaseTypeGLSL* attr_type,
                                                       uint32_t location,
                                                       uint64_t stride,
                                                       uint32_t attr_offset_bytes,
                                                       uint32_t num_instances) {
      // append a description for this attribute
      VkVertexInputAttributeDescription attribute_description = {};
      attribute_description.binding = binding;
      attribute_description.location = location;
      attribute_description.format = attr_type_to_vk_format(attr_type);
      attribute_description.offset = attr_offset_bytes;
      attribute_descriptions_.push_back(attribute_description);

      // capture stride and enforce that it is the same for all attributes
      if (the_stride == 0U) {
        the_stride = stride;
      } else {
        CHECK_EQ(the_stride, stride);
      }
    };

    // tell the layout to process the attributes using the above lambda
    for (auto const& attr_pair : attr_info.attr_pairs) {
      layout_data.layout->bindToMaterial(
          bind_attribute,
          material,
          layout_data.used_bytes,
          attr_pair.first,
          attr_pair.second,
          1);  // should this be num_instances_ ? but it's not used in the lambda!
    }

    // binding description
    VkVertexInputBindingDescription binding_description = {};
    binding_description.binding = binding;
    binding_description.stride = the_stride;
    binding_description.inputRate = vertex_input_rate;
    binding_descriptions_.push_back(binding_description);
  };

  // process vertices?
  if (instanced_attr_info.vbo_and_layout.vertex_buffer) {
    process_vbo(instanced_attr_info, num_vertices_, 0, VK_VERTEX_INPUT_RATE_VERTEX);
  }

  // process instances?
  if (instances_attr_info.vbo_and_layout.vertex_buffer) {
    CHECK(instanced_attr_info.vbo_and_layout.vertex_buffer);
    process_vbo(instances_attr_info, num_instances_, 1, VK_VERTEX_INPUT_RATE_INSTANCE);
  }

  // vertex input state
  vertex_input_state_ci_ = {};
  vertex_input_state_ci_.sType =
      VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  vertex_input_state_ci_.vertexBindingDescriptionCount = binding_descriptions_.size();
  vertex_input_state_ci_.vertexAttributeDescriptionCount = attribute_descriptions_.size();
  vertex_input_state_ci_.pVertexBindingDescriptions = binding_descriptions_.data();
  vertex_input_state_ci_.pVertexAttributeDescriptions = attribute_descriptions_.data();

  // input assembly state
  input_assembly_state_ci_ = {};
  input_assembly_state_ci_.sType =
      VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  input_assembly_state_ci_.topology =
      primitive_topology_to_vk_primitive_topology(getTopology());
  input_assembly_state_ci_.primitiveRestartEnable = VK_FALSE;
}

bool VulkanPrimitiveAssembly::isDirty() const {
  return is_dirty_;
}

void VulkanPrimitiveAssembly::markDirty() {
  is_dirty_ = true;
}

}  // namespace gfx
