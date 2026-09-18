/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Resources/BufferLayout.h"

#include "GfxDriver/Pipeline/Material.h"

namespace gfx {

const std::array<TypeGLSLUqPtr, static_cast<uint32_t>(BufferAttrType::kCOUNT)>
    BaseBufferLayout::attr_type_info = {std::make_unique<TypeGLSL<uint32_t, 1>>(),
                                        std::make_unique<TypeGLSL<uint32_t, 2>>(),
                                        std::make_unique<TypeGLSL<uint32_t, 3>>(),
                                        std::make_unique<TypeGLSL<uint32_t, 4>>(),
                                        std::make_unique<TypeGLSL<int32_t, 1>>(),
                                        std::make_unique<TypeGLSL<int32_t, 2>>(),
                                        std::make_unique<TypeGLSL<int32_t, 3>>(),
                                        std::make_unique<TypeGLSL<int32_t, 4>>(),
                                        std::make_unique<TypeGLSL<float, 1>>(),
                                        std::make_unique<TypeGLSL<float, 2>>(),
                                        std::make_unique<TypeGLSL<float, 3>>(),
                                        std::make_unique<TypeGLSL<float, 4>>(),
                                        std::make_unique<TypeGLSL<double, 1>>(),
                                        std::make_unique<TypeGLSL<double, 2>>(),
                                        std::make_unique<TypeGLSL<double, 3>>(),
                                        std::make_unique<TypeGLSL<double, 4>>(),
                                        std::make_unique<TypeGLSL<uint64_t, 1>>(),
                                        std::make_unique<TypeGLSL<uint64_t, 2>>(),
                                        std::make_unique<TypeGLSL<uint64_t, 3>>(),
                                        std::make_unique<TypeGLSL<uint64_t, 4>>(),
                                        std::make_unique<TypeGLSL<int64_t, 1>>(),
                                        std::make_unique<TypeGLSL<int64_t, 2>>(),
                                        std::make_unique<TypeGLSL<int64_t, 3>>(),
                                        std::make_unique<TypeGLSL<int64_t, 4>>(),
                                        std::make_unique<TypeGLSL<float, 6>>(),
                                        std::make_unique<TypeGLSL<double, 6>>()};

BufferAttrType BaseBufferLayout::getBufferAttrType(const uint32_t a,
                                                   const uint32_t num_components) {
  RUNTIME_EX_ASSERT(num_components == 1,
                    "Only 1 component of uint32_ts are currently supported.");

  return BufferAttrType::kUint;
}

BufferAttrType BaseBufferLayout::getBufferAttrType(const int32_t a,
                                                   const uint32_t num_components) {
  switch (num_components) {
    case 1:
      return BufferAttrType::kInt;
    case 2:
      return BufferAttrType::kVec2i;
    case 3:
      return BufferAttrType::kVec3i;
    case 4:
      return BufferAttrType::kVec4i;
    default:
      THROW_RUNTIME_EX("Invalid number of components " + std::to_string(num_components) +
                       ". Need 1-4 components.");
  }

  return BufferAttrType::kInt;
}

BufferAttrType BaseBufferLayout::getBufferAttrType(const float a,
                                                   const uint32_t num_components) {
  switch (num_components) {
    case 1:
      return BufferAttrType::kFloat;
    case 2:
      return BufferAttrType::kVec2f;
    case 3:
      return BufferAttrType::kVec3f;
    case 4:
      return BufferAttrType::kVec4f;
    case 6:
      return BufferAttrType::kMat3x2f;
    default:
      THROW_RUNTIME_EX("Invalid number of components " + std::to_string(num_components) +
                       ". Need 1-4 or 6 components.");
  }

  return BufferAttrType::kFloat;
}

BufferAttrType BaseBufferLayout::getBufferAttrType(const double a,
                                                   const uint32_t num_components) {
  switch (num_components) {
    case 1:
      return BufferAttrType::kDouble;
    case 2:
      return BufferAttrType::kVec2d;
    case 3:
      return BufferAttrType::kVec3d;
    case 4:
      return BufferAttrType::kVec4d;
    case 6:
      return BufferAttrType::kMat3x2d;
    default:
      THROW_RUNTIME_EX("Invalid number of components " + std::to_string(num_components) +
                       ". Need 1-4 or 6 components.");
  }

  return BufferAttrType::kDouble;
}

BufferAttrType BaseBufferLayout::getBufferAttrType(const uint64_t a,
                                                   const uint32_t num_components) {
  switch (num_components) {
    case 1:
      return BufferAttrType::kUint64;
    case 2:
      return BufferAttrType::kVec2ui64;
    case 3:
      return BufferAttrType::kVec3ui64;
    case 4:
      return BufferAttrType::kVec4ui64;
    default:
      THROW_RUNTIME_EX("Invalid number of components " + std::to_string(num_components) +
                       ". Need 1-4 components.");
  }

  return BufferAttrType::kUint64;
}

BufferAttrType BaseBufferLayout::getBufferAttrType(const int64_t a,
                                                   const uint32_t num_components) {
  switch (num_components) {
    case 1:
      return BufferAttrType::kInt64;
    case 2:
      return BufferAttrType::kVec2i64;
    case 3:
      return BufferAttrType::kVec3i64;
    case 4:
      return BufferAttrType::kVec4i64;
    default:
      THROW_RUNTIME_EX("Invalid number of components " + std::to_string(num_components) +
                       ". Need 1-4 components.");
  }

  return BufferAttrType::kInt64;
}

BaseBufferLayout::BaseBufferLayout(const BaseBufferLayout& layout)
    : layout_type_(layout.layout_type_)
    , attr_map_(layout.attr_map_)
    , item_byte_size_(layout.item_byte_size_) {}

BaseBufferLayout::BaseBufferLayout(BufferLayoutType layout_type)
    : layout_type_(layout_type), item_byte_size_(0) {}

bool BaseBufferLayout::hasAttribute(const std::string& attr_name) const {
  const BufferAttrMap_by_name& name_lookup = attr_map_.get<name>();
  return (name_lookup.find(attr_name) != name_lookup.end());
}

TypeGLSLShPtr BaseBufferLayout::getAttributeTypeGLSL(const std::string& attr_name) const {
  const BufferAttrMap_by_name& name_lookup = attr_map_.get<name>();
  BufferAttrMap_by_name::iterator itr;

  RUNTIME_EX_ASSERT((itr = name_lookup.find(attr_name)) != name_lookup.end(),
                    "BaseBufferLayout::getAttributeTypeGLSL(): attribute \'" + attr_name +
                        "\' does not exist in layout.");

  return itr->type_info->clone();
}

BufferAttrType BaseBufferLayout::getAttributeType(const std::string& attr_name) const {
  // TODO(croot): consolidate this code and the one in getAttributeTypeGLSL()
  // into a single getBufferAttrInfo func or something.
  const BufferAttrMap_by_name& name_lookup = attr_map_.get<name>();
  BufferAttrMap_by_name::iterator itr;

  RUNTIME_EX_ASSERT((itr = name_lookup.find(attr_name)) != name_lookup.end(),
                    "BaseBufferLayout::getAttributeType(): attribute \'" + attr_name +
                        "\' does not exist in layout.");

  return itr->type;
}

const BufferAttrInfo& BaseBufferLayout::getAttributeInfo(
    const std::string& attr_name) const {
  // TODO(croot): consolidate this code with those in the above two functions
  // into a single getBufferAttrInfo func or something.
  const BufferAttrMap_by_name& name_lookup = attr_map_.get<name>();
  BufferAttrMap_by_name::iterator itr;

  RUNTIME_EX_ASSERT((itr = name_lookup.find(attr_name)) != name_lookup.end(),
                    "BaseBufferLayout::getAttributeInfo(): attribute \'" + attr_name +
                        "\' does not exist in layout.");

  return *itr;
}

int32_t BaseBufferLayout::getAttributeByteOffset(const std::string& attr_name) const {
  return getAttributeInfo(attr_name).offset;
}

bool BaseBufferLayout::operator==(const BaseBufferLayout& layout) const {
  if (layout_type_ != layout.layout_type_) {
    return false;
  }

  if (attr_map_.size() != layout.attr_map_.size()) {
    return false;
  }

  for (size_t i = 0; i < attr_map_.size(); ++i) {
    if (attr_map_[i] != layout.attr_map_[i]) {
      return false;
    }
  }
  return true;
}

const BufferAttrInfo& BaseBufferLayout::operator[](const uint32_t i) const {
  RUNTIME_EX_ASSERT(
      i < attr_map_.size(),
      "BaseBufferLayout::operator[]: cannot retrieve attribute info at index: " +
          std::to_string(i) + ". The layout only has " +
          std::to_string(attr_map_.size()) + " attributes.");
  return attr_map_[i];
}

void InterleavedBufferLayout::addAttribute(const std::string& attr_name,
                                           const BufferAttrType type) {
  RUNTIME_EX_ASSERT(!hasAttribute(attr_name) && attr_name.length(),
                    "InterleavedBufferLayout::addAttribute(): attribute \'" + attr_name +
                        "\' already exists in the layout.");

  // TODO(croot), set the stride of all currently existing attrs, or leave
  // that for when the layout is bound to the renderer/shader/VAO

  int32_t enum_val = static_cast<int32_t>(type);
  attr_map_.emplace_back(
      attr_name, type, attr_type_info[enum_val].get(), -1, item_byte_size_);

  item_byte_size_ += attr_type_info[enum_val]->numBytes();
}

void InterleavedBufferLayout::bindToMaterial(
    BindAttributeLambda bind_attribute,
    const Material& material,
    const uint64_t used_bytes,
    const std::string& attr,
    const std::string& shader_attr,
    const uint32_t num_instances_for_attr) const {
  const BufferAttrMap_by_name& name_lookup = attr_map_.get<name>();
  BufferAttrMap_by_name::iterator itr;

  RUNTIME_EX_ASSERT((itr = name_lookup.find(attr)) != name_lookup.end(),
                    "InterleavedBufferLayout::bindToMaterial(): attribute \'" + attr +
                        "\' does not exist in layout.");
  bind_attribute(
      itr->type_info,
      material.getVertexAttributeLocation(shader_attr.length() ? shader_attr : itr->name),
      item_byte_size_,
      itr->offset,
      num_instances_for_attr);
}

void SequentialBufferLayout::addAttribute(const std::string& attr_name,
                                          BufferAttrType type) {
  RUNTIME_EX_ASSERT(!hasAttribute(attr_name) && attr_name.length(),
                    "SequentialBufferLayout::addAttribute(): attribute \'" + attr_name +
                        "\' already exists in the layout.");

  uint32_t enum_val = static_cast<uint32_t>(type);
  attr_map_.emplace_back(attr_name,
                         type,
                         attr_type_info[enum_val].get(),
                         attr_type_info[enum_val]->numBytes(),
                         -1);

  item_byte_size_ += attr_type_info[enum_val]->numBytes();
}

void SequentialBufferLayout::bindToMaterial(BindAttributeLambda bind_attribute,
                                            const Material& material,
                                            const uint64_t used_bytes,
                                            const std::string& attr,
                                            const std::string& shader_attr,
                                            const uint32_t num_instances_for_attr) const {
  RUNTIME_EX_ASSERT(hasAttribute(attr),
                    "SequentialBufferLayout::bindToRenderer(): attribute \'" + attr +
                        "\' doesn't exist in layout.");

  uint64_t attr_offset_bytes = 0u;
  auto num_verts = used_bytes / item_byte_size_;

  for (auto const& element : attr_map_) {
    auto const& attr_type_glsl = element.type_info;
    if (element.name == attr) {
      auto attr_location = material.getVertexAttributeLocation(
          shader_attr.length() ? shader_attr : element.name);
      bind_attribute(attr_type_glsl,
                     attr_location,
                     element.stride,
                     attr_offset_bytes,
                     num_instances_for_attr);
      break;
    }
    attr_offset_bytes += attr_type_glsl->numBytes() * num_verts;
  }
}

}  // namespace gfx
