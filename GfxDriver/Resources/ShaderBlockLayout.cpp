/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Resources/ShaderBlockLayout.h"

#include <sstream>

namespace gfx {

/*
 * buffer attribute templated type explicit instantiations
 */

template <BufferAttrType dataType>
struct BufferAttrTypeSelector {
  using type = void;
  static const int num_components = 1;
};
template <>
struct BufferAttrTypeSelector<BufferAttrType::kUint> {
  using type = unsigned int;
  static const int num_components = 1;
};
template <>
struct BufferAttrTypeSelector<BufferAttrType::kVec2ui> {
  using type = unsigned int;
  static const int num_components = 2;
};
template <>
struct BufferAttrTypeSelector<BufferAttrType::kVec3ui> {
  using type = unsigned int;
  static const int num_components = 3;
};
template <>
struct BufferAttrTypeSelector<BufferAttrType::kVec4ui> {
  using type = unsigned int;
  static const int num_components = 4;
};
template <>
struct BufferAttrTypeSelector<BufferAttrType::kInt> {
  using type = int;
  static const int num_components = 1;
};
template <>
struct BufferAttrTypeSelector<BufferAttrType::kVec2i> {
  using type = int;
  static const int num_components = 2;
};
template <>
struct BufferAttrTypeSelector<BufferAttrType::kVec3i> {
  using type = int;
  static const int num_components = 3;
};
template <>
struct BufferAttrTypeSelector<BufferAttrType::kVec4i> {
  using type = int;
  static const int num_components = 4;
};
template <>
struct BufferAttrTypeSelector<BufferAttrType::kFloat> {
  using type = float;
  static const int num_components = 1;
};
template <>
struct BufferAttrTypeSelector<BufferAttrType::kVec2f> {
  using type = float;
  static const int num_components = 2;
};
template <>
struct BufferAttrTypeSelector<BufferAttrType::kVec3f> {
  using type = float;
  static const int num_components = 3;
};
template <>
struct BufferAttrTypeSelector<BufferAttrType::kVec4f> {
  using type = float;
  static const int num_components = 4;
};
template <>
struct BufferAttrTypeSelector<BufferAttrType::kDouble> {
  using type = double;
  static const int num_components = 1;
};
template <>
struct BufferAttrTypeSelector<BufferAttrType::kVec2d> {
  using type = double;
  static const int num_components = 2;
};
template <>
struct BufferAttrTypeSelector<BufferAttrType::kVec3d> {
  using type = double;
  static const int num_components = 3;
};
template <>
struct BufferAttrTypeSelector<BufferAttrType::kVec4d> {
  using type = double;
  static const int num_components = 4;
};
template <>
struct BufferAttrTypeSelector<BufferAttrType::kUint64> {
  using type = uint64_t;
  static const int num_components = 1;
};
template <>
struct BufferAttrTypeSelector<BufferAttrType::kVec2ui64> {
  using type = uint64_t;
  static const int num_components = 2;
};
template <>
struct BufferAttrTypeSelector<BufferAttrType::kVec3ui64> {
  using type = uint64_t;
  static const int num_components = 3;
};
template <>
struct BufferAttrTypeSelector<BufferAttrType::kVec4ui64> {
  using type = uint64_t;
  static const int num_components = 4;
};
template <>
struct BufferAttrTypeSelector<BufferAttrType::kInt64> {
  using type = int64_t;
  static const int num_components = 1;
};
template <>
struct BufferAttrTypeSelector<BufferAttrType::kVec2i64> {
  using type = int64_t;
  static const int num_components = 2;
};
template <>
struct BufferAttrTypeSelector<BufferAttrType::kVec3i64> {
  using type = int64_t;
  static const int num_components = 3;
};
template <>
struct BufferAttrTypeSelector<BufferAttrType::kVec4i64> {
  using type = int64_t;
  static const int num_components = 4;
};
template <>
struct BufferAttrTypeSelector<BufferAttrType::kMat3x2f> {
  using type = float;
  static const int num_components = 6;
};
template <>
struct BufferAttrTypeSelector<BufferAttrType::kMat3x2d> {
  using type = double;
  static const int num_components = 6;
};
template <>
struct BufferAttrTypeSelector<BufferAttrType::kBool> {
  using type = int;
  static const int num_components = 1;
};

// all shader block layouts must be interleaved, but there are rules for
// the stride/offset for each layout type, hence the reasoning for
// a specific shader block layout.
ShaderBlockLayout::ShaderBlockLayout(ShaderBlockType block_type)
    : BaseBufferLayout(BufferLayoutType::kInterleaved)
    , block_type_(block_type)
    , adding_attrs_(false) {
  item_byte_size_ = 0;
}

ShaderBlockLayout::~ShaderBlockLayout() {}

bool ShaderBlockLayout::operator==(const ShaderBlockLayout& layout) const {
  bool check_attrs = (getLayoutType() == layout.getLayoutType() &&
                      getNumBytesInBlock() == layout.getNumBytesInBlock() &&
                      numAttributes() == layout.numAttributes());

  if (check_attrs) {
    for (int i = 0; i < numAttributes(); ++i) {
      if (operator[](i) != layout[i]) {
        return false;
      }
    }
    return true;
  }
  return false;
}

bool ShaderBlockLayout::operator!=(const ShaderBlockLayout& layout) const {
  return !operator==(layout);
}

void ShaderBlockLayout::beginAddingAttrs() {
  // TODO(croot): should we warn the user that attrs will be cleared (if there are any)
  if (adding_attrs_) {
    return;
  }

  attr_map_.clear();
  adding_attrs_ = true;
}

void ShaderBlockLayout::endAddingAttrs() {
  if (block_type_ == ShaderBlockType::kUniformBuffer) {
    // std430 block padding
    // can be removed with 'scalar' layout type
    static const uint64_t byte_alignment = 4 * sizeof(float);
    uint64_t offset;
    if ((offset = item_byte_size_ % byte_alignment) != 0) {
      item_byte_size_ += byte_alignment - offset;
    }
  }

  adding_attrs_ = false;
}

void ShaderBlockLayout::addAttribute(const std::string& attr_name,
                                     const BufferAttrType type) {
  CHECK(adding_attrs_)
      << "Please call the \"beginAddingAttrs()\" method before adding attributes.";

  RUNTIME_EX_ASSERT(!hasAttribute(attr_name) && attr_name.length(),
                    "ShaderBlockLayout::addAttribute(): attribute " + attr_name +
                        " already exists in the layout.");

  int enum_val = static_cast<int>(type);
  BaseTypeGLSL* type_glsl = attr_type_info[enum_val].get();

  // TODO(croot): there's a lot to do to make this fully functional,
  // i.e. supporting matrices and structs, and arrays of all types
  // Starting with the basics.
  uint64_t data_size = 0;
  int num_components = 0;
  switch (type) {
    case BufferAttrType::kUint:
      data_size = sizeof(BufferAttrTypeSelector<BufferAttrType::kUint>::type);
      num_components = BufferAttrTypeSelector<BufferAttrType::kUint>::num_components;
      break;
    case BufferAttrType::kVec2ui:
      data_size = sizeof(BufferAttrTypeSelector<BufferAttrType::kVec2ui>::type);
      num_components = BufferAttrTypeSelector<BufferAttrType::kVec2ui>::num_components;
      break;
    case BufferAttrType::kVec3ui:
      data_size = sizeof(BufferAttrTypeSelector<BufferAttrType::kVec3ui>::type);
      num_components = BufferAttrTypeSelector<BufferAttrType::kVec3ui>::num_components;
      break;
    case BufferAttrType::kVec4ui:
      data_size = sizeof(BufferAttrTypeSelector<BufferAttrType::kVec4ui>::type);
      num_components = BufferAttrTypeSelector<BufferAttrType::kVec4ui>::num_components;
      break;
    case BufferAttrType::kInt:
      data_size = sizeof(BufferAttrTypeSelector<BufferAttrType::kInt>::type);
      num_components = BufferAttrTypeSelector<BufferAttrType::kInt>::num_components;
      break;
    case BufferAttrType::kVec2i:
      data_size = sizeof(BufferAttrTypeSelector<BufferAttrType::kVec2i>::type);
      num_components = BufferAttrTypeSelector<BufferAttrType::kVec2i>::num_components;
      break;
    case BufferAttrType::kVec3i:
      data_size = sizeof(BufferAttrTypeSelector<BufferAttrType::kVec3i>::type);
      num_components = BufferAttrTypeSelector<BufferAttrType::kVec3i>::num_components;
      break;
    case BufferAttrType::kVec4i:
      data_size = sizeof(BufferAttrTypeSelector<BufferAttrType::kVec4i>::type);
      num_components = BufferAttrTypeSelector<BufferAttrType::kVec4i>::num_components;
      break;
    case BufferAttrType::kFloat:
      data_size = sizeof(BufferAttrTypeSelector<BufferAttrType::kFloat>::type);
      num_components = BufferAttrTypeSelector<BufferAttrType::kFloat>::num_components;
      break;
    case BufferAttrType::kVec2f:
      data_size = sizeof(BufferAttrTypeSelector<BufferAttrType::kVec2f>::type);
      num_components = BufferAttrTypeSelector<BufferAttrType::kVec2f>::num_components;
      break;
    case BufferAttrType::kVec3f:
      data_size = sizeof(BufferAttrTypeSelector<BufferAttrType::kVec3f>::type);
      num_components = BufferAttrTypeSelector<BufferAttrType::kVec3f>::num_components;
      break;
    case BufferAttrType::kVec4f:
      data_size = sizeof(BufferAttrTypeSelector<BufferAttrType::kVec4f>::type);
      num_components = BufferAttrTypeSelector<BufferAttrType::kVec4f>::num_components;
      break;
    case BufferAttrType::kDouble:
      data_size = sizeof(BufferAttrTypeSelector<BufferAttrType::kDouble>::type);
      num_components = BufferAttrTypeSelector<BufferAttrType::kDouble>::num_components;
      break;
    case BufferAttrType::kVec2d:
      data_size = sizeof(BufferAttrTypeSelector<BufferAttrType::kVec2d>::type);
      num_components = BufferAttrTypeSelector<BufferAttrType::kVec2d>::num_components;
      break;
    case BufferAttrType::kVec3d:
      data_size = sizeof(BufferAttrTypeSelector<BufferAttrType::kVec3d>::type);
      num_components = BufferAttrTypeSelector<BufferAttrType::kVec3d>::num_components;
      break;
    case BufferAttrType::kVec4d:
      data_size = sizeof(BufferAttrTypeSelector<BufferAttrType::kVec4d>::type);
      num_components = BufferAttrTypeSelector<BufferAttrType::kVec4d>::num_components;
      break;
    case BufferAttrType::kUint64:
      data_size = sizeof(BufferAttrTypeSelector<BufferAttrType::kUint64>::type);
      num_components = BufferAttrTypeSelector<BufferAttrType::kUint64>::num_components;
      break;
    case BufferAttrType::kVec2ui64:
      data_size = sizeof(BufferAttrTypeSelector<BufferAttrType::kVec2ui64>::type);
      num_components = BufferAttrTypeSelector<BufferAttrType::kVec2ui64>::num_components;
      break;
    case BufferAttrType::kVec3ui64:
      data_size = sizeof(BufferAttrTypeSelector<BufferAttrType::kVec3ui64>::type);
      num_components = BufferAttrTypeSelector<BufferAttrType::kVec3ui64>::num_components;
      break;
    case BufferAttrType::kVec4ui64:
      data_size = sizeof(BufferAttrTypeSelector<BufferAttrType::kVec4ui64>::type);
      num_components = BufferAttrTypeSelector<BufferAttrType::kVec4ui64>::num_components;
      break;
    case BufferAttrType::kInt64:
      data_size = sizeof(BufferAttrTypeSelector<BufferAttrType::kInt64>::type);
      num_components = BufferAttrTypeSelector<BufferAttrType::kInt64>::num_components;
      break;
    case BufferAttrType::kVec2i64:
      data_size = sizeof(BufferAttrTypeSelector<BufferAttrType::kVec2i64>::type);
      num_components = BufferAttrTypeSelector<BufferAttrType::kVec2i64>::num_components;
      break;
    case BufferAttrType::kVec3i64:
      data_size = sizeof(BufferAttrTypeSelector<BufferAttrType::kVec3i64>::type);
      num_components = BufferAttrTypeSelector<BufferAttrType::kVec3i64>::num_components;
      break;
    case BufferAttrType::kVec4i64:
      data_size = sizeof(BufferAttrTypeSelector<BufferAttrType::kVec4i64>::type);
      num_components = BufferAttrTypeSelector<BufferAttrType::kVec4i64>::num_components;
      break;
    case BufferAttrType::kMat3x2f:
      data_size = sizeof(BufferAttrTypeSelector<BufferAttrType::kMat3x2f>::type);
      num_components = BufferAttrTypeSelector<BufferAttrType::kMat3x2f>::num_components;
      break;
    case BufferAttrType::kMat3x2d:
      data_size = sizeof(BufferAttrTypeSelector<BufferAttrType::kMat3x2d>::type);
      num_components = BufferAttrTypeSelector<BufferAttrType::kMat3x2d>::num_components;
      break;
    case BufferAttrType::kBool:
      data_size = sizeof(BufferAttrTypeSelector<BufferAttrType::kBool>::type);
      num_components = BufferAttrTypeSelector<BufferAttrType::kBool>::num_components;
      break;
    case BufferAttrType::kCOUNT:
      CHECK(false);
  }

  const auto offset = addDataFromSizeInternal(data_size, num_components);

  attr_map_.emplace_back(attr_name, type, type_glsl, -1, item_byte_size_);
  item_byte_size_ += offset;
}

std::string ShaderBlockLayout::buildShaderBlockCode(const std::string& block_name,
                                                    const std::string& instance_name) {
  CHECK(!adding_attrs_) << "Please call the endAddingAttrs() method before calling "
                           "methods requiring the all attributes to be added.";

  std::stringstream ss;

  if (block_type_ == ShaderBlockType::kUniformBuffer) {
    ss << "layout(std430) uniform " << block_name << " {\n";
    for (auto& attr_info : attr_map_) {
      ss << "  " << attr_info.type_info->declString() << " " << attr_info.name << ";\n";
    }
    ss << "}";

    if (instance_name.length()) {
      ss << " " << instance_name << ";\n";
    } else {
      ss << ";\n";
    }
  } else if (block_type_ == ShaderBlockType::kStorageBuffer) {
    // also must have valid instance name
    CHECK(!instance_name.empty()) << "SSBOs must have a valid instance name";

    ss << "struct " << block_name << "Type {\n";
    for (auto& attr_info : attr_map_) {
      ss << "  " << attr_info.type_info->declString() << " " << attr_info.name << ";\n";
    }
    ss << "};\n";

    ss << "layout(std430) readonly buffer " << block_name << " {\n";
    ss << "  " << block_name << "Type " << instance_name << "[];\n";
    ss << "};";
  }

  return ss.str();
}

uint64_t ShaderBlockLayout::addDataFromSizeInternal(const uint64_t data_size,
                                                    const uint32_t num_components) {
  if (num_components == 3) {
    return data_size * 4;
  }
  return data_size * num_components;
}

void ShaderBlockLayout::bindToMaterial(BindAttributeLambda bind_attribute,
                                       const Material& material,
                                       const uint64_t used_bytes,
                                       const std::string& attr,
                                       const std::string& shader_attr,
                                       const uint32_t num_instances_per_attr) const {
  THROW_RUNTIME_EX(
      "A ShaderBlockLayout cannot be bound to a shader. It is instead defined by a "
      "shader via a shader storage "
      "block.")
}

}  // namespace gfx
