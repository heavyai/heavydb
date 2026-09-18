/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "GfxDriver/Resources/BufferLayout.h"

namespace gfx {

class ShaderBlockLayout : public BaseBufferLayout {
 public:
  ShaderBlockLayout(ShaderBlockType block_type);
  ~ShaderBlockLayout() override;

  bool operator==(const ShaderBlockLayout& layout) const;
  bool operator!=(const ShaderBlockLayout& layout) const;

  uint64_t getNumBytesInBlock() const {
    CHECK(!adding_attrs_) << "Please call the endAddingAttrs() method before calling "
                             "methods requiring the all attributes to be added.";
    return item_byte_size_;
  }

  uint64_t getNumBytesPerItem() const override { return item_byte_size_; }

  void beginAddingAttrs();
  void endAddingAttrs();

  template <typename T, int num_components = 1>
  void addAttribute(const std::string& attr_name) {
    CHECK(adding_attrs_)
        << "Please call the \"beginAddingAttrs()\" method before adding attributes.";

    RUNTIME_EX_ASSERT(!hasAttribute(attr_name) && attr_name.length(),
                      "ShaderBlockLayout::addAttribute(): attribute " + attr_name +
                          " already exists in the layout.");

    BufferAttrType type = getBufferAttrType(T(0), num_components);
    int enum_val = static_cast<int>(type);
    BaseTypeGLSL* type_glsl = attr_type_info[enum_val].get();

    // TODO(croot): there's a lot to do to make this fully functional,
    // i.e. supporting matrices and structs, and arrays of all types
    // Starting with the basics.

    int data_size = sizeof(T);
    const auto offset = addDataFromSizeInternal(data_size, num_components);

    attr_map_.emplace_back(attr_name, type, type_glsl, -1, item_byte_size_);
    item_byte_size_ += offset;
  }

  void addAttribute(const std::string& attr_name, const BufferAttrType type);

  // Build shader code declaring the shader block (uniform or storage buffer)
  // Storage buffer blocks require a valid instance name
  std::string buildShaderBlockCode(const std::string& block_name,
                                   const std::string& instance_name);

 private:
  uint64_t addDataFromSizeInternal(const uint64_t data_size,
                                   const uint32_t num_components);

  void bindToMaterial(BindAttributeLambda bind_attribute,
                      const Material& material,
                      const uint64_t used_bytes,
                      const std::string& attr = "",
                      const std::string& shader_attr = "",
                      const uint32_t num_instances_per_attr = 0) const override;

  ShaderBlockType block_type_;
  bool adding_attrs_;
};

}  // namespace gfx
