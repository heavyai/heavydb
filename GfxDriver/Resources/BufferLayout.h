/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>

#include <boost/multi_index/hashed_index.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/random_access_index.hpp>
#include <boost/multi_index_container.hpp>

#include "GfxDriver/RenderError.h"
#include "GfxDriver/Resources/Enums.h"
#include "GfxDriver/TypeGLSL.h"
#include "GfxDriver/Types.h"

namespace gfx {

struct BufferAttrInfo {
  std::string name;
  BufferAttrType type;
  BaseTypeGLSL* type_info;
  uint64_t stride;
  uint64_t offset;

  BufferAttrInfo(const BufferAttrInfo& other)
      : name(other.name)
      , type(other.type)
      , type_info(other.type_info)
      , stride(other.stride)
      , offset(other.offset) {}
  BufferAttrInfo(const std::string& name,
                 BufferAttrType type,
                 BaseTypeGLSL* type_info,
                 uint64_t stride,
                 uint64_t offset)
      : name(name), type(type), type_info(type_info), stride(stride), offset(offset) {}

  bool operator==(const BufferAttrInfo& attr_info) const {
    return (type == attr_info.type && stride == attr_info.stride &&
            offset == attr_info.offset);
  }

  bool operator!=(const BufferAttrInfo& attr_info) const {
    return !operator==(attr_info);
  }
};

class BaseBufferLayout {
 public:
  BaseBufferLayout(const BaseBufferLayout& layout);
  BaseBufferLayout(BufferLayoutType layout_type);
  virtual ~BaseBufferLayout() {}

  BufferLayoutType getLayoutType() const { return layout_type_; }
  virtual uint64_t getNumBytesPerItem() const { return item_byte_size_; }

  bool hasAttribute(const std::string& attr_name) const;

  TypeGLSLShPtr getAttributeTypeGLSL(const std::string& attr_name) const;

  BufferAttrType getAttributeType(const std::string& attr_name) const;

  const BufferAttrInfo& getAttributeInfo(const std::string& attr_name) const;
  int getAttributeByteOffset(const std::string& attr_name) const;

  // TODO(croot): add an iterator to iterate over the attributes?
  inline int numAttributes() const { return attr_map_.size(); }
  const BufferAttrInfo& operator[](const uint32_t i) const;

  bool operator==(const BaseBufferLayout& layout) const;
  bool operator!=(const BaseBufferLayout& layout) const { return !operator==(layout); }

  using BindAttributeLambda = std::function<void(gfx::BaseTypeGLSL* attr,
                                                 uint32_t location,
                                                 uint64_t stride,
                                                 uint32_t attr_offset_bytes,
                                                 uint32_t num_instances)>;

  virtual void bindToMaterial(BindAttributeLambda bind_attribute,
                              const Material& material,
                              const uint64_t used_bytes,
                              const std::string& attr = "",
                              const std::string& shader_attr = "",
                              const uint32_t num_instances_for_attr = 0) const = 0;

 protected:
  // tags for boost::multi_index_container
  struct name {};

  using BufferAttrMap = boost::multi_index_container<
      BufferAttrInfo,
      boost::multi_index::indexed_by<
          boost::multi_index::random_access<>,

          // hashed on name
          boost::multi_index::hashed_unique<
              boost::multi_index::tag<name>,
              boost::multi_index::
                  member<BufferAttrInfo, std::string, &BufferAttrInfo::name>>>>;

  using BufferAttrMap_by_name = BufferAttrMap::index<name>::type;

  BufferLayoutType layout_type_;
  BufferAttrMap attr_map_;
  uint64_t item_byte_size_;

  static const std::array<TypeGLSLUqPtr, static_cast<uint32_t>(BufferAttrType::kCOUNT)>
      attr_type_info;
  static BufferAttrType getBufferAttrType(const uint32_t a,
                                          const uint32_t num_components = 1);
  static BufferAttrType getBufferAttrType(const int32_t a,
                                          const uint32_t num_components = 1);
  static BufferAttrType getBufferAttrType(const float a,
                                          const uint32_t num_components = 1);
  static BufferAttrType getBufferAttrType(const double a,
                                          const uint32_t num_components = 1);
  static BufferAttrType getBufferAttrType(const uint64_t a,
                                          const uint32_t num_components = 1);
  static BufferAttrType getBufferAttrType(const int64_t a,
                                          const uint32_t num_components = 1);
};

class InterleavedBufferLayout : public BaseBufferLayout {
 public:
  InterleavedBufferLayout(const InterleavedBufferLayout& layout)
      : BaseBufferLayout(layout) {}
  InterleavedBufferLayout() : BaseBufferLayout(BufferLayoutType::kInterleaved) {}

  void addAttribute(const std::string& attr_name, const BufferAttrType type);

  template <typename T, uint32_t num_components = 1>
  void addAttribute(const std::string& attr_name) {
    RUNTIME_EX_ASSERT(!hasAttribute(attr_name) && attr_name.length(),
                      "InterleavedBufferLayout::addAttribute(): attribute " + attr_name +
                          " already exists in the layout.");

    BufferAttrType type = getBufferAttrType(T(0), num_components);
    int enum_val = static_cast<int>(type);

    attr_map_.emplace_back(
        attr_name, type, attr_type_info[enum_val].get(), -1, item_byte_size_);

    item_byte_size_ += attr_type_info[enum_val]->numBytes();
  }

  void bindToMaterial(BindAttributeLambda bind_attribute,
                      const Material& material,
                      const uint64_t used_bytes,
                      const std::string& attr = "",
                      const std::string& shader_attr = "",
                      const uint32_t num_instances_for_attr = 0) const override;
};

class SequentialBufferLayout : public BaseBufferLayout {
 public:
  SequentialBufferLayout(const SequentialBufferLayout& layout)
      : BaseBufferLayout(layout) {}
  SequentialBufferLayout() : BaseBufferLayout(BufferLayoutType::kSequential) {}

  void addAttribute(const std::string& attr_name, BufferAttrType type);

  template <typename T, int num_components = 1>
  void addAttribute(const std::string& attr_name) {
    RUNTIME_EX_ASSERT(!hasAttribute(attr_name) && attr_name.length(),
                      "SequentialBufferLayout::addAttribute(): attribute " + attr_name +
                          " already exists in the layout.");

    BufferAttrType type = getBufferAttrType(T(0), num_components);
    int enum_val = static_cast<int>(type);
    attr_map_.emplace_back(attr_name,
                           type,
                           attr_type_info[enum_val].get(),
                           attr_type_info[enum_val]->numBytes(),
                           -1);

    item_byte_size_ += attr_type_info[enum_val]->numBytes();
  }

  void bindToMaterial(BindAttributeLambda bind_attribute,
                      const Material& material,
                      const uint64_t used_bytes,
                      const std::string& attr = "",
                      const std::string& shader_attr = "",
                      const uint32_t num_instances_for_attr = 0) const override;
};

}  // namespace gfx
