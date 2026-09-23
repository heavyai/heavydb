/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "GfxDriver/RenderError.h"

namespace gfx {

// POD for API-agnostic reflection results for shaders created by a Builder
// initially an instance of this will be in BuilderImpl::Cache

class ShaderReflection {
 public:
  ShaderReflection();
  ~ShaderReflection() = default;

  struct ItemInfo {
    int set;
    int binding_or_location;
    int offset;
    int block_or_array_size;
    ItemInfo(int set, int binding_or_location, int offset, int block_or_array_size)
        : set{set}
        , binding_or_location{binding_or_location}
        , offset{offset}
        , block_or_array_size{block_or_array_size} {}
    ItemInfo() : ItemInfo(-1, -1, -1, -1) {}
    bool operator==(const ItemInfo& rhs) const {
      return set == rhs.set && binding_or_location == rhs.binding_or_location &&
             offset == rhs.offset && block_or_array_size == rhs.block_or_array_size;
    }
  };

  void initialize();
  void clear();
  ShaderReflection& operator=(const ShaderReflection& rhs);

  void addVertexAttr(std::string_view name, int location, int array_size);
  void addSampler(std::string_view name, int set, int binding, int array_size);
  void addStorageImage(std::string_view name, int set, int binding, int array_size);
  void addUniformBuffer(std::string_view name, int set, int binding, int block_size);
  void addAccelerationStructure(std::string_view name,
                                int set,
                                int binding,
                                int block_size);
  void addShaderStorageBuffer(std::string_view name,
                              int set,
                              int binding,
                              int block_size);
  void addUniformBufferAttr(std::string_view name,
                            int set,
                            int binding,
                            int offset,
                            int size);
  void addShaderStorageBufferAttr(std::string_view name,
                                  int set,
                                  int binding,
                                  int offset,
                                  int size);
  void addFragmentShaderOutputLocation(int location);

  bool hasVertexAttr(std::string_view name) const;
  bool hasSampler(std::string_view name) const;
  bool hasStorageImage(std::string_view name) const;
  bool hasUniformBuffer(std::string_view name) const;
  bool hasShaderStorageBuffer(std::string_view name) const;
  bool hasAccelerationStructure(std::string_view name) const;
  bool hasUniformBufferAttr(std::string_view name) const;
  bool hasShaderStorageBufferAttr(std::string_view name) const;
  bool hasFragmentShaderOutputLocation(int location) const;

  int getVertexAttrLocation(std::string_view name) const;
  int getVertexAttrArraySize(std::string_view name) const;
  int getSamplerSet(std::string_view name) const;
  int getSamplerBinding(std::string_view name) const;
  int getSamplerArraySize(std::string_view name) const;
  int getStorageImageSet(std::string_view name) const;
  int getStorageImageBinding(std::string_view name) const;
  int getStorageImageArraySize(std::string_view name) const;
  int getUniformBufferSet(std::string_view name) const;
  int getUniformBufferBinding(std::string_view name) const;
  int getUniformBufferBlockSize(std::string_view name) const;
  int getShaderStorageBufferSet(std::string_view name) const;
  int getShaderStorageBufferBinding(std::string_view name) const;
  int getShaderStorageBufferBlockSize(std::string_view name) const;
  int getAccelerationStructureSet(std::string_view name) const;
  int getAccelerationStructureBinding(std::string_view name) const;
  int getAccelerationStructureSize(std::string_view name) const;

  const ItemInfo& getUniformBufferAttrItemInfo(std::string_view name) const;
  const ItemInfo& getShaderStorageBufferAttrItemInfo(std::string_view name) const;

  using NameVector = std::vector<std::string_view>;
  const NameVector getAllUniformBufferNames() const;
  const NameVector getAllShaderStorageBufferNames() const;
  const NameVector getAllSamplerNames() const;
  const NameVector getAllStorageImageNames() const;
  const NameVector getAllAccelerationStructureNames() const;
  const NameVector getAllUniformBufferAttrNames() const;
  const NameVector getAllShaderStorageBufferAttrNames() const;

 private:
  class NameToItemInfoMap {
   public:
    using MapType = std::unordered_map<std::string, ItemInfo>;
    bool contains(std::string_view name) const {
      return map_.find(std::string(name)) != map_.end();
    }
    const ItemInfo& find_or_default(std::string_view name) const {
      auto const& itr = map_.find(std::string(name));
      if (itr != map_.end()) {
        return itr->second;
      }
      return kDefaultItemInfo;
    }
    void clear() { map_.clear(); }
    const MapType& the_map() const { return map_; }
    void insert_unless_different(std::string_view name,
                                 ItemInfo value,
                                 std::string_view type_name) {
      RUNTIME_EX_ASSERT(
          map_.try_emplace(std::string(name), value).second,
          std::string(type_name) + " \'" + std::string(name) + "\' repeated");
    }
    NameToItemInfoMap& operator=(const NameToItemInfoMap& rhs) {
      map_ = rhs.map_;
      return *this;
    }

   private:
    MapType map_;
    static const ItemInfo kDefaultItemInfo;
  };

  NameToItemInfoMap samplers_;
  NameToItemInfoMap storage_images_;
  NameToItemInfoMap uniform_buffers_;
  NameToItemInfoMap shader_storage_buffers_;
  NameToItemInfoMap acceleration_structures_;
  NameToItemInfoMap vertex_attr_locations_;
  NameToItemInfoMap uniform_buffer_attrs_;
  NameToItemInfoMap shader_storage_buffer_attrs_;
  std::set<int> fragment_shader_output_locations_;

  void validateSet(int set);
};

}  // namespace gfx
