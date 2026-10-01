/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/ShaderCompiler/ShaderReflection.h"

#include <algorithm>
#include <ostream>

#include "Logger/Logger.h"

namespace gfx {

const ShaderReflection::ItemInfo ShaderReflection::NameToItemInfoMap::kDefaultItemInfo;

ShaderReflection::ShaderReflection() {}

void ShaderReflection::initialize() {
  clear();
}

void ShaderReflection::clear() {
  vertex_attr_locations_.clear();
  samplers_.clear();
  storage_images_.clear();
  uniform_buffers_.clear();
  shader_storage_buffers_.clear();
  acceleration_structures_.clear();
  uniform_buffer_attrs_.clear();
  shader_storage_buffer_attrs_.clear();
}

ShaderReflection& ShaderReflection::operator=(const ShaderReflection& rhs) {
  vertex_attr_locations_ = rhs.vertex_attr_locations_;
  samplers_ = rhs.samplers_;
  storage_images_ = rhs.storage_images_;
  uniform_buffers_ = rhs.uniform_buffers_;
  shader_storage_buffers_ = rhs.shader_storage_buffers_;
  acceleration_structures_ = rhs.acceleration_structures_;
  uniform_buffer_attrs_ = rhs.uniform_buffer_attrs_;
  shader_storage_buffer_attrs_ = rhs.shader_storage_buffer_attrs_;
  return *this;
}

void ShaderReflection::validateSet(int set) {
  CHECK_NE(set, -1);
}

void ShaderReflection::addVertexAttr(std::string_view name,
                                     int location,
                                     int array_size) {
  vertex_attr_locations_.insert_unless_different(
      name, {-1, location, -1, array_size}, "Vertex Attr");
}

void ShaderReflection::addSampler(std::string_view name,
                                  int set,
                                  int binding,
                                  int array_size) {
  validateSet(set);
  samplers_.insert_unless_different(name, {set, binding, -1, array_size}, "Sampler");
}

void ShaderReflection::addStorageImage(std::string_view name,
                                       int set,
                                       int binding,
                                       int array_size) {
  validateSet(set);
  storage_images_.insert_unless_different(
      name, {set, binding, -1, array_size}, "Storage Image");
}

void ShaderReflection::addUniformBuffer(std::string_view name,
                                        int set,
                                        int binding,
                                        int block_size) {
  validateSet(set);
  uniform_buffers_.insert_unless_different(
      name, {set, binding, -1, block_size}, "Uniform Buffer");
}

void ShaderReflection::addShaderStorageBuffer(std::string_view name,
                                              int set,
                                              int binding,
                                              int block_size) {
  validateSet(set);
  shader_storage_buffers_.insert_unless_different(
      name, {set, binding, -1, block_size}, "Shader Storage Buffer");
}

void ShaderReflection::addAccelerationStructure(std::string_view name,
                                                int set,
                                                int binding,
                                                int size) {
  validateSet(set);
  acceleration_structures_.insert_unless_different(
      name, {set, binding, -1, size}, "Acceleration Structure");
}

void ShaderReflection::addUniformBufferAttr(std::string_view name,
                                            int set,
                                            int binding,
                                            int offset,
                                            int size) {
  validateSet(set);
  uniform_buffer_attrs_.insert_unless_different(
      name, {set, binding, offset, size}, "Uniform Buffer Attr");
}

void ShaderReflection::addShaderStorageBufferAttr(std::string_view name,
                                                  int set,
                                                  int binding,
                                                  int offset,
                                                  int size) {
  validateSet(set);
  shader_storage_buffer_attrs_.insert_unless_different(
      name, {set, binding, offset, size}, "Shader Storage Buffer Attr");
}

void ShaderReflection::addFragmentShaderOutputLocation(int location) {
  fragment_shader_output_locations_.insert(location);
}

bool ShaderReflection::hasVertexAttr(std::string_view name) const {
  return vertex_attr_locations_.contains(name);
}

bool ShaderReflection::hasSampler(std::string_view name) const {
  return samplers_.contains(name);
}

bool ShaderReflection::hasStorageImage(std::string_view name) const {
  return storage_images_.contains(name);
}

bool ShaderReflection::hasUniformBuffer(std::string_view name) const {
  return uniform_buffers_.contains(name);
}

bool ShaderReflection::hasShaderStorageBuffer(std::string_view name) const {
  return shader_storage_buffers_.contains(name);
}

bool ShaderReflection::hasAccelerationStructure(std::string_view name) const {
  return acceleration_structures_.contains(name);
}

bool ShaderReflection::hasUniformBufferAttr(std::string_view name) const {
  return uniform_buffer_attrs_.contains(name);
}

bool ShaderReflection::hasShaderStorageBufferAttr(std::string_view name) const {
  return shader_storage_buffer_attrs_.contains(name);
}

bool ShaderReflection::hasFragmentShaderOutputLocation(int location) const {
  auto const itr = fragment_shader_output_locations_.find(location);
  return (itr != fragment_shader_output_locations_.end());
}

int ShaderReflection::getVertexAttrLocation(std::string_view name) const {
  return vertex_attr_locations_.find_or_default(name).binding_or_location;
}

int ShaderReflection::getVertexAttrArraySize(std::string_view name) const {
  return vertex_attr_locations_.find_or_default(name).block_or_array_size;
}

int ShaderReflection::getSamplerSet(std::string_view name) const {
  return samplers_.find_or_default(name).set;
}

int ShaderReflection::getSamplerBinding(std::string_view name) const {
  return samplers_.find_or_default(name).binding_or_location;
}

int ShaderReflection::getSamplerArraySize(std::string_view name) const {
  return samplers_.find_or_default(name).block_or_array_size;
}

int ShaderReflection::getStorageImageSet(std::string_view name) const {
  return storage_images_.find_or_default(name).set;
}

int ShaderReflection::getStorageImageBinding(std::string_view name) const {
  return storage_images_.find_or_default(name).binding_or_location;
}

int ShaderReflection::getStorageImageArraySize(std::string_view name) const {
  return storage_images_.find_or_default(name).block_or_array_size;
}

int ShaderReflection::getUniformBufferSet(std::string_view name) const {
  return uniform_buffers_.find_or_default(name).set;
}

int ShaderReflection::getUniformBufferBinding(std::string_view name) const {
  return uniform_buffers_.find_or_default(name).binding_or_location;
}

int ShaderReflection::getUniformBufferBlockSize(std::string_view name) const {
  return uniform_buffers_.find_or_default(name).block_or_array_size;
}

int ShaderReflection::getShaderStorageBufferSet(std::string_view name) const {
  return shader_storage_buffers_.find_or_default(name).set;
}

int ShaderReflection::getShaderStorageBufferBinding(std::string_view name) const {
  return shader_storage_buffers_.find_or_default(name).binding_or_location;
}

int ShaderReflection::getShaderStorageBufferBlockSize(std::string_view name) const {
  return shader_storage_buffers_.find_or_default(name).block_or_array_size;
}

int ShaderReflection::getAccelerationStructureSet(std::string_view name) const {
  return acceleration_structures_.find_or_default(name).set;
}

int ShaderReflection::getAccelerationStructureBinding(std::string_view name) const {
  return acceleration_structures_.find_or_default(name).binding_or_location;
}

int ShaderReflection::getAccelerationStructureSize(std::string_view name) const {
  return acceleration_structures_.find_or_default(name).block_or_array_size;
}

const ShaderReflection::ItemInfo& ShaderReflection::getUniformBufferAttrItemInfo(
    std::string_view name) const {
  return uniform_buffer_attrs_.find_or_default(name);
}

const ShaderReflection::ItemInfo& ShaderReflection::getShaderStorageBufferAttrItemInfo(
    std::string_view name) const {
  return shader_storage_buffer_attrs_.find_or_default(name);
}

const ShaderReflection::NameVector ShaderReflection::getAllUniformBufferNames() const {
  NameVector names;
  for (auto const& uniform_buffer : uniform_buffers_.the_map()) {
    names.emplace_back(uniform_buffer.first);
  }
  return names;
}

const ShaderReflection::NameVector ShaderReflection::getAllShaderStorageBufferNames()
    const {
  NameVector names;
  for (auto const& shader_storage_buffer : shader_storage_buffers_.the_map()) {
    names.emplace_back(shader_storage_buffer.first);
  }
  return names;
}

const ShaderReflection::NameVector ShaderReflection::getAllSamplerNames() const {
  NameVector names;
  for (auto const& sampler : samplers_.the_map()) {
    names.emplace_back(sampler.first);
  }
  return names;
}

const ShaderReflection::NameVector ShaderReflection::getAllStorageImageNames() const {
  NameVector names;
  for (auto const& storage_image : storage_images_.the_map()) {
    names.emplace_back(storage_image.first);
  }
  return names;
}

const ShaderReflection::NameVector ShaderReflection::getAllAccelerationStructureNames()
    const {
  NameVector names;
  for (auto const& acceleration_structure : acceleration_structures_.the_map()) {
    names.emplace_back(acceleration_structure.first);
  }
  return names;
}

const ShaderReflection::NameVector ShaderReflection::getAllUniformBufferAttrNames()
    const {
  NameVector names;
  for (auto const& uniform_buffer_attr : uniform_buffer_attrs_.the_map()) {
    names.emplace_back(uniform_buffer_attr.first);
  }
  return names;
}

const ShaderReflection::NameVector ShaderReflection::getAllShaderStorageBufferAttrNames()
    const {
  NameVector names;
  for (auto const& shader_storage_buffer_attr : shader_storage_buffer_attrs_.the_map()) {
    names.emplace_back(shader_storage_buffer_attr.first);
  }
  return names;
}

namespace {

// ItemInfo reuses two of its fields for different things depending on the
// category, so each section supplies its own labels for them.
struct ItemLabels {
  const char* binding_or_location;
  const char* block_or_array_size;
};

constexpr ItemLabels kLocationAndArraySize{"location", "array_size"};
constexpr ItemLabels kBindingAndArraySize{"binding", "array_size"};
constexpr ItemLabels kBindingAndBlockSize{"binding", "block_size"};
constexpr ItemLabels kBindingAndSize{"binding", "size"};

using ItemMap = std::unordered_map<std::string, ShaderReflection::ItemInfo>;

// Every field is printed, including the ones a category leaves at -1, so that a
// field which unexpectedly acquires a value shows up as a golden diff rather
// than passing unnoticed.
void write_section(std::ostream& stream,
                   const char* title,
                   const ItemMap& items,
                   const ItemLabels& labels) {
  stream << title << ' ' << items.size() << '\n';

  std::vector<const ItemMap::value_type*> entries;
  entries.reserve(items.size());
  for (auto const& item : items) {
    entries.push_back(&item);
  }
  std::sort(entries.begin(), entries.end(), [](auto const* lhs, auto const* rhs) {
    return lhs->first < rhs->first;
  });

  for (auto const* entry : entries) {
    auto const& info = entry->second;
    stream << "  \"" << entry->first << "\" set=" << info.set << ' '
           << labels.binding_or_location << '=' << info.binding_or_location
           << " offset=" << info.offset << ' ' << labels.block_or_array_size << '='
           << info.block_or_array_size << '\n';
  }
}

}  // namespace

void ShaderReflection::serialize(std::ostream& stream) const {
  write_section(
      stream, "vertex_attrs", vertex_attr_locations_.the_map(), kLocationAndArraySize);
  write_section(stream, "samplers", samplers_.the_map(), kBindingAndArraySize);
  write_section(
      stream, "storage_images", storage_images_.the_map(), kBindingAndArraySize);
  write_section(
      stream, "uniform_buffers", uniform_buffers_.the_map(), kBindingAndBlockSize);
  write_section(stream,
                "shader_storage_buffers",
                shader_storage_buffers_.the_map(),
                kBindingAndBlockSize);
  write_section(stream,
                "acceleration_structures",
                acceleration_structures_.the_map(),
                kBindingAndBlockSize);
  write_section(
      stream, "uniform_buffer_attrs", uniform_buffer_attrs_.the_map(), kBindingAndSize);
  write_section(stream,
                "shader_storage_buffer_attrs",
                shader_storage_buffer_attrs_.the_map(),
                kBindingAndSize);

  // std::set iterates in order, so no explicit sort needed here
  stream << "fragment_output_locations " << fragment_shader_output_locations_.size()
         << '\n';
  for (auto const location : fragment_shader_output_locations_) {
    stream << "  " << location << '\n';
  }
}

}  // namespace gfx
