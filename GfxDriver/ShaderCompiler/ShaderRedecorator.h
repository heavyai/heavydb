/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <map>
#include <tuple>

#include "GfxDriver/ShaderCompiler/Types.h"

namespace gfx {

class ShaderRedecorator {
 public:
  explicit ShaderRedecorator(std::string_view shader_name);
  ~ShaderRedecorator() = default;
  ShaderRedecorator() = delete;

  // Public for use by IoMapResolver
  // These values are one less than glslang::TQualifier::layout*End
  // which are the values representing "undefined" in the IoMapResolver
  // but it's not legal to actually request that they be SET to those
  // values there, which we currently need to do to persist them until
  // our own redecoration step
  // It really shouldn't be this complicated, but I can't find any other
  // way of distinguishing pre-specified values from automatic values
  // @TODO(se) Make mo' better
  static constexpr uint32_t kUninitializedSet = 0x3E;
  static constexpr uint32_t kUninitializedBinding = 0xFFFE;
  static constexpr uint32_t kUninitializedLocation = 0xFFE;

  enum class ResourceType {
    kVertexAttr,
    kUniformBuffer,
    kShaderStorageBuffer,
    kSampledImage,
    kStorageImage,
    kAccelerationStructure,
  };

  void redecorate(spirv_t& spirv,
                  ShaderReflection& reflection,
                  const std::string& template_name);

 private:
  // Every resource lands in this one set. No shader in the library declares an
  // explicit set, so there is never a second one to allocate into.
  static constexpr uint32_t kDescriptorSet = 0U;

  static constexpr uint32_t kMaxBindingsPerSet = kUninitializedBinding - 1U;
  using ReservedBindingEntry = std::tuple<ResourceType, std::string, uint32_t>;
  using ReservedBindings = std::map<uint32_t, ReservedBindingEntry>;

  void redecorateInternal(spirv_t& spirv, ShaderReflection& reflection);

  void reserveBindings(uint32_t first_binding,
                       uint32_t num_bindings,
                       ResourceType resource_type,
                       const std::string& resource_name);

  uint32_t allocateBindings(uint32_t num_bindings,
                            ResourceType resource_type,
                            const std::string& resource_name);

  std::string shader_name_;
  ReservedBindings reserved_bindings_;
  uint32_t num_vertex_attr_locations_;
};

}  // namespace gfx
