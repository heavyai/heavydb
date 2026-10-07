/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <map>
#include <tuple>

#include "GfxDriver/ShaderCompiler/Types.h"

namespace gfx {

// Reads a compiled shader's interface out of its SPIR-V and into a ShaderReflection,
// checking as it goes that no two resources of a material were given the same binding.
// It no longer assigns anything: glslang resolves sets, bindings and locations across
// the material before this ever sees the module. The name is left alone because PR 7
// of the Slang migration deletes the class outright.
class ShaderRedecorator {
 public:
  explicit ShaderRedecorator(std::string_view shader_name);
  ~ShaderRedecorator() = default;
  ShaderRedecorator() = delete;

  enum class ResourceType {
    kVertexAttr,
    kUniformBuffer,
    kShaderStorageBuffer,
    kSampledImage,
    kStorageImage,
    kAccelerationStructure,
  };

  void redecorate(const spirv_t& spirv,
                  ShaderReflection& reflection,
                  const std::string& template_name);

 private:
  // Every resource lands in this one set. No shader in the library declares an explicit
  // set, and glslang's resolver assigns 0 in the absence of one, so there is never a
  // second.
  static constexpr uint32_t kDescriptorSet = 0U;

  // A binding beyond this cannot have come from a layout qualifier, so seeing one means
  // something went wrong in the compiler rather than in the shader. The limit that
  // actually constrains us is the device's maxPerStageDescriptor*, which is enforced
  // when the descriptor set layout is built.
  static constexpr uint32_t kMaxBindingsPerSet = 0xFFFFU;

  using ReservedBindingEntry = std::tuple<ResourceType, std::string>;
  using ReservedBindings = std::map<uint32_t, ReservedBindingEntry>;

  void redecorateInternal(const spirv_t& spirv, ShaderReflection& reflection);

  // Claims one binding for one resource, throwing if a different resource of the
  // material already holds it. Accumulates across the stages, so the clash this is
  // really looking for is a cross-stage one.
  void recordBinding(uint32_t binding,
                     ResourceType resource_type,
                     const std::string& resource_name);

  std::string shader_name_;
  ReservedBindings reserved_bindings_;
};

}  // namespace gfx
