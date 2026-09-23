/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>

#include <boost/noncopyable.hpp>

#include "GfxDriver/ShaderCompiler/ShaderReflection.h"
#include "GfxDriver/ShaderCompiler/Types.h"

namespace gfx {
class ShaderCache : boost::noncopyable {
 public:
  explicit ShaderCache(spirv_t&& spirv,
                       const std::string&& glsl,
                       ShaderReflection&& reflection,
                       const ShaderStage shader_stage,
                       std::string&& entry_point,
                       std::string library_item_filename,
                       std::set<std::string_view>&& external_uniform_buffer_names,
                       uint32_t raytracing_hit_group_index);

  const spirv_t& getSpirv() const;
  const std::string& getGlsl() const;
  const std::string& getEntryPoint() const;
  ShaderReflection& getReflection();
  const ShaderStage getShaderStage() const;
  const bool getUseSpirvToGlslCompilePath() const;
  const std::string& getLibraryItemFilename() const;
  const std::set<std::string_view>& getExternalUniformBufferNames() const;
  const uint32_t getRaytracingHitGroupIndex() const;

 private:
  spirv_t spirv_;
  const std::string glsl_;
  ShaderReflection reflection_;
  const ShaderStage stage_;
  const std::string entry_point_;
  // Used for debugging (when writing out artifacts). Should be derived from library name.
  const std::string library_item_filename_;
  const std::set<std::string_view> external_uniform_buffer_names_;
  const uint32_t raytracing_hit_group_index_;
};
}  // namespace gfx
