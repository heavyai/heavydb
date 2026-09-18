/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/ShaderCompiler/ShaderCache.h"

namespace gfx {

ShaderCache::ShaderCache(spirv_t&& spirv,
                         const std::string&& glsl,
                         ShaderReflection&& reflection,
                         const ShaderStage shader_stage,
                         std::string&& entry_point,
                         std::string library_item_filename,
                         std::set<std::string_view>&& external_uniform_buffer_names,
                         uint32_t raytracing_hit_group_index)
    : spirv_{std::move(spirv)}
    , glsl_{std::move(glsl)}
    , reflection_{std::move(reflection)}
    , stage_{shader_stage}
    , entry_point_{std::move(entry_point)}
    , library_item_filename_{std::move(library_item_filename)}
    , external_uniform_buffer_names_{std::move(external_uniform_buffer_names)}
    , raytracing_hit_group_index_{raytracing_hit_group_index} {}

const spirv_t& ShaderCache::getSpirv() const {
  return spirv_;
}

const std::string& ShaderCache::getGlsl() const {
  return glsl_;
}

ShaderReflection& ShaderCache::getReflection() {
  return reflection_;
}

const ShaderStage ShaderCache::getShaderStage() const {
  return stage_;
}

const std::string& ShaderCache::getEntryPoint() const {
  return entry_point_;
}

const std::string& ShaderCache::getLibraryItemFilename() const {
  return library_item_filename_;
}

const std::set<std::string_view>& ShaderCache::getExternalUniformBufferNames() const {
  return external_uniform_buffer_names_;
}

const uint32_t ShaderCache::getRaytracingHitGroupIndex() const {
  return raytracing_hit_group_index_;
}

}  // namespace gfx
