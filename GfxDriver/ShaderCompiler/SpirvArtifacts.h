/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>

#include "GfxDriver/ShaderCompiler/Types.h"

namespace gfx {
inline constexpr bool shader_artifacts_enabled_in_build() {
#ifdef SHADER_ARTIFACTS_ENABLED
  return true;
#else
  return false;
#endif
}

void write_spirv_artifacts(const std::string& glsl_string,
                           const spirv_t& spv,
                           const spirv_t& opt_spv,
                           const std::string& base_name,
                           ShaderArtifactTypeBits artifacts);

ShaderArtifactTypeBits is_always_save_artifacts_enabled();
ShaderArtifactTypeBits string_to_shader_artifact_type(std::string s);
std::string get_artifact_pathname();

// returns new path string on success or empty string on failure
std::string create_artifact_subdir(const std::string& subdir);

}  // namespace gfx
