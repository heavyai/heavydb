/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Tests/RenderTests/Utils/ShaderUtils.h"

#include "GfxDriver/ShaderCompiler/ShaderCache.h"

namespace gfx {

testing::AssertionResult validate_shader_caches(const ShaderCacheShPtrVector& caches) {
  if (caches.size() == 0u) {
    return testing::AssertionFailure() << "Shader cache vector is empty";
  }
  for (auto const& cache : caches) {
    if (cache->getSpirv().size() == 0u) {
      return testing::AssertionFailure()
             << "Shader cache spir-v is empty for \'"
             << to_string(cache->getShaderStage()) << "\' stage";
    }
  }
  return testing::AssertionSuccess();
}

}  // namespace gfx
