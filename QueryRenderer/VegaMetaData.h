/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>
#include <string>

#include "GfxDriver/ShaderCompiler/Types.h"

namespace QueryRenderer {

struct VegaMetaData {
  // Unique short name use for naming test artifacts
  // Distinct from standard "Title" vega property which has no constraints
  std::string short_name;

  // Debug / Test data
  gfx::ShaderArtifactTypeBits shader_artifacts_to_save =
      gfx::ShaderArtifactTypeBits::kNone;

  // if true, will attempt to clear caches of a current render session to do a fresh
  // render.
  bool clear_caches = false;
};

using VegaMetaDataUqPtr = std::unique_ptr<VegaMetaData>;

}  // namespace QueryRenderer
