/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Marks/MarkPerGpuData.h"

#include "GfxDriver/Resources/ResourceManager.h"

namespace QueryRenderer {

MarkPerGpuData::~MarkPerGpuData() {
  if (instanced_geom_vbo) {
    getResourceManager().destroyBuffer(std::move(instanced_geom_vbo));
  }
  if (instanced_geom_ibo) {
    getResourceManager().destroyBuffer(std::move(instanced_geom_ibo));
  }
  destroyPipelines();
}

void MarkPerGpuData::destroyPipelines() {
  for (auto& pipeline : graphics_pipelines) {
    if (pipeline) {
      getResourceManager().destroyPipeline(std::move(pipeline));
    }
  }
  graphics_pipelines.clear();

  for (auto& pipeline : compute_pipelines) {
    if (pipeline) {
      getResourceManager().destroyPipeline(std::move(pipeline));
    }
  }
  compute_pipelines.clear();
}

}  // namespace QueryRenderer
