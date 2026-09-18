/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "QueryRenderer/PerGpuData.h"

#include <memory>

#include "GfxDriver/Pipeline/Material.h"
#include "GfxDriver/Render/PPLLRender.h"
#include "GfxDriver/Resources/Types.h"

namespace QueryRenderer {

class MarkPerGpuData : public BasePerGpuData {
 public:
  // Standard Mark components
  std::vector<gfx::MaterialUqPtr> fill_materials;
  std::vector<gfx::MaterialUqPtr> stroke_materials;
  std::vector<gfx::PrimitiveAssemblyUqPtr> fill_primitive_assemblies;
  std::vector<gfx::PrimitiveAssemblyUqPtr> stroke_primitive_assemblies;

  std::vector<gfx::resource_ptr<gfx::GraphicsPipeline>> graphics_pipelines;
  std::vector<gfx::resource_ptr<gfx::ComputePipeline>> compute_pipelines;

  // PPLL rendering
  std::unique_ptr<gfx::PPLLRender> ppll_render;

  // Legacy Symbols (geometry) only
  gfx::BufferWrapperUqPtr instanced_geom_vbo;
  gfx::BufferWrapperUqPtr instanced_geom_ibo;
  gfx::VertexBufferRefs vertex_buffer_refs_cache;

  explicit MarkPerGpuData(RootPerGpuData& root_data) : BasePerGpuData{root_data} {}
  ~MarkPerGpuData() override;

  void destroyPipelines();
};

}  // namespace QueryRenderer
