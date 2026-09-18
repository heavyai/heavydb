/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>

#include "GfxDriver/Pipeline/Types.h"
#include "GfxDriver/Resources/ResourcePtr.h"
#include "GfxDriver/Resources/Types.h"

namespace gfx {

//
// Stencil then cover helpers
//
// The StencilThenCoverHelper class is the easiest way to handle required resources
// Separate utilities for creating pipelines and materials are also available
//

// Create a stencil pass pipeline for stencil then cover rendering
// Used internally by StencilThenCoverHelper, but also useful independently
using CreateStencilPipelineReturn = std::tuple<PrimitiveAssemblyUqPtr,
                                               PipelineDescriptorUqPtr,
                                               resource_ptr<GraphicsPipeline>>;

CreateStencilPipelineReturn create_stencil_pipeline(ResourceManager& resource_mgr,
                                                    Material& material,
                                                    PrimitiveAssemblyAttrInfo& attr_info);

// Create cover pass pipeline for stencil then cover rendering
// Used internally by StencilThenCoverHelper, but also useful independently
using CreateCoverPipelineReturn =
    std::pair<PipelineDescriptorUqPtr, resource_ptr<GraphicsPipeline>>;
CreateCoverPipelineReturn create_cover_pipeline(ResourceManager& resource_mgr,
                                                Material& material);

// Create materials using the passed in vertex and fragment templates
// Stencil material does not use a fragment shader
// Cover material renders a full screen pass
using CreateStencilAndCoverMaterialsReturn = std::pair<MaterialUqPtr, MaterialUqPtr>;
CreateStencilAndCoverMaterialsReturn create_stencil_and_cover_materials(
    ResourceManager& resource_mgr,
    const std::string& stencil_vert_template,
    const std::string& cover_frag_template);

//
// StencilThenCoverHelper class
//
// Encapsulate creation and destruction of Materials, Pipelines, and PrimitiveAssemblies
// required for a stencil then cover pass
// buildResources returns the stencil and cover Pipelines required for draw calls
class StencilThenCoverHelper {
 public:
  explicit StencilThenCoverHelper(ResourceManager& resource_mgr);
  ~StencilThenCoverHelper();

  using BuildResourcesReturn = std::pair<Pipeline*, Pipeline*>;
  BuildResourcesReturn buildResources(const std::string& stencil_vert_template,
                                      const std::string& cover_frag_template,
                                      PrimitiveAssemblyAttrInfo& attr_info,
                                      RenderPass& render_pass);
  void destroyResources();

 private:
  ResourceManager& resource_mgr_;
  MaterialUqPtr stencil_material_;
  MaterialUqPtr cover_material_;
  PrimitiveAssemblyUqPtr stencil_primitive_assembly_;
  PipelineDescriptorUqPtr stencil_pipeline_desc_;
  PipelineDescriptorUqPtr cover_pipeline_desc_;
  resource_ptr<GraphicsPipeline> stencil_pipeline_;
  resource_ptr<GraphicsPipeline> cover_pipeline_;
};

}  // namespace gfx
