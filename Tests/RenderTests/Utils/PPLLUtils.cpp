/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Tests/RenderTests/Utils/PPLLUtils.h"

#include "GfxDriver/Pipeline/Material.h"
#include "GfxDriver/Pipeline/Pipeline.h"
#include "GfxDriver/Pipeline/PipelineDescriptor.h"
#include "GfxDriver/Resources/ResourceManager.h"
#include "Logger/Logger.h"

namespace gfx {

CreateStencilPipelineReturn create_stencil_pipeline(
    ResourceManager& resource_mgr,
    Material& material,
    PrimitiveAssemblyAttrInfo& attr_info) {
  auto primitive_assembly = resource_mgr.createPrimitiveAssembly(
      "stencil", PrimitiveTopology::kTriangleFan, material, attr_info);
  auto pipeline_desc = std::make_unique<PipelineDescriptor>();
  pipeline_desc->setEnableStencilTest(true);
  pipeline_desc->setStencilFunc(gfx::StencilFunc::kAlways, 1, 1);
  pipeline_desc->setStencilOp(
      gfx::StencilOp::kInvert, gfx::StencilOp::kInvert, gfx::StencilOp::kInvert);
  pipeline_desc->setEnableColorWrites(false);
  auto pipeline = resource_mgr.createGraphicsPipeline(
      "stencil", material, *pipeline_desc, primitive_assembly.get());

  return {std::move(primitive_assembly), std::move(pipeline_desc), std::move(pipeline)};
};

CreateCoverPipelineReturn create_cover_pipeline(ResourceManager& resource_mgr,
                                                Material& material) {
  // cover pipeline
  auto pipeline_desc = std::make_unique<PipelineDescriptor>();
  pipeline_desc->setEnableStencilTest(true);
  pipeline_desc->setStencilFunc(gfx::StencilFunc::kEqual, 1, 1);
  pipeline_desc->setStencilOp(
      gfx::StencilOp::kZero, gfx::StencilOp::kZero, gfx::StencilOp::kZero);
  pipeline_desc->setEnableColorWrites(true);
  auto pipeline =
      resource_mgr.createGraphicsPipeline("stencil", material, *pipeline_desc);
  return {std::move(pipeline_desc), std::move(pipeline)};
}

CreateStencilAndCoverMaterialsReturn create_stencil_and_cover_materials(
    ResourceManager& resource_mgr,
    const std::string& stencil_vert_template,
    const std::string& cover_frag_template) {
  auto const& shader_mgr = resource_mgr.getShaderManager();
  MaterialUqPtr stencil_material;
  MaterialUqPtr cover_material;

  // Stencil pass material
  {
    auto caches = shader_mgr.createCacheVectorFromTemplate({stencil_vert_template});
    CHECK_NE(caches.size(), 0u);
    CHECK_NE(caches[0]->getSpirv().size(), 0u);

    stencil_material = resource_mgr.createMaterial("stencil", caches);
    CHECK(stencil_material != nullptr);
  }

  // Cover pass material
  {
    auto caches = shader_mgr.createCacheVectorFromTemplate(
        {{"PPLLTests/fullScreenTriangle.vert"}, cover_frag_template});
    CHECK_NE(caches.size(), 0u);
    CHECK_NE(caches[0]->getSpirv().size(), 0u);
    CHECK_NE(caches[1]->getSpirv().size(), 0u);

    cover_material = resource_mgr.createMaterial("cover", caches);
    CHECK(cover_material != nullptr);
  }
  return {std::move(stencil_material), std::move(cover_material)};
}

StencilThenCoverHelper::StencilThenCoverHelper(ResourceManager& resource_mgr)
    : resource_mgr_{resource_mgr} {}

StencilThenCoverHelper::~StencilThenCoverHelper() {
  destroyResources();
}

StencilThenCoverHelper::BuildResourcesReturn StencilThenCoverHelper::buildResources(
    const std::string& stencil_vert_template,
    const std::string& cover_frag_template,
    PrimitiveAssemblyAttrInfo& attr_info,
    RenderPass& render_pass) {
  // materials
  std::tie(stencil_material_, cover_material_) = create_stencil_and_cover_materials(
      resource_mgr_, stencil_vert_template, cover_frag_template);

  // stencil pipeline
  std::tie(stencil_primitive_assembly_, stencil_pipeline_desc_, stencil_pipeline_) =
      create_stencil_pipeline(resource_mgr_, *stencil_material_, attr_info);
  stencil_pipeline_->create(render_pass);

  // cover pipeline
  std::tie(cover_pipeline_desc_, cover_pipeline_) =
      create_cover_pipeline(resource_mgr_, *cover_material_);
  cover_pipeline_->create(render_pass);

  return {stencil_pipeline_.get(), cover_pipeline_.get()};
}

void StencilThenCoverHelper::destroyResources() {
  if (stencil_pipeline_) {
    resource_mgr_.destroyPipeline(std::move(stencil_pipeline_));
  }
  if (cover_pipeline_) {
    resource_mgr_.destroyPipeline(std::move(cover_pipeline_));
  }
  stencil_pipeline_desc_ = nullptr;
  cover_pipeline_desc_ = nullptr;
  stencil_material_ = nullptr;
  cover_material_ = nullptr;
}

}  // namespace gfx
