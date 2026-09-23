/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Utils/MaterialUtils.h"

#include "GfxDriver/DeviceContext.h"
#include "GfxDriver/Pipeline/Material.h"
#include "GfxDriver/Resources/ResourceManager.h"

namespace gfx {

CreateComputePassResult create_compute_pass(
    const DeviceContext& device,
    const std::string& shader_template,
    const std::string& resource_name,
    const BuilderCallback builder_cb,
    const std::vector<SpecializationMapEntry>& specializations,
    const PushConstantRanges& push_constant_ranges) {
  auto subgroup_size_str = std::to_string(device.getLimits().subgroup_size);

  // Material
  auto& resource_mgr = device.getResourceManager();
  auto& shader_mgr = resource_mgr.getShaderManager();
  auto builder = shader_mgr.createBuilderVector({{shader_template}});
  builder[0]->replaceFirstTag("workgroupSize", subgroup_size_str);
  if (builder_cb) {
    builder_cb(*builder[0]);
  }
  auto cache = shader_mgr.createCacheVector(std::move(builder));
  auto material = resource_mgr.createMaterial(resource_name, cache);

  // Pipeline
  auto pipeline = resource_mgr.createComputePipeline(
      resource_name, *material, specializations, push_constant_ranges);
  pipeline->create();

  return {std::move(material), std::move(pipeline)};
}

}  // namespace gfx
