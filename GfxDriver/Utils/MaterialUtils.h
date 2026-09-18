/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <functional>
#include <string>

#include "GfxDriver/Pipeline/Pipeline.h"
#include "GfxDriver/Pipeline/PushConstantRanges.h"
#include "GfxDriver/Pipeline/Types.h"
#include "GfxDriver/Resources/ResourcePtr.h"
#include "GfxDriver/ShaderCompiler/ShaderManager.h"
#include "GfxDriver/Types.h"

namespace gfx {

using BuilderCallback = std::function<void(gfx::ShaderManager::Builder&)>;
using CreateComputePassResult = std::pair<MaterialUqPtr, resource_ptr<ComputePipeline>>;
CreateComputePassResult create_compute_pass(
    const DeviceContext& device,
    const std::string& shader_template,
    const std::string& resource_name,
    const BuilderCallback builder_cb = nullptr,
    const std::vector<SpecializationMapEntry>& specializations = {},
    const PushConstantRanges& push_constant_ranges = {});

}  // namespace gfx
