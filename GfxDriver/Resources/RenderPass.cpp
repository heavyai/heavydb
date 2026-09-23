/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Resources/RenderPass.h"

namespace gfx {

RenderPass::RenderPass(const DeviceContext& device_ctx,
                       std::string_view resource_tracking_string)
    : Resource(device_ctx, resource_tracking_string, ResourceType::kRenderPass) {}

}  // namespace gfx
