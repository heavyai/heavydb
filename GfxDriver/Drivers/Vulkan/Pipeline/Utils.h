/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <ostream>
#include <vector>

#include <vulkan/vulkan.h>

#include "GfxDriver/Pipeline/Types.h"

namespace gfx {

std::vector<VkPushConstantRange> push_constant_ranges_to_vk_push_constant_ranges(
    const PushConstantRanges& ranges);

std::ostream& operator<<(std::ostream& os, const VkDescriptorType value);

}  // namespace gfx
