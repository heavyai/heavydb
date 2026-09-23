/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Drivers/Vulkan/Pipeline/Utils.h"

#include <vulkan/vk_enum_string_helper.h>

#include "GfxDriver/Drivers/Vulkan/Resources/Utils.h"
#include "GfxDriver/Pipeline/PushConstantRanges.h"

namespace gfx {

std::vector<VkPushConstantRange> push_constant_ranges_to_vk_push_constant_ranges(
    const PushConstantRanges& ranges) {
  std::vector<VkPushConstantRange> vk_ranges;
  ranges.visit([&](const auto& range) {
    VkPushConstantRange vk_range = {};
    vk_range.stageFlags = shader_stage_bits_to_vk_shader_stage_flag(range.shader_stages);
    vk_range.offset = range.offset;
    vk_range.size = range.size;
    vk_ranges.push_back(vk_range);
  });

  return vk_ranges;
}

std::ostream& operator<<(std::ostream& os, const VkDescriptorType value) {
  os << string_VkDescriptorType(value);
  return os;
}

}  // namespace gfx
