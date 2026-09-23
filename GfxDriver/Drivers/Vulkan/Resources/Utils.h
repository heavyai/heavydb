/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <vulkan/vulkan.h>

#include "GfxDriver/Resources/Enums.h"
#include "GfxDriver/ShaderCompiler/Types.h"

namespace gfx {

VkImageAspectFlags pixel_format_to_vk_image_aspect_flags(PixelFormat pixel_format);

PixelFormat vk_format_to_pixel_format(VkFormat vk_format);

VkImageUsageFlags image_usage_bits_to_vk_image_usage_bits(ImageUsageBits bits);

VkClearValue pixel_format_to_vk_clear_value(PixelFormat pixel_format);

VkFilter sampler_filter_mode_to_vk_filter(SamplerFilterMode sampler_filter_mode);

VkSamplerAddressMode sampler_wrap_mode_to_vk_sampler_address_mode(
    SamplerWrapMode sampler_wrap_mode);

VkSampleCountFlagBits num_samples_to_vk_sample_flag_bits(uint32_t num_samples);
VkSampleCountFlagBits raster_sample_count_to_vk_sample_flag_bits(RasterSampleCount value);

VkShaderStageFlagBits shader_stage_to_vk_shader_stage_flag(ShaderStage shader_stage);

VkShaderStageFlagBits shader_stage_bits_to_vk_shader_stage_flag(
    ShaderStageBits shader_stage_bits);

VkIndexType index_buffer_data_type_to_vk_index_type(IndexBufferDataType data_type);

VkImageLayout image_layout_to_vk_image_layout(ImageLayout layout,
                                              bool is_color_attachment);

ImageLayout image_usage_bits_to_final_layout(ImageUsageBits bits);

VkAccelerationStructureTypeKHR accel_struct_type_to_vk_accel_struct_type(
    AccelerationStructureType type);

}  // namespace gfx
