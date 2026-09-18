/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Drivers/Vulkan/Resources/Utils.h"

#include "Logger/Logger.h"

namespace gfx {

VkClearValue pixel_format_to_vk_clear_value(PixelFormat pixel_format) {
  VkClearValue rtn;
  switch (pixel_format) {
    case PixelFormat::kR8:
    case PixelFormat::kRG8:
    case PixelFormat::kRGBA8:
    case PixelFormat::kBGRA8:
    case PixelFormat::kR32UI:
    case PixelFormat::kR32I:
      rtn.color = {{0, 0, 0, 0}};
      return rtn;
    case PixelFormat::kDepth:
    case PixelFormat::kDepthHighP:
    case PixelFormat::kDepthStencil:
    case PixelFormat::kDepthStencilHighP:
      rtn.depthStencil = {0.0f, 0};
      return rtn;
    case PixelFormat::kCOUNT:
      CHECK(false);
  }
  UNREACHABLE();
  return VkClearValue{};
}

VkImageAspectFlags pixel_format_to_vk_image_aspect_flags(PixelFormat pixel_format) {
  switch (pixel_format) {
    case PixelFormat::kR8:
    case PixelFormat::kRG8:
    case PixelFormat::kRGBA8:
    case PixelFormat::kBGRA8:
    case PixelFormat::kR32UI:
    case PixelFormat::kR32I:
      return VK_IMAGE_ASPECT_COLOR_BIT;
    case PixelFormat::kDepth:
      // TODO(scb): Base this on hardware format. This is a workaround to handle hardware
      // formats that differ from internal PixelFormat. In this case we prefer D24_S8
      // format for kDepth, which is a packed DS format. Validation will fail if we don't
      // return both aspect bits
      return VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
    case PixelFormat::kDepthHighP:
      return VK_IMAGE_ASPECT_DEPTH_BIT;
    case PixelFormat::kDepthStencil:
    case PixelFormat::kDepthStencilHighP:
      return VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
    case PixelFormat::kCOUNT:
      CHECK(false);
  }
  UNREACHABLE();
  return VkImageAspectFlags{};
}

PixelFormat vk_format_to_pixel_format(VkFormat vk_format) {
  switch (vk_format) {
    case VK_FORMAT_R8_UNORM:
      return PixelFormat::kR8;
    case VK_FORMAT_R8G8_UNORM:
      return PixelFormat::kRG8;
    case VK_FORMAT_R8G8B8A8_UNORM:
      return PixelFormat::kRGBA8;
    case VK_FORMAT_B8G8R8A8_UNORM:
      return PixelFormat::kBGRA8;
    case VK_FORMAT_R32_UINT:
      return PixelFormat::kR32UI;
    case VK_FORMAT_R32_SINT:
      return PixelFormat::kR32I;
    case VK_FORMAT_X8_D24_UNORM_PACK32:
      return PixelFormat::kDepth;
    case VK_FORMAT_D32_SFLOAT:
      return PixelFormat::kDepthHighP;
    case VK_FORMAT_D24_UNORM_S8_UINT:
      return PixelFormat::kDepthStencil;
    case VK_FORMAT_D32_SFLOAT_S8_UINT:
      return PixelFormat::kDepthStencilHighP;
    default:
      CHECK(false) << "Cannot convert VkFormat to PixelFormat";
  }
  return PixelFormat::kCOUNT;
}

VkImageUsageFlags image_usage_bits_to_vk_image_usage_bits(ImageUsageBits bits) {
  VkImageUsageFlags vk_bits{0};
  if (ImageUsageBits::kSampledBit == (bits & ImageUsageBits::kSampledBit)) {
    vk_bits |= VK_IMAGE_USAGE_SAMPLED_BIT;
  }
  if (ImageUsageBits::kStorageBit == (bits & ImageUsageBits::kStorageBit)) {
    vk_bits |= VK_IMAGE_USAGE_STORAGE_BIT;
  }
  if (ImageUsageBits::kColorAttachmentBit ==
      (bits & ImageUsageBits::kColorAttachmentBit)) {
    vk_bits |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
  }
  if (ImageUsageBits::kDepthStencilAttachmentBit ==
      (bits & ImageUsageBits::kDepthStencilAttachmentBit)) {
    vk_bits |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
  }
  if (ImageUsageBits::kTransientAttachmentBit ==
      (bits & ImageUsageBits::kTransientAttachmentBit)) {
    vk_bits |= VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT;
  }
  if (ImageUsageBits::kInputAttachmentBit ==
      (bits & ImageUsageBits::kInputAttachmentBit)) {
    vk_bits |= VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;
  }
  return vk_bits;
}

VkFilter sampler_filter_mode_to_vk_filter(SamplerFilterMode sampler_filter_mode) {
  switch (sampler_filter_mode) {
    case SamplerFilterMode::kLinear:
      return VK_FILTER_LINEAR;
    case SamplerFilterMode::kNearest:
      return VK_FILTER_NEAREST;
  }
  return VK_FILTER_NEAREST;
}

VkSamplerAddressMode sampler_wrap_mode_to_vk_sampler_address_mode(
    SamplerWrapMode sampler_wrap_mode) {
  switch (sampler_wrap_mode) {
    case SamplerWrapMode::kRepeat:
      return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    case SamplerWrapMode::kMirrorRepeat:
      return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    case SamplerWrapMode::kClampEdge:
      return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    case SamplerWrapMode::kClampBorder:
      return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    case SamplerWrapMode::kCOUNT:
      CHECK(false);
  }
  UNREACHABLE();
  return VK_SAMPLER_ADDRESS_MODE_REPEAT;
}

VkSampleCountFlagBits num_samples_to_vk_sample_flag_bits(uint32_t num_samples) {
  switch (num_samples) {
    case 1:
      return VK_SAMPLE_COUNT_1_BIT;
    case 2:
      return VK_SAMPLE_COUNT_2_BIT;
    case 4:
      return VK_SAMPLE_COUNT_4_BIT;
    case 8:
      return VK_SAMPLE_COUNT_8_BIT;
    case 16:
      return VK_SAMPLE_COUNT_16_BIT;
    case 32:
      return VK_SAMPLE_COUNT_32_BIT;
    case 64:
      return VK_SAMPLE_COUNT_64_BIT;
    default:
      CHECK(false) << "Unsupported sample count. Must be power of 2, maximum 64";
  }
  UNREACHABLE();
  return VK_SAMPLE_COUNT_1_BIT;
}

VkSampleCountFlagBits raster_sample_count_to_vk_sample_flag_bits(
    RasterSampleCount value) {
  static constexpr std::array<VkSampleCountFlagBits, 7> vk_bits = {
      VK_SAMPLE_COUNT_1_BIT,
      VK_SAMPLE_COUNT_2_BIT,
      VK_SAMPLE_COUNT_4_BIT,
      VK_SAMPLE_COUNT_8_BIT,
      VK_SAMPLE_COUNT_16_BIT,
      VK_SAMPLE_COUNT_32_BIT,
      VK_SAMPLE_COUNT_64_BIT};
  int index = static_cast<int>(value);
  CHECK_LT(index, 7);
  return vk_bits[index];
}

VkShaderStageFlagBits shader_stage_to_vk_shader_stage_flag(ShaderStage shader_stage) {
  switch (shader_stage) {
    case ShaderStage::kVertex:
      return VK_SHADER_STAGE_VERTEX_BIT;
    case ShaderStage::kFragment:
      return VK_SHADER_STAGE_FRAGMENT_BIT;
    case ShaderStage::kGeometry:
      return VK_SHADER_STAGE_GEOMETRY_BIT;
    case ShaderStage::kTessControl:
      return VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;
    case ShaderStage::kTessEval:
      return VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
    case ShaderStage::kCompute:
      return VK_SHADER_STAGE_COMPUTE_BIT;
    case ShaderStage::kRayGen:
      return VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    case ShaderStage::kAnyHit:
      return VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
    case ShaderStage::kClosestHit:
      return VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    case ShaderStage::kMiss:
      return VK_SHADER_STAGE_MISS_BIT_KHR;
    case ShaderStage::kIntersection:
      return VK_SHADER_STAGE_INTERSECTION_BIT_KHR;
    case ShaderStage::kCallable:
      return VK_SHADER_STAGE_CALLABLE_BIT_KHR;
    case ShaderStage::kMesh:
      return VK_SHADER_STAGE_MESH_BIT_EXT;
    case ShaderStage::kTask:
      return VK_SHADER_STAGE_TASK_BIT_EXT;
    default:
      CHECK(false) << "Unsupported Shader Stage!";
  }
  UNREACHABLE();
  return VK_SHADER_STAGE_FLAG_BITS_MAX_ENUM;
}

VkShaderStageFlagBits shader_stage_bits_to_vk_shader_stage_flag(ShaderStageBits bits) {
  int vk_bits{0};
  if (ShaderStageBits::kVertex == (bits & ShaderStageBits::kVertex)) {
    vk_bits |= VK_SHADER_STAGE_VERTEX_BIT;
  }
  if (ShaderStageBits::kFragment == (bits & ShaderStageBits::kFragment)) {
    vk_bits |= VK_SHADER_STAGE_FRAGMENT_BIT;
  }
  if (ShaderStageBits::kGeometry == (bits & ShaderStageBits::kGeometry)) {
    vk_bits |= VK_SHADER_STAGE_GEOMETRY_BIT;
  }
  if (ShaderStageBits::kCompute == (bits & ShaderStageBits::kCompute)) {
    vk_bits |= VK_SHADER_STAGE_COMPUTE_BIT;
  }
  if (ShaderStageBits::kRayGen == (bits & ShaderStageBits::kRayGen)) {
    vk_bits |= VK_SHADER_STAGE_RAYGEN_BIT_KHR;
  }
  if (ShaderStageBits::kAnyHit == (bits & ShaderStageBits::kAnyHit)) {
    vk_bits |= VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
  }
  if (ShaderStageBits::kClosestHit == (bits & ShaderStageBits::kClosestHit)) {
    vk_bits |= VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
  }
  if (ShaderStageBits::kMiss == (bits & ShaderStageBits::kMiss)) {
    vk_bits |= VK_SHADER_STAGE_MISS_BIT_KHR;
  }
  if (ShaderStageBits::kIntersection == (bits & ShaderStageBits::kIntersection)) {
    vk_bits |= VK_SHADER_STAGE_INTERSECTION_BIT_KHR;
  }
  if (ShaderStageBits::kCallable == (bits & ShaderStageBits::kCallable)) {
    vk_bits |= VK_SHADER_STAGE_CALLABLE_BIT_KHR;
  }
  if (ShaderStageBits::kMesh == (bits & ShaderStageBits::kMesh)) {
    vk_bits |= VK_SHADER_STAGE_MESH_BIT_EXT;
  }
  if (ShaderStageBits::kTask == (bits & ShaderStageBits::kTask)) {
    vk_bits |= VK_SHADER_STAGE_TASK_BIT_EXT;
  }
  return static_cast<VkShaderStageFlagBits>(vk_bits);
}

VkIndexType index_buffer_data_type_to_vk_index_type(IndexBufferDataType data_type) {
  switch (data_type) {
    case IndexBufferDataType::kUnsigned16:
      return VK_INDEX_TYPE_UINT16;
    case IndexBufferDataType::kUnsigned32:
      return VK_INDEX_TYPE_UINT32;
  }
  UNREACHABLE();
  return VK_INDEX_TYPE_UINT16;
}

VkImageLayout image_layout_to_vk_image_layout(ImageLayout layout,
                                              bool is_color_attachment) {
  switch (layout) {
    case ImageLayout::kUndefined:
      return VK_IMAGE_LAYOUT_UNDEFINED;
    case ImageLayout::kAttachment:
      if (is_color_attachment) {
        return VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
      } else {
        return VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
      }
    case ImageLayout::kShaderReadOnly:
      return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    case ImageLayout::kTransferSrc:
      return VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    case ImageLayout::kTransferDst:
      return VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    case ImageLayout::kMapRead:
    case ImageLayout::kMapWrite:
    case ImageLayout::kGeneral:
      return VK_IMAGE_LAYOUT_GENERAL;
    case ImageLayout::kPresentSrc:
      return VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
  }
  UNREACHABLE();
  return VK_IMAGE_LAYOUT_UNDEFINED;
}

ImageLayout image_usage_bits_to_final_layout(ImageUsageBits bits) {
  // return a desirable final layout for an image after a clear or get
  auto const usage_storage = any_bits_set(bits & ImageUsageBits::kStorageBit);
  return usage_storage ? ImageLayout::kGeneral : ImageLayout::kShaderReadOnly;
}

VkAccelerationStructureTypeKHR accel_struct_type_to_vk_accel_struct_type(
    AccelerationStructureType type) {
  switch (type) {
    case AccelerationStructureType::kBottomLevel:
      return VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    case AccelerationStructureType::kTopLevel:
      return VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
  }
  UNREACHABLE();
  return VK_ACCELERATION_STRUCTURE_TYPE_GENERIC_KHR;
}

}  // namespace gfx
