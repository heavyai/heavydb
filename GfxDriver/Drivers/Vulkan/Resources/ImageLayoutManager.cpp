/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Drivers/Vulkan/Resources/ImageLayoutManager.h"

#include "GfxDriver/Drivers/Vulkan/Resources/Utils.h"
#include "Logger/Logger.h"

namespace gfx {

ImageLayoutManager::~ImageLayoutManager() {
  LOG_IF(FATAL, !layout_map_.empty())
      << "ImageLayoutManager has not been properly cleared!";
}

void ImageLayoutManager::addOrSetLayout(const VkImage image, const ImageLayout layout) {
  layout_map_[image] = layout;
}

ImageLayoutManager::map_type::iterator ImageLayoutManager::findMapEntry(
    const VkImage image) {
  auto itr = layout_map_.find(image);
  CHECK(itr != layout_map_.end()) << "Failed to find VkImage in ImageLayout map";
  return itr;
}

const ImageLayout ImageLayoutManager::getCurrentLayout(const VkImage image) {
  return findMapEntry(image)->second;
}

namespace {
std::pair<VkAccessFlags, VkImageLayout> access_mask_and_image_layout(
    const ImageLayout texture_image_layout,
    const bool is_color_format) {
  VkAccessFlags access_mask{0};
  VkImageLayout image_layout{VK_IMAGE_LAYOUT_UNDEFINED};
  switch (texture_image_layout) {
    case ImageLayout::kUndefined:
      access_mask = 0;
      image_layout = VK_IMAGE_LAYOUT_UNDEFINED;
      break;
    case ImageLayout::kGeneral:
      access_mask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
      image_layout = VK_IMAGE_LAYOUT_GENERAL;
      break;
    case ImageLayout::kAttachment:
      if (is_color_format) {
        access_mask =
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;
        image_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
      } else {
        access_mask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                      VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        image_layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
      }
      break;
    case ImageLayout::kShaderReadOnly:
      access_mask = VK_ACCESS_SHADER_READ_BIT;
      image_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      break;
    case ImageLayout::kTransferSrc:
      access_mask = VK_ACCESS_TRANSFER_READ_BIT;
      image_layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
      break;
    case ImageLayout::kTransferDst:
      access_mask = VK_ACCESS_TRANSFER_WRITE_BIT;
      image_layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
      break;
    case ImageLayout::kMapRead:
      access_mask = VK_ACCESS_MEMORY_READ_BIT;
      image_layout = VK_IMAGE_LAYOUT_GENERAL;
      break;
    case ImageLayout::kMapWrite:
      access_mask = VK_ACCESS_MEMORY_WRITE_BIT;
      image_layout = VK_IMAGE_LAYOUT_GENERAL;
      break;
    case ImageLayout::kPresentSrc:
      // according to spec docs for VkPresentInfoKHR
      access_mask = 0;
      image_layout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
      break;
  }
  return {access_mask, image_layout};
}
}  // namespace

VkPipelineStageFlags get_stage_mask_for_layout(const ImageLayout layout,
                                               const bool is_color_format) {
  switch (layout) {
    case ImageLayout::kAttachment:
      if (is_color_format) {
        return VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
      } else {
        return VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
               VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
      }
    case ImageLayout::kShaderReadOnly:
      return VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    case ImageLayout::kTransferSrc:
    case ImageLayout::kTransferDst:
      return VK_PIPELINE_STAGE_TRANSFER_BIT;
    case ImageLayout::kPresentSrc:
      return VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    case ImageLayout::kUndefined:
      return VK_PIPELINE_STAGE_NONE_KHR;
    case ImageLayout::kGeneral:
      return VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    case ImageLayout::kMapRead:
    case ImageLayout::kMapWrite:
      LOG(FATAL) << "unable to get pipeline stage for ImageLayout: " << layout;
      break;
  }
  UNREACHABLE();
  return VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
}

VkPipelineStageFlags get_stage_mask_for_vk_layout(const VkImageLayout layout) {
  switch (layout) {
    case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
      return VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
      return VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
             VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
      return VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
    case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
      return VK_PIPELINE_STAGE_TRANSFER_BIT;
    default:
      LOG(FATAL) << "unable to get pipeline stage for VkImageLayout: " << layout;
  }
  UNREACHABLE();
  return VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
}

void ImageLayoutManager::transitionToLayout(
    const VkImage image,
    const ImageLayout new_layout,
    VulkanCommandBuffer& cmd_buffer,
    const std::optional<VkPipelineStageFlags> in_stage_mask,
    const VkPipelineStageFlags out_stage_mask,
    const VkImageSubresourceRange& subresource_range) {
  auto itr = findMapEntry(image);
  bool is_color_format = subresource_range.aspectMask & VK_IMAGE_ASPECT_COLOR_BIT;

  if (itr->second != new_layout) {
    auto [in_access_mask, in_vk_layout] =
        access_mask_and_image_layout(itr->second, is_color_format);
    auto [out_access_mask, out_vk_layout] =
        access_mask_and_image_layout(new_layout, is_color_format);

    auto in_stage_mask_to_use =
        in_stage_mask ? *in_stage_mask
                      : get_stage_mask_for_layout(itr->second, is_color_format);

    if (in_stage_mask && in_stage_mask == VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT) {
      in_access_mask = 0;
    }

    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.srcAccessMask = in_access_mask;
    barrier.dstAccessMask = out_access_mask;
    barrier.oldLayout = in_vk_layout;
    barrier.newLayout = out_vk_layout;
    barrier.image = image;
    barrier.subresourceRange = subresource_range;

    vkCmdPipelineBarrier(cmd_buffer.getHandle(),
                         in_stage_mask_to_use,
                         out_stage_mask,
                         0,
                         0,
                         nullptr,
                         0,
                         nullptr,
                         1,
                         &barrier);

    itr->second = new_layout;
  }
}

void ImageLayoutManager::transitionToLayout(
    const std::vector<VkImage>& color_images,
    const uint32_t num_color_layers,
    const std::optional<ImageAndFormat> depth_image_and_format,
    const ImageLayout new_layout,
    VulkanCommandBuffer& cmd_buffer,
    const std::optional<VkPipelineStageFlags> in_stage_mask,
    const VkPipelineStageFlags out_stage_mask) {
  // Get the current layout and ensure all images are in the same layout
  std::vector<map_type::iterator> color_iterators;

  // Find map entries for all the color images
  for (auto const& image : color_images) {
    auto itr = findMapEntry(image);
    if (itr->second != new_layout) {
      color_iterators.push_back(itr);
    }
  }

  // Create VkImageMemoryBarriers for all the color images we need to transition
  std::vector<VkImageMemoryBarrier> barriers;
  VkPipelineStageFlags in_stage_mask_to_use = in_stage_mask ? *in_stage_mask : 0u;

  VkImageSubresourceRange subresource_range{
      VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, num_color_layers};
  if (!color_iterators.empty()) {
    auto [out_access_mask, out_vk_layout] =
        access_mask_and_image_layout(new_layout, true);

    for (auto const& map_iterator : color_iterators) {
      auto [in_access_mask, in_vk_layout] =
          access_mask_and_image_layout(map_iterator->second, true);
      if (!in_stage_mask) {
        in_stage_mask_to_use |= get_stage_mask_for_vk_layout(in_vk_layout);
      }

      VkImageMemoryBarrier barrier = {};
      barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
      barrier.srcAccessMask = in_access_mask;
      barrier.dstAccessMask = out_access_mask;
      barrier.oldLayout = in_vk_layout;
      barrier.newLayout = out_vk_layout;
      barrier.image = map_iterator->first;
      barrier.subresourceRange = subresource_range;
      barriers.push_back(barrier);
    }
  }

  // Check if we have a depth image to transition to kAttachment
  if (depth_image_and_format && (new_layout == ImageLayout::kAttachment)) {
    auto const& [depth_vk_image, depth_format] = *depth_image_and_format;
    auto itr = findMapEntry(depth_vk_image);
    if (itr->second != ImageLayout::kAttachment) {
      auto [in_access_mask, in_vk_layout] =
          access_mask_and_image_layout(itr->second, false);
      auto [out_access_mask, out_vk_layout] =
          access_mask_and_image_layout(new_layout, false);

      // aspect may be depth or depth + stencil
      subresource_range.aspectMask = pixel_format_to_vk_image_aspect_flags(depth_format);
      subresource_range.layerCount = 1;  // no depth arrays

      VkImageMemoryBarrier barrier = {};
      barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
      barrier.srcAccessMask = in_access_mask;
      barrier.dstAccessMask = out_access_mask;
      barrier.oldLayout = in_vk_layout;
      barrier.newLayout = out_vk_layout;
      barrier.image = depth_vk_image;
      barrier.subresourceRange = subresource_range;
      barriers.push_back(barrier);

      // It's now safe to update our layouts since no further trappable errors
      // or exceptions can occur, so update the map
      itr->second = new_layout;
    }
  }

  for (auto const& map_iterator : color_iterators) {
    map_iterator->second = new_layout;
  }

  if (!barriers.empty()) {
    vkCmdPipelineBarrier(cmd_buffer.getHandle(),
                         in_stage_mask_to_use,
                         out_stage_mask,
                         0,
                         0,
                         nullptr,
                         0,
                         nullptr,
                         barriers.size(),
                         barriers.data());
  }
}

void ImageLayoutManager::remove(const VkImage image) {
  layout_map_.erase(image);
}

}  // namespace gfx
