/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <optional>
#include <unordered_map>
#include <vector>

#include <vulkan/vulkan.h>
#include <boost/noncopyable.hpp>

#include "GfxDriver/Drivers/Vulkan/Commands/VulkanCommandBuffers.h"
#include "GfxDriver/Resources/Enums.h"

namespace gfx {

VkPipelineStageFlags get_stage_mask_for_layout(const ImageLayout layout,
                                               const bool is_color_format);

VkPipelineStageFlags get_stage_mask_for_vk_layout(const VkImageLayout layout);

class ImageLayoutManager : boost::noncopyable {
 public:
  using ImageAndFormat = std::pair<VkImage, PixelFormat>;

  ImageLayoutManager() = default;
  ~ImageLayoutManager();

  void addOrSetLayout(const VkImage image, const ImageLayout layout);
  const ImageLayout getCurrentLayout(const VkImage image);

  void transitionToLayout(const VkImage image,
                          const ImageLayout layout,
                          VulkanCommandBuffer& cmd_buffer,
                          const std::optional<VkPipelineStageFlags> in_stage_mask,
                          const VkPipelineStageFlags out_stage_mask,
                          const VkImageSubresourceRange& subresource_range);

  void transitionToLayout(const std::vector<VkImage>& color_images,
                          const uint32_t num_color_layers,
                          const std::optional<ImageAndFormat> depth_image_and_format,
                          const ImageLayout layout,
                          VulkanCommandBuffer& cmd_buffer,
                          const std::optional<VkPipelineStageFlags> in_stage_mask,
                          const VkPipelineStageFlags out_stage_mask);

  void remove(const VkImage image);

 private:
  using map_type = std::unordered_map<VkImage, ImageLayout>;

  map_type layout_map_;
  map_type::iterator findMapEntry(const VkImage image);
};

}  // namespace gfx
