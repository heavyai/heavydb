/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <map>

#include <vulkan/vulkan.h>

#include "GfxDriver/Drivers/Vulkan/VulkanMemoryMgr.h"
#include "GfxDriver/Resources/Texture.h"
#include "GfxDriver/Types.h"

namespace gfx {

class VulkanTexture : public Texture {
 public:
  explicit VulkanTexture(const DeviceContext& device_ctx,
                         std::string_view resource_tracking_string,
                         uint32_t width,
                         uint32_t height,
                         uint32_t depth,
                         PixelFormat pixel_format,
                         uint32_t num_samples,
                         bool is_array_texture,
                         ImageUsageBits extra_usage_bits,
                         TextureSamplerState sampler_state = TextureSamplerState(),
                         const void* pixel_data = nullptr);
  ~VulkanTexture() override;
  VulkanTexture() = delete;

  ResourceHandle getResourceHandle() const override {
    return reinterpret_cast<ResourceHandle>(vk_image_);
  }
  VkImage getImage() const { return vk_image_; }
  VkImageView getImageView() const { return vk_image_view_; }
  VkSampler getSampler() const { return vk_sampler_; }
  VulkanAllocation* getMemoryAllocation() const { return vk_memory_allocation_.get(); }
  ImageUsageBits getUsageBits() const { return usage_bits_; }

  uint64_t getGpuAllocationSize() const override;

  void resize(const uint32_t width, const uint32_t height, const uint32_t depth) override;

  void clearPixels() override;
  void clearPixelsToValue(const ClearTextureValue& value) override;

  void setPixels(const uint32_t width,
                 const uint32_t height,
                 const uint32_t depth,
                 const PixelFormat pixel_format,
                 const void* pixel_data) override;

  // this function can be called on MS textures for debugging purposes
  // but should not be done in normal operation as it is not performant
  void getPixels(const uint32_t width,
                 const uint32_t height,
                 const uint32_t depth,
                 const PixelFormat pixel_format,
                 void* pixel_data,
                 const uint64_t buffer_size) const override;

  ViewCreateResult createView(const uint32_t view_id,
                              const PixelFormat pixel_format) override;
  bool destroyView(const uint32_t view_id) override;
  bool hasView(const uint32_t view_id) const override;
  ResourceHandle getViewHandle(const uint32_t view_id) const override;
  PixelFormat getViewPixelFormat(const uint32_t view_id) const override;

 private:
  void initResource(const void* pixel_data) override;
  void cleanupResourceBase() override;
  void makeEmpty() override;
  void rebuild(uint32_t width, uint32_t height, uint32_t depth);

  VkImage vk_image_;
  VkImageView vk_image_view_;
  VkSampler vk_sampler_;
  ImageUsageBits usage_bits_;
  VkImageUsageFlags vk_usage_bits_;

  VulkanMemoryMgr::allocation_ptr vk_memory_allocation_;
  VkMemoryRequirements vk_memory_requirements_;

  struct ViewInfo {
    VkImageView vk_image_view;
    PixelFormat pixel_format;
    ViewInfo(VkImageView vk_image_view, const PixelFormat pixel_format)
        : vk_image_view{vk_image_view}, pixel_format{pixel_format} {}
  };

  std::map<uint32_t, ViewInfo> image_view_map_;

  void setPixelsInternal(const uint32_t width,
                         const uint32_t height,
                         const void* pixel_data,
                         const bool from_init_resource);
};

}  // namespace gfx
