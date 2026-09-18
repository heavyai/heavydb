/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Drivers/Vulkan/Resources/VulkanTexture.h"

#include <iostream>

#include "GfxDriver/Commands/CommandList.h"
#include "GfxDriver/Drivers/Vulkan/Commands/StagingContext.h"
#include "GfxDriver/Drivers/Vulkan/Resources/Utils.h"
#include "GfxDriver/Drivers/Vulkan/Resources/VulkanResourceManager.h"
#include "GfxDriver/Drivers/Vulkan/VulkanDeviceContext.h"
#include "GfxDriver/Drivers/Vulkan/VulkanResult.h"
#include "GfxDriver/Resources/Texture.h"
#include "GfxDriver/Utils/LoggingUtils.h"
#include "Shared/scope.h"

namespace gfx {

VulkanTexture::VulkanTexture(const DeviceContext& device_ctx,
                             std::string_view resource_tracking_string,
                             uint32_t width,
                             uint32_t height,
                             uint32_t depth,
                             PixelFormat pixel_format,
                             uint32_t num_samples,
                             bool is_array_texture,
                             ImageUsageBits extra_usage_bits,
                             TextureSamplerState sampler_state,
                             const void* pixel_data)
    : Texture(device_ctx,
              resource_tracking_string,
              width,
              height,
              depth,
              pixel_format,
              num_samples,
              is_array_texture,
              std::move(sampler_state),
              pixel_data)
    , vk_image_{VK_NULL_HANDLE}
    , vk_image_view_{VK_NULL_HANDLE}
    , vk_sampler_{VK_NULL_HANDLE}
    , usage_bits_{extra_usage_bits | ImageUsageBits::kSampledBit}
    , vk_usage_bits_{image_usage_bits_to_vk_image_usage_bits(usage_bits_) |
                     VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT}
    , vk_memory_requirements_({}) {
  RUNTIME_EX_ASSERT(width > 0 && height > 0 && depth > 0,
                    "Invalid dimensions for the texture array. Dimensions must be > 0");
  initResource(pixel_data);
}

VulkanTexture::~VulkanTexture() {
  cleanupResource();
}

void VulkanTexture::initResource(const void* pixel_data) {
  RUNTIME_EX_ASSERT(num_samples_ > 0,
                    "Invalid number of samples " + std::to_string(num_samples_));

  const VulkanDeviceContext& vk_device =
      static_cast<const VulkanDeviceContext&>(getDeviceContext());

  bool allow_export = any_bits_set(getDeviceContext().getCapabilityBits() &
                                   DeviceCapabilityBits::kImageMemoryExport) &&
                      ((usage_bits_ & ImageUsageBits::kExternalApiBit) ==
                       ImageUsageBits::kExternalApiBit);

  if (allow_export) {
    CHECK(any_bits_set(vk_device.getCapabilityBits() &
                       DeviceCapabilityBits::kImageMemoryExport));
  }

  if (!is_array_texture_) {
    CHECK_EQ(depth_, 1u);
  }

  auto resource_name = getTrackingData().origin;

  // Ensure we clean up any dangling Vulkan bits if initialization fails
  bool is_complete = false;
  ScopeGuard scope_guard = [&is_complete, this] {
    if (!is_complete) {
      cleanupResourceBase();
    }
  };

  //
  // create image
  //

  // create info
  VkImageCreateInfo image_info = {};
  image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  image_info.imageType = VK_IMAGE_TYPE_2D;
  image_info.extent.width = width_;
  image_info.extent.height = height_;
  image_info.extent.depth = 1;
  image_info.arrayLayers = depth_;
  image_info.format = vk_device.pixelFormatToVkFormat(pixel_format_);
  image_info.samples = num_samples_to_vk_sample_flag_bits(num_samples_);
  image_info.flags = is_color_pixel_format(pixel_format_) &&
                             any_bits_set(usage_bits_ & ImageUsageBits::kMutableViewBit)
                         ? VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT
                         : 0;

  // more create info (TBD)
  image_info.mipLevels = 1;
  image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
  image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  image_info.usage = vk_usage_bits_;
  image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  // memory export
  VkExternalMemoryImageCreateInfo ext_mem_image_ci = {};
  if (allow_export) {
    ext_mem_image_ci.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
    ext_mem_image_ci.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    image_info.pNext = &ext_mem_image_ci;
  }

  // Out of memory error logging callback
  auto oom_logging_cb = [&](std::ostream& os) {
    os << "Texture create info:\n";
    StreamStatFormatter oom_logger(os);
    oom_logger("Width", width_);
    oom_logger("Height", height_);
    oom_logger("Depth", depth_);
  };

  // create
  VkResult result =
      vkCreateImage(vk_device.getHandle(), &image_info, nullptr, &vk_image_);
  CHECK_OOM_VKRESULT(
      result, "creating image", resource_name, 0, vk_device, oom_logging_cb);

  // name it
  vk_device.nameVulkanObject(VK_OBJECT_TYPE_IMAGE, vk_image_, resource_name);

  // memory requirements
  vk_memory_requirements_ = {};
  vkGetImageMemoryRequirements(
      vk_device.getHandle(), vk_image_, &vk_memory_requirements_);

  // allocate memory
  vk_memory_allocation_ =
      vk_device.getMemoryManager().alloc(resource_name,
                                         vk_memory_requirements_,
                                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                         allow_export,
                                         false,
                                         -1,
                                         oom_logging_cb);

  // bind memory to image
  vkBindImageMemory(
      vk_device.getHandle(), vk_image_, vk_memory_allocation_->getHandle(), 0);

  // add to ImageLayoutManager
  static_cast<VulkanResourceManager*>(&vk_device.getResourceManager())
      ->getImageLayoutManager()
      .addOrSetLayout(vk_image_, ImageLayout::kUndefined);

  //
  // create image view
  //

  VkImageViewCreateInfo view_info = {};
  view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  view_info.image = vk_image_;
  view_info.viewType =
      is_array_texture_ ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
  view_info.format = vk_device.pixelFormatToVkFormat(pixel_format_);
  view_info.subresourceRange.aspectMask =
      pixel_format_to_vk_image_aspect_flags(pixel_format_);
  view_info.subresourceRange.baseMipLevel = 0;
  view_info.subresourceRange.levelCount = 1;  // @TODO mip support
  view_info.subresourceRange.baseArrayLayer = 0;
  view_info.subresourceRange.layerCount = depth_;

  result = vkCreateImageView(vk_device.getHandle(), &view_info, nullptr, &vk_image_view_);
  CHECK_OOM_VKRESULT(
      result, "creating image view", resource_name, 0, vk_device, oom_logging_cb);

  // name it
  vk_device.nameVulkanObject(VK_OBJECT_TYPE_IMAGE_VIEW, vk_image_view_, resource_name);

  //
  // create sampler?
  //

  // always create a sampler for now regardless of TextureSamplerState values
  // we can choose not to use the sampler at descriptor-set-write time if this
  // texture is used as a storage image
  // @TODO(se) simplify TextureSamplerState and allow for explicit "no sampler"
  bool needs_sampler = true;

  if (needs_sampler) {
    VkSamplerCreateInfo sampler_info = {};
    sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sampler_info.magFilter =
        sampler_filter_mode_to_vk_filter(getSamplerState().mag_filter_mode);
    sampler_info.minFilter =
        sampler_filter_mode_to_vk_filter(getSamplerState().min_filter_mode);

    sampler_info.addressModeU =
        sampler_wrap_mode_to_vk_sampler_address_mode(getSamplerState().wrap_mode_s);
    sampler_info.addressModeV =
        sampler_wrap_mode_to_vk_sampler_address_mode(getSamplerState().wrap_mode_t);
    sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;

    sampler_info.anisotropyEnable = VK_FALSE;  // @TODO enable aniso?
    sampler_info.maxAnisotropy = 0.0f;

    sampler_info.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;

    // enable to allow specifying coordinates in [0,textureSize] form
    sampler_info.unnormalizedCoordinates = VK_FALSE;

    // allow compare with fixed value during filtering
    // generally useful for shadow PCF
    sampler_info.compareEnable = VK_FALSE;
    sampler_info.compareOp = VK_COMPARE_OP_ALWAYS;

    // mip-mapping
    sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;  // @TODO support mipmapping
    sampler_info.mipLodBias = 0;
    sampler_info.minLod = 0.0f;
    sampler_info.maxLod = 1.0f;

    result = vkCreateSampler(vk_device.getHandle(), &sampler_info, nullptr, &vk_sampler_);
    CHECK_OOM_VKRESULT(
        result, "creating sampler", resource_name, 0, vk_device, oom_logging_cb);

    // name it
    vk_device.nameVulkanObject(VK_OBJECT_TYPE_SAMPLER, vk_sampler_, resource_name);
  }

  if (pixel_data) {
    setPixelsInternal(width_, height_, pixel_data, true);
  }

  is_complete = true;
  setUsable();
}

Texture::ViewCreateResult VulkanTexture::createView(const uint32_t view_id,
                                                    const PixelFormat pixel_format) {
  CHECK(vk_image_);
  CHECK_NE(view_id, 0u) << "view_id 0 is not valid in createView";
  if (pixel_format != pixel_format_) {
    CHECK(any_bits_set(usage_bits_ & ImageUsageBits::kMutableViewBit));
  }

  auto did_replace_view = destroyView(view_id);

  const VulkanDeviceContext& vk_device =
      static_cast<const VulkanDeviceContext&>(getDeviceContext());

  VkImageViewCreateInfo view_info = {};
  view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  view_info.image = vk_image_;
  view_info.viewType =
      is_array_texture_ ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
  view_info.format = vk_device.pixelFormatToVkFormat(pixel_format);
  view_info.subresourceRange.aspectMask =
      pixel_format_to_vk_image_aspect_flags(pixel_format);

  view_info.subresourceRange.baseMipLevel = 0;
  view_info.subresourceRange.levelCount = 1;
  view_info.subresourceRange.baseArrayLayer = 0;
  view_info.subresourceRange.layerCount = depth_;

  VkImageView new_image_view = VK_NULL_HANDLE;
  auto result =
      vkCreateImageView(vk_device.getHandle(), &view_info, nullptr, &new_image_view);
  CHECK_OOM_VKRESULT(result,
                     "creating image view",
                     getTrackingData().origin,
                     0,
                     vk_device,
                     std::nullopt);

  auto [itr, did_insert] =
      image_view_map_.try_emplace(view_id, new_image_view, pixel_format);
  if (!did_insert) {
    vkDestroyImageView(vk_device.getHandle(), new_image_view, nullptr);
    return {0u, false};
  } else {
    // name it
    vk_device.nameVulkanObject(
        VK_OBJECT_TYPE_IMAGE_VIEW,
        new_image_view,
        getTrackingData().origin + "(" + std::to_string(view_id) + ")");
    return {reinterpret_cast<ResourceHandle>(new_image_view), did_replace_view};
  }
}

bool VulkanTexture::destroyView(const uint32_t view_id) {
  CHECK_NE(view_id, 0u) << "view_id 0 is not valid in destroyView";
  bool did_erase = false;
  if (image_view_map_.count(view_id)) {
    try {
      auto& vi = image_view_map_.at(view_id);
      if (vi.vk_image_view != VK_NULL_HANDLE) {
        vkDestroyImageView(
            static_cast<const VulkanDeviceContext&>(getDeviceContext()).getHandle(),
            vi.vk_image_view,
            nullptr);
        vi.vk_image_view = VK_NULL_HANDLE;
      }
      image_view_map_.erase(view_id);
      did_erase = true;
    } catch (std::out_of_range& err) {
      THROW_RUNTIME_EX("View id " + std::to_string(view_id) + " out of range");
    }
  }
  return did_erase;
}

bool VulkanTexture::hasView(const uint32_t view_id) const {
  return view_id == 0 ? true : image_view_map_.count(view_id);
}

ResourceHandle VulkanTexture::getViewHandle(const uint32_t view_id) const {
  try {
    return view_id == 0 ? reinterpret_cast<ResourceHandle>(vk_image_view_)
                        : reinterpret_cast<ResourceHandle>(
                              image_view_map_.at(view_id).vk_image_view);
  } catch (std::out_of_range& err) {
    THROW_RUNTIME_EX("View id " + std::to_string(view_id) + " out of range");
  }
}

PixelFormat VulkanTexture::getViewPixelFormat(const uint32_t view_id) const {
  try {
    return view_id == 0 ? pixel_format_ : image_view_map_.at(view_id).pixel_format;
  } catch (std::out_of_range& err) {
    THROW_RUNTIME_EX("View id " + std::to_string(view_id) + " out of range");
  }
}

void VulkanTexture::cleanupResourceBase() {
  const VulkanDeviceContext& vk_device =
      static_cast<const VulkanDeviceContext&>(getDeviceContext());
  auto device_handle = vk_device.getHandle();

  if (vk_sampler_ != VK_NULL_HANDLE) {
    vkDestroySampler(device_handle, vk_sampler_, nullptr);
    vk_sampler_ = VK_NULL_HANDLE;
  }

  if (vk_image_view_ != VK_NULL_HANDLE) {
    vkDestroyImageView(device_handle, vk_image_view_, nullptr);
    vk_image_view_ = VK_NULL_HANDLE;
  }

  for (auto& vi : image_view_map_) {
    if (vi.second.vk_image_view != VK_NULL_HANDLE) {
      vkDestroyImageView(device_handle, vi.second.vk_image_view, nullptr);
      vi.second.vk_image_view = VK_NULL_HANDLE;
    }
  }
  image_view_map_.clear();

  if (vk_image_ != VK_NULL_HANDLE) {
    static_cast<VulkanResourceManager*>(&vk_device.getResourceManager())
        ->getImageLayoutManager()
        .remove(vk_image_);
    vkDestroyImage(device_handle, vk_image_, nullptr);
    vk_image_ = VK_NULL_HANDLE;
  }

  if (vk_memory_allocation_) {
    vk_device.getMemoryManager().free(std::move(vk_memory_allocation_));
  }

  vk_memory_requirements_ = {};

  makeEmpty();
}

uint64_t VulkanTexture::getGpuAllocationSize() const {
  // Images are currently always device local
  return vk_memory_allocation_ ? vk_memory_allocation_->size() : 0ULL;
}

void VulkanTexture::makeEmpty() {
  width_ = 0;
  height_ = 0;
  depth_ = 0;
}

void VulkanTexture::resize(uint32_t width, uint32_t height, uint32_t depth) {
  if (width != width_ || height != height_ || depth != depth_) {
    validateUsability(__FILE__, __LINE__);

    // Save view info
    auto orig_views = image_view_map_;

    cleanupResourceBase();
    width_ = width;
    height_ = height;
    depth_ = depth;
    initResource(nullptr);

    // Rebuild views
    for (auto const& [id, info] : orig_views) {
      createView(id, info.pixel_format);
    }
  }
}

void VulkanTexture::setPixels(const uint32_t width,
                              const uint32_t height,
                              const uint32_t depth,
                              const PixelFormat pixel_format,
                              const void* pixel_data) {
  CHECK_EQ(pixel_format, pixel_format_);
  CHECK_EQ(num_samples_, 1U);

  CHECK_LE(width, width_);
  CHECK_LE(height, height_);
  CHECK_LE(depth, depth_);

  auto const pixel_size = pixelFormatDataSize(pixel_format_);
  const uint64_t num_bytes = width_ * height_ * depth_ * pixel_size;

  auto const& vk_device = static_cast<const VulkanDeviceContext&>(getDeviceContext());
  auto& staging = vk_device.getStagingContext();

  auto locked_staging_buffer = staging.acquireStagingBuffer(num_bytes);
  auto* buffer_data = locked_staging_buffer.buffer;
  CHECK(buffer_data);

  ScopeGuard release_staging = [&] {
    vk_device.getStagingContext().releaseStagingBuffer(std::move(locked_staging_buffer),
                                                       this);
  };

  auto const src_line_bytes = width * pixel_size;
  auto const dst_line_bytes = width_ * pixel_size;
  auto const src_image_bytes = src_line_bytes * height;
  auto const dst_image_bytes = dst_line_bytes * height_;
  for (uint32_t layer = 0; layer < depth; layer++) {
    auto const* src_line =
        reinterpret_cast<const unsigned char*>(pixel_data) + (layer * src_image_bytes);
    auto* dst_line =
        reinterpret_cast<unsigned char*>(buffer_data) + (layer * dst_image_bytes);
    for (uint32_t y = 0; y < height; y++) {
      std::memcpy(dst_line, src_line, src_line_bytes);
      src_line += src_line_bytes;
      dst_line += dst_line_bytes;
    }
  }
}

void VulkanTexture::getPixels(const uint32_t width,
                              const uint32_t height,
                              const uint32_t depth,
                              const PixelFormat pixel_format,
                              void* pixel_data,
                              const uint64_t buffer_size) const {
  CHECK_EQ(pixel_format, pixel_format_);

  // must support sub-region
  CHECK_LE(width, width_);
  CHECK_LE(height, height_);
  CHECK_LE(depth, depth_);

  auto const& vk_device = static_cast<const VulkanDeviceContext&>(getDeviceContext());
  auto& staging = vk_device.getStagingContext();

  uint64_t expected_buffer_size =
      width * height * depth * pixelFormatDataSize(pixel_format);
  CHECK_EQ(buffer_size, expected_buffer_size);

  if (num_samples_ == 1) {
    staging.getPixels(getImage(),
                      width,
                      height,
                      depth,
                      pixel_format_,
                      static_cast<std::byte*>(pixel_data),
                      buffer_size);
  } else {
    CHECK_EQ(depth, 1u) << "multi-sample array texture not supported";
#if !NDEBUG
    auto const msg =
        "Performance Warning: VulkanTexture::getPixels() called on multi-sample texture";
    LOG(WARNING) << msg;
    std::cout << msg << std::endl;
#endif
    gfx::resource_ptr<Texture> temp_texture;
    ScopeGuard destroy_temp = [&]() {
      if (temp_texture) {
        vk_device.getResourceManager().destroyTexture(std::move(temp_texture));
      }
    };
    temp_texture =
        vk_device.getResourceManager().createTexture("getPixels multi-sample temp",
                                                     width,
                                                     height,
                                                     1,
                                                     pixel_format,
                                                     1,
                                                     false,
                                                     usage_bits_,
                                                     TextureSamplerState());
    auto const& temp_vulkan_texture = static_cast<const VulkanTexture&>(*temp_texture);
    staging.copyOrResolvePixels(vk_image_,
                                temp_vulkan_texture.getImage(),
                                width,
                                height,
                                pixel_format,
                                num_samples_,
                                true);
    staging.getPixels(temp_vulkan_texture.getImage(),
                      width,
                      height,
                      1,
                      pixel_format_,
                      static_cast<std::byte*>(pixel_data),
                      buffer_size);
  }
}

void VulkanTexture::setPixelsInternal(const uint32_t width,
                                      const uint32_t height,
                                      const void* pixel_data,
                                      const bool from_init_resource) {
  auto const pixel_size = pixelFormatDataSize(pixel_format_);
  const uint64_t num_bytes = width_ * height_ * pixel_size;

  auto const& vk_device = static_cast<const VulkanDeviceContext&>(getDeviceContext());
  auto& staging = vk_device.getStagingContext();

  auto locked_staging_buffer = staging.acquireStagingBuffer(num_bytes);
  auto* buffer_data = locked_staging_buffer.buffer;
  CHECK(buffer_data);

  ScopeGuard release_staging = [&] {
    vk_device.getStagingContext().releaseStagingBuffer(std::move(locked_staging_buffer),
                                                       this);
  };

  auto const src_line_bytes = width * pixel_size;
  auto const dst_line_bytes = width_ * pixel_size;
  auto const* src_line = reinterpret_cast<const unsigned char*>(pixel_data);
  auto* dst_line = reinterpret_cast<unsigned char*>(buffer_data);
  for (uint32_t y = 0; y < height; y++) {
    std::memcpy(dst_line, src_line, src_line_bytes);
    src_line += src_line_bytes;
    dst_line += dst_line_bytes;
  }
}

void VulkanTexture::clearPixels() {
  auto& cmd_list = getDeviceContext().getCommandList();
  cmd_list.clearTexture(*this).flush("Texture clear pixels");
}

void VulkanTexture::clearPixelsToValue(const ClearTextureValue& value) {
  auto& cmd_list = getDeviceContext().getCommandList();
  cmd_list.clearTextureToValue(*this, value).flush("Texture clear pixels to value");
}

}  // namespace gfx
