/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "GfxDriver/Resources/Enums.h"
#include "GfxDriver/Resources/Resource.h"
#include "GfxDriver/Resources/TextureSamplerState.h"
#include "GfxDriver/Types.h"

namespace gfx {

class Texture : public Resource {
 public:
  explicit Texture(const DeviceContext& device_ctx,
                   std::string_view resource_tracking_string,
                   uint32_t width,
                   uint32_t height,
                   uint32_t depth,
                   PixelFormat pixel_format,
                   uint32_t num_samples,
                   bool is_array_texture,
                   TextureSamplerState sampler_state = TextureSamplerState(),
                   const void* pixel_data = nullptr);
  ~Texture() override;
  Texture() = delete;

  uint32_t getWidth() const;
  uint32_t getHeight() const;
  uint32_t getDepth() const;
  PixelFormat getPixelFormat() const;
  uint32_t getNumSamples() const;
  bool isArrayTexture() const;
  const TextureSamplerState& getSamplerState() const;

  virtual void resize(const uint32_t width,
                      const uint32_t height,
                      const uint32_t depth) = 0;

  // clearPixels performs an immediate CommandList flush in Vulkan
  virtual void clearPixels() = 0;
  virtual void clearPixelsToValue(const ClearTextureValue& value) = 0;

  virtual void setPixels(const uint32_t width,
                         const uint32_t height,
                         const uint32_t depth,
                         const PixelFormat pixel_format,
                         const void* pixel_data) = 0;
  virtual void getPixels(const uint32_t width,
                         const uint32_t height,
                         const uint32_t depth,
                         const PixelFormat pixel_format,
                         void* pixel_data,
                         const uint64_t buffer_size) const = 0;

  //
  // View support
  //
  // Create additional views of the underlying image
  // Requires ImageUsageBit::kMutableView when pixel_format differs from original image
  // view_id 0 cannot be used in createView
  using ViewCreateResult = std::pair<ResourceHandle, bool>;
  virtual ViewCreateResult createView(const uint32_t view_id,
                                      const PixelFormat pixel_format) = 0;
  // Destroy existing view
  // view_id 0 cannot be used in destroyView
  virtual bool destroyView(const uint32_t view_id) = 0;

  // Check if a valid view exists for view_id
  // view_id 0 refers to the default view and always returns true
  virtual bool hasView(const uint32_t view_id) const = 0;

  // Get the ResourceHandle for an existing view_id (view must have been created)
  // view_id 0 returns the default view for the texture (created automatically)
  virtual ResourceHandle getViewHandle(const uint32_t view_id) const = 0;

  // Get the PixelFormat for an existing view_id (view must have been created)
  // view_id 0 returns the PixelFormat for the base texture
  virtual PixelFormat getViewPixelFormat(const uint32_t view_id) const = 0;

 protected:
  uint32_t width_;
  uint32_t height_;
  uint32_t depth_;
  PixelFormat pixel_format_;
  uint32_t num_samples_;
  bool is_array_texture_;
  TextureSamplerState sampler_state_;

  virtual void initResource(const void* pixel_data) = 0;
};

};  // namespace gfx
