/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Resources/Texture.h"

#include "Logger/Logger.h"

namespace gfx {

Texture::Texture(const DeviceContext& device_ctx,
                 std::string_view resource_tracking_string,
                 uint32_t width,
                 uint32_t height,
                 uint32_t depth,
                 PixelFormat pixel_format,
                 uint32_t num_samples,
                 bool is_array_texture,
                 TextureSamplerState sampler_state,
                 const void* pixel_data)
    : Resource(device_ctx, resource_tracking_string, ResourceType::kTexture)
    , width_{width}
    , height_{height}
    , depth_{depth}
    , pixel_format_{pixel_format}
    , num_samples_{num_samples}
    , is_array_texture_{is_array_texture}
    , sampler_state_{std::move(sampler_state)} {
  if (pixel_data) {
    CHECK(!is_array_texture) << "Initializing array texture is currently unsupported";
  }
}

Texture::~Texture() {}

uint32_t Texture::getWidth() const {
  return width_;
}

uint32_t Texture::getHeight() const {
  return height_;
}

uint32_t Texture::getDepth() const {
  return depth_;
}

PixelFormat Texture::getPixelFormat() const {
  return pixel_format_;
}

uint32_t Texture::getNumSamples() const {
  return num_samples_;
}

bool Texture::isArrayTexture() const {
  return is_array_texture_;
}

const TextureSamplerState& Texture::getSamplerState() const {
  return sampler_state_;
}

}  // namespace gfx
