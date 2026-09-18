/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>
#include <vector>

#include "GfxDriver/DeviceContext.h"
#include "GfxDriver/Resources/Framebuffer.h"

namespace gfx {

//
// get_pixels_from_framebuffer
//
struct GetPixelsResult {
  uint32_t width = 0;
  uint32_t height = 0;
  gfx::PixelFormat pixel_format = PixelFormat::kRGBA8;
  std::vector<uint8_t> pixels;
};

GetPixelsResult get_pixels_from_framebuffer(
    gfx::Framebuffer& framebuffer,
    const gfx::Framebuffer::Attachment attachment);

//
// get_pixels_from_texture
//
// returns width, height, PixelFormat, and vector of pixels
//
GetPixelsResult get_pixels_from_texture(const gfx::Texture& texture);

//
// write_png_image
//
// write any attachment consisting of 1, 2, 3, or 4 8-bit channels to a PNG file
//
void write_png_image(gfx::Framebuffer& framebuffer,
                     const gfx::Framebuffer::Attachment attachment,
                     const std::string& filename);

void write_png_image(const std::vector<uint8_t>& pixels,
                     const uint32_t width,
                     const uint32_t height,
                     const std::string& filename);

void write_png_image(const GetPixelsResult& pixel_result, const std::string& filename);

//
// read_png_image
//
// returns width, height, num channels, and vector of uint8_t pixels
//
struct ReadImageResult {
  int width;
  int height;
  int num_channels;
  std::vector<uint8_t> pixels;
};

ReadImageResult read_png_image(const std::string& filename, const bool is_required);

ReadImageResult read_png_image_from_memory(const unsigned char* array,
                                           uint32_t array_size);

}  // namespace gfx
