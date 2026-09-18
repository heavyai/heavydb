/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Utils/ImageUtils.h"

#include <boost/filesystem.hpp>

#include <iostream>
#include <string>
#include <vector>

#include "GfxDriver/Resources/AttachmentManager.h"
#include "GfxDriver/Resources/Texture.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "ThirdParty/stb/stb_image_write.h"
#define STB_IMAGE_IMPLEMENTATION
#include "ThirdParty/stb/stb_image.h"

namespace filesystem = boost::filesystem;

namespace gfx {

//
// remap_pixel_buffer_to_rgba
//
// TODO: need hardware format for depth
void remap_byte_buffer_to_rgba(const PixelFormat pixel_format,
                               std::vector<uint8_t>& pixels,
                               DriverType driver_type) {
  // For d24_s8, Vulkan packs depth into low bits
  auto unpack_d24_s8 = [](uint8_t* byte_ptr) -> float {
    uint32_t ival = *reinterpret_cast<uint32_t*>(byte_ptr);
    float depth = static_cast<float>(ival & 0x00FFFFFF);
    depth /= 16777216.f;  // 2^24 to normalize
    return depth;
  };

  auto unpack_d32 = [](uint8_t* byte_ptr) -> float {
    return *reinterpret_cast<float*>(byte_ptr);
  };

  std::function<float(uint8_t*)> remap_func;
  const auto stride = 4;
  switch (pixel_format) {
    case PixelFormat::kDepth:
    case PixelFormat::kDepthStencil:
      remap_func = unpack_d24_s8;
      break;
    case PixelFormat::kDepthHighP:
    case PixelFormat::kDepthStencilHighP:
      remap_func = unpack_d32;
      break;
    default:
      CHECK(false) << "PixelFormat not supported by remap_byte_buffer_to_rgba";
  }

  // Get min / max values
  uint8_t* pixel_data = pixels.data();
  auto const pixel_data_size = pixels.size();
  float min = 1.0f;
  float max = 0.0f;
  uint8_t* ptr = pixel_data;
  uint8_t* read_end = pixel_data + pixel_data_size;
  for (; ptr < read_end; ptr += stride) {
    float val = fabs(remap_func(ptr));
    min = val > 0.0f ? std::min(val, min) : min;
    max = std::max(val, max);
  }

  // remap values 0..255
  float scale = max > 0.0f ? 1.0f / (max - min) : 0.0f;
  float offset = min;
  for (ptr = pixel_data; ptr < read_end; ptr += stride) {
    float val = remap_func(ptr);
    val = (val > 0.0f) ? (val - offset) * scale : 0.0f;
    uint8_t uval = static_cast<uint8_t>(val * 255.0f);
    ptr[0] = uval;
    ptr[1] = uval;
    ptr[2] = uval;
    ptr[3] = 255;
  }
#if 0
  std::cout << "min: " << min << " max: " << max << std::endl;
#endif
}

//
// get_pixels_from_framebuffer
//
GetPixelsResult get_pixels_from_framebuffer(Framebuffer& framebuffer,
                                            const Framebuffer::Attachment attachment) {
  auto texture = framebuffer.getAttachmentManager().getAttachmentTexture(attachment);
  CHECK(texture);
  return get_pixels_from_texture(*texture);
}

//
// get_pixels_from_texture
//
GetPixelsResult get_pixels_from_texture(const Texture& texture) {
  auto width = texture.getWidth();
  auto height = texture.getHeight();
  auto pixel_format = texture.getPixelFormat();
  auto data_size = pixelFormatDataSize(pixel_format);

  std::vector<uint8_t> pixels(width * height * data_size, 0);
  texture.getPixels(width, height, 1, pixel_format, pixels.data(), pixels.size());

  // remap format to 32-bit RGBA
  if (!is_color_pixel_format(pixel_format)) {
    remap_byte_buffer_to_rgba(
        pixel_format, pixels, texture.getDeviceContext().getDriverType());
  } else if (pixel_format == PixelFormat::kR32UI) {
    // Force what will become alpha to be opaque
    for (size_t i = 0; i < pixels.size(); i += 4) {
      pixels[i + 3] = 255;
    }
  }
  return {width, height, pixel_format, std::move(pixels)};
}

//
// write_png_image
//
void write_png_image(Framebuffer& framebuffer,
                     const Framebuffer::Attachment attachment,
                     const std::string& filename) {
  auto [width, height, pixel_format, pixels] =
      get_pixels_from_framebuffer(framebuffer, attachment);
  write_png_image(pixels, width, height, filename);
}

void write_png_image(const std::vector<uint8_t>& pixels,
                     const uint32_t width,
                     const uint32_t height,
                     const std::string& filename) {
  auto result =
      stbi_write_png(filename.c_str(), width, height, 4, pixels.data(), width * 4);
  CHECK_NE(result, 0);
}

void write_png_image(const GetPixelsResult& pixel_result, const std::string& filename) {
  auto const& [width, height, pixel_format, pixels] = pixel_result;
  CHECK(is_color_pixel_format(pixel_format));
  stbi_write_png(filename.c_str(), width, height, 4, pixels.data(), width * 4);
}

//
// read_png_image
//
ReadImageResult read_png_image(const std::string& filename, const bool is_required) {
  int width;
  int height;
  int num_channels;
  if (boost::filesystem::exists(filename)) {
    auto* raw_data = stbi_load(filename.c_str(), &width, &height, &num_channels, 0);
    CHECK(raw_data);
    std::vector<uint8_t> data(raw_data, raw_data + (width * height * num_channels));
    stbi_image_free(raw_data);
    return {width, height, num_channels, data};
  } else if (is_required) {
    CHECK(false) << "Unable to load png file: " << filename;
  }
  return {0, 0, 0, {}};
}

ReadImageResult read_png_image_from_memory(const unsigned char* array,
                                           uint32_t array_size) {
  int width;
  int height;
  int num_channels;
  auto* raw_data =
      stbi_load_from_memory(array, array_size, &width, &height, &num_channels, 0);
  CHECK(raw_data);
  std::vector<uint8_t> data(raw_data, raw_data + (width * height * num_channels));
  stbi_image_free(raw_data);
  return {width, height, num_channels, data};
}
}  // namespace gfx
