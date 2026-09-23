/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Tests/RenderTests/Utils/GoldenImage.h"

#include <boost/filesystem.hpp>

#include <iostream>
#include <string>
#include <vector>

#include "GfxDriver/Resources/Texture.h"
#include "GfxDriver/Utils/ImageUtils.h"

using namespace gfx;
namespace filesystem = boost::filesystem;

//
// GoldenImage class
//
namespace {
static std::string build_png_filename(const std::string& base,
                                      const std::string& file,
                                      const std::string& driver_suffix,
                                      const bool append_driver_suffix) {
  if (append_driver_suffix) {
    return base + file + driver_suffix + ".png";
  } else {
    return base + file + ".png";
  }
}
}  // namespace

GoldenImage::GoldenImage(std::string_view base_path,
                         gfx::DriverType driver_type,
                         bool force_write)
    : base_path_{base_path}
    , driver_type_{driver_type}
    , driver_suffix_{"_vk"}
    , force_write_{force_write} {
  if (force_write_) {
    std::cerr << "GoldenImage is in force_write mode and will overwrite golden images"
              << std::endl;
  }
}

testing::AssertionResult GoldenImage::write(Framebuffer& framebuffer,
                                            const Framebuffer::Attachment attachment,
                                            const std::string& base_filename,
                                            const bool append_driver_suffix) const {
  auto png_filename =
      build_png_filename(base_path_, base_filename, driver_suffix_, append_driver_suffix);
  write_png_image(framebuffer, attachment, png_filename);

  // Check if file written successfully
  if (!filesystem::exists(png_filename)) {
    return testing::AssertionFailure() << "Failed to write golden image file";
  }
  return testing::AssertionSuccess();
}

testing::AssertionResult GoldenImage::write(const std::vector<uint8_t>& pixels,
                                            const uint32_t width,
                                            const uint32_t height,
                                            const std::string& base_filename,
                                            const bool append_driver_suffix) const {
  auto png_filename =
      build_png_filename(base_path_, base_filename, driver_suffix_, append_driver_suffix);
  write_png_image(pixels, width, height, png_filename);

  // Check if file written successfully
  if (!filesystem::exists(png_filename)) {
    return testing::AssertionFailure() << "Failed to write golden image file";
  }
  return testing::AssertionSuccess();
}

testing::AssertionResult GoldenImage::write(const gfx::Texture& texture,
                                            const std::string& base_filename,
                                            const bool append_driver_suffix) const {
  auto pixels = get_pixels_from_texture(texture);
  auto png_filename =
      build_png_filename(base_path_, base_filename, driver_suffix_, append_driver_suffix);
  write_png_image(pixels, png_filename);

  // Check if file written successfully
  if (!filesystem::exists(png_filename)) {
    return testing::AssertionFailure() << "Failed to write golden image file";
  }
  return testing::AssertionSuccess();
}

testing::AssertionResult GoldenImage::autoCompareDrivers(
    Framebuffer& framebuffer,
    const Framebuffer::Attachment attachment,
    const std::string& base_filename,
    const uint32_t failed_pixel_count_tolerance,
    const uint8_t max_delta_per_channel_tolerance) const {
  if (driver_type_ == gfx::DriverType::kVulkan) {
    return write(framebuffer, Framebuffer::Attachment::kColor0, base_filename);
  } else {
    return compare(framebuffer,
                   Framebuffer::Attachment::kColor0,
                   base_filename,
                   false,
                   failed_pixel_count_tolerance,
                   max_delta_per_channel_tolerance);
  }
}

testing::AssertionResult GoldenImage::compare(
    Framebuffer& framebuffer,
    const Framebuffer::Attachment attachment,
    const std::string& base_filename,
    const bool append_driver_suffix,
    const uint32_t failed_pixel_count_tolerance,
    const uint8_t max_delta_per_channel_tolerance) const {
  // Get images
  auto [width, height, format, pixels] =
      get_pixels_from_framebuffer(framebuffer, attachment);

  return compare(pixels,
                 width,
                 height,
                 pixelFormatDataSize(format),
                 base_filename,
                 append_driver_suffix,
                 failed_pixel_count_tolerance,
                 max_delta_per_channel_tolerance);
}

testing::AssertionResult GoldenImage::compare(
    const Texture& texture,
    const std::string& base_filename,
    const bool append_driver_suffix,
    const uint32_t failed_pixel_count_tolerance,
    const uint8_t max_delta_per_channel_tolerance) const {
  auto image = get_pixels_from_texture(texture);

  return compare(image.pixels,
                 image.width,
                 image.height,
                 pixelFormatDataSize(image.pixel_format),
                 base_filename,
                 append_driver_suffix,
                 failed_pixel_count_tolerance,
                 max_delta_per_channel_tolerance);
}

testing::AssertionResult GoldenImage::compare(
    const std::vector<uint8_t>& pixels,
    const uint32_t width,
    const uint32_t height,
    const uint32_t pixel_format_channels,
    const std::string& base_filename,
    const bool append_driver_suffix,
    const uint32_t failed_pixel_count_tolerance,
    const uint8_t max_delta_per_channel_tolerance) const {
  if (force_write_) {
    return write(pixels, width, height, base_filename, append_driver_suffix);
  } else {
    auto png_filename = build_png_filename(
        base_path_, base_filename, driver_suffix_, append_driver_suffix);

    auto [golden_image_width,
          golden_image_height,
          golden_image_channels,
          golden_image_pixels] = read_png_image(png_filename, true);

    // Check if golden image exists
    if (!filesystem::exists(png_filename)) {
      return testing::AssertionFailure() << "Golden image file does not exist";
    }
    // Compare metadata and pixels
    if (width != static_cast<uint32_t>(golden_image_width)) {
      return testing::AssertionFailure()
             << "Render width does not match: golden_image=" << golden_image_width
             << " but render=" << width;
    }
    if (height != static_cast<uint32_t>(golden_image_height)) {
      return testing::AssertionFailure()
             << "Render height does not match: golden_image=" << golden_image_height
             << " but render=" << height;
    }
    if (pixel_format_channels != static_cast<uint32_t>(golden_image_channels)) {
      return testing::AssertionFailure()
             << "Image channel count does not match, (wrong PixelFormat?): golden_image="
             << golden_image_channels << " but render=" << pixel_format_channels;
    }
    if (pixels != golden_image_pixels) {
      uint32_t failed_pixel_count = 0;
      size_t num_pixels = pixels.size() / pixel_format_channels;
      auto* pixel = pixels.data();
      auto* golden = golden_image_pixels.data();
      uint8_t max_channel_delta = 0;
      for (size_t i = 0; i < num_pixels;
           ++i, pixel += pixel_format_channels, golden += pixel_format_channels) {
        uint32_t did_pixel_fail = 0u;
        for (uint32_t channel = 0; channel < pixel_format_channels; ++channel) {
          uint8_t channel_delta =
              static_cast<uint8_t>(abs(static_cast<int16_t>(pixel[channel]) -
                                       static_cast<int16_t>(golden[channel])));
          if (channel_delta) {
            max_channel_delta = std::max(channel_delta, max_channel_delta);
            did_pixel_fail = 1u;
          }
        }
        failed_pixel_count += did_pixel_fail;
      }
      if ((failed_pixel_count > failed_pixel_count_tolerance) ||
          (max_channel_delta > max_delta_per_channel_tolerance)) {
        return testing::AssertionFailure()
               << failed_pixel_count
               << " pixels do not match. Max delta = " << +max_channel_delta;
      }
    }
    return testing::AssertionSuccess();
  }
}

void GoldenImage::writeImageVector(
    const std::vector<gfx::resource_ptr<gfx::Texture>>& textures,
    const std::string& base_filename,
    const bool append_driver_suffix) const {
  for (size_t i = 0; i < textures.size(); ++i) {
    auto pixels = get_pixels_from_texture(*textures[i]);
    std::string png_filename;
    if (append_driver_suffix) {
      png_filename = base_path_ + base_filename + driver_suffix_;
    } else {
      png_filename = base_path_ + base_filename;
    }
    png_filename += "_" + std::to_string(i) + ".png";
    write_png_image(pixels, png_filename);
  }
}
