/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "GfxDriver/Enums.h"
#include "GfxDriver/Resources/Framebuffer.h"

//
// GoldenImage class
//
class GoldenImage {
 public:
  explicit GoldenImage(std::string_view base_filename,
                       gfx::DriverType driver_type,
                       bool force_write = false);

  // Compare pixels to golden image
  // Tolerances should only be used as a last resort, for renders where driver or
  // architecture differences create mismatches
  // failed_pixels_count_tolerance: maximum number of pixels that do not match exactly
  // max_delta_per_channel_tolerance: maximum allowed difference in any channel of any
  //   pixel (0-255)
  testing::AssertionResult compare(
      gfx::Framebuffer& framebuffer,
      const gfx::Framebuffer::Attachment attachment,
      const std::string& base_filename,
      const bool append_driver_suffix = false,
      const uint32_t failed_pixel_count_tolerance = 0u,
      const uint8_t max_delta_per_channel_tolerance = 0u) const;

  testing::AssertionResult compare(
      const gfx::Texture& texture,
      const std::string& base_filename,
      const bool append_driver_suffix = false,
      const uint32_t failed_pixel_count_tolerance = 0u,
      const uint8_t max_delta_per_channel_tolerance = 0u) const;

  testing::AssertionResult compare(
      const std::vector<uint8_t>& pixels,
      const uint32_t width,
      const uint32_t height,
      const uint32_t pixel_format_channels,
      const std::string& base_filename,
      const bool append_driver_suffix = false,
      const uint32_t failed_pixel_count_tolerance = 0u,
      const uint8_t max_delta_per_channel_tolerance = 0u) const;

  testing::AssertionResult write(gfx::Framebuffer& framebuffer,
                                 const gfx::Framebuffer::Attachment attachment,
                                 const std::string& base_filename,
                                 const bool append_driver_suffix = false) const;

  testing::AssertionResult write(const std::vector<uint8_t>& pixels,
                                 const uint32_t width,
                                 const uint32_t height,
                                 const std::string& base_filename,
                                 const bool append_driver_suffix = false) const;

  testing::AssertionResult write(const gfx::Texture& texture,
                                 const std::string& base_filename,
                                 const bool append_driver_suffix = false) const;

  // Since Vulkan always runs first, write out the image with Vulkan, but then do a
  // compare when the test for a different future driver runs. This allows editing
  // the shader and comparing without having to recompile
  testing::AssertionResult autoCompareDrivers(
      gfx::Framebuffer& framebuffer,
      const gfx::Framebuffer::Attachment attachment,
      const std::string& filename,
      const uint32_t failed_pixel_count_tolerance = 0u,
      const uint8_t max_delta_per_channel_tolerance = 0u) const;

  void writeImageVector(const std::vector<gfx::resource_ptr<gfx::Texture>>& textures,
                        const std::string& base_filename,
                        const bool append_driver_suffix = false) const;

 private:
  std::string base_path_;
  gfx::DriverType driver_type_;
  std::string driver_suffix_;
  bool force_write_;
};
