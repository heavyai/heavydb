/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace QueryRenderer {

struct RenderPixels {
  std::vector<std::byte> pixels;
  uint32_t width = 0;
  uint32_t height = 0;
  // empty and assignable
  RenderPixels() = default;
  // for returning valid images
  RenderPixels(std::vector<std::byte>&& p, const uint32_t w, const uint32_t h)
      : pixels{std::move(p)}, width{w}, height{h} {}
  // for returning empty images
  RenderPixels(const uint32_t w, const uint32_t h) : width{w}, height{h} {
    pixels.resize(width * height * 4, std::byte(0));
  }
};

struct RenderRequestInfo {
  RenderPixels renderData;
  int64_t total_execution_time_ms = 0;
  int64_t total_render_time_ms = 0;
};

}  // namespace QueryRenderer
