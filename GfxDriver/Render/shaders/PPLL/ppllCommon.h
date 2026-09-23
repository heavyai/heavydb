/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * ppllCommon.h
 *
 * include file for use in GLSL and CPP files
 * Declares structures used for host<->device communication when using PPLL
 * Shader uniform blocks must use `scalar` layout to guarantee alignment match
 *
 */

#ifndef PPLL_COMMON_H
#define PPLL_COMMON_H

#ifdef __cplusplus
#include <cstdint>
#include <iomanip>
#include <ostream>
#endif  // __cplusplus

// Image
struct ImageInfo {
  uint32_t width;
  uint32_t height;
  uint32_t num_pixels;
};

// Image tile
struct Tile {
  int32_t x;
  int32_t y;
  uint32_t w;
  uint32_t h;
};

// Counting pass stats
struct PPLLFragmentStats {
  uint64_t total_fragment_count;
  uint64_t deep_pixel_fragment_count;
  uint32_t max_per_pixel_fragment_count;

#ifdef __cplusplus
  void reset() {
    total_fragment_count = 0ull;
    deep_pixel_fragment_count = 0ull;
    max_per_pixel_fragment_count = 0u;
  }
  void print(std::ostream& os, int name_width = 25, int value_width = 11) const {
    auto format = [&](std::string_view name) -> std::ostream& {
      os << std::left << std::setw(name_width) << name << std::right
         << std::setw(value_width);
      return os;
    };
    format("Fragments:") << total_fragment_count << "\n";
    format("Deep pixel fragments:") << deep_pixel_fragment_count << "\n";
    format("Max fragments per pixel:") << max_per_pixel_fragment_count;
  }
#endif  // __cplusplus
};

// Resolved stats after composite
struct PPLLResolveStats {
  uint32_t overflow_pixel_count;
  uint32_t max_unique_ids;
};

#endif  // PPLL_COMMON_H
