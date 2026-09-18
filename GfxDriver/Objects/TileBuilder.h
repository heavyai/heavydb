/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <iostream>

#include "Logger/Logger.h"

namespace gfx {

/**
 * Rect2D
 *
 * Simple POD data struct to use as the default tile type
 * Supports negative origin coordinates
 * */
struct Rect2D {
  int32_t x{0};
  int32_t y{0};
  uint32_t w{0u};
  uint32_t h{0u};

  Rect2D() = default;
  Rect2D(int32_t x, int32_t y, uint32_t w, uint32_t h) : x{x}, y{y}, w{w}, h{h} {}

  bool operator==(const Rect2D& other) const {
    return x == other.x && y == other.y && w == other.w && h == other.h;
  }
  bool operator!=(const Rect2D& other) const { return !operator==(other); }

  friend std::ostream& operator<<(std::ostream& os, const Rect2D& r) {
    os << "[" << r.x << ", " << r.y << ", " << r.w << ", " << r.h << "]";
    return os;
  }
};

/// Compile time optional pretty printing of the queue during build
static constexpr bool kPrintQueue = false;
void pretty_print_tile(int32_t x, int32_t y, uint32_t w, uint32_t h);

/// Template function for setting a tile. Allows extending the builder to other tile types
template <typename T>
void set_tile_func(T& tile, int32_t x, int32_t y, uint32_t w, uint32_t h);

/// set_tile_func specialization for Rect2D
template <>
void set_tile_func<Rect2D>(Rect2D& r, int32_t x, int32_t y, uint32_t w, uint32_t h);

/**
 * build_tile_queue
 * Fills a container of QueueType with rectangular tiles of TileType
 *
 * Defaults to std::vector<Rect2D>
 *
 * Tiles are generated along x (rows) then y (columns)
 * Partial remainder tiles are generated along the right and bottom
 *
 * For image dimensions smaller than the tile size, generates single tile of image size
 *
 * usage examples:
 *   std::vector<Rect2D> tiles;
 *   build_tile_queue(tiles, 320, 240, 20, 20);
 *
 *   std::deque<Rect2D> tiles;
 *   build_tile_queue<Rect2D, std::dequeue<Rect2D>(tiles, 320, 240, 20, 20);
 *
 *   std::vector<VkRect2D> tiles;
 *   build_tile_queue<VkRect2D>(320, 240, 20, 20);
 * */
template <typename TileType = Rect2D,
          typename QueueType = std::vector<TileType>,
          void (*set_tile)(TileType&, int32_t, int32_t, uint32_t, uint32_t) =
              set_tile_func<TileType>>
void build_tile_queue(QueueType& queue,
                      uint32_t w,
                      uint32_t h,
                      uint32_t tile_w,
                      uint32_t tile_h) {
  CHECK_GT(w, 0u);
  CHECK_GT(h, 0u);
  CHECK_GT(tile_w, 0u);
  CHECK_GT(tile_h, 0u);

  uint32_t num_rows;
  int32_t row_full_tiles_w;  // width of full sized tiles portion
  uint32_t row_extra;        // width of fractional portion
  if (w > tile_w) {
    num_rows = w / tile_w;
    row_full_tiles_w = static_cast<int32_t>(num_rows * tile_w);
    row_extra = w % tile_w;
    if (row_extra) {
      num_rows++;
    }
  } else {
    num_rows = 1u;
    row_full_tiles_w = 0u;
    row_extra = w;
  }

  uint32_t num_cols;
  int32_t col_full_tiles_h;  // height of full sized tiles portion
  uint32_t col_extra;        // height of fractional portion
  if (h > tile_h) {
    num_cols = h / tile_h;
    col_full_tiles_h = static_cast<int32_t>(num_cols * tile_h);
    col_extra = h % tile_h;
    if (col_extra) {
      num_cols++;
    }
  } else {
    num_cols = 1u;
    col_full_tiles_h = 0u;
    col_extra = h;
  }

  queue.resize(num_rows * num_cols);

  if constexpr (kPrintQueue) {  // NOLINT
    std::cout << "queue  w: " << w << "  h: " << h << "  tile_w: " << tile_w
              << "  tile_h: " << tile_h << "  num_tiles: " << queue.size() << std::endl;
  }

  //
  // Fill the queue container
  //
  auto elem = queue.begin();
  int32_t y = 0;
  for (; y < col_full_tiles_h; y += tile_h) {
    CHECK_LT(y, static_cast<int32_t>(h));
    int32_t x = 0;
    // Full width tiles
    for (; x < row_full_tiles_w; x += tile_w, ++elem) {
      CHECK_LT(x, static_cast<int32_t>(w));
      CHECK(elem != queue.end());
      set_tile_func(*elem, x, y, tile_w, tile_h);
      if constexpr (kPrintQueue) {  // NOLINT
        pretty_print_tile(x, y, tile_w, tile_h);
      }
    }
    // Partial width tile along right edge
    if (row_extra) {
      CHECK(elem != queue.end());
      set_tile_func(*elem, x, y, row_extra, tile_h);
      if constexpr (kPrintQueue) {  // NOLINT
        pretty_print_tile(x, y, row_extra, tile_h);
      }
      elem++;
    }
  }

  // Partial height tiles along bottom edge
  if (col_extra) {
    int32_t x = 0;
    for (; x < row_full_tiles_w; x += tile_w, ++elem) {
      CHECK(elem != queue.end());
      set_tile_func(*elem, x, y, tile_w, col_extra);
      if constexpr (kPrintQueue) {  // NOLINT
        pretty_print_tile(x, y, tile_w, col_extra);
      }
    }

    // Partial tile in bottom right corner
    if (row_extra) {
      CHECK(elem != queue.end());
      set_tile_func(*elem, x, y, row_extra, col_extra);
      if constexpr (kPrintQueue) {  // NOLINT
        pretty_print_tile(x, y, row_extra, col_extra);
      }
      elem++;
    }
  }

  // Ensure we filled the entire queue container
  CHECK(elem == queue.end());
}

}  // namespace gfx
