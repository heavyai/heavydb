/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Objects/TileBuilder.h"

#include <iomanip>

namespace gfx {

void pretty_print_tile(int32_t x, int32_t y, uint32_t w, uint32_t h) {
  std::cout << "x: " << std::left << std::setw(6) << x << "y: " << std::left
            << std::setw(6) << y;
  std::cout << "w: " << std::left << std::setw(6) << w << "h: " << std::left
            << std::setw(6) << h;
  std::cout << std::endl;
}

template <>
void set_tile_func<Rect2D>(Rect2D& r, int32_t x, int32_t y, uint32_t w, uint32_t h) {
  r.w = w;
  r.h = h;
  r.x = x;
  r.y = y;
}
}  // namespace gfx
