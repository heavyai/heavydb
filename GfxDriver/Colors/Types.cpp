/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Colors/Types.h"

#include <ostream>

namespace gfx {

template <>
ColorType getColorType<ColorRGBA>() {
  return ColorType::RGBA;
}

template <>
ColorType getColorType<ColorHSL>() {
  return ColorType::HSL;
}

template <>
ColorType getColorType<ColorLAB>() {
  return ColorType::LAB;
}

template <>
ColorType getColorType<ColorHCL>() {
  return ColorType::HCL;
}

std::string to_string(const ColorType type) {
  switch (type) {
    case ColorType::RGBA:
      return "RGBA";
    case ColorType::HSL:
      return "HSL";
    case ColorType::LAB:
      return "LAB";
    case ColorType::HCL:
      return "HCL";
    default:
      return "Unknown Color";
  }

  return "";
}

}  // namespace gfx

std::ostream& operator<<(std::ostream& os, const gfx::ColorType colorType) {
  os << gfx::to_string(colorType);
  return os;
}
