/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "GfxDriver/Math/Constants.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <type_traits>

namespace gfx {

namespace ColorValidators {

template <typename T>
struct PassThruValidation {
  inline T operator()(const T& v) const { return v; }
};

template <typename T, const int defaultLo, const int defaultHi>
struct ClampValidation {
  ClampValidation() : lo(static_cast<T>(defaultLo)), hi(static_cast<T>(defaultHi)) {}
  ClampValidation(const T& lo, const T& hi) : lo(lo), hi(hi) {}
  T operator()(const T& v) const { return Math::clamp<T>(v, lo, hi); }

 private:
  T lo;
  T hi;
};

template <typename T, const int defaultLo, const int defaultHi>
struct WrapValidation {
  WrapValidation() : lo(static_cast<T>(defaultLo)), hi(static_cast<T>(defaultHi)) {}
  WrapValidation(const T& lo, const T& hi) : lo(lo), hi(hi) {}
  T operator()(const T& v) const {
    auto val = v;
    if (val < lo) {
      val = hi - std::fmod(lo - val, hi - lo);
    }
    return lo + std::fmod(val - lo, hi - lo);
  }

 private:
  T lo;
  T hi;
};

using Clamp0to1f = ClampValidation<float, 0, 1>;
using AngleWrap0to360 = WrapValidation<float, 0, 360>;

}  // namespace ColorValidators

namespace PackedFloatColorConverters {
struct ConvertUInt8To0to1 {
  float operator()(const uint8_t& val) const {
    return Math::clamp<float>(float(val) / 255.0f, 0.0f, 1.0f);
  }
  uint8_t inverse(const float val) const {
    return static_cast<uint8_t>(Math::clamp(val, 0.0f, 1.0f) * 255.0f);
  }
};
}  // namespace PackedFloatColorConverters

struct ColorInitializer;

enum class ColorType { RGBA, HSL, LAB, HCL, INVALID };
using ColorArray = std::array<float, 4>;
class ColorRGBA;
class ColorHSL;
class ColorLAB;
class ColorHCL;
class ColorUnion;

template <class T>
struct is_color
    : std::integral_constant<
          bool,
          std::is_same<ColorRGBA, typename std::remove_cv<T>::type>::value ||
              std::is_same<ColorHSL, typename std::remove_cv<T>::type>::value ||
              std::is_same<ColorLAB, typename std::remove_cv<T>::type>::value ||
              std::is_same<ColorHCL, typename std::remove_cv<T>::type>::value> {};

template <class T, class TT>
struct is_specific_color
    : std::integral_constant<bool,
                             std::is_same<TT, typename std::remove_cv<T>::type>::value> {
};

template <class T>
struct is_color_union
    : std::integral_constant<
          bool,
          std::is_same<ColorUnion, typename std::remove_cv<T>::type>::value> {};

template <class T, std::enable_if_t<gfx::is_color<T>::value>* = nullptr>
ColorType getColorType() {
  return ColorType::RGBA;
}

template <>
ColorType getColorType<ColorRGBA>();

template <>
ColorType getColorType<ColorHSL>();

template <>
ColorType getColorType<ColorLAB>();

template <>
ColorType getColorType<ColorHCL>();

std::string to_string(const ColorType type);

template <typename T>
using EnableIfColorType = std::enable_if_t<gfx::is_color<T>::value>;

template <typename T, typename ColorType>
using EnableIfSpecificColorType =
    std::enable_if_t<gfx::is_specific_color<T, ColorType>::value>;

}  // namespace gfx

std::ostream& operator<<(std::ostream& os, const gfx::ColorType colorType);
