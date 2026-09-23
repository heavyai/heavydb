/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "GfxDriver/Resources/Enums.h"

namespace gfx {

struct TextureSamplerState {
  SamplerFilterMode min_filter_mode;
  SamplerFilterMode mag_filter_mode;
  SamplerWrapMode wrap_mode_s;
  SamplerWrapMode wrap_mode_t;
  // TODO as needed:
  // sampler_mipmap_mode
  // anisotropy

  constexpr TextureSamplerState()
      : min_filter_mode{SamplerFilterMode::kNearest}
      , mag_filter_mode{SamplerFilterMode::kLinear}
      , wrap_mode_s{SamplerWrapMode::kClampEdge}
      , wrap_mode_t{SamplerWrapMode::kClampEdge} {}

  constexpr TextureSamplerState(const SamplerFilterMode min_filter_mode,
                                const SamplerFilterMode mag_filter_mode,
                                const SamplerWrapMode wrap_mode_s,
                                const SamplerWrapMode wrap_mode_t)
      : min_filter_mode{min_filter_mode}
      , mag_filter_mode{mag_filter_mode}
      , wrap_mode_s{wrap_mode_s}
      , wrap_mode_t{wrap_mode_t} {}

  constexpr bool operator==(const TextureSamplerState& rhs) const {
    return (min_filter_mode == rhs.min_filter_mode &&
            mag_filter_mode == rhs.mag_filter_mode && wrap_mode_s == rhs.wrap_mode_s &&
            wrap_mode_t == rhs.wrap_mode_t);
  }

  constexpr bool operator!=(const TextureSamplerState& rhs) const {
    return !(operator==(rhs));
  }
};

static constexpr TextureSamplerState default_texture_sampler_state_linear = {
    SamplerFilterMode::kNearest,
    SamplerFilterMode::kLinear,
    SamplerWrapMode::kClampEdge,
    SamplerWrapMode::kClampEdge};

static constexpr TextureSamplerState default_texture_sampler_state_nearest = {
    SamplerFilterMode::kNearest,
    SamplerFilterMode::kNearest,
    SamplerWrapMode::kClampEdge,
    SamplerWrapMode::kClampEdge};

static constexpr TextureSamplerState get_default_sampler_state_for_format(
    const PixelFormat format) {
  switch (format) {
    case PixelFormat::kR8:
    case PixelFormat::kRG8:
    case PixelFormat::kRGBA8:
    case PixelFormat::kBGRA8:
      return default_texture_sampler_state_linear;
    case PixelFormat::kR32UI:
    case PixelFormat::kR32I:
    case PixelFormat::kDepth:
    case PixelFormat::kDepthHighP:
    case PixelFormat::kDepthStencil:
    case PixelFormat::kDepthStencilHighP:
    // Include kCOUNT to allow the function to be constexpr. This will still catch cases
    // when a format is added since there's no default, but at the cost of allowing kCOUNT
    // to be passed
    case PixelFormat::kCOUNT:
      return default_texture_sampler_state_nearest;
  }
  // NOTE: UNREACHABLE can't be used in a constexpr function
  return default_texture_sampler_state_nearest;
}

}  // namespace gfx
