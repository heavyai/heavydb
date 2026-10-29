/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <glm/common.hpp>

#include "GfxDriver/Utils/NoiseUtils.h"

class HeightfieldCallback {
 public:
  enum class GeneratorType { kNoise, kCell, kChips };

  GeneratorType generator{GeneratorType::kCell};
  float height{0.3f};
  bool invert{false};
  float low{0.3f};
  float high{0.5f};
  float noise_size{3.0f};
  int noise_levels{4};
  float H{1.5f};
  float lacunarity{3.0f};
  gfx::CellDistanceMetric distance_metric{gfx::CellDistanceMetric::kEuclideanSquared};

  float operator()(float u, float v) {
    float f = 0.0f;
    switch (generator) {
      case GeneratorType::kNoise:
        f = gfx::fractal_noise(noise_levels, u, v, noise_size, 0.5f, 0.0f, 1.0f);
        break;
      case GeneratorType::kCell:
        f = wCell_fbm_basic(glm::vec3(u, v, 1.0f) * noise_size,
                            noise_levels,
                            H,
                            lacunarity,
                            distance_metric);
        break;
      case GeneratorType::kChips:
        f = wCell_fbm_chips(glm::vec3(u, v, 1.0f) * noise_size,
                            noise_levels,
                            H,
                            lacunarity,
                            distance_metric);
        break;
    }
    f = glm::smoothstep(low, high, f) * height;
    if (invert) {
      f = 1.0f - f;
    }
    return f;
  }
};
