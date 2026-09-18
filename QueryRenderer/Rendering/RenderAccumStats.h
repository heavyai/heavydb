/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cmath>
#include <cstdint>
#include <memory>

struct RenderAccumStats {
  uint32_t min{0u};
  uint32_t max{0u};
  uint64_t count{0ull};
  uint32_t numpixels{0};
  uint64_t sqdiffsum{0ull};
  double avg{0.0};
  double variance{0.0};
  double stddev{0.0};

  RenderAccumStats() = default;
  explicit RenderAccumStats(const uint32_t min,
                            const uint32_t max,
                            const uint64_t count,
                            const uint32_t numpixels,
                            const uint64_t sqdiffsum)
      : min{min}
      , max{max}
      , count{count}
      , numpixels{numpixels}
      , sqdiffsum{sqdiffsum}
      , avg{numpixels > 0u ? double(count) / double(numpixels) : 0.0}
      , variance{numpixels > 0u ? double(sqdiffsum) / double(numpixels) : 0.0}
      , stddev{std::sqrt(variance)} {}
};

using RenderAccumStatsUqPtr = std::unique_ptr<RenderAccumStats>;
