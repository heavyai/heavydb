/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

namespace gfx {

// Number of tiles to use when tiling
const int kNumPPLLTiles = 4;

// Number of tiles stored in the tiles shared uniform buffer
// +1 for full image "tile" holding image width and height
const int kPPLLTilesUBOSize = kNumPPLLTiles + 1;

// Maximum ratio of fragments to polygons when determining if tiling should be used
// Computed as fragment count / poly count
const float kMaxPPLLFragmentCountToPolyRatio = 0.5f;  // 1 fragment per 2 polys

// Fragment buffer size threshold that will force tiling on, regardless of poly to
// fragment ratio
const uint32_t kMaxPPLLFullFragmentBufferSize = 50000000;

// Maximum number of primitive batches to use
const uint32_t kMaxNumPPLLPrimitiveBatches = 50;

// Number of batches to use when sizing the stats SSBO
// +1 for storing the sum for each tile across all batches
const uint32_t kPPLLStatsUBOPrimitiveBatches = kMaxNumPPLLPrimitiveBatches + 1;

}  // namespace gfx
