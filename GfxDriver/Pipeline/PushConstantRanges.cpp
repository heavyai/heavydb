/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Pipeline/PushConstantRanges.h"

#include <algorithm>

#include "Logger/Logger.h"

namespace gfx {

PushConstantRanges::PushConstantRanges(std::initializer_list<PushConstantRange> l)
    : ranges_(l) {}

PushConstantRanges& PushConstantRanges::operator=(const PushConstantRanges& rhs) {
  ranges_ = rhs.ranges_;
  return *this;
}

void PushConstantRanges::clear() {
  ranges_.clear();
}

void PushConstantRanges::insert(ShaderStageBits shader_stages,
                                uint32_t offset,
                                uint32_t size) {
  // Ensure size < min guaranteed size (iGPU only supports 128 bytes)
  CHECK_LE(offset + size, 128u);
  ranges_.emplace_back(shader_stages, offset, size);
}

// Pass a vector of PushConstantRanges and overwrite existing ranges
void PushConstantRanges::set(std::vector<PushConstantRange> new_ranges) {
  for (auto const& range : new_ranges) {
    // Ensure size < min guaranteed size (iGPU only supports 128 bytes)
    CHECK_LE(range.offset + range.size, 128u);
  }
  ranges_ = std::move(new_ranges);
}

void PushConstantRanges::visit(
    std::function<void(const PushConstantRange& r)> visitor) const {
  std::for_each(ranges_.begin(), ranges_.end(), visitor);
}

}  // namespace gfx
