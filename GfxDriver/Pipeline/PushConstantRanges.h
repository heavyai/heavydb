/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include "GfxDriver/ShaderCompiler/Types.h"

namespace gfx {

//
// PushConstantRange struct
//
// Defines affected shader stages, byte offset, and byte size of a push constant
// block
struct PushConstantRange {
  explicit PushConstantRange(ShaderStageBits shader_stages,
                             uint32_t offset,
                             uint32_t size)
      : shader_stages{shader_stages}, offset{offset}, size{size} {}
  ShaderStageBits shader_stages;
  uint32_t offset;
  uint32_t size;
};

//
// PushConstantRanges class
//
// Container to wrap a vector of PushConstantRanges
class PushConstantRanges {
 public:
  PushConstantRanges() = default;
  PushConstantRanges(std::initializer_list<PushConstantRange> l);
  PushConstantRanges& operator=(const PushConstantRanges& rhs);

  // Clear the vector
  void clear();

  // Insert a new PushConstantRange into the vector
  void insert(ShaderStageBits shader_stages, uint32_t offset, uint32_t size);

  // Pass a vector of PushConstantRanges and overwrite existing ranges.
  // new_ranges will be std::moved allowing the vector copy to be skipped if
  // new_ranges is movable
  void set(std::vector<PushConstantRange> new_ranges);

  // Iterate all PushConstantRange structs, invoking the visitor funcion
  void visit(std::function<void(const PushConstantRange& r)> visitor) const;

 private:
  std::vector<PushConstantRange> ranges_;
};

}  // namespace gfx
