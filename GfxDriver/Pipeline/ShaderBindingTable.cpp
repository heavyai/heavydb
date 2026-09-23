/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Pipeline/ShaderBindingTable.h"

#include "Logger/Logger.h"

namespace gfx {

ShaderBindingTable::Entry shader_stage_to_sbt_entry(const ShaderStage stage) {
  using e = ShaderBindingTable::Entry;
  switch (stage) {
    case ShaderStage::kRayGen:
      return e::kRayGen;
    case ShaderStage::kMiss:
      return e::kMiss;
    case ShaderStage::kAnyHit:
    case ShaderStage::kClosestHit:
    case ShaderStage::kIntersection:
      return e::kHit;
    case ShaderStage::kCallable:
      return e::kCallable;
    default:
      CHECK(false) << "Invalid ShaderStage '" << to_string(stage) << "' for SBTRegion";
  }
  UNREACHABLE();
  return e::kCount;
}

std::ostream& operator<<(std::ostream& os, const ShaderBindingTable::Entry value) {
  using e = ShaderBindingTable::Entry;
  switch (value) {
    case e::kRayGen:
      os << "Raygen";
      break;
    case e::kMiss:
      os << "Miss";
      break;
    case e::kHit:
      os << "Hit";
      break;
    case e::kCallable:
      os << "Callable";
      break;
    case e::kCount:
      UNREACHABLE();
      break;
  }
  return os;
}

}  // namespace gfx
