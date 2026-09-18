/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Pipeline/Pipeline.h"

namespace gfx {

std::ostream& operator<<(std::ostream& os,
                         const RaytracingPipeline::ShaderGroup::Type value) {
  using e = RaytracingPipeline::ShaderGroup::Type;
  switch (value) {
    case e::kGeneral:
      os << "General";
      break;
    case e::kTriangleHit:
      os << "TriangleHit";
      break;
    case e::kProceduralHit:
      os << "ProceduralHit";
      break;
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const RaytracingPipeline::ShaderGroup& value) {
  os << "Type: " << value.type << "  SBTRegion: " << value.sbt_entry << "  Handle: ";
  for (auto v : value.handle_data) {
    os << +v << " ";
  }
  return os;
}

}  // namespace gfx
