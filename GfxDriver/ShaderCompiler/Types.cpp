/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/ShaderCompiler/Types.h"

namespace gfx {

std::string to_string(const ShaderStage value) {
  switch (value) {
    case ShaderStage::kVertex:
      return "Vertex";
    case ShaderStage::kFragment:
      return "Fragment";
    case ShaderStage::kGeometry:
      return "Geometry";
    case ShaderStage::kTessControl:
      return "TessControl";
    case ShaderStage::kTessEval:
      return "TessEval";
    case ShaderStage::kCompute:
      return "Compute";
    case ShaderStage::kRayGen:
      return "RayGen";
    case ShaderStage::kAnyHit:
      return "AnyHit";
    case ShaderStage::kClosestHit:
      return "ClosestHit";
    case ShaderStage::kMiss:
      return "Miss";
    case ShaderStage::kIntersection:
      return "Intersection";
    case ShaderStage::kCallable:
      return "Callable";
    case ShaderStage::kMesh:
      return "Mesh";
    case ShaderStage::kTask:
      return "Task";
  }
  return "";
}

}  // end namespace gfx

std::ostream& operator<<(std::ostream& os, const gfx::ShaderStage value) {
  os << gfx::to_string(value);
  return os;
}
