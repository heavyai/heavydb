/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>

namespace QueryRenderer {

enum class ProjectionType { kMercator = 0, kUndefined };

class Projection;
using ProjectionShPtr = std::shared_ptr<Projection>;
using ProjectionWkPtr = std::weak_ptr<Projection>;

class ProjectionShader;
using ProjectionShaderShPtr = std::shared_ptr<ProjectionShader>;
struct ProjectionShaderShPtrPair {
  ProjectionShaderShPtr x;
  ProjectionShaderShPtr y;
};
}  // namespace QueryRenderer
