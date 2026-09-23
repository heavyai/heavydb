/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>

#include "QueryRenderer/Scales/Types.h"

namespace QueryRenderer {

class JSONLocation;
class QueryRendererContext;

ScaleShPtr createScale(const JSONLocation& json_loc,
                       QueryRendererContext& ctx,
                       const std::string& name = "",
                       // TODO(croot): expose default as a constant somewhere
                       ScaleType type = ScaleType::kUndefined);

}  // namespace QueryRenderer
