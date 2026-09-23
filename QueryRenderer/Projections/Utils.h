/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <rapidjson/document.h>
#include <rapidjson/pointer.h>

#include "QueryRenderer/Projections/Types.h"
#include "QueryRenderer/Types.h"
#include "QueryRenderer/Utils/RapidJSONUtils.h"

namespace QueryRenderer {

namespace JSONSchema_v1 {
namespace Projections {
// mercator proj props
constexpr char kBoundsProp[] = "bounds";
constexpr char kXProp[] = "x";
constexpr char kYProp[] = "y";
constexpr char kNameProp[] = "name";
constexpr char kTypeProp[] = "type";
}  // namespace Projections
}  // namespace JSONSchema_v1

std::string getProjectionNameFromJSONObj(const JSONLocation& json_loc);

ProjectionType getProjectionTypeFromJSONObj(const JSONLocation& json_loc);

ProjectionShPtr createProjection(const JSONLocation& json_loc,
                                 QueryRendererContext& ctx,
                                 const std::string& name = "",
                                 ProjectionType type = ProjectionType::kUndefined);
}  // namespace QueryRenderer
