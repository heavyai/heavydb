/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "QueryRenderer/Marks/Enums.h"
#include "QueryRenderer/Marks/Types.h"
#include "QueryRenderer/Types.h"
#include "QueryRenderer/Utils/RapidJSONUtils.h"

namespace QueryRenderer {

namespace JSONSchema_v1 {
namespace Marks {
constexpr char kTypeProp[] = "type";
constexpr char kFromProp[] = "from";
constexpr char kDataProp[] = "data";
constexpr char kTransformProp[] = "transform";
constexpr char kProjectionProp[] = "projection";
constexpr char kPropertiesProp[] = "properties";
constexpr char kFieldProp[] = "field";
constexpr char kValueProp[] = "value";
constexpr char kScaleProp[] = "scale";
constexpr char kColorSpaceProp[] = "colorSpace";
}  // namespace Marks
}  // namespace JSONSchema_v1

GeomType getMarkTypeFromJSONObj(const JSONLocation& json_loc);
BaseMarkUqPtr createMark(const JSONLocation& json_loc, QueryRendererContext& ctx);

}  // namespace QueryRenderer
