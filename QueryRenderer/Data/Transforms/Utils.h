/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <rapidjson/document.h>
#include <rapidjson/pointer.h>

#include "QueryRenderer/Data/Transforms/Types.h"
#include "QueryRenderer/Data/Types.h"
#include "QueryRenderer/Types.h"
#include "QueryRenderer/Utils/RapidJSONUtils.h"

namespace QueryRenderer {

namespace JSONSchema_v1 {
namespace Xform {
constexpr char kFieldProp[] = "fields";
constexpr char kOpsProp[] = "ops";
constexpr char kAsProp[] = "as";
constexpr char kTypeProp[] = "type";
constexpr char kApproximateProp[] = "approximate";
constexpr char kNumBinsProp[] = "numBins";
constexpr char kNumQuantilesProp[] = "numQuantiles";
constexpr char kIncludeExtremaProp[] = "includeExtrema";
constexpr char kExprProp[] = "expr";
}  // namespace Xform
}  // namespace JSONSchema_v1

XformShPtr createTransform(QueryRendererContext& ctx,
                           const BaseDataTableShPtr& source_data_ptr,
                           const JSONLocation& obj_loc);

}  // namespace QueryRenderer
