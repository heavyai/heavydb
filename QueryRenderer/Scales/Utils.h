/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <any>

#include <rapidjson/document.h>
#include <rapidjson/pointer.h>

#include "GfxDriver/Colors/Types.h"
#include "QueryRenderer/Data/Types.h"
#include "QueryRenderer/Scales/Types.h"
#include "QueryRenderer/Types.h"
#include "QueryRenderer/Utils/RapidJSONUtils.h"

namespace QueryRenderer {

namespace JSONSchema_v1 {
namespace Scales {
constexpr char kNameProp[] = "name";
constexpr char kTypeProp[] = "type";
constexpr char kDomainProp[] = "domain";
constexpr char kRangeProp[] = "range";
constexpr char kDataProp[] = "data";
constexpr char kDefaultProp[] = "default";
constexpr char kNullValueProp[] = "nullValue";
constexpr char kClampProp[] = "clamp";
constexpr char kExponentProp[] = "exponent";
constexpr char kAccumulatorProp[] = "accumulator";
constexpr char kDensityMinProp[] = "minDensityCnt";
constexpr char kDensityMaxProp[] = "maxDensityCnt";
constexpr char kPctCatProp[] = "pctCategory";
constexpr char kPctCatMarginProp[] = "pctCategoryMargin";
constexpr char kFieldProp[] = "field";
constexpr char kFieldsProp[] = "fields";
constexpr char kInterpolatorProp[] = "interpolator";
}  // namespace Scales
}  // namespace JSONSchema_v1

std::vector<std::string> getFieldsFromDataRef(const JSONLocation& data_loc,
                                              const QueryRendererContext& ctx,
                                              const BaseDataTableShPtr& table);

std::string getScaleNameFromJSONObj(const JSONLocation& json_loc);
ScaleType getScaleTypeFromJSONObj(const JSONLocation& json_loc);
QueryDataType getScaleDomainDataTypeFromJSONObj(const JSONLocation& json_loc,
                                                const QueryRendererContext& ctx,
                                                const ScaleType scaleType);
QueryDataType getScaleRangeDataTypeFromJSONObj(const JSONLocation& json_loc,
                                               const QueryRendererContext& ctx,
                                               const ScaleType scale_type);
std::pair<gfx::ColorType, ScaleInterpType> getScaleRangeColorTypeFromJSONObj(
    const JSONLocation& json_loc);

ScaleInterpType getScaleInterpTypeFromJSONObj(const JSONLocation& json_loc);
AccumulatorType getScaleAccumulatorTypeFromJSONObj(const JSONLocation& json_loc);

bool isScaleDomainCompatible(const ScaleType scale_type, const QueryDataType domain_type);
bool isScaleRangeCompatible(const ScaleType scale_type, const QueryDataType range_type);
bool areTypesCompatible(const QueryDataType src_type, const QueryDataType in_type);
bool areTypesCompatible(const std::type_info& src_type, const std::type_info& in_type);

// TODO(scb): use ScaleManager
QueryDataType convertTypeIdToDataType(const std::type_info& src_type_id);

template <>
gfx::ColorRGBA convertType(const QueryDataType type,
                           const std::any& value,
                           const bool ignore_null);

template <>
gfx::ColorHSL convertType(const QueryDataType type,
                          const std::any& value,
                          const bool ignore_null);

template <>
gfx::ColorLAB convertType(const QueryDataType type,
                          const std::any& value,
                          const bool ignore_null);

template <>
gfx::ColorHCL convertType(const QueryDataType type,
                          const std::any& value,
                          const bool ignore_null);

}  // namespace QueryRenderer
