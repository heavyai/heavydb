/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "QueryRenderer/Data/Types.h"
#include "QueryRenderer/Types.h"
#include "QueryRenderer/Utils/RapidJSONUtils.h"

namespace QueryRenderer {

namespace JSONSchema_v1 {
namespace Data {
constexpr char kNameProp[] = "name";
constexpr char kFormatProp[] = "format";
constexpr char kTypeProp[] = "type";
constexpr char kSqlProp[] = "sql";
constexpr char kEnableHitTestingProp[] = "enableHitTesting";
constexpr char kValuesProp[] = "values";
constexpr char kUrlProp[] = "url";
constexpr char kSourceProp[] = "source";
constexpr char kDbNameProp[] = "dbTableName";

// "line" format type
constexpr char kCoordsProp[] = "coords";
constexpr char kXCoordProp[] = "x";
constexpr char kYCoordProp[] = "y";
constexpr char kLayoutProp[] = "layout";
constexpr char kFromProp[] = "from";

// cross_section 1d/2d format type
constexpr char kXYCrossSectionProp[] = "xyCrossSection";
constexpr char kCrossSectionNumPointsProp[] = "numPoints";
constexpr char kCrossSectionNumPointsXProp[] = "numPointsX";
constexpr char kCrossSectionNumPointsYProp[] = "numPointsY";
constexpr char kCrossSectionDWithinDistanceProp[] = "dWithinDistance";

// "polys" format type
constexpr char kPolysKeyProp[] = "polysKey";
constexpr char kFactsKeyProp[] = "factsKey";
constexpr char kFactsTableNameProp[] = "factsTableName";
constexpr char kAggExprProp[] = "aggExpr";
constexpr char kFilterExprProp[] = "filterExpr";
constexpr char kEnableInSituPolysProp[] = "enableInSituPolys";
constexpr char kGeoColumnProp[] = "geocolumn";

// "source" format type
constexpr char kTransformProp[] = "transform";
}  // namespace Data
}  // namespace JSONSchema_v1

std::string getDataTableNameFromJSONObj(const JSONLocation& json_loc);
std::string getSourcedDataTableNameFromJSONObj(const JSONLocation& json_loc);
std::pair<DataInputFormat, DataOutputFormat> getDataIOFormatsFromJSONObj(
    const JSONLocation& json_loc);
BaseDataTableShPtr createDataTable(const JSONLocation& json_loc,
                                   QueryRendererContext& ctx,
                                   const std::string& name = "");

QueryDataLayoutShPtr getDataLayoutForAttribute(const BaseDataTableShPtr& in_data,
                                               const std::string& attr_name);

}  // namespace QueryRenderer
