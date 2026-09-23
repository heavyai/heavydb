/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Projections/Utils.h"
#include "QueryRenderer/Projections/Projection.h"

namespace QueryRenderer {

std::string getProjectionNameFromJSONObj(const JSONLocation& json_loc) {
  RUNTIME_EX_ASSERT(json_loc.isObject(),
                    RapidJSONUtils::createJsonParseError(
                        json_loc, "projection items must be JSON objects."));

  const auto name_loc = json_loc.getMember(JSONSchema_v1::Projections::kNameProp);
  RUNTIME_EX_ASSERT(name_loc.isValid() && name_loc.isString(),
                    RapidJSONUtils::createJsonParseError(
                        name_loc.isValid() ? name_loc : json_loc,
                        "projection objects must contain a \"" +
                            std::string(JSONSchema_v1::Projections::kNameProp) +
                            "\" string property."));

  return name_loc.getString();
}

ProjectionType getProjectionTypeFromJSONObj(const JSONLocation& json_loc) {
  ProjectionType rtn = ProjectionType::kMercator;

  RUNTIME_EX_ASSERT(json_loc.isObject(),
                    RapidJSONUtils::createJsonParseError(
                        json_loc, "projection items must be JSON objects."));

  const auto type_loc = json_loc.getMember(JSONSchema_v1::Projections::kTypeProp);
  if (type_loc.isValid()) {
    RUNTIME_EX_ASSERT(type_loc.isString(),
                      RapidJSONUtils::createJsonParseError(
                          type_loc,
                          "\"" + std::string(JSONSchema_v1::Projections::kTypeProp) +
                              "\" property in projection objects must be a string."));

    std::string strProjType(type_loc.getString());

    if (strProjType == "mercator") {
      rtn = ProjectionType::kMercator;
    } else {
      THROW_RUNTIME_EX(RapidJSONUtils::createJsonParseError(
          type_loc, "projection type \"" + strProjType + "\" is not a supported type."));
    }
  }

  return rtn;
}

ProjectionShPtr createProjection(const JSONLocation& json_loc,
                                 QueryRendererContext& ctx,
                                 const std::string& name,
                                 ProjectionType type) {
  ProjectionShPtr ret{nullptr};
  std::string projection_name{name};
  if (!projection_name.length()) {
    projection_name = getProjectionNameFromJSONObj(json_loc);
  }

  RUNTIME_EX_ASSERT(
      projection_name.length() > 0,
      RapidJSONUtils::createJsonParseError(
          json_loc,
          "Projections must have a \"" +
              std::string(JSONSchema_v1::Projections::kNameProp) + "\" property."));

  ProjectionType projection_type{type};
  if (projection_type == ProjectionType::kUndefined) {
    projection_type = getProjectionTypeFromJSONObj(json_loc);
  }

  switch (projection_type) {
    case ProjectionType::kMercator: {
      ret = std::make_shared<MercatorProjection>(
          json_loc, ctx, projection_name, projection_type);
      break;
    }
    default: {
      THROW_RUNTIME_EX(RapidJSONUtils::createJsonParseError(
          json_loc, "Projection type for \"" + projection_name + "\" is undefined."));
    }
  }

  return ret;
}

}  // namespace QueryRenderer
