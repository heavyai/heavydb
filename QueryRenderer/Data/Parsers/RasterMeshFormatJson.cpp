/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Data/Parsers/RasterMeshFormatJson.h"

#include "QueryRenderer/Data/Utils.h"

namespace QueryRenderer {

void RasterMeshFormatJson::validateCoordinateProp(const JSONLocation& parent_loc,
                                                  const std::string_view prop_name) {
  const auto coordinate_loc = parent_loc.getMember(std::string(prop_name));
  RUNTIME_EX_ASSERT(coordinate_loc.isValid() && coordinate_loc.isString(),
                    RapidJSONUtils::createJsonParseError(
                        coordinate_loc.isValid() ? coordinate_loc : parent_loc,
                        "The '" + std::string(prop_name) + "' " +
                            std::string(JSONSchema_v1::Data::kCoordsProp) +
                            " property must exist and be a string"));
}

void RasterMeshFormatJson::validate(const JSONLocation& parent_loc) {
  const auto coords_loc = parent_loc.getMember(JSONSchema_v1::Data::kCoordsProp);
  RUNTIME_EX_ASSERT(coords_loc.isValid() && coords_loc.isObject(),
                    RapidJSONUtils::createJsonParseError(
                        coords_loc,
                        "The '" + std::string(JSONSchema_v1::Data::kCoordsProp) +
                            "' format property must exist and be an object."));

  validateCoordinateProp(coords_loc, JSONSchema_v1::Data::kXCoordProp);
  validateCoordinateProp(coords_loc, JSONSchema_v1::Data::kYCoordProp);
}

void RasterMeshFormatJson::parse(RasterMeshFormatJson& mesh_format_json,
                                 const JSONLocation& parent_loc) {
  auto const coords_loc = parent_loc.getMember(JSONSchema_v1::Data::kCoordsProp);
  CHECK(coords_loc.isValid());
  CHECK(coords_loc.isObject());

  auto const x_loc = coords_loc.getMember(JSONSchema_v1::Data::kXCoordProp);
  CHECK(x_loc.isValid());
  CHECK(x_loc.isString());
  mesh_format_json.x_coord_name = x_loc.getString();

  auto const y_loc = coords_loc.getMember(JSONSchema_v1::Data::kYCoordProp);
  CHECK(y_loc.isValid());
  CHECK(y_loc.isString());
  mesh_format_json.y_coord_name = y_loc.getString();
}

}  // namespace QueryRenderer
