/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Data/Parsers/CrossSectionFormatJson.h"

#include "QueryRenderer/Data/Utils.h"

namespace QueryRenderer {

void CrossSectionFormatJson::validate(const JSONLocation& parent_loc) {
  RasterMeshFormatJson::validate(parent_loc);

  // the coords location should have been vaildated appropriately in the above
  // RasterMeshFormatJson::validate() call
  auto const coords_loc = parent_loc.getMember(JSONSchema_v1::Data::kCoordsProp);
  CHECK(coords_loc.isValid());

  const auto xy_section_loc =
      parent_loc.getMember(JSONSchema_v1::Data::kXYCrossSectionProp);

  RUNTIME_EX_ASSERT(xy_section_loc.isValid() && xy_section_loc.isArray(),
                    RapidJSONUtils::createJsonParseError(
                        xy_section_loc.isValid() ? xy_section_loc : parent_loc,
                        "The '" + std::string(JSONSchema_v1::Data::kXYCrossSectionProp) +
                            "' " + " property must exist and be an array"));

  RUNTIME_EX_ASSERT(xy_section_loc.size() == 2,
                    RapidJSONUtils::createJsonParseError(
                        xy_section_loc,
                        "The '" + std::string(JSONSchema_v1::Data::kXYCrossSectionProp) +
                            "' property must be an array of size 2. It is of size " +
                            std::to_string(xy_section_loc.size())));

  for (auto i = 0u; i < xy_section_loc.size(); ++i) {
    auto const xy_section_item_loc =
        xy_section_loc.getArrayMember(i, JSONValueType::kArray);
    RUNTIME_EX_ASSERT(xy_section_item_loc.size() == 2,
                      RapidJSONUtils::createJsonParseError(
                          xy_section_item_loc,
                          "The elements of the '" +
                              std::string(JSONSchema_v1::Data::kXYCrossSectionProp) +
                              "' array must be an array of 2 floating-pt values."));

    for (auto j = 0u; j < xy_section_item_loc.size(); ++j) {
      // NOTE: the getArrayMember() method below handles the validation for us with the
      // kNumber argument, so we do not need to validate that it's numeric
      auto const cross_section_coord_value_loc =
          xy_section_item_loc.getArrayMember(j, JSONValueType::kNumber);
      CHECK(cross_section_coord_value_loc.isNumber());
    }
  }
}

void CrossSectionFormatJson::parse(CrossSectionFormatJson& cross_section_format_json,
                                   const JSONLocation& parent_loc) {
  RasterMeshFormatJson::parse(cross_section_format_json, parent_loc);
  auto const coords_loc = parent_loc.getMember(JSONSchema_v1::Data::kCoordsProp);

  // build out the cross section line from the json
  const auto xy_section_loc =
      parent_loc.getMember(JSONSchema_v1::Data::kXYCrossSectionProp);
  CHECK(xy_section_loc.isValid());
  CHECK(xy_section_loc.isArray());
  CHECK_EQ(xy_section_loc.size(), 2u);
  for (auto i = 0u; i < xy_section_loc.size(); ++i) {
    auto const xy_section_item_loc =
        xy_section_loc.getArrayMember(i, JSONValueType::kArray);
    CHECK(xy_section_item_loc.isArray());
    CHECK_EQ(xy_section_item_loc.size(), 2u);
    for (auto j = 0u; j < xy_section_item_loc.size(); ++j) {
      auto const cross_section_coord_value_loc =
          xy_section_item_loc.getArrayMember(j, JSONValueType::kNumber);
      cross_section_format_json.linestring[i][j] =
          cross_section_coord_value_loc.getDouble();
    }
  }
}

}  // namespace QueryRenderer
