/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Data/Parsers/SqlQueryLineFormatJson.h"

#include "QueryRenderer/Data/Utils.h"

namespace QueryRenderer {

namespace {

bool is_short_form(const JSONLocation& format_loc) {
  if (format_loc.isString()) {
    // regular lines
    return true;
  } else if (format_loc.isObject()) {
    auto type_loc = format_loc.getMember(JSONSchema_v1::Data::kTypeProp);
    if (type_loc.isValid() && type_loc.isString() &&
        type_loc.getString() == "cross_section1d") {
      // cross section 1D
      return true;
    }
  }
  return false;
}

}  // namespace

void SqlQueryLineFormatJson::validate(const JSONLocation& parent_loc) {
  const auto coords_loc = parent_loc.getMember(JSONSchema_v1::Data::kCoordsProp);
  RUNTIME_EX_ASSERT(coords_loc.isValid() && coords_loc.isObject(),
                    RapidJSONUtils::createJsonParseError(
                        (coords_loc.isValid() ? coords_loc : parent_loc),
                        "The '" + std::string(JSONSchema_v1::Data::kCoordsProp) +
                            "' format property must exist and be an object."));

  const auto x_loc = coords_loc.getMember(JSONSchema_v1::Data::kXCoordProp);
  RUNTIME_EX_ASSERT(x_loc.isValid() && x_loc.isArray(),
                    RapidJSONUtils::createJsonParseError(
                        x_loc.isValid() ? x_loc : coords_loc,
                        "The '" + std::string(JSONSchema_v1::Data::kXCoordProp) +
                            "' property must exist and be an array."));

  const auto y_loc = coords_loc.getMember(JSONSchema_v1::Data::kYCoordProp);
  RUNTIME_EX_ASSERT(y_loc.isValid() && y_loc.isArray(),
                    RapidJSONUtils::createJsonParseError(
                        y_loc.isValid() ? y_loc : coords_loc,
                        "The '" + std::string(JSONSchema_v1::Data::kYCoordProp) +
                            "' property must exist and be an array."));

  RUNTIME_EX_ASSERT(
      x_loc.size() == y_loc.size(),
      RapidJSONUtils::createJsonParseError(
          coords_loc,
          "The '" + std::string(JSONSchema_v1::Data::kXCoordProp) + "' and '" +
              std::string(JSONSchema_v1::Data::kYCoordProp) +
              "' array properties must be the same size. The former is of size " +
              std::to_string(x_loc.size()) + ", the latter is of size " +
              std::to_string(y_loc.size())));

  for (size_t i = 0; i < x_loc.size(); ++i) {
    const auto x_array_loc = x_loc[i];
    RUNTIME_EX_ASSERT(
        x_array_loc.isString() || x_array_loc.isObject(),
        RapidJSONUtils::createJsonParseError(
            x_array_loc,
            "Coordinates for the '" + std::string(JSONSchema_v1::Data::kXCoordProp) +
                "' array property must be either strings (column "
                "references) or objects (array column references)"));

    const auto y_array_loc = y_loc[i];
    RUNTIME_EX_ASSERT(
        y_array_loc.isString() || y_array_loc.isObject(),
        RapidJSONUtils::createJsonParseError(
            y_array_loc,
            "Coordinates for the '" + std::string(JSONSchema_v1::Data::kYCoordProp) +
                "' array property must be either strings (column "
                "references) or objects (array column references)"));
  }
}

QueryDataLayout::LayoutType SqlQueryLineFormatJson::SetVertexLayout(
    const JSONLocation& data_loc) {
  const auto format_loc = data_loc.getMember(JSONSchema_v1::Data::kFormatProp);
  if (!data_loc.isValid() || format_loc.isString()) {
    // no "format" or it's a string means in-situ
    return QueryRenderer::QueryDataLayout::LayoutType::kVertexInterleaved;
  }

  RUNTIME_EX_ASSERT(format_loc.isObject(),
                    RapidJSONUtils::createJsonParseError(
                        format_loc,
                        "\"" + std::string(JSONSchema_v1::Data::kFormatProp) +
                            "\" must be an object or string"));

  const auto layout_loc = format_loc.getMember(JSONSchema_v1::Data::kLayoutProp);
  if (layout_loc.isValid()) {
    if (layout_loc.getString() == "sequential") {
      return QueryRenderer::QueryDataLayout::LayoutType::kVertexSequential;
    } else if (layout_loc.getString() == "interleaved") {
      return QueryRenderer::QueryDataLayout::LayoutType::kVertexInterleaved;
    }
    THROW_RUNTIME_EX(RapidJSONUtils::createJsonParseError(
        layout_loc,
        "\"" + std::string(JSONSchema_v1::Data::kLayoutProp) +
            "\" must be \"interleaved\" or \"sequential\""))
  }
  return QueryRenderer::QueryDataLayout::LayoutType::kVertexInterleaved;
}

SqlQueryLineFormatJson::LineVertexQueryTargetInfo
SqlQueryLineFormatJson::SetVertexQueryTargets(
    const JSONLocation& data_loc,
    const QueryColumnInfoMap& query_column_info_map,
    const QueryDataLayout::LayoutType vertex_layout) {
  std::unordered_map<unsigned int, LineVertexColumnInfo> vertex_target_map;

  // Separate query targets by rendering buffer
  std::vector<unsigned int> primary_vertex_query_indices;
  std::vector<unsigned int> secondary_vertex_query_indices;

  auto GetIndexForTargetColumn = [&](const JSONLocation& col_loc) {
    const std::string col_name = col_loc.getString();
    auto query_col_info_map_itr = query_column_info_map.find(col_name);
    if (query_col_info_map_itr == query_column_info_map.end()) {
      THROW_RUNTIME_EX(RapidJSONUtils::createJsonParseError(
          col_loc,
          "Cannot find vertex column with name \"" + col_name +
              "\" in the query. All vertex columns must be "
              "included in the SQL query for "
              "line rendering."));
    }
    return query_col_info_map_itr->second.first;
  };

  auto get_vertex_info_for_column = [&](const JSONLocation& item_loc) {
    if (item_loc.isString()) {
      return LineVertexColumnInfo({item_loc.getString(),
                                   static_cast<int>(GetIndexForTargetColumn(item_loc)),
                                   /*is_reference=*/false});
    } else if (item_loc.isObject()) {
      auto from_loc = item_loc.getMember(JSONSchema_v1::Data::kFromProp);
      RUNTIME_EX_ASSERT(from_loc.isValid() && from_loc.isString(),
                        RapidJSONUtils::createJsonParseError(
                            from_loc.isValid() ? from_loc : item_loc,
                            "Vertex coordinate reference object must contain a \"from\" "
                            "member and it must be a string."));
      return LineVertexColumnInfo({from_loc.getString(),
                                   static_cast<int>(GetIndexForTargetColumn(from_loc)),
                                   /*is_reference=*/true});
    } else {
      // Note that this check should never be tripped, since both coordinate arrays are
      // checked and validated in SqlQueryLineDataTable
      CHECK(false) << "Error: Unsupported array member type.";
    }
    return LineVertexColumnInfo();
  };

  // NOTE: all the json props have been previously validated so no need to check that they
  // exist or are the appropriate type

  // short form?
  const auto format_loc = data_loc.getMember(JSONSchema_v1::Data::kFormatProp);
  if (is_short_form(format_loc)) {
    // short-form only supports LINESTRING
    for (const auto& entry : query_column_info_map) {
      if (entry.second.second == kLINESTRING || entry.second.second == kMULTILINESTRING) {
        LineVertexColumnInfo lvci{entry.first, int(entry.second.first), false};
        vertex_target_map.insert(std::make_pair(lvci.target_column_index, lvci));
        primary_vertex_query_indices.push_back(lvci.target_column_index);
        return {vertex_target_map,
                primary_vertex_query_indices,
                secondary_vertex_query_indices};
      }
    }
    THROW_RUNTIME_EX(RapidJSONUtils::createJsonParseError(
        format_loc, "Short-form Line Render only supports LINESTRING"));
  }

  const auto coords_loc = format_loc.getMember(JSONSchema_v1::Data::kCoordsProp);
  const auto x_loc = coords_loc.getMember(JSONSchema_v1::Data::kXCoordProp);
  const auto y_loc = coords_loc.getMember(JSONSchema_v1::Data::kYCoordProp);

  for (size_t i = 0; i < x_loc.size(); ++i) {
    auto x_item_loc = x_loc[i];
    auto y_item_loc = y_loc[i];
    auto x_col_obj = get_vertex_info_for_column(x_item_loc);
    auto y_col_obj = get_vertex_info_for_column(y_item_loc);
    if (!x_col_obj.is_reference && !y_col_obj.is_reference) {
      vertex_target_map.insert(std::make_pair(x_col_obj.target_column_index, x_col_obj));
      primary_vertex_query_indices.push_back(x_col_obj.target_column_index);
      vertex_target_map.insert(std::make_pair(y_col_obj.target_column_index, y_col_obj));
      if (vertex_layout == QueryDataLayout::LayoutType::kVertexSequential) {
        secondary_vertex_query_indices.push_back(y_col_obj.target_column_index);
      } else {
        primary_vertex_query_indices.push_back(y_col_obj.target_column_index);
      }

    } else if (x_col_obj.is_reference && y_col_obj.is_reference) {
      THROW_RUNTIME_EX(RapidJSONUtils::createJsonParseError(
          coords_loc, "The x and y vertex column entries cannot both be references."));
    } else {
      // TODO(adb): we could probably remove this limitation. Is there a use case?
      if (x_col_obj.name != y_col_obj.name) {
        THROW_RUNTIME_EX(RapidJSONUtils::createJsonParseError(
            y_item_loc,
            "A referencing column must be in the same position in the coordinates array "
            "as the column it references. "
            "The x column \"" +
                x_col_obj.name + "\" does not match the y column \"" + y_col_obj.name +
                "\"."));
      }
      // pick the reference column, since it stores the layout. assume all layouts are
      // (x,y) [for now]
      if (x_col_obj.is_reference) {
        vertex_target_map.insert(
            std::make_pair(x_col_obj.target_column_index, x_col_obj));
        primary_vertex_query_indices.push_back(x_col_obj.target_column_index);
      } else {
        // y must be the reference
        vertex_target_map.insert(
            std::make_pair(y_col_obj.target_column_index, y_col_obj));
        if (vertex_layout ==
            QueryRenderer::QueryDataLayout::LayoutType::kVertexSequential) {
          secondary_vertex_query_indices.push_back(y_col_obj.target_column_index);
        } else {
          primary_vertex_query_indices.push_back(y_col_obj.target_column_index);
        }
      }
    }
  }
  return {
      vertex_target_map, primary_vertex_query_indices, secondary_vertex_query_indices};
}

bool SqlQueryLineFormatJson::ShouldAttemptInSituRender(const JSONLocation& data_loc) {
  const auto format_loc = data_loc.getMember(JSONSchema_v1::Data::kFormatProp);
  if (format_loc.isValid() && format_loc.isString()) {
    // no need to check string value or we wouldn't be here
    // short-form Vega for LINESTRING only
    return true;
  }
  // @TODO support in-situ render for long-form Vega coords mapping
  return false;
}

}  // namespace QueryRenderer
