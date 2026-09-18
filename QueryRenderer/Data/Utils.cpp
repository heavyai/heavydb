/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Data/Utils.h"

#include "GfxDriver/RenderError.h"
#include "GfxDriver/RenderLogger.h"
#include "QueryRenderer/Data/EmbeddedLineDataTable.h"
#include "QueryRenderer/Data/EmbeddedPolyDataTable.h"
#include "QueryRenderer/Data/EmbeddedRowDataTable.h"
#include "QueryRenderer/Data/QueryLineDataTable.h"
#include "QueryRenderer/Data/QueryMeshDataTable.h"
#include "QueryRenderer/Data/QueryPolyDataTable.h"
#include "QueryRenderer/Data/QueryRowDataTable.h"
#include "QueryRenderer/Data/QuerySourceDataTable.h"
#include "QueryRenderer/QueryRendererContext.h"
#include "QueryRenderer/Utils/RapidJSONUtils.h"
#include "Shared/StringTransform.h"

namespace QueryRenderer {

std::string getDataTableNameFromJSONObj(const JSONLocation& json_loc) {
  RUNTIME_EX_ASSERT(json_loc.isObject(),
                    RapidJSONUtils::createJsonParseError(
                        json_loc, "A data object in the JSON must be an object."));

  auto const name_loc = json_loc.getMember(JSONSchema_v1::Data::kNameProp);
  RUNTIME_EX_ASSERT(name_loc.isValid() && name_loc.isString(),
                    RapidJSONUtils::createJsonParseError(
                        name_loc.isValid() ? name_loc : json_loc,
                        "A data object must contain a \"" +
                            std::string(JSONSchema_v1::Data::kNameProp) +
                            "\" property and it must be a string"));

  return name_loc.getString();
}

std::string getSourcedDataTableNameFromJSONObj(const JSONLocation& json_loc) {
  RUNTIME_EX_ASSERT(json_loc.isObject(),
                    RapidJSONUtils::createJsonParseError(
                        json_loc, "A data object in the JSON must be an object."));

  auto const src_loc = json_loc.getMember(JSONSchema_v1::Data::kSourceProp);
  RUNTIME_EX_ASSERT(src_loc.isValid() && src_loc.isString(),
                    RapidJSONUtils::createJsonParseError(
                        (src_loc.isValid() ? src_loc : json_loc),
                        "A sourced data table must contain a \"" +
                            std::string(JSONSchema_v1::Data::kSourceProp) +
                            "\" property and it must be a string"));

  return src_loc.getString();
}

std::pair<DataInputFormat, DataOutputFormat> getDataIOFormatsFromJSONObj(
    const JSONLocation& json_loc) {
  DataInputFormat input_format = DataInputFormat::kUnsupported;
  DataOutputFormat output_format = DataOutputFormat::kRows;

  RUNTIME_EX_ASSERT(json_loc.isObject(),
                    RapidJSONUtils::createJsonParseError(
                        json_loc, "A data table in the JSON must be an object."));

  const auto format_loc = json_loc.getMember(JSONSchema_v1::Data::kFormatProp);
  if (format_loc.isValid()) {
    if (format_loc.isString()) {
      std::string format = format_loc.getString();
      if (format == "polys") {
        output_format = DataOutputFormat::kPolys;
      } else if (format == "lines") {
        output_format = DataOutputFormat::kLines;
      } else {
        THROW_RUNTIME_EX(RapidJSONUtils::createJsonParseError(
            format_loc, "Unsupported table format type \"" + format + "\""));
      }
    } else if (format_loc.isObject()) {
      auto format_type_loc = format_loc.getMember(JSONSchema_v1::Data::kTypeProp);
      if (format_type_loc.isValid()) {
        RUNTIME_EX_ASSERT(format_type_loc.isString(),
                          RapidJSONUtils::createJsonParseError(
                              format_type_loc,
                              "The '" + std::string(JSONSchema_v1::Data::kTypeProp) +
                                  "' property in the '" +
                                  std::string(JSONSchema_v1::Data::kFormatProp) +
                                  "' object must be a string."));

        auto const type = format_type_loc.getString();
        switch (get_data_format_from_string(to_lower(type))) {
          case DataFormatType::kLines:
            output_format = DataOutputFormat::kLines;
            break;
          case DataFormatType::kRasterMesh2d:
          case DataFormatType::kCrossSection2d:
            output_format = DataOutputFormat::kMesh2d;
            break;
          case DataFormatType::kCrossSection1d:
            output_format = DataOutputFormat::kLines;
            break;
          case DataFormatType::kUnknown:
            THROW_RUNTIME_EX(RapidJSONUtils::createJsonParseError(
                format_type_loc, "Unsupported table format type \"" + type + "\""));
            break;
        }
      }
    } else {
      RUNTIME_EX_ASSERT(format_loc.isString(),
                        RapidJSONUtils::createJsonParseError(
                            format_loc,
                            "The format of a data table is declared as a wrong type. "
                            "It must be a string (polys) or an object (lines/mesh2d)."));
    }
  }

  auto data_type_loc = json_loc.getMember(JSONSchema_v1::Data::kSqlProp);
  if (data_type_loc.isValid()) {
    RUNTIME_EX_ASSERT(data_type_loc.isString(),
                      RapidJSONUtils::createJsonParseError(
                          data_type_loc,
                          "Cannot get data table's type - the sql property for a data "
                          "table must be a string."));
    input_format = DataInputFormat::kSQL;
  } else if ((data_type_loc = json_loc.getMember(JSONSchema_v1::Data::kValuesProp))
                 .isValid()) {
    input_format = DataInputFormat::kEmbedded;
  } else if ((data_type_loc = json_loc.getMember(JSONSchema_v1::Data::kUrlProp))
                 .isValid()) {
    input_format = DataInputFormat::kURL;
  } else if ((data_type_loc = json_loc.getMember(JSONSchema_v1::Data::kSourceProp))
                 .isValid()) {
    input_format = DataInputFormat::kSourced;
  }

  RUNTIME_EX_ASSERT(
      input_format != DataInputFormat::kUnsupported,
      RapidJSONUtils::createJsonParseError(
          json_loc,
          "Cannot get data table's type - the data table's type is not supported."));

  return std::make_pair(input_format, output_format);
}

BaseDataTableShPtr createDataTable(const JSONLocation& json_loc,
                                   QueryRendererContext& ctx,
                                   const std::string& name) {
  RENDER_LOG_SCOPE() << "name: " << name;
  std::string table_name{name};
  if (!table_name.length()) {
    table_name = getDataTableNameFromJSONObj(json_loc);
  } else {
    RUNTIME_EX_ASSERT(
        json_loc.isObject(),
        RapidJSONUtils::createJsonParseError(
            json_loc,
            "Cannot create data table - A data object in the JSON must be an object."));
  }

  RUNTIME_EX_ASSERT(
      table_name.length(),
      RapidJSONUtils::createJsonParseError(json_loc,
                                           "Cannot create data table - The data table "
                                           "has an empty name. It must have a name."));

  auto const [input_format, output_format] = getDataIOFormatsFromJSONObj(json_loc);
  RENDER_LOG() << "input_format: " << input_format;
  BaseDataTableShPtr rtn;
  switch (input_format) {
    case DataInputFormat::kSQL:
      switch (output_format) {
        case DataOutputFormat::kRows:
          rtn = std::make_shared<SqlQueryRowDataTableJSON>(ctx, table_name, json_loc);
          break;
        case DataOutputFormat::kPolys:
          rtn = std::make_shared<SqlQueryPolyDataTableJSON>(ctx, table_name, json_loc);
          break;
        case DataOutputFormat::kLines:
          rtn = std::make_shared<SqlQueryLineDataTableJSON>(ctx, table_name, json_loc);
          break;
        case DataOutputFormat::kMesh2d:
          rtn = std::make_shared<SqlQueryMeshDataTableJSON>(ctx, table_name, json_loc);
          break;
        default:
          THROW_RUNTIME_EX(RapidJSONUtils::createJsonParseError(
              json_loc,
              "Cannot create data table \"" + table_name + "\". " +
                  to_string(input_format) + " is not a supported table."));
      }
      break;
    case DataInputFormat::kEmbedded:
    case DataInputFormat::kURL:
      switch (output_format) {
        case DataOutputFormat::kRows:
          rtn = std::make_shared<EmbeddedRowDataTable>(ctx,
                                                       table_name,
                                                       json_loc,
                                                       input_format,
                                                       ctx.doHitTest(),
                                                       EmbeddedDataVboType::kInterleaved);
          break;

        case DataOutputFormat::kPolys:
          rtn =
              std::make_shared<EmbeddedPolyDataTable>(ctx,
                                                      table_name,
                                                      json_loc,
                                                      input_format,
                                                      ctx.doHitTest(),
                                                      EmbeddedDataVboType::kInterleaved);
          break;

        case DataOutputFormat::kLines:
          rtn =
              std::make_shared<EmbeddedLineDataTable>(ctx,
                                                      table_name,
                                                      json_loc,
                                                      input_format,
                                                      ctx.doHitTest(),
                                                      EmbeddedDataVboType::kInterleaved);
          break;

        default:
          THROW_RUNTIME_EX(RapidJSONUtils::createJsonParseError(
              json_loc,
              "Cannot create data table \"" + table_name + "\". " +
                  to_string(input_format) + " is not a supported table."));
      }
      break;
    case DataInputFormat::kSourced:
      switch (output_format) {
        case DataOutputFormat::kRows:
          rtn = std::make_shared<QuerySourceDataTable>(ctx, table_name, json_loc);
          break;
        default:
          THROW_RUNTIME_EX(RapidJSONUtils::createJsonParseError(
              json_loc,
              "Cannot create data table \"" + table_name + "\". " +
                  to_string(input_format) + " is not a supported table."));
      }
      break;
    default:
      THROW_RUNTIME_EX(RapidJSONUtils::createJsonParseError(
          json_loc,
          "Cannot create data table \"" + table_name +
              "\". It is not a supported table. Supported tables "
              "must have an \"sql\", \"values\" or \"url\" "
              "property."));
  }
  CHECK(rtn);

  return rtn;
}

QueryDataLayoutShPtr getDataLayoutForAttribute(const BaseDataTableShPtr& in_data,
                                               const std::string& attr_name) {
  QueryDataLayoutShPtr rtn_layout;
  auto data = std::dynamic_pointer_cast<BaseQueryDataTableSQLJSON>(in_data);
  if (data) {
    rtn_layout = data->getVboQueryDataLayout();
    if (!rtn_layout || !rtn_layout->hasAttribute(attr_name)) {
      rtn_layout = data->getSsboQueryDataLayout();
      if (rtn_layout && !rtn_layout->hasAttribute(attr_name)) {
        rtn_layout = nullptr;
      }
    }
  }
  return rtn_layout;
}

}  // namespace QueryRenderer
