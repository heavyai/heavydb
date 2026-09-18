/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Data/Types.h"

#include "GfxDriver/RenderError.h"
#include "QueryRenderer/Utils/StringUtils.h"

namespace QueryRenderer {

using ::gfx::BufferAttrType;
using ::gfx::ColorHCL;
using ::gfx::ColorHSL;
using ::gfx::ColorLAB;
using ::gfx::ColorRGBA;
using ::gfx::ColorUnion;

std::string to_string(QueryDataType data_type) {
  switch (data_type) {
    case QueryDataType::UINT:
      return "UINT";
    case QueryDataType::INT:
      return "INT";
    case QueryDataType::FLOAT:
      return "FLOAT";
    case QueryDataType::DOUBLE:
      return "DOUBLE";
    case QueryDataType::UINT64:
      return "UINT64";
    case QueryDataType::INT64:
      return "INT64";
    case QueryDataType::COLOR:
      return "COLOR";
    case QueryDataType::STRING:
      return "STRING";
    case QueryDataType::BOOL:
      return "BOOL";
    case QueryDataType::LINE_JOIN_ENUM:
      return "LINE_JOIN_ENUM";
    case QueryDataType::SYMBOL_SHAPE_ENUM:
      return "SYMBOL_SHAPE_ENUM";
    case QueryDataType::ANGLE_UNIT_ENUM:
      return "ANGLE_UNIT_ENUM";
    case QueryDataType::POLYGON_DOUBLE:
      return "POLYGON_DOUBLE";
    case QueryDataType::LINE_DOUBLE:
      return "LINE_DOUBLE";
  }
  return "";
}

std::string to_string(DataInputFormat data_input_format) {
  switch (data_input_format) {
    case DataInputFormat::kSQL:
      return "SqlQuery";
    case DataInputFormat::kEmbedded:
      return "Embedded";
    case DataInputFormat::kURL:
      return "Url";
    case DataInputFormat::kSourced:
      return "Sourced";
    case DataInputFormat::kTransformed:
      return "Transformed";
    case DataInputFormat::kUnsupported:
      return "Unsupported";
  }
  UNREACHABLE();
  return "";
}

std::string to_string(DataOutputFormat data_output_format) {
  switch (data_output_format) {
    case DataOutputFormat::kRows:
      return "BasicVbo";
    case DataOutputFormat::kPolys:
      return "Poly";
    case DataOutputFormat::kLines:
      return "Line";
    case DataOutputFormat::kMesh2d:
      return "Mesh2d";
    case DataOutputFormat::kUnsupported:
      return "Unsupported";
  }
  UNREACHABLE();
  return "";
}

bool isExternallySourcedInputFormat(const DataInputFormat data_input_format) {
  switch (data_input_format) {
    case DataInputFormat::kSQL:
    case DataInputFormat::kEmbedded:
    case DataInputFormat::kURL:
      return true;
    case DataInputFormat::kSourced:
    case DataInputFormat::kTransformed:
    case DataInputFormat::kUnsupported:
      return false;
  }
  UNREACHABLE();
  return false;
}

std::string getDataInputFormatsAsStr() {
  return enum_to_string<DataInputFormat>(
      static_cast<DataInputFormat>(0),
      DataInputFormat::kUnsupported,
      static_cast<std::string (*)(DataInputFormat)>(&to_string),
      [](const DataInputFormat input_format) {
        return isExternallySourcedInputFormat(input_format);
      });
}

template <>
QueryDataType TypeToQueryDataTypeSelector<uint32_t>::getQueryDataType() {
  return QueryDataType::UINT;
}

template <>
QueryDataType TypeToQueryDataTypeSelector<int32_t>::getQueryDataType() {
  return QueryDataType::INT;
}

template <>
QueryDataType TypeToQueryDataTypeSelector<float>::getQueryDataType() {
  return QueryDataType::FLOAT;
}

template <>
QueryDataType TypeToQueryDataTypeSelector<uint64_t>::getQueryDataType() {
  return QueryDataType::UINT64;
}

template <>
QueryDataType TypeToQueryDataTypeSelector<int64_t>::getQueryDataType() {
  return QueryDataType::INT64;
}

template <>
QueryDataType TypeToQueryDataTypeSelector<double>::getQueryDataType() {
  return QueryDataType::DOUBLE;
}

QueryDataType TypeToQueryDataTypeSelector<std::string>::getQueryDataType() {
  return QueryDataType::STRING;
}

bool isArithmeticQueryDataType(const QueryDataType type) {
  return type == QueryDataType::UINT || type == QueryDataType::INT ||
         type == QueryDataType::FLOAT || type == QueryDataType::DOUBLE ||
         type == QueryDataType::UINT64 || type == QueryDataType::INT64;
}

QueryDataType convertToQueryDataType(const gfx::BufferAttrType attr_type) {
  switch (attr_type) {
    case BufferAttrType::kUint:
      return QueryDataType::UINT;
    case BufferAttrType::kInt:
      return QueryDataType::INT;
    case BufferAttrType::kFloat:
      return QueryDataType::FLOAT;
    case BufferAttrType::kDouble:
      return QueryDataType::DOUBLE;
    case BufferAttrType::kVec4f:
      return QueryDataType::COLOR;
    case BufferAttrType::kUint64:
      return QueryDataType::UINT64;
    case BufferAttrType::kInt64:
      return QueryDataType::INT64;
    default:
      THROW_RUNTIME_EX(
          "Buffer attribute type: " + std::to_string(static_cast<int>(attr_type)) +
          " cannot be converted into a query data type.");
  }

  return QueryDataType::INT;
}

QueryDataType convertToQueryDataType(const SQLTypeInfo& type_info) {
  if (type_info.is_string() && type_info.get_compression() == kENCODING_DICT) {
    return QueryDataType::STRING;
  }

  switch (type_info.get_type()) {
    case kBOOLEAN:
    case kTINYINT:
    case kSMALLINT:
    case kINT:
      return QueryDataType::INT;
    case kNUMERIC:
    case kDECIMAL:
    case kBIGINT:
      return QueryDataType::INT64;
    case kFLOAT:
      return QueryDataType::FLOAT;
    case kDOUBLE:
      return QueryDataType::DOUBLE;
    default:
      THROW_RUNTIME_EX("SQL type: " + type_info.get_type_name() +
                       " with compression: " + type_info.get_compression_name() +
                       " cannot be converted into a query data type.");
  }

  CHECK(false) << "SQL type: " << type_info.get_type_name()
               << ", compression: " << type_info.get_compression_name();
  return QueryDataType::INT;
}

std::ostream& operator<<(std::ostream& os, QueryDataType value) {
  os << to_string(value);
  return os;
}

std::ostream& operator<<(std::ostream& os, DataInputFormat input_format) {
  os << to_string(input_format);
  return os;
}

std::ostream& operator<<(std::ostream& os, DataOutputFormat output_format) {
  os << to_string(output_format);
  return os;
}

}  // namespace QueryRenderer
