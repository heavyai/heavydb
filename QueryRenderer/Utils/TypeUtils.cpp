/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Utils/TypeUtils.h"

#include "GfxDriver/Resources/Enums.h"

using gfx::BufferAttrType;

namespace QueryRenderer {

SQLTypeInfo get_float_equivalent_type(const SQLTypeInfo& type) {
  switch (type.get_size()) {
    case 1:
    case 2:
    case 4:
      return SQLTypeInfo(kFLOAT, type.get_notnull());
    case 8:
      return SQLTypeInfo(kDOUBLE, type.get_notnull());
  }
  CHECK(false) << "Cannot determine floating-pt equivalent type for sql type "
               << type.get_type_name() << " of size: " << type.get_size();
  return SQLTypeInfo();
}

QueryDataType get_float_equivalent_type(const QueryDataType data_type) {
  switch (data_type) {
    case QueryDataType::INT:
    case QueryDataType::UINT:
    case QueryDataType::FLOAT:
      return QueryDataType::FLOAT;
    case QueryDataType::UINT64:
    case QueryDataType::INT64:
    case QueryDataType::DOUBLE:
      return QueryDataType::DOUBLE;
    default:
      throw std::runtime_error("Cannot convert " + to_string(data_type) +
                               " to floating-pt");
  }
}

BufferAttrType query_sql_type_to_render_type(const std::string& attr,
                                             const SQLTypeInfo& ti) {
  if (ti.is_array()) {
    // @TODO simon.eves
    // only those array sub-types necessary for geo tables
    if (ti.get_subtype() == SQLTypes::kINT) {
      // ring sizes array is INT
      return BufferAttrType::kInt64;
    } else if (ti.get_subtype() == SQLTypes::kTINYINT ||
               ti.get_subtype() == SQLTypes::kDOUBLE) {
      // coords arrays are TINYINT
      // bounds arrays are DOUBLE
      return BufferAttrType::kDouble;
    }
  }
  if (ti.is_fp()) {
    return BufferAttrType::kDouble;
  } else if (ti.is_integer() || ti.is_boolean() || ti.is_decimal()) {
    return BufferAttrType::kInt64;
  } else if (ti.is_string()) {
    if (ti.get_compression() != kENCODING_DICT) {
      throw std::runtime_error("The attr \"" + attr + "\" has a sql type of " +
                               ti.get_type_name() +
                               " which is not supported in render queries. String "
                               "columns must be dictionary encoded.");
    }
    return BufferAttrType::kInt64;
  }
  if (!IS_GEO(ti.get_type())) {
    throw std::runtime_error("The attr \"" + attr + "\" has a sql type of " +
                             ti.get_type_name() +
                             " which is not supported in render queries.");
  }
  return BufferAttrType::kInt64;
}

BufferAttrType sql_type_to_render_type(const std::string& attr, const SQLTypeInfo& ti) {
  if (ti.is_string()) {
    CHECK_EQ(kENCODING_DICT, ti.get_compression());
    return BufferAttrType::kInt64;
  }
  CHECK(!ti.is_array());
  switch (ti.get_type()) {
    case kBOOLEAN:
    case kSMALLINT:
    case kINT:
    case kTINYINT:
      return BufferAttrType::kInt;
    case kDECIMAL:
    case kBIGINT:
      return BufferAttrType::kInt64;
    case kFLOAT:
      return BufferAttrType::kFloat;
    case kDOUBLE:
      return BufferAttrType::kDouble;
    default:
      throw std::runtime_error("The attr \"" + attr + "\" has a sql type of " +
                               ti.get_type_name() +
                               " which is not currently supported by render queries.");
  }
}

SQLTypeInfo render_type_to_sql_type(const BufferAttrType buffer_attr_type) {
  switch (buffer_attr_type) {
    case BufferAttrType::kBool:
    case BufferAttrType::kUint:
    case BufferAttrType::kInt:
      return SQLTypeInfo(kINT, true);
    case BufferAttrType::kVec2ui:
    case BufferAttrType::kVec3ui:
    case BufferAttrType::kVec4ui:
    case BufferAttrType::kVec2i:
    case BufferAttrType::kVec3i:
    case BufferAttrType::kVec4i:
      return SQLTypeInfo(kARRAY, 0, 0, true, kENCODING_NONE, 0, kINT);
    case BufferAttrType::kFloat:
      return SQLTypeInfo(kFLOAT, true);
    case BufferAttrType::kVec2f:
    case BufferAttrType::kVec3f:
    case BufferAttrType::kVec4f:
      return SQLTypeInfo(kARRAY, 0, 0, true, kENCODING_NONE, 0, kFLOAT);
    case BufferAttrType::kDouble:
      return SQLTypeInfo(kDOUBLE, true);
    case BufferAttrType::kVec2d:
    case BufferAttrType::kVec3d:
    case BufferAttrType::kVec4d:
      return SQLTypeInfo(kARRAY, 0, 0, true, kENCODING_NONE, 0, kDOUBLE);
    case BufferAttrType::kUint64:
    case BufferAttrType::kInt64:
      return SQLTypeInfo(kBIGINT, true);
    case BufferAttrType::kVec2ui64:
    case BufferAttrType::kVec3ui64:
    case BufferAttrType::kVec4ui64:
    case BufferAttrType::kVec2i64:
    case BufferAttrType::kVec3i64:
    case BufferAttrType::kVec4i64:
      return SQLTypeInfo(kARRAY, 0, 0, true, kENCODING_NONE, 0, kBIGINT);
    case BufferAttrType::kMat3x2f:
    case BufferAttrType::kMat3x2d:
      CHECK(false) << "MAT3x2F/D not supported as an SQLType";
    case BufferAttrType::kCOUNT:
      CHECK(false);
  }
  CHECK(false) << buffer_attr_type;
  return SQLTypeInfo();
}

}  // namespace QueryRenderer
