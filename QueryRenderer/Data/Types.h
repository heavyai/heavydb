/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>

#include "GfxDriver/Colors/Types.h"
#include "GfxDriver/TypeGLSL.h"
#include "Shared/EnumBitmaskOps.h"
#include "Shared/sqltypes.h"

namespace QueryRenderer {

// @TODO(se) 7/28/20 style these enum values, but not yet

enum class QueryDataType {
  UINT = 0,
  INT,
  FLOAT,
  DOUBLE,
  UINT64,
  INT64,
  COLOR,
  STRING,
  BOOL,
  LINE_JOIN_ENUM,
  SYMBOL_SHAPE_ENUM,
  ANGLE_UNIT_ENUM,
  POLYGON_DOUBLE,
  LINE_DOUBLE
};

enum class DataInputFormat {
  kSQL = 0,
  kEmbedded,
  kURL,
  kSourced,
  kTransformed,
  kUnsupported  // leave UNSUPPORTED as last enum
};

enum class DataOutputFormat {
  kRows = 0,
  kPolys,
  kLines,
  kMesh2d,
  kUnsupported  // leave as last enum
};

// Layout changed flags for query data table.
enum class QDTLayoutChangedFlags : uint8_t {
  kNone = 0x00,
  kVboContents = 0x01,          // vbo contents
  kVboOffset = 0x02,            // vbo layout offset
  kVboContentsOrOffset = 0x03,  // vbo contents or offset
  kSsboContents = 0x04          // ssbo contents
};

bool isExternallySourcedInputFormat(const DataInputFormat);
std::string getDataInputFormatsAsStr();

class BaseDataTable;
using BaseDataTableUqPtr = std::unique_ptr<BaseDataTable>;
using BaseDataTableShPtr = std::shared_ptr<BaseDataTable>;

class BaseQueryDataTableSQLJSON;
using QueryDataTableSQLJSONUqPtr = std::unique_ptr<BaseQueryDataTableSQLJSON>;
using QueryDataTableSQLJSONShPtr = std::shared_ptr<BaseQueryDataTableSQLJSON>;

class QueryDataTableSQL;

class BasePolyDataTable;
class BaseLineDataTable;

class SqlQueryPolyDataTableJSON;
using SqlQueryPolyDataTableShPtr = std::shared_ptr<SqlQueryPolyDataTableJSON>;

class SqlQueryLineDataTableJSON;
using SqlQueryLineDataTableShPtr = std::shared_ptr<SqlQueryLineDataTableJSON>;

class QuerySourceDataTable;
using QuerySourceDataTableShPtr = std::shared_ptr<QuerySourceDataTable>;

class QueryDataTableQueues;

std::string to_string(QueryDataType data_type);
std::string to_string(DataInputFormat data_input_format);
std::string to_string(DataOutputFormat data_output_format);

template <typename T, class Enable = void>
struct TypeToQueryDataTypeSelector {
  using BufferType = T;
  static constexpr int numComponents() { return 1; }
  static QueryDataType getQueryDataType() {
    CHECK(false) << "Needs to be defined in a specialization";
    return QueryDataType::INT;
  }
  static gfx::TypeGLSLShPtr getTypeGLSLPtr() {
    return std::make_shared<gfx::TypeGLSL<BufferType, numComponents()>>();
  }
};

template <typename T>
struct TypeToQueryDataTypeSelector<
    T,
    typename std::enable_if_t<gfx::is_color<T>::value || gfx::is_color_union<T>::value>> {
  using BufferType = float;
  static constexpr int numComponents() { return 4; }
  static QueryDataType getQueryDataType() { return QueryDataType::COLOR; }
  static gfx::TypeGLSLShPtr getTypeGLSLPtr() {
    return std::make_shared<gfx::TypeGLSL<BufferType, numComponents()>>();
  }
};

template <>
QueryDataType TypeToQueryDataTypeSelector<uint32_t>::getQueryDataType();

template <>
QueryDataType TypeToQueryDataTypeSelector<int32_t>::getQueryDataType();

template <>
QueryDataType TypeToQueryDataTypeSelector<float>::getQueryDataType();

template <>
QueryDataType TypeToQueryDataTypeSelector<uint64_t>::getQueryDataType();

template <>
QueryDataType TypeToQueryDataTypeSelector<int64_t>::getQueryDataType();

template <>
QueryDataType TypeToQueryDataTypeSelector<double>::getQueryDataType();

template <>
struct TypeToQueryDataTypeSelector<std::string> {
  using BufferType = int32_t;
  static constexpr int numComponents() { return 1; }
  static QueryDataType getQueryDataType();
  static gfx::TypeGLSLShPtr getTypeGLSLPtr() {
    return std::make_shared<gfx::TypeGLSL<BufferType, numComponents()>>();
  }
};

bool isArithmeticQueryDataType(const QueryDataType type);
QueryDataType convertToQueryDataType(const gfx::BufferAttrType attr_type);
QueryDataType convertToQueryDataType(const SQLTypeInfo& type_info);

template <QueryDataType dataType>
struct QueryDataTypeSelector {
  using type = void;
};
template <>
struct QueryDataTypeSelector<QueryDataType::UINT> {
  using type = uint32_t;
};
template <>
struct QueryDataTypeSelector<QueryDataType::INT> {
  using type = int32_t;
};
template <>
struct QueryDataTypeSelector<QueryDataType::FLOAT> {
  using type = float;
};
template <>
struct QueryDataTypeSelector<QueryDataType::DOUBLE> {
  using type = double;
};
template <>
struct QueryDataTypeSelector<QueryDataType::UINT64> {
  using type = uint64_t;
};
template <>
struct QueryDataTypeSelector<QueryDataType::INT64> {
  using type = int64_t;
};
template <>
struct QueryDataTypeSelector<QueryDataType::COLOR> {
  using type = gfx::ColorUnion;
};
template <>
struct QueryDataTypeSelector<QueryDataType::STRING> {
  using type = std::string;
};
template <>
struct QueryDataTypeSelector<QueryDataType::BOOL> {
  using type = bool;
};
template <>
struct QueryDataTypeSelector<QueryDataType::LINE_JOIN_ENUM> {
  using type = unsigned int;
};
template <>
struct QueryDataTypeSelector<QueryDataType::SYMBOL_SHAPE_ENUM> {
  using type = unsigned int;
};
template <>
struct QueryDataTypeSelector<QueryDataType::ANGLE_UNIT_ENUM> {
  using type = unsigned int;
};
template <>
struct QueryDataTypeSelector<QueryDataType::POLYGON_DOUBLE> {
  using type = double;
};

std::ostream& operator<<(std::ostream&, QueryDataType);
std::ostream& operator<<(std::ostream&, DataInputFormat);
std::ostream& operator<<(std::ostream&, DataOutputFormat);

}  // namespace QueryRenderer

ENABLE_BITMASK_OPS(::QueryRenderer::QDTLayoutChangedFlags);
