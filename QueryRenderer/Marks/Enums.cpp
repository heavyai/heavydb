/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Marks/Enums.h"

#include "Logger/Logger.h"
#include "QueryRenderer/Interop/InteropBufferHandle.h"
#include "QueryRenderer/Utils/StringUtils.h"

namespace QueryRenderer {

namespace {
template <typename EnumType, int Count>
inline int convert_string_to_enum(const std::string& val,
                                  const std::array<std::string, Count>& lut) {
  std::string upper_case = makeUpperCase(val);
  for (int i = 0; i < Count; ++i) {
    if (upper_case == lut[i]) {
      return i;
    }
  }
  return -1;
}
}  // namespace

// GeomType (Mark type)
static std::array<std::string, static_cast<int>(GeomType::kCOUNT)> geom_type_strings =
    {"POINTS", "POLYS", "SYMBOL", "LINES", "LEGACYSYMBOL", "WINDBARB", "MESH2D"};

std::string to_string(const GeomType value) {
  CHECK_NE(value, GeomType::kCOUNT);
  return geom_type_strings[static_cast<int>(value)];
}

int convertStringToGeomTypeEnum(const std::string& val) {
  return convert_string_to_enum<GeomType, static_cast<int>(GeomType::kCOUNT)>(
      val, geom_type_strings);
}

// LineJoinType
static std::array<std::string, static_cast<int>(LineJoinType::kCOUNT)>
    line_join_type_strings = {"BEVEL", "ROUND", "MITER"};

std::string to_string(const LineJoinType value) {
  CHECK_NE(value, LineJoinType::kCOUNT);
  return line_join_type_strings[static_cast<int>(value)];
}

int convertStringToLineJoinEnum(const std::string& val) {
  return convert_string_to_enum<LineJoinType, static_cast<int>(LineJoinType::kCOUNT)>(
      val, line_join_type_strings);
}

// SymbolShapeType
static std::array<std::string, static_cast<int>(SymbolShapeType::kCOUNT)>
    symbol_shape_type_strings = {"CIRCLE",
                                 "SQUARE",
                                 "CROSS",
                                 "DIAMOND",
                                 "TRIANGLE-UP",
                                 "TRIANGLE-DOWN",
                                 "TRIANGLE-RIGHT",
                                 "TRIANGLE-LEFT",
                                 "HEXAGON-HORIZ",
                                 "HEXAGON-VERT",
                                 "WEDGE",
                                 "ARROW",
                                 "AIRPLANE"};

std::string to_string(const SymbolShapeType value) {
  CHECK_NE(value, SymbolShapeType::kCOUNT);
  return symbol_shape_type_strings[static_cast<int>(value)];
}

int convertStringToSymbolShapeEnum(const std::string& val) {
  return convert_string_to_enum<SymbolShapeType,
                                static_cast<int>(SymbolShapeType::kCOUNT)>(
      val, symbol_shape_type_strings);
}

// AngleUnit
static std::array<std::string, static_cast<int>(AngleUnit::kCOUNT)>
    symbol_angle_unit_strings = {"RADIANS", "DEGREES"};

int convertStringToAngleUnitEnum(const std::string& val) {
  return convert_string_to_enum<AngleUnit, static_cast<int>(AngleUnit::kCOUNT)>(
      val, symbol_angle_unit_strings);
}

std::string to_string(const AngleUnit value) {
  CHECK_NE(value, AngleUnit::kCOUNT);
  return symbol_angle_unit_strings[static_cast<int>(value)];
}

}  // namespace QueryRenderer

std::ostream& operator<<(std::ostream& os, const QueryRenderer::GeomType value) {
  CHECK_NE(value, QueryRenderer::GeomType::kCOUNT);
  os << QueryRenderer::geom_type_strings[static_cast<int>(value)];
  return os;
}

std::ostream& operator<<(std::ostream& os, const QueryRenderer::LineJoinType value) {
  CHECK_NE(value, QueryRenderer::LineJoinType::kCOUNT);
  os << QueryRenderer::line_join_type_strings[static_cast<int>(value)];
  return os;
}

std::ostream& operator<<(std::ostream& os, const QueryRenderer::SymbolShapeType value) {
  CHECK_NE(value, QueryRenderer::SymbolShapeType::kCOUNT);
  os << QueryRenderer::symbol_shape_type_strings[static_cast<int>(value)];
  return os;
}

std::ostream& operator<<(std::ostream& os, const QueryRenderer::AngleUnit value) {
  CHECK_NE(value, QueryRenderer::AngleUnit::kCOUNT);
  os << QueryRenderer::symbol_angle_unit_strings[static_cast<int>(value)];
  return os;
}
