/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>

namespace QueryRenderer {

enum class GeomType {
  kPoints = 0,
  kPolys,
  kSymbols,
  kLines,
  kLegacySymbols,
  kWindBarbs,
  kMesh2d,
  kCOUNT
};

enum class LineJoinType { kBevel = 0, kRound, kMiter, kCOUNT };

enum class SymbolShapeType {
  kCircle = 0,
  kSquare,
  kCross,
  kDiamond,
  kTriangleUp,
  kTriangleDown,
  kTriangleRight,
  kTriangleLeft,
  kHexagonHoriz,
  kHexagonVert,
  kWedge,
  kArrow,
  kAirplane,
  kCOUNT
};

enum class AngleUnit { kRadians, kDegrees, kCOUNT };

enum class MarkGpuResourceSlot { kFill, kStroke };

std::string to_string(const LineJoinType value);
int convertStringToLineJoinEnum(const std::string& val);

std::string to_string(const SymbolShapeType value);
int convertStringToSymbolShapeEnum(const std::string& val);

std::string to_string(const AngleUnit value);
int convertStringToAngleUnitEnum(const std::string& val);

std::string to_string(const GeomType value);
int convertStringToGeomTypeEnum(const std::string& val);

}  // namespace QueryRenderer

std::ostream& operator<<(std::ostream& os, const ::QueryRenderer::GeomType value);
std::ostream& operator<<(std::ostream& os, const ::QueryRenderer::LineJoinType value);
std::ostream& operator<<(std::ostream& os, const ::QueryRenderer::SymbolShapeType value);
std::ostream& operator<<(std::ostream& os, const ::QueryRenderer::AngleUnit value);
