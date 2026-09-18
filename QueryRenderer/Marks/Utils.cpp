/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Marks/Utils.h"

#include "QueryRenderer/Marks/LineMark.h"
#include "QueryRenderer/Marks/Mesh2dMark.h"
#include "QueryRenderer/Marks/PointMark.h"
#include "QueryRenderer/Marks/PolyMark.h"
#include "QueryRenderer/Marks/SymbolMark.h"
#include "QueryRenderer/Marks/SymbolMark_Proc.h"
#include "QueryRenderer/Marks/WindBarbMark.h"
#include "QueryRenderer/Utils/RapidJSONUtils.h"

namespace QueryRenderer {

GeomType getMarkTypeFromJSONObj(const JSONLocation& json_loc) {
  const auto type_loc = json_loc.getMember(JSONSchema_v1::Marks::kTypeProp);
  RUNTIME_EX_ASSERT(
      type_loc.isValid() && type_loc.isString(),
      RapidJSONUtils::createJsonParseError(
          (type_loc.isValid() ? type_loc : json_loc),
          "A mark object must have a \"" + std::string(JSONSchema_v1::Marks::kTypeProp) +
              "\" string property."));

  auto geom_type = convertStringToGeomTypeEnum(type_loc.getString());
  RUNTIME_EX_ASSERT(
      geom_type >= 0,
      RapidJSONUtils::createJsonParseError(
          type_loc, "A mark of type \"" + type_loc.getString() + "\" is unsupported."));

  return static_cast<GeomType>(geom_type);
}

BaseMarkUqPtr createMark(const JSONLocation& json_loc, QueryRendererContext& ctx) {
  RUNTIME_EX_ASSERT(
      json_loc.isObject(),
      RapidJSONUtils::createJsonParseError(json_loc, "Marks must be objects."));

  switch (getMarkTypeFromJSONObj(json_loc)) {
    case GeomType::kPoints:
      return std::make_unique<PointMark>(json_loc, ctx);
    case GeomType::kPolys:
      return std::make_unique<PolyMark>(json_loc, ctx);
    case GeomType::kSymbols:
      return std::make_unique<SymbolMark_Proc>(json_loc, ctx);
    case GeomType::kLines:
      return std::make_unique<LineMark>(json_loc, ctx);
    case GeomType::kLegacySymbols:
      return std::make_unique<SymbolMark>(json_loc, ctx);
    case GeomType::kWindBarbs:
      return std::make_unique<WindBarbMark>(json_loc, ctx);
    case GeomType::kMesh2d:
      return std::make_unique<Mesh2dMark>(json_loc, ctx);
    case GeomType::kCOUNT:
      CHECK(false) << "Invalid GeomType";
      return nullptr;
  }

  return BaseMarkUqPtr();
}

}  // namespace QueryRenderer
