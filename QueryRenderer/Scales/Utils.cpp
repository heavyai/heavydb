/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Scales/Utils.h"

#include <boost/algorithm/string/join.hpp>

#include "GfxDriver/Colors/ColorHCL.h"
#include "GfxDriver/Colors/ColorHSL.h"
#include "GfxDriver/Colors/ColorLAB.h"
#include "GfxDriver/Colors/ColorRGBA.h"
#include "GfxDriver/Colors/Utils.h"
#include "QueryRenderer/Data/BaseDataTable.h"
#include "QueryRenderer/Data/BaseQueryDataTable.h"
#include "QueryRenderer/QueryRendererContext.h"
#include "QueryRenderer/Utils/StringUtils.h"

namespace QueryRenderer {

using ::gfx::ColorHCL;
using ::gfx::ColorHSL;
using ::gfx::ColorLAB;
using ::gfx::ColorRGBA;

std::vector<std::string> getFieldsFromDataRef(const JSONLocation& data_loc,
                                              const QueryRendererContext& ctx,
                                              const BaseDataTableShPtr& table) {
  JSONLocation field_loc;
  RUNTIME_EX_ASSERT(
      ((field_loc = data_loc.getMember(JSONSchema_v1::Scales::kFieldProp)).isValid() &&
       field_loc.isString()) ||
          ((field_loc = data_loc.getMember(JSONSchema_v1::Scales::kFieldsProp))
               .isValid() &&
           field_loc.isArray() && field_loc.size()),
      RapidJSONUtils::createJsonParseError(
          field_loc.isValid() ? field_loc : data_loc,
          "Data reference object must have a \"" +
              std::string(JSONSchema_v1::Scales::kFieldProp) + "\" (string) or \"" +
              std::string(JSONSchema_v1::Scales::kFieldsProp) +
              "\" (array of strings) property."));

  std::vector<std::string> column_names;
  if (field_loc.isString()) {
    column_names = {field_loc.getString()};
  } else {
    for (size_t i = 0; i < field_loc.size(); ++i) {
      const auto field_item_loc = field_loc[i];
      RUNTIME_EX_ASSERT(
          field_item_loc.isString(),
          RapidJSONUtils::createJsonParseError(
              field_item_loc,
              "All items in the \"" + std::string(JSONSchema_v1::Scales::kFieldsProp) +
                  "\" property must be strings."));
      const std::string col_name = field_item_loc.getString();
      auto json_data = std::dynamic_pointer_cast<BaseQueryDataTableSQLJSON>(table);
      RUNTIME_EX_ASSERT(
          table->hasAttribute(col_name),
          RapidJSONUtils::createJsonParseError(
              field_item_loc,
              "\"" + col_name + "\" does not exist in the data table \"" +
                  (json_data ? json_data->getName() : "(unknown)") + "\"."));
      column_names.emplace_back(field_item_loc.getString());
    }
  }

  return column_names;
}

namespace {
QueryDataType getDataTypeFromDataRefJSONObj(const JSONLocation& json_loc,
                                            const QueryRendererContext& ctx) {
  RUNTIME_EX_ASSERT(json_loc.isObject(),
                    RapidJSONUtils::createJsonParseError(
                        json_loc, "Data reference is not a JSON object."));

  const auto data_loc = json_loc.getMember(JSONSchema_v1::Scales::kDataProp);
  RUNTIME_EX_ASSERT(
      data_loc.isValid() && data_loc.isString(),
      RapidJSONUtils::createJsonParseError(
          (data_loc.isValid() ? data_loc : json_loc),
          "Data reference object doesn't contain a \"" +
              std::string(JSONSchema_v1::Scales::kDataProp) + "\" string property."));
  auto data_table_name = data_loc.getString();
  auto table = ctx.getDataTable(data_table_name);
  RUNTIME_EX_ASSERT(
      table != nullptr,
      RapidJSONUtils::createJsonParseError(
          data_loc, "Data table \"" + data_table_name + "\" doesn't exist."));

  auto column_names = getFieldsFromDataRef(json_loc, ctx, table);
  CHECK(column_names.size());

  auto rtn_type = table->getAttributeType(column_names[0]);
  for (size_t i = 1; i < column_names.size(); ++i) {
    auto item_type = table->getAttributeType(column_names[i]);

    auto higher_type = QueryDataType::UINT;
    RUNTIME_EX_ASSERT(
        RapidJSONUtils::getHigherOrderDataType(higher_type, rtn_type, item_type),
        "Scale domain item at index " + std::to_string(i) +
            " has an incompatible type. " + to_string(rtn_type) +
            " is not compatible with " + to_string(item_type));
    if (higher_type != rtn_type) {
      rtn_type = higher_type;
    }
  }

  return rtn_type;
}
}  // namespace

std::string getScaleNameFromJSONObj(const JSONLocation& json_loc) {
  RUNTIME_EX_ASSERT(json_loc.isObject(),
                    RapidJSONUtils::createJsonParseError(
                        json_loc, "Scale items must be JSON objects."));

  const auto name_loc = json_loc.getMember(JSONSchema_v1::Scales::kNameProp);
  RUNTIME_EX_ASSERT(
      name_loc.isValid() && name_loc.isString(),
      RapidJSONUtils::createJsonParseError(
          name_loc.isValid() ? name_loc : json_loc,
          "Scale objects must contain a \"" +
              std::string(JSONSchema_v1::Scales::kNameProp) + "\" string property."));

  return name_loc.getString();
}

ScaleType getScaleTypeFromJSONObj(const JSONLocation& json_loc) {
  // TODO(croot): expose default as a static attr.
  ScaleType rtn = ScaleType::kLinear;

  RUNTIME_EX_ASSERT(json_loc.isObject(),
                    RapidJSONUtils::createJsonParseError(
                        json_loc, "Scale items must be JSON objects."));

  const auto type_loc = json_loc.getMember(JSONSchema_v1::Scales::kTypeProp);
  if (type_loc.isValid()) {
    RUNTIME_EX_ASSERT(type_loc.isString(),
                      RapidJSONUtils::createJsonParseError(
                          type_loc,
                          "\"" + std::string(JSONSchema_v1::Scales::kTypeProp) +
                              "\" property in scale objects must be a string."));
    std::string str_scale_type(type_loc.getString());
    if (str_scale_type == "linear") {
      rtn = ScaleType::kLinear;
    } else if (str_scale_type == "log") {
      rtn = ScaleType::kLog;
    } else if (str_scale_type == "pow") {
      rtn = ScaleType::kPow;
    } else if (str_scale_type == "sqrt") {
      rtn = ScaleType::kSqrt;
    } else if (str_scale_type == "ordinal") {
      rtn = ScaleType::kOrdinal;
    } else if (str_scale_type == "quantize") {
      rtn = ScaleType::kQuantize;
    } else if (str_scale_type == "threshold") {
      rtn = ScaleType::kThreshold;
    } else {
      THROW_RUNTIME_EX(RapidJSONUtils::createJsonParseError(
          type_loc, "Scale type \"" + str_scale_type + "\" is not a supported type."));
    }
  }

  return rtn;
}

QueryDataType getScaleDomainDataTypeFromJSONObj(const JSONLocation& json_loc,
                                                const QueryRendererContext& ctx,
                                                const ScaleType scale_type) {
  bool is_object = false;

  // TODO(croot): expose "domain" as a const somewhere.
  const auto domain_loc = json_loc.getMember(JSONSchema_v1::Scales::kDomainProp);
  RUNTIME_EX_ASSERT(domain_loc.isValid() && ((is_object = domain_loc.isObject()) ||
                                             (domain_loc.isArray() && domain_loc.size())),
                    RapidJSONUtils::createJsonParseError(
                        (domain_loc.isValid() ? domain_loc : json_loc),
                        "\"" + std::string(JSONSchema_v1::Scales::kDomainProp) +
                            "\" property for scales must exist "
                            "and must be an object or an array."));

  QueryDataType domain_type{QueryDataType::UINT};

  if (is_object) {
    domain_type = getDataTypeFromDataRefJSONObj(domain_loc, ctx);
  } else {
    // TODO(croot): Probably need to have specific classes to correspond
    // to the different scales. For example, ordinal/categorical scales
    // can support strings for domain values. Others shouldn't.
    // Will allow all domains to accept all strings for now.

    AnyDataType domain_any_type, item_any_type;
    size_t start_idx = 0;
    switch (scale_type) {
      case ScaleType::kLog:
      case ScaleType::kSqrt:
      case ScaleType::kPow:
        // TODO(croot): support float?
        domain_any_type.set(QueryDataType::DOUBLE, double(0));
        start_idx = 0;
        break;
      default: {
        const auto domain_item_loc = domain_loc[0];
        domain_any_type = RapidJSONUtils::getAnyDataFromJSONObj(domain_item_loc, true);
        start_idx = 1;
        break;
      }
    }

    // If the domain is a single-value array, just take the type from the JSON obj
    domain_type = domain_any_type.getType();
    for (size_t i = start_idx; i < domain_loc.size(); ++i) {
      const auto domain_item_loc = domain_loc[i];
      item_any_type = RapidJSONUtils::getAnyDataFromJSONObj(domain_item_loc, true);

      RUNTIME_EX_ASSERT(
          RapidJSONUtils::getHigherOrderDataType(
              domain_type, domain_any_type, item_any_type),
          RapidJSONUtils::createJsonParseError(
              domain_item_loc,
              "Scale domain item at index " + std::to_string(i) +
                  " has an incompatible type. " + to_string(domain_any_type.getType()) +
                  " is not compatible with " + to_string(item_any_type.getType())));
      if (domain_type != domain_any_type.getType()) {
        domain_any_type = item_any_type;
      }
    }
  }

  auto accum_type = getScaleAccumulatorTypeFromJSONObj(json_loc);
  if (accum_type == AccumulatorType::kDensity) {
    RUNTIME_EX_ASSERT(RapidJSONUtils::getHigherOrderDataType(
                          domain_type, domain_type, QueryDataType::FLOAT),
                      RapidJSONUtils::createJsonParseError(
                          domain_loc,
                          "Density accumulation scale domain of type " +
                              to_string(domain_type) + " is not FLOAT compatible."));
  }

  return domain_type;
}

QueryDataType getScaleRangeDataTypeFromJSONObj(const JSONLocation& json_loc,
                                               const QueryRendererContext& ctx,
                                               const ScaleType scale_type) {
  bool is_object = false;
  bool is_string;

  const auto range_loc = json_loc.getMember(JSONSchema_v1::Scales::kRangeProp);
  RUNTIME_EX_ASSERT(
      range_loc.isValid() &&
          ((is_object = range_loc.isObject()) || (is_string = range_loc.isString()) ||
           (range_loc.isArray() && range_loc.size())),
      RapidJSONUtils::createJsonParseError(
          (range_loc.isValid() ? range_loc : json_loc),
          "\"" + std::string(JSONSchema_v1::Scales::kRangeProp) +
              "\" property for scales must exist and must be an object or a string."));

  QueryDataType range_type{QueryDataType::UINT};

  if (is_object) {
    range_type = getDataTypeFromDataRefJSONObj(range_loc, ctx);
  } else if (is_string) {
    std::string str_val = makeLowerCase(range_loc.getString());
    bool is_width_height;
    RUNTIME_EX_ASSERT((is_width_height = str_val == "width" || str_val == "height") ||
                          str_val == "symbol",
                      RapidJSONUtils::createJsonParseError(
                          range_loc,
                          "Invalid \"" + std::string(JSONSchema_v1::Scales::kRangeProp) +
                              "\" string property for "
                              "scales. Only string literals supported are "
                              "\"width\", \"height\", and \"symbol\""));

    range_type =
        (is_width_height ? QueryDataType::FLOAT : QueryDataType::SYMBOL_SHAPE_ENUM);
  } else {
    auto range_item_loc = range_loc[0];
    AnyDataType range_any_type = RapidJSONUtils::getAnyDataFromJSONObj(range_item_loc),
                item_any_type;
    range_type = range_any_type.getType();

    for (size_t i = 1; i < range_loc.size(); ++i) {
      range_item_loc = range_loc[i];
      item_any_type = RapidJSONUtils::getAnyDataFromJSONObj(range_item_loc);

      RUNTIME_EX_ASSERT(
          RapidJSONUtils::getHigherOrderDataType(
              range_type, range_any_type, item_any_type),
          RapidJSONUtils::createJsonParseError(
              range_item_loc,
              "Scale range item at index " + std::to_string(i) +
                  " has an incompatible type. " + to_string(range_any_type.getType()) +
                  " is not compatible with " + to_string(item_any_type.getType())));
      if (range_type != range_any_type.getType()) {
        range_any_type = item_any_type;
      }
    }
  }

  // TODO(croot): this should be more appropriately handled by a parser specific to each
  // scale type. Since a refactor regarding a decoupling of json parser and operators,
  // I'll leave save that for then rather than deal with it now.
  switch (scale_type) {
    case ScaleType::kOrdinal: {
      const auto default_loc = json_loc.getMember(JSONSchema_v1::Scales::kDefaultProp);
      if (default_loc.isValid()) {
        auto item_type = RapidJSONUtils::getDataTypeFromJSONObj(default_loc);
        RUNTIME_EX_ASSERT(
            RapidJSONUtils::getHigherOrderDataType(range_type, range_type, item_type),
            RapidJSONUtils::createJsonParseError(
                default_loc,
                "The scale of type " + to_string(scale_type) + " has a range of type " +
                    to_string(range_type) + " which is not compatible with a \"" +
                    std::string(JSONSchema_v1::Scales::kDefaultProp) +
                    "\" property of type " + to_string(item_type) + "."));
      }
    }  // let pass thru to default case to check for nullValue
    default: {
      const auto null_value_loc =
          json_loc.getMember(JSONSchema_v1::Scales::kNullValueProp);
      if (null_value_loc.isValid()) {
        auto item_type = RapidJSONUtils::getDataTypeFromJSONObj(null_value_loc);
        RUNTIME_EX_ASSERT(
            RapidJSONUtils::getHigherOrderDataType(range_type, range_type, item_type),
            RapidJSONUtils::createJsonParseError(
                null_value_loc,
                "The scale of type " + to_string(scale_type) + " has a range of type " +
                    to_string(range_type) + " which is not compatible with a \"" +
                    std::string(JSONSchema_v1::Scales::kNullValueProp) +
                    "\" property of type " + to_string(item_type) + "."));
      }
      break;
    }
  }

  return range_type;
}

namespace {
gfx::ColorType getColorTypeFromScaleInterpType(const ScaleInterpType interp_type) {
  switch (interp_type) {
    case ScaleInterpType::kRgb:
      return gfx::ColorType::RGBA;
    case ScaleInterpType::kHsl:
    case ScaleInterpType::kHslLong:
      return gfx::ColorType::HSL;
    case ScaleInterpType::kLab:
      return gfx::ColorType::LAB;
    case ScaleInterpType::kHcl:
    case ScaleInterpType::kHclLong:
      return gfx::ColorType::HCL;
    case ScaleInterpType::kUndefined:
      return gfx::ColorType::INVALID;
  }
  return gfx::ColorType::INVALID;
}
}  // namespace

ScaleInterpType getScaleInterpTypeFromJSONObj(const JSONLocation& json_loc) {
  CHECK(json_loc.isValid() && json_loc.isObject())
      << RapidJSONUtils::getPointerPath(json_loc.getPathRef());
  auto interp_loc = json_loc.getMember(JSONSchema_v1::Scales::kInterpolatorProp);
  if (!interp_loc.isValid()) {
    return ScaleInterpType::kUndefined;
  }

  RUNTIME_EX_ASSERT(
      interp_loc.isString(),
      RapidJSONUtils::createJsonParseError(
          interp_loc,
          "Scale interpolators must be a string of one of the following: [" +
              boost::algorithm::join(getScaleInterpTypes(), ", ") + "]"));

  const auto str_val = std::string(interp_loc.getString());
  static std::vector<std::regex> interp_regex;
  if (!interp_regex.size()) {
    auto interp_strings = getScaleInterpTypes();
    for (auto& interp_str : interp_strings) {
      interp_regex.emplace_back("^\\s*" + interp_str + "\\s*$",
                                std::regex_constants::icase);
    }
  }

  for (size_t i = 0; i < interp_regex.size(); ++i) {
    if (std::regex_match(str_val, interp_regex[i])) {
      return static_cast<ScaleInterpType>(i);
    }
  }

  THROW_RUNTIME_EX(RapidJSONUtils::createJsonParseError(
      interp_loc,
      "Interpolator \"" + str_val +
          "\" is not a supported interpolator type. Supported validators are [" +
          boost::algorithm::join(getScaleInterpTypes(), ", ") + "]"));
  return ScaleInterpType::kUndefined;
}

std::pair<gfx::ColorType, ScaleInterpType> getScaleRangeColorTypeFromJSONObj(
    const JSONLocation& json_loc) {
  gfx::ColorType color_type = gfx::ColorType::RGBA;

  CHECK(json_loc.isObject());
  const auto range_loc = json_loc.getMember(JSONSchema_v1::Scales::kRangeProp);
  CHECK(range_loc.isValid() && range_loc.isArray() && range_loc.size());

  const auto range_item_loc = range_loc[0];
  CHECK(range_item_loc.isString());

  auto interp_type = getScaleInterpTypeFromJSONObj(json_loc);
  color_type = getColorTypeFromScaleInterpType(interp_type);
  if (color_type == gfx::ColorType::INVALID) {
    color_type = gfx::getColorTypeFromColorString(range_item_loc.getString());
  }

  return std::make_pair(color_type, interp_type);
}

AccumulatorType getScaleAccumulatorTypeFromJSONObj(const JSONLocation& json_loc) {
  auto rtn = AccumulatorType::kUndefined;
  const auto accum_loc = json_loc.getMember(JSONSchema_v1::Scales::kAccumulatorProp);
  if (accum_loc.isValid()) {
    auto item_type = RapidJSONUtils::getDataTypeFromJSONObj(accum_loc, true);

    RUNTIME_EX_ASSERT(
        item_type == QueryDataType::STRING,
        RapidJSONUtils::createJsonParseError(
            accum_loc,
            "Scale \"" + std::string(JSONSchema_v1::Scales::kAccumulatorProp) +
                "\" property must be a string."));

    auto accum_type_str = makeUpperCase(accum_loc.getString());

    if (accum_type_str == "MIN") {
      rtn = AccumulatorType::kMin;
    } else if (accum_type_str == "MAX") {
      rtn = AccumulatorType::kMax;
    } else if (accum_type_str == "BLEND") {
      rtn = AccumulatorType::kBlend;
    } else if (accum_type_str == "DENSITY") {
      rtn = AccumulatorType::kDensity;
    } else if (accum_type_str == "PCT") {
      rtn = AccumulatorType::kPct;
    } else {
      THROW_RUNTIME_EX(RapidJSONUtils::createJsonParseError(
          accum_loc,
          "\"" + accum_loc.getString() +
              "\" is not a supported type for the scale property \"" +
              std::string(JSONSchema_v1::Scales::kAccumulatorProp) + "\"."));
    }
  }

  return rtn;
}

bool isScaleDomainCompatible(const ScaleType scale_type,
                             const QueryDataType domain_type) {
  // TODO(croot): put this in the scale classes? It's easier to do here because
  // otherwise we'd have to do template specializations, which would require a lot
  // of extra code, so keeping here for now.
  switch (scale_type) {
    case ScaleType::kLog:
    case ScaleType::kPow:
    case ScaleType::kSqrt:
      return (domain_type == QueryDataType::DOUBLE ||
              domain_type == QueryDataType::FLOAT);
    case ScaleType::kQuantize:
    case ScaleType::kThreshold:
      return (
          domain_type == QueryDataType::UINT || domain_type == QueryDataType::INT ||
          domain_type == QueryDataType::UINT64 || domain_type == QueryDataType::INT64 ||
          domain_type == QueryDataType::FLOAT || domain_type == QueryDataType::DOUBLE);
    default:
      return true;
  }

  return true;
}

bool isScaleRangeCompatible(const ScaleType scale_type, const QueryDataType range_type) {
  return true;
}

bool areTypesCompatible(const QueryDataType src_type, const QueryDataType in_type) {
  if (src_type == in_type) {
    return true;
  }

  switch (src_type) {
    case QueryDataType::UINT:
    case QueryDataType::INT:
      return (in_type == QueryDataType::UINT || in_type == QueryDataType::INT);
    case QueryDataType::FLOAT:
      return (in_type == QueryDataType::UINT || in_type == QueryDataType::INT ||
              in_type == QueryDataType::FLOAT);
    case QueryDataType::UINT64:
    case QueryDataType::INT64:
      return (in_type == QueryDataType::UINT || in_type == QueryDataType::INT ||
              in_type == QueryDataType::UINT64 || in_type == QueryDataType::INT64);
    case QueryDataType::DOUBLE:
      return (in_type == QueryDataType::UINT || in_type == QueryDataType::INT ||
              in_type == QueryDataType::FLOAT || in_type == QueryDataType::UINT64 ||
              in_type == QueryDataType::INT64 || in_type == QueryDataType::DOUBLE);
    default:
      return false;
  }
}

bool areTypesCompatible(const std::type_info& src_type, const std::type_info& in_type) {
  return areTypesCompatible(convertTypeIdToDataType(src_type),
                            convertTypeIdToDataType(in_type));
}

QueryDataType convertTypeIdToDataType(const std::type_info& src_type_id) {
  if (src_type_id == typeid(int)) {
    return QueryDataType::INT;
  } else if (src_type_id == typeid(unsigned int)) {
    return QueryDataType::UINT;
  } else if (src_type_id == typeid(float)) {
    return QueryDataType::FLOAT;
  } else if (src_type_id == typeid(int64_t)) {
    return QueryDataType::INT64;
  } else if (src_type_id == typeid(uint64_t)) {
    return QueryDataType::UINT64;
  } else if (src_type_id == typeid(double)) {
    return QueryDataType::DOUBLE;
  } else if (src_type_id == typeid(gfx::ColorRGBA)) {
    return QueryDataType::COLOR;
  } else if (src_type_id == typeid(gfx::ColorHSL)) {
    return QueryDataType::COLOR;
  } else if (src_type_id == typeid(gfx::ColorLAB)) {
    return QueryDataType::COLOR;
  } else if (src_type_id == typeid(gfx::ColorHCL)) {
    return QueryDataType::COLOR;
  } else {
    THROW_RUNTIME_EX("Type id: " + std::string(src_type_id.name()) +
                     " cannot be converted to a QueryDataType");
  }
  return QueryDataType::INT;
}

template <typename C0, typename C1, typename C2, typename C3>
static C0 convertColorType(const QueryDataType type, const std::any& value) {
  RUNTIME_EX_ASSERT(type == QueryDataType::COLOR,
                    "Converting " + to_string(type) + " to a " +
                        std::string(typeid(C0).name()) + " is unsupported.");

  C0 rtn;
  try {
    rtn = std::any_cast<C0>(value);
  } catch (const std::bad_any_cast&) {
    try {
      auto c1 = std::any_cast<C1>(value);
      convertColor(c1, rtn);
    } catch (const std::bad_any_cast&) {
      try {
        auto c2 = std::any_cast<C2>(value);
        convertColor(c2, rtn);
      } catch (const std::bad_any_cast&) {
        auto c3 = std::any_cast<C3>(value);
        convertColor(c3, rtn);
      }
    }
  }

  return rtn;
}

template <>
ColorRGBA convertType(const QueryDataType type,
                      const std::any& value,
                      const bool ignore_null) {
  return convertColorType<ColorRGBA, ColorHSL, ColorLAB, ColorHCL>(type, value);
}

template <>
ColorHSL convertType(const QueryDataType type,
                     const std::any& value,
                     const bool ignore_null) {
  return convertColorType<ColorHSL, ColorRGBA, ColorLAB, ColorHCL>(type, value);
}

template <>
ColorLAB convertType(const QueryDataType type,
                     const std::any& value,
                     const bool ignore_null) {
  return convertColorType<ColorLAB, ColorRGBA, ColorHSL, ColorHCL>(type, value);
}

template <>
ColorHCL convertType(const QueryDataType type,
                     const std::any& value,
                     const bool ignore_null) {
  return convertColorType<ColorHCL, ColorRGBA, ColorHSL, ColorLAB>(type, value);
}

}  // namespace QueryRenderer
