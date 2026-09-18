/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Scales/ScaleFactory.h"

#include "QueryRenderer/Scales/Scale.h"
#include "QueryRenderer/Scales/Utils.h"

// Scale type instantiation
#define ENABLE_ORDINAL 1
#define ENABLE_QUANTITATIVE 1
#define ENABLE_QUANTIZE 1
#define ENABLE_THRESHOLD 1

// Domain type instantiation
#define ENABLE_NUMERIC_TYPES 1
#define ENABLE_COLOR_TYPES 1
#define ENABLE_STRING_TYPES 1
#define ENABLE_OTHER_TYPES 1  // bool, shape enum

// Single type: ENABLE_NUMERIC_TYPES 1, all other ENABLE_*_TYPES 0
// set this to 1 which will then only enable UINT for domain and range
#define ENABLE_SINGLE_TYPE 0

#if ENABLE_ORDINAL
#include "QueryRenderer/Scales/OrdinalScale.h"
#endif
#if ENABLE_QUANTITATIVE
#include "QueryRenderer/Scales/QuantitativeScale.h"
#endif
#if ENABLE_QUANTIZE
#include "QueryRenderer/Scales/QuantizeScale.h"
#endif
#if ENABLE_THRESHOLD
#include "QueryRenderer/Scales/ThresholdScale.h"
#endif

#include "GfxDriver/Colors/ColorHCL.h"
#include "GfxDriver/Colors/ColorHSL.h"
#include "GfxDriver/Colors/ColorLAB.h"
#include "GfxDriver/Colors/ColorRGBA.h"
#include "GfxDriver/Colors/Utils.h"

#include <type_traits>

namespace QueryRenderer {

using ::gfx::ColorHCL;
using ::gfx::ColorHSL;
using ::gfx::ColorLAB;
using ::gfx::ColorRGBA;

namespace {
template <typename DomainType, typename RangeType>
ScaleShPtr createScalePtr(
    const JSONLocation& json_loc,
    QueryRendererContext& ctx,
    const QueryDataType domain_type,
    const QueryDataType range_type,
    const std::string& scale_name,
    const ScaleType scale_type,
    const ScaleInterpType interp_type = ScaleInterpType::kUndefined) {
  ScaleShPtr base_scale_ptr =
      std::make_shared<BaseScale>(json_loc, ctx, scale_name, scale_type);
  CHECK(base_scale_ptr);
  std::unique_ptr<ScaleImplBase> scale_impl_ptr;
  switch (scale_type) {
#if ENABLE_QUANTITATIVE
    case ScaleType::kLinear:
      scale_impl_ptr = std::make_unique<QuantitativeScale<DomainType, RangeType>>(
          json_loc, ctx, *base_scale_ptr.get(), domain_type, range_type, interp_type);
      break;

    // log, pow, and sqrt only supported for double domain types
    // TODO: support float?
    // NOTE: DomainType should have been validated before hand (scb:??)
    case ScaleType::kLog:
    case ScaleType::kPow:
    case ScaleType::kSqrt: {
      // TODO(croot): support float?
      bool is_valid = std::is_same_v<DomainType, double>;
      CHECK(is_valid);
      scale_impl_ptr = std::make_unique<QuantitativeScale<double, RangeType>>(
          json_loc, ctx, *base_scale_ptr.get(), domain_type, range_type, interp_type);
    }
#endif
    break;

    case ScaleType::kOrdinal:
#if ENABLE_ORDINAL
      scale_impl_ptr = std::make_unique<OrdinalScale<DomainType, RangeType>>(
          json_loc, ctx, *base_scale_ptr.get(), domain_type, range_type);
#endif
      break;

    case ScaleType::kQuantize:
#if ENABLE_QUANTIZE
      scale_impl_ptr = std::make_unique<QuantizeScale<DomainType, RangeType>>(
          json_loc, ctx, *base_scale_ptr.get(), domain_type, range_type);
#endif
      break;

    case ScaleType::kThreshold:
#if ENABLE_THRESHOLD
      scale_impl_ptr = std::make_unique<ThresholdScale<DomainType, RangeType>>(
          json_loc, ctx, *base_scale_ptr.get(), domain_type, range_type);
#endif
      break;

    default:
      THROW_RUNTIME_EX(RapidJSONUtils::createJsonParseError(
          json_loc, "Scale type " + to_string(scale_type) + " is unsupported."));
  }

  CHECK(scale_impl_ptr);
  base_scale_ptr->setScaleImpl(std::move(scale_impl_ptr));

  ScaleDomainRangeDataUqPtr domain_data_ptr =
      std::make_unique<ScaleDomainRangeData<DomainType>>(
          ctx, true, "domain", domain_type, false);
  CHECK(domain_data_ptr);
  ScaleDomainRangeDataUqPtr range_data_ptr =
      std::make_unique<ScaleDomainRangeData<RangeType>>(
          ctx, false, "range", range_type, true);
  CHECK(range_data_ptr);

  // TODO(scb): move ownership to scale_impl_ptr?
  base_scale_ptr->setDomainData(std::move(domain_data_ptr));
  base_scale_ptr->setRangeData(std::move(range_data_ptr));

  base_scale_ptr->updateFromJSONObj(json_loc);

  return base_scale_ptr;
}

template <typename DomainType>
ScaleShPtr createRangeColorScalePtr(const JSONLocation& json_loc,
                                    QueryRendererContext& ctx,
                                    const QueryDataType domain_type,
                                    const QueryDataType range_type,
                                    const std::string& scale_name,
                                    ScaleType scale_type) {
  CHECK(range_type == QueryDataType::COLOR);
  auto const [color_type, interp_type] = getScaleRangeColorTypeFromJSONObj(json_loc);

  switch (color_type) {
    case gfx::ColorType::RGBA:
      return createScalePtr<DomainType, ColorRGBA>(
          json_loc, ctx, domain_type, range_type, scale_name, scale_type, interp_type);
    case gfx::ColorType::HSL:
      return createScalePtr<DomainType, ColorHSL>(
          json_loc, ctx, domain_type, range_type, scale_name, scale_type, interp_type);
    case gfx::ColorType::LAB:
      return createScalePtr<DomainType, ColorLAB>(
          json_loc, ctx, domain_type, range_type, scale_name, scale_type, interp_type);
    case gfx::ColorType::HCL:
      return createScalePtr<DomainType, ColorHCL>(
          json_loc, ctx, domain_type, range_type, scale_name, scale_type, interp_type);
    default:
      THROW_RUNTIME_EX(RapidJSONUtils::createJsonParseError(
          json_loc,
          "Unsupported color type: " + std::to_string(static_cast<int>(color_type)) +
              ". Cannot create scale with color range."));
  }
  return nullptr;
}

template <typename DomainType>
ScaleShPtr createRangeSymbolShapeEnumPtr(const JSONLocation& json_loc,
                                         QueryRendererContext& ctx,
                                         const QueryDataType domain_type,
                                         const QueryDataType range_type,
                                         const std::string& scale_name,
                                         ScaleType scale_type) {
  CHECK(range_type == QueryDataType::SYMBOL_SHAPE_ENUM);
  RUNTIME_EX_ASSERT(
      scale_type == ScaleType::kOrdinal || scale_type == ScaleType::kThreshold,
      RapidJSONUtils::createJsonParseError(json_loc,
                                           "Symbol shapes can only be used as the range "
                                           "for ordinal or threshold scales."));
  return createScalePtr<DomainType, unsigned int>(
      json_loc, ctx, domain_type, range_type, scale_name, scale_type);
}

template <typename RangeType>
ScaleShPtr createDomainColorScalePtr(const JSONLocation& json_loc,
                                     QueryRendererContext& ctx,
                                     const QueryDataType domain_type,
                                     const QueryDataType range_type,
                                     const std::string& scale_name,
                                     ScaleType scale_type) {
  CHECK(domain_type == QueryDataType::COLOR);
  const auto domain_loc = json_loc.getMember(JSONSchema_v1::Scales::kDomainProp);
  CHECK(domain_loc.isValid() && domain_loc.isArray() && domain_loc.size());

  const auto domain_item_loc = domain_loc[0];
  CHECK(domain_loc.isString());
  auto color_type = gfx::getColorTypeFromColorString(domain_loc.getString());
  switch (color_type) {
    case gfx::ColorType::RGBA:
      return createScalePtr<ColorRGBA, RangeType>(
          json_loc, ctx, domain_type, range_type, scale_name, scale_type);
    case gfx::ColorType::HSL:
      return createScalePtr<ColorHSL, RangeType>(
          json_loc, ctx, domain_type, range_type, scale_name, scale_type);
    case gfx::ColorType::LAB:
      return createScalePtr<ColorLAB, RangeType>(
          json_loc, ctx, domain_type, range_type, scale_name, scale_type);
    case gfx::ColorType::HCL:
      return createScalePtr<ColorHCL, RangeType>(
          json_loc, ctx, domain_type, range_type, scale_name, scale_type);
    default:
      THROW_RUNTIME_EX(RapidJSONUtils::createJsonParseError(
          domain_item_loc,
          "Unsupported color type: " + std::to_string(static_cast<int>(color_type)) +
              ". Cannot create scale with color range."));
  }
  return nullptr;
}

}  // namespace

ScaleShPtr createScale(const JSONLocation& json_loc,
                       QueryRendererContext& ctx,
                       const std::string& name,
                       ScaleType type) {
  std::string scale_name(name);
  if (!scale_name.length()) {
    scale_name = getScaleNameFromJSONObj(json_loc);
  }

  RUNTIME_EX_ASSERT(scale_name.length() > 0,
                    RapidJSONUtils::createJsonParseError(
                        json_loc, "Scales must have a \"name\" property"));

  ScaleType scale_type(type);
  if (scale_type == ScaleType::kUndefined) {
    scale_type = getScaleTypeFromJSONObj(json_loc);
  }

  RUNTIME_EX_ASSERT(scale_type != ScaleType::kUndefined,
                    RapidJSONUtils::createJsonParseError(
                        json_loc, "Scale type for \"" + scale_name + "\" is undefined."));

  QueryDataType domain_type =
      getScaleDomainDataTypeFromJSONObj(json_loc, ctx, scale_type);
  QueryDataType range_type = getScaleRangeDataTypeFromJSONObj(json_loc, ctx, scale_type);

  RUNTIME_EX_ASSERT(
      isScaleDomainCompatible(scale_type, domain_type),
      RapidJSONUtils::createJsonParseError(json_loc,
                                           "Domain type " + to_string(domain_type) +
                                               " is not supported for a " +
                                               to_string(scale_type) + " scale."));

  RUNTIME_EX_ASSERT(
      isScaleRangeCompatible(scale_type, range_type),
      RapidJSONUtils::createJsonParseError(json_loc,
                                           "Range type " + to_string(domain_type) +
                                               " is not supported for a " +
                                               to_string(scale_type) + " scale."));

  switch (domain_type) {
#if ENABLE_NUMERIC_TYPES

    case QueryDataType::UINT:
      switch (range_type) {
        case QueryDataType::UINT:
          return createScalePtr<unsigned int, unsigned int>(
              json_loc, ctx, domain_type, range_type, scale_name, scale_type);
#if !ENABLE_SINGLE_TYPE
        case QueryDataType::INT:
          return createScalePtr<unsigned int, int>(
              json_loc, ctx, domain_type, range_type, scale_name, scale_type);
        case QueryDataType::FLOAT:
          return createScalePtr<unsigned int, float>(
              json_loc, ctx, domain_type, range_type, scale_name, scale_type);
        case QueryDataType::UINT64:
          return createScalePtr<unsigned int, uint64_t>(
              json_loc, ctx, domain_type, range_type, scale_name, scale_type);
        case QueryDataType::INT64:
          return createScalePtr<unsigned int, int64_t>(
              json_loc, ctx, domain_type, range_type, scale_name, scale_type);
        case QueryDataType::DOUBLE:
          return createScalePtr<unsigned int, double>(
              json_loc, ctx, domain_type, range_type, scale_name, scale_type);
        case QueryDataType::COLOR:
          return createRangeColorScalePtr<unsigned int>(
              json_loc, ctx, domain_type, range_type, scale_name, scale_type);
        case QueryDataType::SYMBOL_SHAPE_ENUM:
          return createRangeSymbolShapeEnumPtr<unsigned int>(
              json_loc, ctx, domain_type, range_type, scale_name, scale_type);
#endif
        default:
          THROW_RUNTIME_EX(RapidJSONUtils::createJsonParseError(
              json_loc,
              "Range type is unsupported: " +
                  std::to_string(static_cast<int>(range_type))));
      }
#if !ENABLE_SINGLE_TYPE
    case QueryDataType::INT:
      switch (range_type) {
        case QueryDataType::UINT:
          return createScalePtr<int, unsigned int>(
              json_loc, ctx, domain_type, range_type, scale_name, scale_type);
        case QueryDataType::INT:
          return createScalePtr<int, int>(
              json_loc, ctx, domain_type, range_type, scale_name, scale_type);
        case QueryDataType::FLOAT:
          return createScalePtr<int, float>(
              json_loc, ctx, domain_type, range_type, scale_name, scale_type);
        case QueryDataType::UINT64:
          return createScalePtr<int, uint64_t>(
              json_loc, ctx, domain_type, range_type, scale_name, scale_type);
        case QueryDataType::INT64:
          return createScalePtr<int, int64_t>(
              json_loc, ctx, domain_type, range_type, scale_name, scale_type);
        case QueryDataType::DOUBLE:
          return createScalePtr<int, double>(
              json_loc, ctx, domain_type, range_type, scale_name, scale_type);
        case QueryDataType::COLOR:
          return createRangeColorScalePtr<int>(
              json_loc, ctx, domain_type, range_type, scale_name, scale_type);
        case QueryDataType::SYMBOL_SHAPE_ENUM:
          return createRangeSymbolShapeEnumPtr<int>(
              json_loc, ctx, domain_type, range_type, scale_name, scale_type);
        default:
          THROW_RUNTIME_EX(RapidJSONUtils::createJsonParseError(
              json_loc,
              "Range type is unsupported: " +
                  std::to_string(static_cast<int>(range_type))));
      }
    case QueryDataType::UINT64:
      switch (range_type) {
        case QueryDataType::UINT:
          return createScalePtr<uint64_t, unsigned int>(
              json_loc, ctx, domain_type, range_type, scale_name, scale_type);
        case QueryDataType::INT:
          return createScalePtr<uint64_t, int>(
              json_loc, ctx, domain_type, range_type, scale_name, scale_type);
        case QueryDataType::FLOAT:
          return createScalePtr<uint64_t, float>(
              json_loc, ctx, domain_type, range_type, scale_name, scale_type);
        case QueryDataType::UINT64:
          return createScalePtr<uint64_t, uint64_t>(
              json_loc, ctx, domain_type, range_type, scale_name, scale_type);
        case QueryDataType::INT64:
          return createScalePtr<uint64_t, int64_t>(
              json_loc, ctx, domain_type, range_type, scale_name, scale_type);
        case QueryDataType::DOUBLE:
          return createScalePtr<uint64_t, double>(
              json_loc, ctx, domain_type, range_type, scale_name, scale_type);
        case QueryDataType::COLOR:
          return createRangeColorScalePtr<uint64_t>(
              json_loc, ctx, domain_type, range_type, scale_name, scale_type);
        case QueryDataType::SYMBOL_SHAPE_ENUM:
          return createRangeSymbolShapeEnumPtr<uint64_t>(
              json_loc, ctx, domain_type, range_type, scale_name, scale_type);
        default:
          THROW_RUNTIME_EX(RapidJSONUtils::createJsonParseError(
              json_loc,
              "Range type is unsupported: " +
                  std::to_string(static_cast<int>(range_type))));
      }
    case QueryDataType::INT64:
      switch (range_type) {
        case QueryDataType::UINT:
          return createScalePtr<int64_t, unsigned int>(
              json_loc, ctx, domain_type, range_type, scale_name, scale_type);
        case QueryDataType::INT:
          return createScalePtr<int64_t, int>(
              json_loc, ctx, domain_type, range_type, scale_name, scale_type);
        case QueryDataType::FLOAT:
          return createScalePtr<int64_t, float>(
              json_loc, ctx, domain_type, range_type, scale_name, scale_type);
        case QueryDataType::UINT64:
          return createScalePtr<int64_t, uint64_t>(
              json_loc, ctx, domain_type, range_type, scale_name, scale_type);
        case QueryDataType::INT64:
          return createScalePtr<int64_t, int64_t>(
              json_loc, ctx, domain_type, range_type, scale_name, scale_type);
        case QueryDataType::DOUBLE:
          return createScalePtr<int64_t, double>(
              json_loc, ctx, domain_type, range_type, scale_name, scale_type);
        case QueryDataType::COLOR:
          return createRangeColorScalePtr<int64_t>(
              json_loc, ctx, domain_type, range_type, scale_name, scale_type);
        case QueryDataType::SYMBOL_SHAPE_ENUM:
          return createRangeSymbolShapeEnumPtr<int64_t>(
              json_loc, ctx, domain_type, range_type, scale_name, scale_type);
        default:
          THROW_RUNTIME_EX(RapidJSONUtils::createJsonParseError(
              json_loc,
              "Range type is unsupported: " +
                  std::to_string(static_cast<int>(range_type))));
      }
    case QueryDataType::FLOAT:
      switch (range_type) {
        case QueryDataType::UINT:
          return createScalePtr<float, unsigned int>(
              json_loc, ctx, domain_type, range_type, scale_name, scale_type);
        case QueryDataType::INT:
          return createScalePtr<float, int>(
              json_loc, ctx, domain_type, range_type, scale_name, scale_type);
        case QueryDataType::FLOAT:
          return createScalePtr<float, float>(
              json_loc, ctx, domain_type, range_type, scale_name, scale_type);
        case QueryDataType::UINT64:
          return createScalePtr<float, uint64_t>(
              json_loc, ctx, domain_type, range_type, scale_name, scale_type);
        case QueryDataType::INT64:
          return createScalePtr<float, int64_t>(
              json_loc, ctx, domain_type, range_type, scale_name, scale_type);
        case QueryDataType::DOUBLE:
          return createScalePtr<float, double>(
              json_loc, ctx, domain_type, range_type, scale_name, scale_type);
        case QueryDataType::COLOR:
          return createRangeColorScalePtr<float>(
              json_loc, ctx, domain_type, range_type, scale_name, scale_type);
        case QueryDataType::SYMBOL_SHAPE_ENUM:
          return createRangeSymbolShapeEnumPtr<float>(
              json_loc, ctx, domain_type, range_type, scale_name, scale_type);
        default:
          THROW_RUNTIME_EX(RapidJSONUtils::createJsonParseError(
              json_loc,
              "Range type is unsupported: " +
                  std::to_string(static_cast<int>(range_type))));
      }
#endif

    case QueryDataType::DOUBLE:
      switch (range_type) {
        case QueryDataType::UINT:
          return createScalePtr<double, unsigned int>(
              json_loc, ctx, domain_type, range_type, scale_name, scale_type);
        case QueryDataType::INT:
          return createScalePtr<double, int>(
              json_loc, ctx, domain_type, range_type, scale_name, scale_type);
        case QueryDataType::FLOAT:
          return createScalePtr<double, float>(
              json_loc, ctx, domain_type, range_type, scale_name, scale_type);
        case QueryDataType::UINT64:
          return createScalePtr<double, uint64_t>(
              json_loc, ctx, domain_type, range_type, scale_name, scale_type);
        case QueryDataType::INT64:
          return createScalePtr<double, int64_t>(
              json_loc, ctx, domain_type, range_type, scale_name, scale_type);
        case QueryDataType::DOUBLE:
          return createScalePtr<double, double>(
              json_loc, ctx, domain_type, range_type, scale_name, scale_type);
        case QueryDataType::COLOR:
          return createRangeColorScalePtr<double>(
              json_loc, ctx, domain_type, range_type, scale_name, scale_type);
        case QueryDataType::SYMBOL_SHAPE_ENUM:
          return createRangeSymbolShapeEnumPtr<double>(
              json_loc, ctx, domain_type, range_type, scale_name, scale_type);
        default:
          THROW_RUNTIME_EX(RapidJSONUtils::createJsonParseError(
              json_loc,
              "Range type is unsupported: " +
                  std::to_string(static_cast<int>(range_type))));
      }
#endif

#if ENABLE_COLOR_TYPES
    case QueryDataType::COLOR:
      switch (range_type) {
        case QueryDataType::UINT:
          return createDomainColorScalePtr<unsigned int>(
              json_loc, ctx, domain_type, range_type, scale_name, scale_type);
        case QueryDataType::INT:
          return createDomainColorScalePtr<int>(
              json_loc, ctx, domain_type, range_type, scale_name, scale_type);
        case QueryDataType::FLOAT:
          return createDomainColorScalePtr<float>(
              json_loc, ctx, domain_type, range_type, scale_name, scale_type);
        case QueryDataType::UINT64:
          return createDomainColorScalePtr<uint64_t>(
              json_loc, ctx, domain_type, range_type, scale_name, scale_type);
        case QueryDataType::INT64:
          return createDomainColorScalePtr<int64_t>(
              json_loc, ctx, domain_type, range_type, scale_name, scale_type);
        case QueryDataType::DOUBLE:
          return createDomainColorScalePtr<double>(
              json_loc, ctx, domain_type, range_type, scale_name, scale_type);
        case QueryDataType::COLOR:
          return createRangeColorScalePtr<ColorRGBA>(
              json_loc, ctx, domain_type, range_type, scale_name, scale_type);
        case QueryDataType::SYMBOL_SHAPE_ENUM:
          RUNTIME_EX_ASSERT(
              scale_type == ScaleType::kOrdinal,
              RapidJSONUtils::createJsonParseError(
                  json_loc,
                  "Symbol shapes can only be used as the range for ordinal scales."));
          return createDomainColorScalePtr<unsigned int>(
              json_loc, ctx, domain_type, range_type, scale_name, scale_type);
        default:
          THROW_RUNTIME_EX(RapidJSONUtils::createJsonParseError(
              json_loc,
              "Range type is unsupported: " +
                  std::to_string(static_cast<int>(range_type))));
      }
#endif
#if ENABLE_STRING_TYPES
    case QueryDataType::STRING:
      switch (range_type) {
        case QueryDataType::UINT:
          return createScalePtr<std::string, unsigned int>(
              json_loc, ctx, domain_type, range_type, scale_name, scale_type);
        case QueryDataType::INT:
          return createScalePtr<std::string, int>(
              json_loc, ctx, domain_type, range_type, scale_name, scale_type);
        case QueryDataType::FLOAT:
          return createScalePtr<std::string, float>(
              json_loc, ctx, domain_type, range_type, scale_name, scale_type);
        case QueryDataType::UINT64:
          return createScalePtr<std::string, uint64_t>(
              json_loc, ctx, domain_type, range_type, scale_name, scale_type);
        case QueryDataType::INT64:
          return createScalePtr<std::string, int64_t>(
              json_loc, ctx, domain_type, range_type, scale_name, scale_type);
        case QueryDataType::DOUBLE:
          return createScalePtr<std::string, double>(
              json_loc, ctx, domain_type, range_type, scale_name, scale_type);
        case QueryDataType::COLOR:
          return createRangeColorScalePtr<std::string>(
              json_loc, ctx, domain_type, range_type, scale_name, scale_type);
        case QueryDataType::SYMBOL_SHAPE_ENUM:
          return createRangeSymbolShapeEnumPtr<std::string>(
              json_loc, ctx, domain_type, range_type, scale_name, scale_type);
        default:
          THROW_RUNTIME_EX(RapidJSONUtils::createJsonParseError(
              json_loc,
              "Range type is unsupported: " +
                  std::to_string(static_cast<int>(range_type))));
      }
#endif
#if ENABLE_OTHER_TYPES
    case QueryDataType::BOOL:
      // NOTE: using unsigned ints for booleans. Doing this shouldn't be too wasteful
      // memory-wise as the domains/ranges of any boolean type scale should be no more
      // than two. Ultimately using unsigned ints for 2 reasons:
      // 1) the inability to get a bool pointer for std::vector<bool>
      //    (i.e. bool* boolptr = &boolvec[0]; // this results in errors)
      // 2) although glsl supports a bool type, setting a uniform boolean array
      //    requires that the booleans be stored in 32-bit registers when passed as
      //    uniforms
      // TODO(croot): one way to possibly avert doing this is to use a
      // ScaleDomainRangeData class template specialization for booleans
      switch (range_type) {
        case QueryDataType::UINT:
          return createScalePtr<unsigned int, unsigned int>(
              json_loc, ctx, domain_type, range_type, scale_name, scale_type);
        case QueryDataType::INT:
          return createScalePtr<unsigned int, int>(
              json_loc, ctx, domain_type, range_type, scale_name, scale_type);
        case QueryDataType::FLOAT:
          return createScalePtr<unsigned int, float>(
              json_loc, ctx, domain_type, range_type, scale_name, scale_type);
        case QueryDataType::UINT64:
          return createScalePtr<unsigned int, uint64_t>(
              json_loc, ctx, domain_type, range_type, scale_name, scale_type);
        case QueryDataType::INT64:
          return createScalePtr<unsigned int, int64_t>(
              json_loc, ctx, domain_type, range_type, scale_name, scale_type);
        case QueryDataType::DOUBLE:
          return createScalePtr<unsigned int, double>(
              json_loc, ctx, domain_type, range_type, scale_name, scale_type);
        case QueryDataType::COLOR:
          return createRangeColorScalePtr<unsigned int>(
              json_loc, ctx, domain_type, range_type, scale_name, scale_type);
        case QueryDataType::SYMBOL_SHAPE_ENUM:
          return createRangeSymbolShapeEnumPtr<unsigned int>(
              json_loc, ctx, domain_type, range_type, scale_name, scale_type);
        default:
          THROW_RUNTIME_EX(RapidJSONUtils::createJsonParseError(
              json_loc,
              "Range type is unsupported: " +
                  std::to_string(static_cast<int>(range_type))));
      }
    case QueryDataType::SYMBOL_SHAPE_ENUM:
      THROW_RUNTIME_EX(RapidJSONUtils::createJsonParseError(
          json_loc, "Symbol shapes are not supported as domain values."));
#endif
    default:
      THROW_RUNTIME_EX(
          RapidJSONUtils::createJsonParseError(json_loc, "Domain type is unsupported."));
  }
  CHECK(false);
  return nullptr;
}

}  // namespace QueryRenderer
