/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Data/EmbeddedDataUtils.h"

#include "GfxDriver/Colors/ColorHCL.h"
#include "GfxDriver/Colors/ColorHSL.h"
#include "GfxDriver/Colors/ColorLAB.h"
#include "GfxDriver/Colors/ColorRGBA.h"
#include "GfxDriver/Colors/Utils.h"

using ::gfx::ColorHCL;
using ::gfx::ColorHSL;
using ::gfx::ColorLAB;
using ::gfx::ColorRGBA;

namespace QueryRenderer {

template <>
void TDataColumn<ColorRGBA>::push_back(const std::string& val) {
  column_data_->emplace_back(val);
}

template <>
void TDataColumn<ColorRGBA>::initFromRowMajorJSONObj(const JSONLocation& json_loc) {
  RUNTIME_EX_ASSERT(json_loc.isArray(),
                    RapidJSONUtils::createJsonParseError(
                        json_loc, "Row-major data object is not an array."));

  ColorRGBA color;
  for (size_t i = 0; i < json_loc.size(); ++i) {
    const auto array_item_loc = json_loc[i];
    RUNTIME_EX_ASSERT(
        array_item_loc.isObject(),
        RapidJSONUtils::createJsonParseError(
            array_item_loc,
            "Item " + std::to_string(i) +
                "in data array must be an object for row-major-defined data."));

    const auto col_loc = array_item_loc.getMember(column_name);
    RUNTIME_EX_ASSERT(
        col_loc.isValid(),
        RapidJSONUtils::createJsonParseError(
            array_item_loc,
            "column \"" + column_name +
                "\" does not exist in row-major-defined data item " + std::to_string(i)));

    color.initFromCSSString(col_loc.getString());
    column_data_->push_back(color);
  }
}

template <>
void TDataColumn<ColorHSL>::push_back(const std::string& val) {
  column_data_->emplace_back(val);
}

template <>
void TDataColumn<ColorHSL>::initFromRowMajorJSONObj(const JSONLocation& json_loc) {
  RUNTIME_EX_ASSERT(json_loc.isArray(),
                    RapidJSONUtils::createJsonParseError(
                        json_loc, "Row-major data object is not an array."));

  ColorHSL color;
  for (size_t i = 0; i < json_loc.size(); ++i) {
    const auto array_item_loc = json_loc[i];
    RUNTIME_EX_ASSERT(
        array_item_loc.isObject(),
        RapidJSONUtils::createJsonParseError(
            array_item_loc,
            "Item " + std::to_string(i) +
                "in data array must be an object for row-major-defined data."));

    const auto col_loc = array_item_loc.getMember(column_name);
    RUNTIME_EX_ASSERT(
        col_loc.isValid(),
        RapidJSONUtils::createJsonParseError(
            array_item_loc,
            "column \"" + column_name +
                "\" does not exist in row-major-defined data item " + std::to_string(i)));

    color.initFromCSSString(col_loc.getString());

    column_data_->push_back(color);
  }
}

template <>
void TDataColumn<ColorLAB>::push_back(const std::string& val) {
  column_data_->emplace_back(val);
}

template <>
void TDataColumn<ColorLAB>::initFromRowMajorJSONObj(const JSONLocation& json_loc) {
  RUNTIME_EX_ASSERT(json_loc.isArray(),
                    RapidJSONUtils::createJsonParseError(
                        json_loc, "Row-major data object is not an array."));

  ColorLAB color;
  for (size_t i = 0; i < json_loc.size(); ++i) {
    const auto array_item_loc = json_loc[i];
    RUNTIME_EX_ASSERT(
        array_item_loc.isObject(),
        RapidJSONUtils::createJsonParseError(
            array_item_loc,
            "Item " + std::to_string(i) +
                "in data array must be an object for row-major-defined data."));

    const auto col_loc = array_item_loc.getMember(column_name);
    RUNTIME_EX_ASSERT(
        col_loc.isValid(),
        RapidJSONUtils::createJsonParseError(
            array_item_loc,
            "column \"" + column_name +
                "\" does not exist in row-major-defined data item " + std::to_string(i)));

    color.initFromCSSString(col_loc.getString());

    column_data_->push_back(color);
  }
}

template <>
void TDataColumn<ColorHCL>::push_back(const std::string& val) {
  column_data_->emplace_back(val);
}

template <>
void TDataColumn<ColorHCL>::initFromRowMajorJSONObj(const JSONLocation& json_loc) {
  RUNTIME_EX_ASSERT(json_loc.isArray(),
                    RapidJSONUtils::createJsonParseError(
                        json_loc, "Row-major data object is not an array."));

  ColorHCL color;
  for (size_t i = 0; i < json_loc.size(); ++i) {
    const auto array_item_loc = json_loc[i];
    RUNTIME_EX_ASSERT(
        array_item_loc.isObject(),
        RapidJSONUtils::createJsonParseError(
            array_item_loc,
            "Item " + std::to_string(i) +
                "in data array must be an object for row-major-defined data."));

    const auto col_loc = array_item_loc.getMember(column_name);
    RUNTIME_EX_ASSERT(
        col_loc.isValid(),
        RapidJSONUtils::createJsonParseError(
            array_item_loc,
            "column \"" + column_name +
                "\" does not exist in row-major-defined data item " + std::to_string(i)));

    color.initFromCSSString(col_loc.getString());

    column_data_->push_back(color);
  }
}

DataColumnUqPtr create_data_column_from_row_major_obj(const std::string& column_name,
                                                      const JSONLocation& row_item_loc,
                                                      const JSONLocation& array_loc) {
  if (row_item_loc.isInt()) {
    return std::make_unique<TDataColumn<int>>(
        column_name, array_loc, DataColumn::InitType::kRowMajor);
  } else if (row_item_loc.isUint()) {
    return std::make_unique<TDataColumn<unsigned int>>(
        column_name, array_loc, DataColumn::InitType::kRowMajor);
  } else if (row_item_loc.isDouble()) {
    // TODO(croot): How do we properly handle floats?
    return std::make_unique<TDataColumn<double>>(
        column_name, array_loc, DataColumn::InitType::kRowMajor);
  } else if (row_item_loc.isBool()) {
    // TODO(croot): How do we properly handle bools?
    return std::make_unique<TDataColumn<unsigned int>>(
        column_name, array_loc, DataColumn::InitType::kRowMajor);
  } else {
    THROW_RUNTIME_EX(RapidJSONUtils::createJsonParseError(
        row_item_loc,
        "Cannot create data column for column \"" + column_name +
            "\". The JSON data for the column is not supported."));
  }
  return nullptr;
}

DataColumnUqPtr create_color_data_column_from_row_major_obj(
    const std::string& column_name,
    const JSONLocation& row_item_loc,
    const JSONLocation& array_loc) {
  RUNTIME_EX_ASSERT(
      row_item_loc.isString(),
      RapidJSONUtils::createJsonParseError(row_item_loc,
                                           "Cannot create color column \"" + column_name +
                                               "\". Colors must be defined as strings."));

  auto colorType = gfx::getColorTypeFromColorString(row_item_loc.getString());
  switch (colorType) {
    case gfx::ColorType::RGBA:
      return std::make_unique<TDataColumn<ColorRGBA>>(
          column_name, array_loc, DataColumn::InitType::kRowMajor);
      break;
    case gfx::ColorType::HSL:
      return std::make_unique<TDataColumn<ColorHSL>>(
          column_name, array_loc, DataColumn::InitType::kRowMajor);
      break;
    case gfx::ColorType::LAB:
      return std::make_unique<TDataColumn<ColorLAB>>(
          column_name, array_loc, DataColumn::InitType::kRowMajor);
      break;
    case gfx::ColorType::HCL:
      return std::make_unique<TDataColumn<ColorHCL>>(
          column_name, array_loc, DataColumn::InitType::kRowMajor);
      break;
    default:
      THROW_RUNTIME_EX("Color type " + std::to_string(static_cast<int>(colorType)) +
                       " is not a supported data table column type");
      break;
  }
  return nullptr;
}

}  // namespace QueryRenderer
