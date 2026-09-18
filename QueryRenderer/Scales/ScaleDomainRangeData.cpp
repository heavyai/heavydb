/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryRenderer/Scales/ScaleDomainRangeData.h"

#include "GfxDriver/Colors/ColorUnion.h"

using ::gfx::ColorHCL;
using ::gfx::ColorHSL;
using ::gfx::ColorLAB;
using ::gfx::ColorRGBA;
using ::gfx::ColorUnion;
using ::gfx::TypeGLSLShPtr;

namespace QueryRenderer {

/*
 * RGBA specializations
 */
template <>
double ScaleDomainRangeData<ColorRGBA>::getDifference(const double divisor) const {
  THROW_RUNTIME_EX(
      "Mathematical operations on color objects aren't currently supported.");
  return 0.0;
}

template <>
ColorRGBA ScaleDomainRangeData<ColorRGBA>::getDataValueFromJSONObj(
    const JSONLocation& json_loc) {
  CHECK(json_loc.isValid()) << RapidJSONUtils::getPointerPath(json_loc.getPathRef());
  return ColorRGBA(ColorUnion(json_loc.getString()));
}

template <>
std::pair<bool, bool> ScaleDomainRangeData<ColorRGBA>::updateDataFromDataRef(
    const ScaleType type,
    const JSONLocation& data_loc,
    const BaseDataTableShPtr& table) {
  THROW_RUNTIME_EX(RapidJSONUtils::createJsonParseError(
      data_loc, "Color domain/ranges do not currently support data references."));
  return std::make_pair(false, false);
}

template <>
void ScaleDomainRangeData<ColorRGBA>::setFromStringValue(const JSONLocation& str_loc,
                                                         ScaleType type) {
  CHECK(false) << RapidJSONUtils::getPointerPath(str_loc.getPathRef());
}

/*
 * HSL specializations
 */
template <>
double ScaleDomainRangeData<ColorHSL>::getDifference(const double divisor) const {
  THROW_RUNTIME_EX(
      "Mathematical operations on color objects aren't currently supported.");
  return 0.0;
}

template <>
ColorHSL ScaleDomainRangeData<ColorHSL>::getDataValueFromJSONObj(
    const JSONLocation& json_loc) {
  CHECK(json_loc.isValid()) << RapidJSONUtils::getPointerPath(json_loc.getPathRef());
  return ColorHSL(ColorUnion(json_loc.getString()));
}

template <>
std::pair<bool, bool> ScaleDomainRangeData<ColorHSL>::updateDataFromDataRef(
    const ScaleType type,
    const JSONLocation& json_loc,
    const BaseDataTableShPtr& table) {
  THROW_RUNTIME_EX(RapidJSONUtils::createJsonParseError(
      json_loc, "Color domain/ranges do not currently support data references."));
  return std::make_pair(false, false);
}

template <>
void ScaleDomainRangeData<ColorHSL>::setFromStringValue(const JSONLocation& str_loc,
                                                        ScaleType type) {
  CHECK(false) << RapidJSONUtils::getPointerPath(str_loc.getPathRef());
}

/*
 * LAB specializations
 */
template <>
double ScaleDomainRangeData<ColorLAB>::getDifference(const double divisor) const {
  THROW_RUNTIME_EX(
      "Mathematical operations on color objects aren't currently supported.");
  return 0.0;
}

template <>
ColorLAB ScaleDomainRangeData<ColorLAB>::getDataValueFromJSONObj(
    const JSONLocation& json_loc) {
  CHECK(json_loc.isValid()) << RapidJSONUtils::getPointerPath(json_loc.getPathRef());
  return ColorLAB(ColorUnion(json_loc.getString()));
}

template <>
std::pair<bool, bool> ScaleDomainRangeData<ColorLAB>::updateDataFromDataRef(
    const ScaleType type,
    const JSONLocation& data_loc,
    const BaseDataTableShPtr& table) {
  THROW_RUNTIME_EX(RapidJSONUtils::createJsonParseError(
      data_loc, "Color domain/ranges do not currently support data references."));
  return std::make_pair(false, false);
}

template <>
void ScaleDomainRangeData<ColorLAB>::setFromStringValue(const JSONLocation& str_loc,
                                                        ScaleType type) {
  CHECK(false) << RapidJSONUtils::getPointerPath(str_loc.getPathRef());
}

/*
 * HCL specializations
 */
template <>
double ScaleDomainRangeData<ColorHCL>::getDifference(const double divisor) const {
  THROW_RUNTIME_EX(
      "Mathematical operations on color objects aren't currently supported.");
  return 0.0;
}

template <>
ColorHCL ScaleDomainRangeData<ColorHCL>::getDataValueFromJSONObj(
    const JSONLocation& json_loc) {
  CHECK(json_loc.isValid()) << RapidJSONUtils::getPointerPath(json_loc.getPathRef());
  return ColorHCL(ColorUnion(json_loc.getString()));
}

template <>
std::pair<bool, bool> ScaleDomainRangeData<ColorHCL>::updateDataFromDataRef(
    const ScaleType type,
    const JSONLocation& data_loc,
    const BaseDataTableShPtr& table) {
  THROW_RUNTIME_EX(RapidJSONUtils::createJsonParseError(
      data_loc, "Color domain/ranges do not currently support data references."));
  return std::make_pair(false, false);
}

template <>
void ScaleDomainRangeData<ColorHCL>::setFromStringValue(const JSONLocation& str_loc,
                                                        ScaleType type) {
  CHECK(false) << RapidJSONUtils::getPointerPath(str_loc.getPathRef());
}

/*
 * string specializations
 */

template <>
double ScaleDomainRangeData<std::string>::getDifference(const double divisor) const {
  THROW_RUNTIME_EX("Cannot run mathematical operations on string objects.");
  return 0.0;
}

template <>
std::string ScaleDomainRangeData<std::string>::getDataValueFromJSONObj(
    const JSONLocation& json_loc) {
  CHECK(json_loc.isValid()) << RapidJSONUtils::getPointerPath(json_loc.getPathRef());
  return json_loc.getString();
}

template <>
std::vector<std::string> ScaleDomainRangeData<std::string>::getDataFromEmbeddedDataRef(
    const std::vector<std::string>& column_names,
    BaseDataTable* data_ref) {
  throw std::runtime_error(
      "String domain/ranges do not support embedded data references");
  return {};
}

template <>
std::vector<std::string> ScaleDomainRangeData<std::string>::getDataFromSourcedDataRef(
    const std::vector<std::string>& column_names,
    BaseDataTable* data_ref) {
  auto source_data = dynamic_cast<QuerySourceDataTable*>(data_ref);
  CHECK(source_data);
  const auto array_col =
      BaseScaleDomainRangeData::validateArrayColsFromDataRef(column_names, source_data);
  if (!array_col.size()) {
    throw std::runtime_error("The output \"" + column_names[0] +
                             "\" from source data table \"" + source_data->getName() +
                             "\" is not an array output. String domain/ranges can only "
                             "reference outputs that result in string arrays.");
  }
  return source_data->getTypedVectorData<std::string>(array_col);
}

template <>
void ScaleDomainRangeData<std::string>::setFromStringValue(const JSONLocation& str_loc,
                                                           ScaleType type) {
  CHECK(false) << RapidJSONUtils::getPointerPath(str_loc.getPathRef());
}

}  // namespace QueryRenderer
