/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>

#include <boost/lexical_cast.hpp>

#include "QueryRenderer/Data/Types.h"
#include "QueryRenderer/Utils/RapidJSONUtils.h"

namespace QueryRenderer {

enum class EmbeddedDataVboType { kSequential, kInterleaved };

//
// Embedded column data creation
//
struct TypelessColumnData {
  void* data;
  size_t num_items;
  size_t num_bytes_per_item;
};

class DataColumn {
 public:
  enum class InitType { kRowMajor, kColumnMajor };

  std::string column_name;

  explicit DataColumn(const std::string& name) : column_name(name) {}
  virtual ~DataColumn() {}

  virtual int size() = 0;

  virtual QueryDataType getColumnType() = 0;
  virtual TypelessColumnData getTypelessColumnData() = 0;
  virtual void push_back(const std::string& val) = 0;

  // for multi_index_container tag
  struct ColumnName {};
};

using DataColumnUqPtr = std::unique_ptr<DataColumn>;
using DataColumnShPtr = std::shared_ptr<DataColumn>;

template <typename T>
class TDataColumn : public DataColumn {
 public:
  TDataColumn(const std::string& name, int size = 0)
      : DataColumn(name), column_data_(new std::vector<T>(size)) {}
  TDataColumn(const std::string& name, const JSONLocation& json_loc, InitType init_type)
      : DataColumn(name), column_data_(new std::vector<T>()) {
    if (init_type == DataColumn::InitType::kRowMajor) {
      initFromRowMajorJSONObj(json_loc);
    } else {
      initFromColMajorJSONObj(json_loc);
    }
  }
  ~TDataColumn() override {}

  T& operator[](unsigned int i) { return (*column_data_)[i]; }

  void push_back(const std::string& val) override {
    // TODO(croot): this would throw a boost::bad_lexical_cast error
    // if the conversion can't be done.. I may need to throw
    // a MapD-compliant exception
    column_data_->push_back(boost::lexical_cast<T>(val));
  }

  QueryDataType getColumnType() override {
    return TypeToQueryDataTypeSelector<T>::getQueryDataType();
  }

  std::shared_ptr<std::vector<T>> getColumnData() { return column_data_; }
  TypelessColumnData getTypelessColumnData() override {
    return TypelessColumnData(
        {static_cast<void*>(&(*column_data_)[0]), column_data_->size(), sizeof(T)});
  };

  std::pair<T, T> getExtrema() {
    auto result = std::minmax_element(column_data_->begin(), column_data_->end());

    RUNTIME_EX_ASSERT(
        result.first != column_data_->end() && result.second != column_data_->end(),
        std::string(*this) + " getExtrema(): cannot find the extrema of the column.");

    return std::make_pair(*result.first, *result.second);
  }

  int size() override { return column_data_->size(); }

  operator std::string() const {
    return "TDataColumn<" + std::string(typeid(T).name()) +
           ">(column name: " + column_name + ")";
  }

 private:
  std::shared_ptr<std::vector<T>> column_data_;

  void initFromRowMajorJSONObj(const JSONLocation& json_loc) {
    RUNTIME_EX_ASSERT(json_loc.isArray(),
                      RapidJSONUtils::createJsonParseError(
                          json_loc, "Row-major data object is not an array."));

    for (size_t i = 0; i < json_loc.size(); ++i) {
      const auto array_item_loc = json_loc[i];
      RUNTIME_EX_ASSERT(
          array_item_loc.isObject(),
          RapidJSONUtils::createJsonParseError(
              array_item_loc,
              "Item " + std::to_string(i) +
                  "in data array must be an object for row-major-defined data."));
      const auto column_loc = array_item_loc.getMember(column_name);
      RUNTIME_EX_ASSERT(column_loc.isValid(),
                        RapidJSONUtils::createJsonParseError(
                            array_item_loc,
                            "column \"" + column_name +
                                "\" does not exist in row-major-defined data item " +
                                std::to_string(i)));

      column_data_->push_back(RapidJSONUtils::getNumValFromJSONObj<T>(column_loc));
    }
  }

  void initFromColMajorJSONObj(const JSONLocation& json_loc) {
    THROW_RUNTIME_EX("Column-major data is not yet supported.");
  }
};

template <>
void TDataColumn<gfx::ColorRGBA>::push_back(const std::string& val);

template <>
void TDataColumn<gfx::ColorRGBA>::initFromRowMajorJSONObj(const JSONLocation& json_loc);

template <>
void TDataColumn<gfx::ColorHSL>::push_back(const std::string& val);

template <>
void TDataColumn<gfx::ColorHSL>::initFromRowMajorJSONObj(const JSONLocation& json_loc);

template <>
void TDataColumn<gfx::ColorLAB>::push_back(const std::string& val);

template <>
void TDataColumn<gfx::ColorLAB>::initFromRowMajorJSONObj(const JSONLocation& json_loc);

template <>
void TDataColumn<gfx::ColorHCL>::push_back(const std::string& val);

template <>
void TDataColumn<gfx::ColorHCL>::initFromRowMajorJSONObj(const JSONLocation& json_loc);

const std::string kDefaultIdColumnName = "rowid";

DataColumnUqPtr create_data_column_from_row_major_obj(const std::string& column_name,
                                                      const JSONLocation& row_item_loc,
                                                      const JSONLocation& array_loc);

DataColumnUqPtr create_color_data_column_from_row_major_obj(
    const std::string& column_name,
    const JSONLocation& row_item_loc,
    const JSONLocation& array_loc);

}  // namespace QueryRenderer
