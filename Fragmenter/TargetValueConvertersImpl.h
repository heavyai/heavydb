/*
 * Copyright 2018, OmniSci, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef TARGET_VALUE_CONVERTERS_IMPL_H_
#define TARGET_VALUE_CONVERTERS_IMPL_H_

#include "TargetValueConverters.h"

namespace Importer_NS {
std::vector<uint8_t> compress_coords(std::vector<double>& coords, const SQLTypeInfo& ti);
}  // namespace Importer_NS

template <typename SOURCE_TYPE, typename TARGET_TYPE>
struct NumericValueConverter : public TargetValueConverter {
  using ColumnDataPtr = std::unique_ptr<TARGET_TYPE, CheckedMallocDeleter<TARGET_TYPE>>;

  ColumnDataPtr column_data_;
  TARGET_TYPE null_value_;
  SOURCE_TYPE null_check_value_;
  bool do_null_check_;

  boost_variant_accessor<SOURCE_TYPE> SOURCE_TYPE_ACCESSOR;

  NumericValueConverter(const ColumnDescriptor* cd,
                        size_t num_rows,
                        TARGET_TYPE nullValue,
                        SOURCE_TYPE nullCheckValue,
                        bool doNullCheck)
      : TargetValueConverter(cd)
      , null_value_(nullValue)
      , null_check_value_(nullCheckValue)
      , do_null_check_(doNullCheck) {
    if (num_rows) {
      allocateColumnarData(num_rows);
    }
  }

  virtual ~NumericValueConverter() {}

  auto&& takeColumnarData() { return std::move(column_data_); }

  virtual void allocateColumnarData(size_t num_rows) {
    CHECK(num_rows > 0);
    column_data_ = ColumnDataPtr(
        reinterpret_cast<TARGET_TYPE*>(checked_malloc(num_rows * sizeof(TARGET_TYPE))));
  }

  void convertToColumnarFormat(size_t row, const ScalarTargetValue* scalarValue) {
    auto mapd_p = checked_get<SOURCE_TYPE>(row, scalarValue, SOURCE_TYPE_ACCESSOR);
    auto val = *mapd_p;

    if (do_null_check_ && null_check_value_ == val) {
      column_data_.get()[row] = null_value_;
    } else {
      column_data_.get()[row] = static_cast<TARGET_TYPE>(val);
    }
  }

  virtual void convertToColumnarFormat(size_t row, const TargetValue* value) {
    auto scalarValue =
        checked_get<ScalarTargetValue>(row, value, SCALAR_TARGET_VALUE_ACCESSOR);
    convertToColumnarFormat(row, scalarValue);
  }

  virtual void addDataBlocksToInsertData(Fragmenter_Namespace::InsertData& insertData) {
    DataBlockPtr dataBlock;
    dataBlock.numbersPtr = reinterpret_cast<int8_t*>(column_data_.get());
    insertData.data.push_back(dataBlock);
    insertData.columnIds.push_back(column_descriptor_->columnId);
  }
};

template <typename TARGET_TYPE>
struct DictionaryValueConverter : public NumericValueConverter<int64_t, TARGET_TYPE> {
  const DictDescriptor* source_dict_desc_;
  const DictDescriptor* target_dict_desc_;

  const StringDictionary* source_dict_;

  DictionaryValueConverter(const Catalog_Namespace::Catalog& cat,
                           const ColumnDescriptor* sourceDescriptor,
                           const ColumnDescriptor* targetDescriptor,
                           size_t num_rows,
                           TARGET_TYPE nullValue,
                           int64_t nullCheckValue,
                           bool doNullCheck,
                           bool expect_always_strings)
      : NumericValueConverter<int64_t, TARGET_TYPE>(targetDescriptor,
                                                    num_rows,
                                                    nullValue,
                                                    nullCheckValue,
                                                    doNullCheck) {
    source_dict_desc_ =
        cat.getMetadataForDict(sourceDescriptor->columnType.get_comp_param(), true);
    target_dict_desc_ =
        cat.getMetadataForDict(targetDescriptor->columnType.get_comp_param(), true);

    source_dict_ = nullptr;

    if (!expect_always_strings) {
      if (source_dict_desc_) {
        source_dict_ = source_dict_desc_->stringDict.get();
      }
    }
  }

  virtual ~DictionaryValueConverter() {}

  void convertToColumnarFormatFromDict(size_t row, const ScalarTargetValue* scalarValue) {
    auto mapd_p = checked_get<int64_t>(row, scalarValue, this->SOURCE_TYPE_ACCESSOR);
    auto val = *mapd_p;

    if (this->do_null_check_ && this->null_check_value_ == val) {
      this->column_data_.get()[row] = this->null_value_;
    } else {
      std::string strVal = source_dict_->getString(val);
      auto newVal = target_dict_desc_->stringDict->getOrAdd(strVal);
      this->column_data_.get()[row] = (TARGET_TYPE)newVal;
    }
  }

  virtual void convertToColumnarFormatFromString(size_t row,
                                                 const ScalarTargetValue* scalarValue) {
    auto mapd_p =
        checked_get<NullableString>(row, scalarValue, this->NULLABLE_STRING_ACCESSOR);

    const auto mapd_str_p = checked_get<std::string>(row, mapd_p, this->STRING_ACCESSOR);

    if (this->do_null_check_ && (nullptr == mapd_str_p || *mapd_str_p == "")) {
      this->column_data_.get()[row] = this->null_value_;
    } else {
      auto newVal = target_dict_desc_->stringDict->getOrAdd(*mapd_str_p);
      this->column_data_.get()[row] = (TARGET_TYPE)newVal;
    }
  }

  void convertToColumnarFormat(size_t row, const ScalarTargetValue* scalarValue) {
    if (source_dict_) {
      convertToColumnarFormatFromDict(row, scalarValue);
    } else {
      convertToColumnarFormatFromString(row, scalarValue);
    }
  }

  virtual void convertToColumnarFormat(size_t row, const TargetValue* value) {
    auto scalarValue =
        checked_get<ScalarTargetValue>(row, value, this->SCALAR_TARGET_VALUE_ACCESSOR);

    convertToColumnarFormat(row, scalarValue);
  }
};

struct StringValueConverter : public TargetValueConverter {
  std::unique_ptr<std::vector<std::string>> column_data_;

  StringValueConverter(const ColumnDescriptor* cd, size_t num_rows)
      : TargetValueConverter(cd) {
    if (num_rows) {
      allocateColumnarData(num_rows);
    }
  }

  virtual ~StringValueConverter() {}

  virtual void allocateColumnarData(size_t num_rows) {
    CHECK(num_rows > 0);
    column_data_ = std::make_unique<std::vector<std::string>>(num_rows);
  }

  virtual void convertToColumnarFormat(size_t row, const TargetValue* value) {
    auto scalarValue =
        checked_get<ScalarTargetValue>(row, value, SCALAR_TARGET_VALUE_ACCESSOR);
    auto mapd_p = checked_get<NullableString>(row, scalarValue, NULLABLE_STRING_ACCESSOR);

    const auto mapd_str_p = checked_get<std::string>(row, mapd_p, STRING_ACCESSOR);

    if (nullptr != mapd_str_p) {
      (*column_data_)[row] = *mapd_str_p;
    } else {
      (*column_data_)[row] = "";
    }
  }

  virtual void addDataBlocksToInsertData(Fragmenter_Namespace::InsertData& insertData) {
    DataBlockPtr dataBlock;
    dataBlock.stringsPtr = column_data_.get();
    insertData.data.push_back(dataBlock);
    insertData.columnIds.push_back(column_descriptor_->columnId);
  }
};

struct DateValueConverter : public NumericValueConverter<int64_t, int64_t> {
  DateValueConverter(const ColumnDescriptor* targetDescriptor,
                     size_t num_rows,
                     int64_t nullValue,
                     int64_t nullCheckValue,
                     bool doNullCheck)
      : NumericValueConverter<int64_t, int64_t>(targetDescriptor,
                                                num_rows,
                                                nullValue,
                                                nullCheckValue,
                                                doNullCheck) {}

  virtual ~DateValueConverter() {}

  void convertToColumnarFormat(size_t row, const ScalarTargetValue* scalarValue) {
    auto mapd_p = checked_get<int64_t>(row, scalarValue, this->SOURCE_TYPE_ACCESSOR);
    auto val = *mapd_p;

    if (this->do_null_check_ && this->null_check_value_ == val) {
      this->column_data_.get()[row] = this->null_value_;
    } else {
      this->column_data_.get()[row] = static_cast<int64_t>(val / SECSPERDAY);
    }
  }

  virtual void convertToColumnarFormat(size_t row, const TargetValue* value) {
    auto scalarValue =
        checked_get<ScalarTargetValue>(row, value, this->SCALAR_TARGET_VALUE_ACCESSOR);
    convertToColumnarFormat(row, scalarValue);
  }
};

template <typename ELEMENT_CONVERTER>
struct ArrayValueConverter : public TargetValueConverter {
  std::unique_ptr<std::vector<ArrayDatum>> column_data_;
  std::unique_ptr<ELEMENT_CONVERTER> element_converter_;
  SQLTypeInfo element_type_info_;
  bool do_check_null_;

  boost_variant_accessor<std::vector<ScalarTargetValue>> SCALAR_VECTOR_ACCESSOR;

  ArrayValueConverter(const ColumnDescriptor* cd,
                      size_t num_rows,
                      std::unique_ptr<ELEMENT_CONVERTER> element_converter,
                      bool do_check_null)
      : TargetValueConverter(cd)
      , element_converter_(std::move(element_converter))
      , element_type_info_(cd->columnType.get_elem_type())
      , do_check_null_(do_check_null) {
    if (num_rows) {
      allocateColumnarData(num_rows);
    }
  }

  virtual ~ArrayValueConverter() {}

  virtual void allocateColumnarData(size_t num_rows) {
    CHECK(num_rows > 0);
    column_data_ = std::make_unique<std::vector<ArrayDatum>>(num_rows);
  }

  virtual void convertToColumnarFormat(size_t row, const TargetValue* value) {
    const auto scalarValueVector =
        checked_get<std::vector<ScalarTargetValue>>(row, value, SCALAR_VECTOR_ACCESSOR);

    element_converter_->allocateColumnarData(scalarValueVector->size());

    int elementIndex = 0;
    for (const auto& scalarValue : *scalarValueVector) {
      element_converter_->convertToColumnarFormat(elementIndex++, &scalarValue);
    }

    bool is_null = do_check_null_ && 0 == scalarValueVector->size();
    typename ELEMENT_CONVERTER::ColumnDataPtr ptr =
        element_converter_->takeColumnarData();
    int8_t* arrayData = reinterpret_cast<int8_t*>(ptr.release());
    (*column_data_)[row] = ArrayDatum(
        scalarValueVector->size() * element_type_info_.get_size(), arrayData, is_null);
  }

  virtual void addDataBlocksToInsertData(Fragmenter_Namespace::InsertData& insertData) {
    DataBlockPtr dataBlock;
    dataBlock.arraysPtr = column_data_.get();
    insertData.data.push_back(dataBlock);
    insertData.columnIds.push_back(column_descriptor_->columnId);
  }
};

struct GeoPointValueConverter : public TargetValueConverter {
  const ColumnDescriptor* coords_column_descriptor_;

  std::unique_ptr<std::vector<std::string>> column_data_;
  std::unique_ptr<std::vector<ArrayDatum>> signed_compressed_coords_data_;

  GeoPointValueConverter(const Catalog_Namespace::Catalog& cat,
                         size_t num_rows,
                         const ColumnDescriptor* logicalColumnDescriptor)
      : TargetValueConverter(logicalColumnDescriptor) {
    coords_column_descriptor_ = cat.getMetadataForColumn(
        column_descriptor_->tableId, column_descriptor_->columnId + 1);
    CHECK(coords_column_descriptor_);

    allocateColumnarData(num_rows);
  }

  virtual ~GeoPointValueConverter() {}

  virtual void allocateColumnarData(size_t num_rows) {
    CHECK(num_rows > 0);
    column_data_ = std::make_unique<std::vector<std::string>>(num_rows);
    signed_compressed_coords_data_ = std::make_unique<std::vector<ArrayDatum>>(num_rows);
  }

  boost_variant_accessor<GeoPointTargetValue> GEO_POINT_VALUE_ACCESSOR;

  inline ArrayDatum toCompressedCoords(
      const std::shared_ptr<std::vector<double>>& coords) {
    const auto compressed_coords_vector =
        Importer_NS::compress_coords(*coords, column_descriptor_->columnType);

    uint8_t* compressed_coords_array = reinterpret_cast<uint8_t*>(
        checked_malloc(sizeof(uint8_t) * compressed_coords_vector.size()));
    memcpy(compressed_coords_array,
           &compressed_coords_vector[0],
           compressed_coords_vector.size());

    return ArrayDatum((int)compressed_coords_vector.size(),
                      reinterpret_cast<int8_t*>(compressed_coords_array),
                      false);
  }

  virtual void convertToColumnarFormat(size_t row, const TargetValue* value) {
    auto geoValue = checked_get<GeoTargetValue>(row, value, GEO_TARGET_VALUE_ACCESSOR);
    auto geoPoint =
        checked_get<GeoPointTargetValue>(row, geoValue, GEO_POINT_VALUE_ACCESSOR);

    (*column_data_)[row] = "";
    (*signed_compressed_coords_data_)[row] = toCompressedCoords(geoPoint->coords);
  }

  virtual void addDataBlocksToInsertData(Fragmenter_Namespace::InsertData& insertData) {
    DataBlockPtr logical, coords;

    logical.stringsPtr = column_data_.get();
    coords.arraysPtr = signed_compressed_coords_data_.get();

    insertData.data.emplace_back(logical);
    insertData.columnIds.emplace_back(column_descriptor_->columnId);

    insertData.data.emplace_back(coords);
    insertData.columnIds.emplace_back(coords_column_descriptor_->columnId);
  }
};

inline std::vector<double> compute_bounds_of_coords(
    const std::shared_ptr<std::vector<double>>& coords) {
  std::vector<double> bounds(4);
  constexpr auto DOUBLE_MAX = std::numeric_limits<double>::max();
  constexpr auto DOUBLE_MIN = std::numeric_limits<double>::lowest();
  bounds[0] = DOUBLE_MAX;
  bounds[1] = DOUBLE_MAX;
  bounds[2] = DOUBLE_MIN;
  bounds[3] = DOUBLE_MIN;
  auto size_coords = coords->size();

  for (size_t i = 0; i < size_coords; i += 2) {
    double x = (*coords)[i];
    double y = (*coords)[i + 1];

    bounds[0] = std::min(bounds[0], x);
    bounds[1] = std::min(bounds[1], y);
    bounds[2] = std::max(bounds[2], x);
    bounds[3] = std::max(bounds[3], y);
  }
  return bounds;
}

template <typename ELEM_TYPE>
inline ArrayDatum to_array_datum(const std::vector<ELEM_TYPE>& vector) {
  ELEM_TYPE* array =
      reinterpret_cast<ELEM_TYPE*>(checked_malloc(sizeof(ELEM_TYPE) * vector.size()));
  memcpy(array, vector.data(), vector.size() * sizeof(ELEM_TYPE));

  return ArrayDatum(
      (int)(vector.size() * sizeof(ELEM_TYPE)), reinterpret_cast<int8_t*>(array), false);
}

template <typename ELEM_TYPE>
inline ArrayDatum to_array_datum(const std::shared_ptr<std::vector<ELEM_TYPE>>& vector) {
  return to_array_datum(*vector.get());
}

struct GeoLinestringValueConverter : public GeoPointValueConverter {
  const ColumnDescriptor* bounds_column_descriptor_;

  std::unique_ptr<std::vector<ArrayDatum>> bounds_data_;

  GeoLinestringValueConverter(const Catalog_Namespace::Catalog& cat,
                              size_t num_rows,
                              const ColumnDescriptor* logicalColumnDescriptor)
      : GeoPointValueConverter(cat, num_rows, logicalColumnDescriptor) {
    bounds_column_descriptor_ = cat.getMetadataForColumn(
        column_descriptor_->tableId, column_descriptor_->columnId + 2);
    CHECK(bounds_column_descriptor_);

    allocateColumnarData(num_rows);
  }

  virtual ~GeoLinestringValueConverter() {}

  virtual void allocateColumnarData(size_t num_rows) {
    CHECK(num_rows > 0);
    GeoPointValueConverter::allocateColumnarData(num_rows);
    bounds_data_ = std::make_unique<std::vector<ArrayDatum>>(num_rows);
  }

  boost_variant_accessor<GeoLineStringTargetValue> GEO_LINESTRING_VALUE_ACCESSOR;

  virtual void convertToColumnarFormat(size_t row, const TargetValue* value) {
    auto geoValue = checked_get<GeoTargetValue>(row, value, GEO_TARGET_VALUE_ACCESSOR);
    auto geoLinestring = checked_get<GeoLineStringTargetValue>(
        row, geoValue, GEO_LINESTRING_VALUE_ACCESSOR);

    (*column_data_)[row] = "";
    (*signed_compressed_coords_data_)[row] = toCompressedCoords(geoLinestring->coords);
    auto bounds = compute_bounds_of_coords(geoLinestring->coords);
    (*bounds_data_)[row] = to_array_datum(bounds);
  }

  virtual void addDataBlocksToInsertData(Fragmenter_Namespace::InsertData& insertData) {
    GeoPointValueConverter::addDataBlocksToInsertData(insertData);

    DataBlockPtr bounds;

    bounds.arraysPtr = bounds_data_.get();

    insertData.data.emplace_back(bounds);
    insertData.columnIds.emplace_back(bounds_column_descriptor_->columnId);
  }
};

struct GeoPolygonValueConverter : public GeoPointValueConverter {
  const ColumnDescriptor* ring_sizes_column_descriptor_;
  const ColumnDescriptor* bounds_column_descriptor_;
  const ColumnDescriptor* render_group_column_descriptor_;
  Importer_NS::RenderGroupAnalyzer render_group_analyzer_;

  std::unique_ptr<std::vector<ArrayDatum>> ring_sizes_data_;
  std::unique_ptr<std::vector<ArrayDatum>> bounds_data_;
  std::unique_ptr<int32_t[]> render_group_data_;

  GeoPolygonValueConverter(const Catalog_Namespace::Catalog& cat,
                           size_t num_rows,
                           const ColumnDescriptor* logicalColumnDescriptor)
      : GeoPointValueConverter(cat, num_rows, logicalColumnDescriptor) {
    ring_sizes_column_descriptor_ = cat.getMetadataForColumn(
        column_descriptor_->tableId, column_descriptor_->columnId + 2);
    CHECK(ring_sizes_column_descriptor_);
    bounds_column_descriptor_ = cat.getMetadataForColumn(
        column_descriptor_->tableId, column_descriptor_->columnId + 3);
    CHECK(bounds_column_descriptor_);
    render_group_column_descriptor_ = cat.getMetadataForColumn(
        column_descriptor_->tableId, column_descriptor_->columnId + 4);
    CHECK(render_group_column_descriptor_);

    if (num_rows) {
      allocateColumnarData(num_rows);
    }
  }

  virtual ~GeoPolygonValueConverter() {}

  virtual void allocateColumnarData(size_t num_rows) {
    GeoPointValueConverter::allocateColumnarData(num_rows);
    ring_sizes_data_ = std::make_unique<std::vector<ArrayDatum>>(num_rows);
    bounds_data_ = std::make_unique<std::vector<ArrayDatum>>(num_rows);
    render_group_data_ = std::make_unique<int32_t[]>(num_rows);
  }

  boost_variant_accessor<GeoPolyTargetValue> GEO_POLY_VALUE_ACCESSOR;

  virtual void convertToColumnarFormat(size_t row, const TargetValue* value) {
    auto geoValue = checked_get<GeoTargetValue>(row, value, GEO_TARGET_VALUE_ACCESSOR);
    auto geoPoly =
        checked_get<GeoPolyTargetValue>(row, geoValue, GEO_POLY_VALUE_ACCESSOR);

    (*column_data_)[row] = "";
    (*signed_compressed_coords_data_)[row] = toCompressedCoords(geoPoly->coords);
    (*ring_sizes_data_)[row] = to_array_datum(geoPoly->ring_sizes);
    auto bounds = compute_bounds_of_coords(geoPoly->coords);
    (*bounds_data_)[row] = to_array_datum(bounds);
    render_group_data_[row] =
        render_group_analyzer_.insertBoundsAndReturnRenderGroup(bounds);
  }

  virtual void addDataBlocksToInsertData(Fragmenter_Namespace::InsertData& insertData) {
    GeoPointValueConverter::addDataBlocksToInsertData(insertData);

    DataBlockPtr ringSizes, bounds, renderGroup;

    ringSizes.arraysPtr = ring_sizes_data_.get();
    bounds.arraysPtr = bounds_data_.get();
    renderGroup.numbersPtr = reinterpret_cast<int8_t*>(render_group_data_.get());

    insertData.data.emplace_back(ringSizes);
    insertData.columnIds.emplace_back(ring_sizes_column_descriptor_->columnId);

    insertData.data.emplace_back(bounds);
    insertData.columnIds.emplace_back(bounds_column_descriptor_->columnId);

    insertData.data.emplace_back(renderGroup);
    insertData.columnIds.emplace_back(render_group_column_descriptor_->columnId);
  }
};

struct GeoMultiPolygonValueConverter : public GeoPointValueConverter {
  const ColumnDescriptor* ring_sizes_column_descriptor_;
  const ColumnDescriptor* ring_sizes_solumn_descriptor_;
  const ColumnDescriptor* bounds_column_descriptor_;
  const ColumnDescriptor* render_group_column_descriptor_;
  Importer_NS::RenderGroupAnalyzer render_group_analyzer_;

  std::unique_ptr<std::vector<ArrayDatum>> ring_sizes_data_;
  std::unique_ptr<std::vector<ArrayDatum>> poly_rings_data_;
  std::unique_ptr<std::vector<ArrayDatum>> bounds_data_;
  std::unique_ptr<int32_t[]> render_group_data_;

  GeoMultiPolygonValueConverter(const Catalog_Namespace::Catalog& cat,
                                size_t num_rows,
                                const ColumnDescriptor* logicalColumnDescriptor)
      : GeoPointValueConverter(cat, num_rows, logicalColumnDescriptor) {
    ring_sizes_column_descriptor_ = cat.getMetadataForColumn(
        column_descriptor_->tableId, column_descriptor_->columnId + 2);
    CHECK(ring_sizes_column_descriptor_);
    ring_sizes_solumn_descriptor_ = cat.getMetadataForColumn(
        column_descriptor_->tableId, column_descriptor_->columnId + 3);
    CHECK(ring_sizes_column_descriptor_);
    bounds_column_descriptor_ = cat.getMetadataForColumn(
        column_descriptor_->tableId, column_descriptor_->columnId + 4);
    CHECK(bounds_column_descriptor_);
    render_group_column_descriptor_ = cat.getMetadataForColumn(
        column_descriptor_->tableId, column_descriptor_->columnId + 5);
    CHECK(render_group_column_descriptor_);

    if (num_rows) {
      allocateColumnarData(num_rows);
    }
  }

  virtual ~GeoMultiPolygonValueConverter() {}

  virtual void allocateColumnarData(size_t num_rows) {
    GeoPointValueConverter::allocateColumnarData(num_rows);
    ring_sizes_data_ = std::make_unique<std::vector<ArrayDatum>>(num_rows);
    poly_rings_data_ = std::make_unique<std::vector<ArrayDatum>>(num_rows);
    bounds_data_ = std::make_unique<std::vector<ArrayDatum>>(num_rows);
    render_group_data_ = std::make_unique<int32_t[]>(num_rows);
  }

  boost_variant_accessor<GeoMultiPolyTargetValue> GEO_MULTI_POLY_VALUE_ACCESSOR;

  virtual void convertToColumnarFormat(size_t row, const TargetValue* value) {
    auto geoValue = checked_get<GeoTargetValue>(row, value, GEO_TARGET_VALUE_ACCESSOR);
    auto geoMultiPoly = checked_get<GeoMultiPolyTargetValue>(
        row, geoValue, GEO_MULTI_POLY_VALUE_ACCESSOR);

    (*column_data_)[row] = "";
    (*signed_compressed_coords_data_)[row] = toCompressedCoords(geoMultiPoly->coords);
    (*ring_sizes_data_)[row] = to_array_datum(geoMultiPoly->ring_sizes);
    (*poly_rings_data_)[row] = to_array_datum(geoMultiPoly->poly_rings);
    auto bounds = compute_bounds_of_coords(geoMultiPoly->coords);
    (*bounds_data_)[row] = to_array_datum(bounds);
    render_group_data_[row] =
        render_group_analyzer_.insertBoundsAndReturnRenderGroup(bounds);
  }

  virtual void addDataBlocksToInsertData(Fragmenter_Namespace::InsertData& insertData) {
    GeoPointValueConverter::addDataBlocksToInsertData(insertData);

    DataBlockPtr ringSizes, polyRings, bounds, renderGroup;

    ringSizes.arraysPtr = ring_sizes_data_.get();
    polyRings.arraysPtr = poly_rings_data_.get();
    bounds.arraysPtr = bounds_data_.get();
    renderGroup.numbersPtr = reinterpret_cast<int8_t*>(render_group_data_.get());

    insertData.data.emplace_back(ringSizes);
    insertData.columnIds.emplace_back(ring_sizes_column_descriptor_->columnId);

    insertData.data.emplace_back(polyRings);
    insertData.columnIds.emplace_back(ring_sizes_solumn_descriptor_->columnId);

    insertData.data.emplace_back(bounds);
    insertData.columnIds.emplace_back(bounds_column_descriptor_->columnId);

    insertData.data.emplace_back(renderGroup);
    insertData.columnIds.emplace_back(render_group_column_descriptor_->columnId);
  }
};

#endif
