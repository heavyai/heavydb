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

#include <atomic>
#include <future>
#include <thread>

namespace Importer_NS {
std::vector<uint8_t> compress_coords(std::vector<double>& coords, const SQLTypeInfo& ti);
}  // namespace Importer_NS

template <typename SOURCE_TYPE, typename TARGET_TYPE>
struct NumericValueConverter : public TargetValueConverter {
  using ColumnDataPtr = std::unique_ptr<TARGET_TYPE, CheckedMallocDeleter<TARGET_TYPE>>;
  using ElementsBufferColumnPtr = ColumnDataPtr;

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

  ~NumericValueConverter() override {}

  void allocateColumnarData(size_t num_rows) override {
    CHECK(num_rows > 0);
    column_data_ = ColumnDataPtr(
        reinterpret_cast<TARGET_TYPE*>(checked_malloc(num_rows * sizeof(TARGET_TYPE))));
  }

  ElementsBufferColumnPtr allocateColumnarBuffer(size_t num_rows) {
    CHECK(num_rows > 0);
    return ElementsBufferColumnPtr(
        reinterpret_cast<TARGET_TYPE*>(checked_malloc(num_rows * sizeof(TARGET_TYPE))));
  }

  void convertElementToColumnarFormat(
      size_t row,
      typename ElementsBufferColumnPtr::pointer columnData,
      const ScalarTargetValue* scalarValue) {
    auto mapd_p = checked_get<SOURCE_TYPE>(row, scalarValue, SOURCE_TYPE_ACCESSOR);
    auto val = *mapd_p;

    if (do_null_check_ && null_check_value_ == val) {
      columnData[row] = null_value_;
    } else {
      columnData[row] = static_cast<TARGET_TYPE>(val);
    }
  }

  void convertToColumnarFormat(size_t row, const ScalarTargetValue* scalarValue) {
    convertElementToColumnarFormat(row, column_data_.get(), scalarValue);
  }

  void convertToColumnarFormat(size_t row, const TargetValue* value) override {
    auto scalarValue =
        checked_get<ScalarTargetValue>(row, value, SCALAR_TARGET_VALUE_ACCESSOR);
    convertToColumnarFormat(row, scalarValue);
  }

  auto processArrayBuffer(auto buffer) { return std::move(buffer); }

  void addDataBlocksToInsertData(Fragmenter_Namespace::InsertData& insertData) override {
    DataBlockPtr dataBlock;
    dataBlock.numbersPtr = reinterpret_cast<int8_t*>(column_data_.get());
    insertData.data.push_back(dataBlock);
    insertData.columnIds.push_back(column_descriptor_->columnId);
  }
};

template <typename TARGET_TYPE>
struct DictionaryValueConverter : public NumericValueConverter<int64_t, TARGET_TYPE> {
  using ElementsDataColumnPtr =
      typename NumericValueConverter<int64_t, TARGET_TYPE>::ColumnDataPtr;

  using ElementsBufferColumnPtr = std::unique_ptr<std::vector<std::string>>;

  ElementsBufferColumnPtr column_buffer_;

  const DictDescriptor* target_dict_desc_;

  DictionaryValueConverter(const Catalog_Namespace::Catalog& cat,
                           const ColumnDescriptor* sourceDescriptor,
                           const ColumnDescriptor* targetDescriptor,
                           size_t num_rows,
                           TARGET_TYPE nullValue,
                           int64_t nullCheckValue,
                           bool doNullCheck)
      : NumericValueConverter<int64_t, TARGET_TYPE>(targetDescriptor,
                                                    num_rows,
                                                    nullValue,
                                                    nullCheckValue,
                                                    doNullCheck) {
    target_dict_desc_ =
        cat.getMetadataForDict(targetDescriptor->columnType.get_comp_param(), true);
    CHECK(target_dict_desc_);

    if (num_rows) {
      column_buffer_ = allocateColumnarBuffer(num_rows);
    }
  }

  ~DictionaryValueConverter() override {}

  ElementsBufferColumnPtr allocateColumnarBuffer(size_t num_rows) {
    CHECK(num_rows > 0);
    return std::make_unique<std::vector<std::string>>(num_rows);
  }

  void convertToColumnarFormatFromString(
      size_t row,
      typename ElementsBufferColumnPtr::pointer columnBuffer,
      const ScalarTargetValue* scalarValue) {
    auto mapd_p =
        checked_get<NullableString>(row, scalarValue, this->NULLABLE_STRING_ACCESSOR);

    const auto mapd_str_p = checked_get<std::string>(row, mapd_p, this->STRING_ACCESSOR);

    if (this->do_null_check_ && (nullptr == mapd_str_p)) {
      // TODO: using empty string for nulls
      columnBuffer->at(row) = "";
    } else {
      CHECK(mapd_str_p);
      columnBuffer->at(row) = *mapd_str_p;
    }
  }

  void convertElementToColumnarFormat(
      size_t row,
      typename ElementsBufferColumnPtr::pointer columnBuffer,
      const ScalarTargetValue* scalarValue) {
    convertToColumnarFormatFromString(row, columnBuffer, scalarValue);
  }

  void convertToColumnarFormat(size_t row, const ScalarTargetValue* scalarValue) {
    convertElementToColumnarFormat(row, this->column_buffer_.get(), scalarValue);
  }

  void convertToColumnarFormat(size_t row, const TargetValue* value) override {
    auto scalarValue =
        checked_get<ScalarTargetValue>(row, value, this->SCALAR_TARGET_VALUE_ACCESSOR);

    convertToColumnarFormat(row, scalarValue);
  }

  typename NumericValueConverter<int64_t, TARGET_TYPE>::ColumnDataPtr processArrayBuffer(
      ElementsBufferColumnPtr buffer) {
    typename NumericValueConverter<int64_t, TARGET_TYPE>::ColumnDataPtr data =
        typename NumericValueConverter<int64_t, TARGET_TYPE>::ColumnDataPtr(
            reinterpret_cast<TARGET_TYPE*>(
                checked_malloc(buffer->size() * sizeof(TARGET_TYPE))));

    TARGET_TYPE* columnDataPtr = reinterpret_cast<TARGET_TYPE*>(data.get());
    target_dict_desc_->stringDict->getOrAddBulk(
        *reinterpret_cast<std::vector<std::string>*>(buffer.get()), columnDataPtr);
    return std::move(data);
  }

  void finalizeDataBlocksForInsertData() override {
    if (column_buffer_) {
      TARGET_TYPE* columnDataPtr =
          reinterpret_cast<TARGET_TYPE*>(this->column_data_.get());
      std::vector<std::string>* dataPtr =
          reinterpret_cast<std::vector<std::string>*>(column_buffer_.get());
      target_dict_desc_->stringDict->getOrAddBulk(*dataPtr, columnDataPtr);
      column_buffer_ = nullptr;
    }
  }

  void addDataBlocksToInsertData(Fragmenter_Namespace::InsertData& insertData) override {
    finalizeDataBlocksForInsertData();
    DataBlockPtr dataBlock;
    dataBlock.numbersPtr = reinterpret_cast<int8_t*>(this->column_data_.get());
    insertData.data.push_back(dataBlock);
    insertData.columnIds.push_back(this->column_descriptor_->columnId);
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

  ~StringValueConverter() override {}

  void allocateColumnarData(size_t num_rows) override {
    CHECK(num_rows > 0);
    column_data_ = std::make_unique<std::vector<std::string>>(num_rows);
  }

  void convertToColumnarFormat(size_t row, const TargetValue* value) override {
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

  void addDataBlocksToInsertData(Fragmenter_Namespace::InsertData& insertData) override {
    DataBlockPtr dataBlock;
    dataBlock.stringsPtr = column_data_.get();
    insertData.data.push_back(dataBlock);
    insertData.columnIds.push_back(column_descriptor_->columnId);
  }
};

template <typename ELEMENT_CONVERTER>
struct ArrayValueConverter : public TargetValueConverter {
  std::unique_ptr<
      std::vector<std::pair<size_t, typename ELEMENT_CONVERTER::ElementsBufferColumnPtr>>>
      column_buffer_;
  std::unique_ptr<std::vector<ArrayDatum>> column_data_;
  std::unique_ptr<ELEMENT_CONVERTER> element_converter_;
  SQLTypeInfo element_type_info_;
  bool do_check_null_;

  boost_variant_accessor<ArrayTargetValue> ARRAY_VALUE_ACCESSOR;

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

  ~ArrayValueConverter() override {}

  void allocateColumnarData(size_t num_rows) override {
    CHECK(num_rows > 0);
    column_data_ = std::make_unique<std::vector<ArrayDatum>>(num_rows);
    column_buffer_ = std::make_unique<std::vector<
        std::pair<size_t, typename ELEMENT_CONVERTER::ElementsBufferColumnPtr>>>(
        num_rows);
  }

  void convertToColumnarFormat(size_t row, const TargetValue* value) override {
    const auto arrayValue =
        checked_get<ArrayTargetValue>(row, value, ARRAY_VALUE_ACCESSOR);
    CHECK(arrayValue);
    if (arrayValue->is_initialized()) {
      const auto& vec = arrayValue->get();
      bool is_null = false;
      if (vec.size()) {
        typename ELEMENT_CONVERTER::ElementsBufferColumnPtr elementBuffer =
            element_converter_->allocateColumnarBuffer(vec.size());

        int elementIndex = 0;
        for (const auto& scalarValue : vec) {
          element_converter_->convertElementToColumnarFormat(
              elementIndex++, elementBuffer.get(), &scalarValue);
        }

        column_buffer_->at(row) = {vec.size(), std::move(elementBuffer)};

      } else {
        // Empty, not NULL
        (*column_data_)[row] = ArrayDatum(0, nullptr, is_null, DoNothingDeleter());
      }
    } else {
      // TODO: what does it mean if do_check_null_ is set to false and we get a NULL?
      // CHECK(do_check_null_);  // May need to check
      bool is_null = true;  // do_check_null_;
      (*column_data_)[row] = ArrayDatum(0, nullptr, is_null, DoNothingDeleter());
      (*column_data_)[row].is_null = is_null;
    }
  }

  void finalizeDataBlocksForInsertData() override {
    if (column_buffer_) {
      std::atomic<size_t> row_idx{0};
      auto row_buffer_finalizer = [this, &row_idx](int tid) {
        auto row = row_idx.fetch_add(1);

        if (row >= column_buffer_->size()) {
          return;
        }

        auto& element = (column_buffer_->at(row));
        bool is_null = false;
        if (element.second) {
          typename ELEMENT_CONVERTER::ColumnDataPtr data =
              element_converter_->processArrayBuffer(std::move(element.second));
          int8_t* arrayData = reinterpret_cast<int8_t*>(data.release());
          (*column_data_)[row] = ArrayDatum(
              element.first * element_type_info_.get_size(), arrayData, is_null);
        }
      };

      std::vector<std::future<void>> worker_threads;
      const int num_worker_threads = std::thread::hardware_concurrency();

      if (column_buffer_->size() / num_worker_threads > 10) {
        for (int i = 0; i < num_worker_threads; ++i) {
          worker_threads.push_back(
              std::async(std::launch::async, row_buffer_finalizer, i));
        }

        for (auto& child : worker_threads) {
          child.wait();
        }
      } else {
        row_buffer_finalizer(0);
      }

      column_buffer_ = nullptr;
    }
  }

  void addDataBlocksToInsertData(Fragmenter_Namespace::InsertData& insertData) override {
    finalizeDataBlocksForInsertData();
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

  ~GeoPointValueConverter() override {}

  void allocateColumnarData(size_t num_rows) override {
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

  void convertToColumnarFormat(size_t row, const TargetValue* value) override {
    auto geoValue = checked_get<GeoTargetValue>(row, value, GEO_TARGET_VALUE_ACCESSOR);
    auto geoPoint =
        checked_get<GeoPointTargetValue>(row, geoValue, GEO_POINT_VALUE_ACCESSOR);

    (*column_data_)[row] = "";
    (*signed_compressed_coords_data_)[row] = toCompressedCoords(geoPoint->coords);
  }

  void addDataBlocksToInsertData(Fragmenter_Namespace::InsertData& insertData) override {
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

  ~GeoLinestringValueConverter() override {}

  void allocateColumnarData(size_t num_rows) override {
    CHECK(num_rows > 0);
    GeoPointValueConverter::allocateColumnarData(num_rows);
    bounds_data_ = std::make_unique<std::vector<ArrayDatum>>(num_rows);
  }

  boost_variant_accessor<GeoLineStringTargetValue> GEO_LINESTRING_VALUE_ACCESSOR;

  void convertToColumnarFormat(size_t row, const TargetValue* value) override {
    auto geoValue = checked_get<GeoTargetValue>(row, value, GEO_TARGET_VALUE_ACCESSOR);
    auto geoLinestring = checked_get<GeoLineStringTargetValue>(
        row, geoValue, GEO_LINESTRING_VALUE_ACCESSOR);

    (*column_data_)[row] = "";
    (*signed_compressed_coords_data_)[row] = toCompressedCoords(geoLinestring->coords);
    auto bounds = compute_bounds_of_coords(geoLinestring->coords);
    (*bounds_data_)[row] = to_array_datum(bounds);
  }

  void addDataBlocksToInsertData(Fragmenter_Namespace::InsertData& insertData) override {
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

  ~GeoPolygonValueConverter() override {}

  void allocateColumnarData(size_t num_rows) override {
    GeoPointValueConverter::allocateColumnarData(num_rows);
    ring_sizes_data_ = std::make_unique<std::vector<ArrayDatum>>(num_rows);
    bounds_data_ = std::make_unique<std::vector<ArrayDatum>>(num_rows);
    render_group_data_ = std::make_unique<int32_t[]>(num_rows);
  }

  boost_variant_accessor<GeoPolyTargetValue> GEO_POLY_VALUE_ACCESSOR;

  void convertToColumnarFormat(size_t row, const TargetValue* value) override {
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

  void addDataBlocksToInsertData(Fragmenter_Namespace::InsertData& insertData) override {
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

  ~GeoMultiPolygonValueConverter() override {}

  void allocateColumnarData(size_t num_rows) override {
    GeoPointValueConverter::allocateColumnarData(num_rows);
    ring_sizes_data_ = std::make_unique<std::vector<ArrayDatum>>(num_rows);
    poly_rings_data_ = std::make_unique<std::vector<ArrayDatum>>(num_rows);
    bounds_data_ = std::make_unique<std::vector<ArrayDatum>>(num_rows);
    render_group_data_ = std::make_unique<int32_t[]>(num_rows);
  }

  boost_variant_accessor<GeoMultiPolyTargetValue> GEO_MULTI_POLY_VALUE_ACCESSOR;

  void convertToColumnarFormat(size_t row, const TargetValue* value) override {
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

  void addDataBlocksToInsertData(Fragmenter_Namespace::InsertData& insertData) override {
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
