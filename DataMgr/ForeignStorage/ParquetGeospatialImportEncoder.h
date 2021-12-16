/*
 * Copyright 2021 OmniSci, Inc.
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

#pragma once

#include <parquet/schema.h>
#include <parquet/types.h>
#include "GeospatialEncoder.h"
#include "ParquetEncoder.h"
#include "TypedParquetStorageBuffer.h"

namespace foreign_storage {

class ParquetGeospatialImportEncoder : public ParquetEncoder,
                                       public GeospatialEncoder,
                                       public ParquetImportEncoder {
 public:
  ParquetGeospatialImportEncoder()
      : ParquetEncoder(nullptr)
      , GeospatialEncoder()
      , current_batch_offset_(0)
      , invalid_indices_(nullptr) {}

  ParquetGeospatialImportEncoder(std::list<Chunk_NS::Chunk>& chunks)
      : ParquetEncoder(nullptr)
      , GeospatialEncoder(chunks)
      , current_batch_offset_(0)
      , invalid_indices_(nullptr)
      , base_column_buffer_(nullptr)
      , coords_column_buffer_(nullptr)
      , bounds_column_buffer_(nullptr)
      , ring_sizes_column_buffer_(nullptr)
      , poly_rings_column_buffer_(nullptr)
      , render_group_column_buffer_(nullptr) {
    CHECK(geo_column_info_->type.is_geometry());

    const auto geo_column_type = geo_column_info_->type.get_type();

    base_column_buffer_ = dynamic_cast<TypedParquetStorageBuffer<std::string>*>(
        chunks.begin()->getBuffer());
    CHECK(base_column_buffer_);

    // initialize coords column
    coords_column_buffer_ = dynamic_cast<TypedParquetStorageBuffer<ArrayDatum>*>(
        getBuffer(chunks, geo_column_type, COORDS));
    CHECK(coords_column_buffer_);

    // initialize bounds column
    if (hasBoundsColumn()) {
      bounds_column_buffer_ = dynamic_cast<TypedParquetStorageBuffer<ArrayDatum>*>(
          getBuffer(chunks, geo_column_type, BOUNDS));
      CHECK(bounds_column_buffer_);
    }

    // initialize ring sizes column & render group column
    if (hasRingSizesColumn()) {
      ring_sizes_column_buffer_ = dynamic_cast<TypedParquetStorageBuffer<ArrayDatum>*>(
          getBuffer(chunks, geo_column_type, RING_SIZES));
      CHECK(ring_sizes_column_buffer_);
    }
    if (hasRenderGroupColumn()) {
      render_group_column_buffer_ = getBuffer(chunks, geo_column_type, RENDER_GROUP);
      CHECK(render_group_column_buffer_);
    }

    // initialize poly rings column
    if (hasPolyRingsColumn()) {
      poly_rings_column_buffer_ = dynamic_cast<TypedParquetStorageBuffer<ArrayDatum>*>(
          getBuffer(chunks, geo_column_type, POLY_RINGS));
      CHECK(poly_rings_column_buffer_);
    }
  }

  void validateAndAppendData(const int16_t* def_levels,
                             const int16_t* rep_levels,
                             const int64_t values_read,
                             const int64_t levels_read,
                             int8_t* values,
                             const SQLTypeInfo& column_type, /* may not be used */
                             InvalidRowGroupIndices& invalid_indices) override {
    invalid_indices_ = &invalid_indices;  // used in assembly algorithm
    appendData(def_levels, rep_levels, values_read, levels_read, values);
  }

  void eraseInvalidIndicesInBuffer(
      const InvalidRowGroupIndices& invalid_indices) override {
    if (invalid_indices.empty()) {
      return;
    }
    base_column_buffer_->eraseInvalidData(invalid_indices);
    coords_column_buffer_->eraseInvalidData(invalid_indices);
    if (hasBoundsColumn()) {
      bounds_column_buffer_->eraseInvalidData(invalid_indices);
    }
    if (hasRingSizesColumn()) {
      ring_sizes_column_buffer_->eraseInvalidData(invalid_indices);
    }
    if (hasPolyRingsColumn()) {
      poly_rings_column_buffer_->eraseInvalidData(invalid_indices);
    }
    if (hasRenderGroupColumn()) {
      render_group_column_buffer_->setSize(
          sizeof(int32_t) *
          (render_group_column_buffer_->size() - invalid_indices.size()));
    }
  }

  void appendData(const int16_t* def_levels,
                  const int16_t* rep_levels,
                  const int64_t values_read,
                  const int64_t levels_read,
                  int8_t* values) override {
    auto parquet_data_ptr = reinterpret_cast<const parquet::ByteArray*>(values);

    clearDatumBuffers();

    for (int64_t i = 0, j = 0; i < levels_read; ++i) {
      clearParseBuffers();
      if (def_levels[i] == 0) {
        processNullGeoElement();
      } else {
        CHECK(j < values_read);
        auto& byte_array = parquet_data_ptr[j++];
        auto geo_string_view = std::string_view{
            reinterpret_cast<const char*>(byte_array.ptr), byte_array.len};
        try {
          processGeoElement(geo_string_view);
        } catch (const std::runtime_error& error) {
          CHECK(invalid_indices_);
          invalid_indices_->insert(current_batch_offset_ + i);
          /// add null if failed
          clearParseBuffers();
          processNullGeoElement();
        }
      }
    }

    appendArrayDatumsToBuffer();

    appendBaseAndRenderGroupData(levels_read);

    current_batch_offset_ += levels_read;
  }

 private:
  void appendArrayDatumsIfApplicable(TypedParquetStorageBuffer<ArrayDatum>* column_buffer,
                                     const std::vector<ArrayDatum>& datum_buffer) {
    if (column_buffer) {
      for (const auto& datum : datum_buffer) {
        column_buffer->appendElement(datum);
      }
    } else {
      CHECK(datum_buffer.empty());
    }
  }

  void appendArrayDatumsToBuffer() {
    appendArrayDatumsIfApplicable(coords_column_buffer_, coords_datum_buffer_);
    appendArrayDatumsIfApplicable(bounds_column_buffer_, bounds_datum_buffer_);
    appendArrayDatumsIfApplicable(ring_sizes_column_buffer_, ring_sizes_datum_buffer_);
    appendArrayDatumsIfApplicable(poly_rings_column_buffer_, poly_rings_datum_buffer_);
  }

  void appendBaseAndRenderGroupData(const int64_t row_count) {
    for (int64_t i = 0; i < row_count; ++i) {
      base_column_buffer_->appendElement("");
    }
    if (render_group_column_buffer_) {
      render_group_values_.resize(row_count, 0);
      auto data_ptr = reinterpret_cast<int8_t*>(render_group_values_.data());
      render_group_column_buffer_->append(data_ptr, sizeof(int32_t) * row_count);
    }
  }

  AbstractBuffer* getBuffer(std::list<Chunk_NS::Chunk>& chunks,
                            const SQLTypes sql_type,
                            GeoColumnType geo_column_type) {
    auto chunk = getIteratorForGeoColumnType(chunks, sql_type, geo_column_type);
    auto buffer = chunk->getBuffer();
    return buffer;
  }

  int64_t current_batch_offset_;
  InvalidRowGroupIndices* invalid_indices_;
  TypedParquetStorageBuffer<std::string>* base_column_buffer_;
  TypedParquetStorageBuffer<ArrayDatum>* coords_column_buffer_;
  TypedParquetStorageBuffer<ArrayDatum>* bounds_column_buffer_;
  TypedParquetStorageBuffer<ArrayDatum>* ring_sizes_column_buffer_;
  TypedParquetStorageBuffer<ArrayDatum>* poly_rings_column_buffer_;
  AbstractBuffer* render_group_column_buffer_;
};

}  // namespace foreign_storage
