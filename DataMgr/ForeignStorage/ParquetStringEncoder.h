/*
 * Copyright 2020 OmniSci, Inc.
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

#include "LazyParquetChunkLoader.h"
#include "ParquetInPlaceEncoder.h"
#include "StringDictionary/StringDictionary.h"

#include <parquet/schema.h>
#include <parquet/types.h>

namespace foreign_storage {

template <typename V>
class ParquetStringEncoder : public TypedParquetInPlaceEncoder<V, V> {
 public:
  ParquetStringEncoder(Data_Namespace::AbstractBuffer* buffer,
                       StringDictionary* string_dictionary,
                       ChunkMetadata* chunk_metadata)
      : TypedParquetInPlaceEncoder<V, V>(buffer, sizeof(V), sizeof(V))
      , string_dictionary_(string_dictionary)
      , chunk_metadata_(chunk_metadata)
      , encode_buffer_(LazyParquetChunkLoader::batch_reader_num_elements * sizeof(V))
      , min_(std::numeric_limits<V>::max())
      , max_(std::numeric_limits<V>::lowest())
      , current_batch_offset_(0)
      , invalid_indices_(nullptr) {}

  void validateAndAppendData(const int16_t* def_levels,
                             const int16_t* rep_levels,
                             const int64_t values_read,
                             const int64_t levels_read,
                             int8_t* values,
                             const SQLTypeInfo& column_type, /* may not be used */
                             InvalidRowGroupIndices& invalid_indices) override {
    auto parquet_data_ptr = reinterpret_cast<const parquet::ByteArray*>(values);
    for (int64_t i = 0, j = 0; i < levels_read; ++i) {
      if (def_levels[i]) {
        CHECK(j < values_read);
        auto& byte_array = parquet_data_ptr[j++];
        if (byte_array.len > StringDictionary::MAX_STRLEN) {
          invalid_indices.insert(current_batch_offset_ + i);
        }
      }
    }
    current_batch_offset_ += levels_read;
    encodeAndCopyContiguous(values, encode_buffer_.data(), values_read);
    appendData(def_levels, rep_levels, values_read, levels_read, values);
  }

  void appendData(const int16_t* def_levels,
                  const int16_t* rep_levels,
                  const int64_t values_read,
                  const int64_t levels_read,
                  int8_t* values) override {
    encodeAndCopyContiguous(values, encode_buffer_.data(), values_read);
    TypedParquetInPlaceEncoder<V, V>::appendData(
        def_levels, rep_levels, values_read, levels_read, encode_buffer_.data());
  }

  void encodeAndCopyContiguous(const int8_t* parquet_data_bytes,
                               int8_t* omnisci_data_bytes,
                               const size_t num_elements) override {
    CHECK(string_dictionary_);
    auto parquet_data_ptr =
        reinterpret_cast<const parquet::ByteArray*>(parquet_data_bytes);
    auto omnisci_data_ptr = reinterpret_cast<V*>(omnisci_data_bytes);
    std::vector<std::string_view> string_views;
    string_views.reserve(num_elements);
    for (size_t i = 0; i < num_elements; ++i) {
      auto& byte_array = parquet_data_ptr[i];
      if (byte_array.len <= StringDictionary::MAX_STRLEN) {
        string_views.emplace_back(reinterpret_cast<const char*>(byte_array.ptr),
                                  byte_array.len);
      } else {
        string_views.emplace_back(nullptr, 0);
      }
    }
    string_dictionary_->getOrAddBulk(string_views, omnisci_data_ptr);
    updateMetadataStats(num_elements, omnisci_data_bytes);
  }

  void encodeAndCopy(const int8_t* parquet_data_bytes,
                     int8_t* omnisci_data_bytes) override {
    TypedParquetInPlaceEncoder<V, V>::copy(parquet_data_bytes, omnisci_data_bytes);
  }

  std::shared_ptr<ChunkMetadata> getRowGroupMetadata(
      const parquet::RowGroupMetaData* group_metadata,
      const int parquet_column_index,
      const SQLTypeInfo& column_type) override {
    auto metadata = ParquetEncoder::getRowGroupMetadata(
        group_metadata, parquet_column_index, column_type);
    auto column_metadata = group_metadata->ColumnChunk(parquet_column_index);
    metadata->numBytes = ParquetInPlaceEncoder::omnisci_data_type_byte_size_ *
                         column_metadata->num_values();
    return metadata;
  }

 protected:
  bool encodingIsIdentityForSameTypes() const override { return true; }

 private:
  void updateMetadataStats(int64_t values_read, int8_t* values) {
    if (!chunk_metadata_) {
      return;
    }
    V* data_ptr = reinterpret_cast<V*>(values);
    for (int64_t i = 0; i < values_read; ++i) {
      min_ = std::min<V>(data_ptr[i], min_);
      max_ = std::max<V>(data_ptr[i], max_);
    }
    chunk_metadata_->fillChunkStats(min_, max_, false);
  }

  StringDictionary* string_dictionary_;
  ChunkMetadata* chunk_metadata_;
  std::vector<int8_t> encode_buffer_;

  V min_, max_;

  int64_t current_batch_offset_;
  InvalidRowGroupIndices* invalid_indices_;
};

}  // namespace foreign_storage
