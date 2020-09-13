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
                       const std::shared_ptr<ChunkMetadata>& chunk_metadata)
      : TypedParquetInPlaceEncoder<V, V>(buffer, sizeof(V), sizeof(V))
      , string_dictionary_(string_dictionary)
      , chunk_metadata_(chunk_metadata)
      , encode_buffer_(LazyParquetChunkLoader::batch_reader_num_elements * sizeof(V))
      , min_(std::numeric_limits<V>::max())
      , max_(std::numeric_limits<V>::lowest())
      , has_nulls_(false)
      , num_elements_(0) {}

  void appendData(int16_t* def_levels,
                  int64_t values_read,
                  int64_t levels_read,
                  int8_t* values) override {
    V* data_ptr = reinterpret_cast<V*>(encode_buffer_.data());
    auto parquet_data_ptr = reinterpret_cast<const parquet::ByteArray*>(values);
    std::vector<std::string_view> string_views;
    string_views.reserve(values_read);
    for (int64_t i = 0; i < values_read; ++i) {
      auto& byte_array = parquet_data_ptr[i];
      string_views.emplace_back(reinterpret_cast<const char*>(byte_array.ptr),
                                byte_array.len);
    }
    string_dictionary_->getOrAddBulk(string_views, data_ptr);
    updateMetadata(values_read, levels_read, encode_buffer_.data());
    TypedParquetInPlaceEncoder<V, V>::appendData(
        def_levels, values_read, levels_read, encode_buffer_.data());
  }

 protected:
  void encodeAndCopy(const int8_t* parquet_data_bytes,
                     int8_t* omnisci_data_bytes) override {
    TypedParquetInPlaceEncoder<V, V>::copy(parquet_data_bytes, omnisci_data_bytes);
  }

  bool encodingIsIdentityForSameTypes() const override { return true; }

 private:
  void updateMetadata(int64_t values_read, int64_t levels_read, int8_t* values) {
    num_elements_ += levels_read;
    V* data_ptr = reinterpret_cast<V*>(values);
    for (int64_t i = 0; i < values_read; ++i) {
      min_ = std::min<V>(data_ptr[i], min_);
      max_ = std::max<V>(data_ptr[i], max_);
    }
    if (values_read < levels_read) {
      has_nulls_ = true;
    }
    chunk_metadata_->fillChunkStats(min_, max_, has_nulls_);
    chunk_metadata_->numBytes = num_elements_ * sizeof(V);
    chunk_metadata_->numElements = num_elements_;
  }

  StringDictionary* string_dictionary_;
  std::shared_ptr<ChunkMetadata> chunk_metadata_;
  std::vector<int8_t> encode_buffer_;

  V min_, max_;
  bool has_nulls_;
  size_t num_elements_;
};

}  // namespace foreign_storage
