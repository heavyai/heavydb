/*
 * Copyright 2022 HEAVY.AI, Inc.
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
#include "ParquetEncoder.h"

#include <parquet/schema.h>
#include <parquet/types.h>

namespace foreign_storage {

class ParquetStringNoneEncoder : public ParquetEncoder {
 public:
  ParquetStringNoneEncoder(Data_Namespace::AbstractBuffer* buffer,
                           Data_Namespace::AbstractBuffer* index_buffer)
      : ParquetEncoder(buffer)
      , index_buffer_(index_buffer)
      , encode_buffer_(LazyParquetChunkLoader::batch_reader_num_elements *
                       sizeof(StringOffsetT)) {}

  void appendData(const int16_t* def_levels,
                  const int16_t* rep_levels,
                  const int64_t values_read,
                  const int64_t levels_read,
                  int8_t* values) override {
    CHECK(levels_read > 0);
    writeInitialOffsetIfApplicable();

    auto parquet_data_ptr = reinterpret_cast<const parquet::ByteArray*>(values);
    auto offsets = reinterpret_cast<StringOffsetT*>(encode_buffer_.data());
    auto last_offset = buffer_->size();

    size_t total_len = 0;
    for (int64_t i = 0, j = 0; i < levels_read; ++i) {
      if (def_levels[i]) {
        CHECK(j < values_read);
        auto& byte_array = parquet_data_ptr[j++];
        if (is_error_tracking_enabled_ && byte_array.len > StringDictionary::MAX_STRLEN) {
          // no-op, or effectively inserting a null: total_len += 0;
        } else {
          total_len += byte_array.len;
        }
      }
      offsets[i] = last_offset + total_len;
    }
    index_buffer_->append(encode_buffer_.data(), levels_read * sizeof(StringOffsetT));

    encode_buffer_.resize(std::max<size_t>(total_len, encode_buffer_.size()));
    buffer_->reserve(buffer_->size() + total_len);
    total_len = 0;
    for (int64_t i = 0, j = 0; i < levels_read; ++i) {
      if (def_levels[i]) {
        CHECK(j < values_read);
        auto& byte_array = parquet_data_ptr[j++];
        if (is_error_tracking_enabled_ && byte_array.len > StringDictionary::MAX_STRLEN) {
          ParquetEncoder::invalid_indices_.insert(ParquetEncoder::current_chunk_offset_ +
                                                  i);
        } else {
          memcpy(encode_buffer_.data() + total_len, byte_array.ptr, byte_array.len);
          total_len += byte_array.len;
        }
      } else if (is_error_tracking_enabled_ &&
                 ParquetEncoder::column_type_
                     .get_notnull()) {  // item is null for NOT NULL column
        ParquetEncoder::invalid_indices_.insert(ParquetEncoder::current_chunk_offset_ +
                                                i);
      }
    }
    if (is_error_tracking_enabled_) {
      ParquetEncoder::current_chunk_offset_ += levels_read;
    }
    buffer_->append(encode_buffer_.data(), total_len);
  }

  void appendDataTrackErrors(const int16_t* def_levels,
                             const int16_t* rep_levels,
                             const int64_t values_read,
                             const int64_t levels_read,
                             int8_t* values) override {
    CHECK(is_error_tracking_enabled_);
    appendData(def_levels, rep_levels, values_read, levels_read, values);
  }

 private:
  void writeInitialOffsetIfApplicable() {
    if (!index_buffer_->size()) {
      // write the initial starting offset
      StringOffsetT zero = 0;
      index_buffer_->append(reinterpret_cast<int8_t*>(&zero), sizeof(StringOffsetT));
    }
  }

  Data_Namespace::AbstractBuffer* index_buffer_;
  std::vector<int8_t> encode_buffer_;
};

}  // namespace foreign_storage
