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

  void appendData(int16_t* def_levels,
                  int64_t values_read,
                  int64_t levels_read,
                  int8_t* values) override {
    if (!buffer_->size()) {
      CHECK(levels_read > 0);
      // write the initial starting offset
      StringOffsetT zero = 0;
      index_buffer_->append(reinterpret_cast<int8_t*>(&zero), sizeof(StringOffsetT));
    }

    auto parquet_data_ptr = reinterpret_cast<const parquet::ByteArray*>(values);
    auto offsets = reinterpret_cast<StringOffsetT*>(encode_buffer_.data());
    auto last_offset = buffer_->size();

    size_t total_len = 0;
    for (int64_t i = 0, j = 0; i < levels_read; ++i) {
      if (def_levels[i]) {
        CHECK(j < values_read);
        auto& byte_array = parquet_data_ptr[j++];
        total_len += byte_array.len;
      }
      offsets[i] = last_offset + total_len;
    }
    index_buffer_->append(encode_buffer_.data(), levels_read * sizeof(StringOffsetT));

    encode_buffer_.resize(std::max<size_t>(total_len, encode_buffer_.size()));
    buffer_->reserve(total_len);
    total_len = 0;
    for (int64_t i = 0, j = 0; i < levels_read; ++i) {
      if (def_levels[i]) {
        CHECK(j < values_read);
        auto& byte_array = parquet_data_ptr[j++];
        memcpy(encode_buffer_.data() + total_len, byte_array.ptr, byte_array.len);
        total_len += byte_array.len;
      }
    }
    buffer_->append(encode_buffer_.data(), total_len);
  }

 private:
  Data_Namespace::AbstractBuffer* index_buffer_;
  std::vector<int8_t> encode_buffer_;
};

}  // namespace foreign_storage
