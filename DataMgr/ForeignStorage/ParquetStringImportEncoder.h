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

#include "ParquetEncoder.h"
#include "TypedParquetStorageBuffer.h"

#include <parquet/schema.h>
#include <parquet/types.h>

namespace foreign_storage {

class ParquetStringImportEncoder : public ParquetEncoder {
 public:
  ParquetStringImportEncoder(Data_Namespace::AbstractBuffer* buffer)
      : ParquetEncoder(buffer)
      , string_buffer_(dynamic_cast<TypedParquetStorageBuffer<std::string>*>(buffer)) {
    CHECK(string_buffer_);  // verify dynamic_cast succeeded
  }

  void appendData(const int16_t* def_levels,
                  const int16_t* rep_levels,
                  const int64_t values_read,
                  const int64_t levels_read,
                  int8_t* values) override {
    auto parquet_data_ptr = reinterpret_cast<const parquet::ByteArray*>(values);
    string_buffer_->reserveNumElements(levels_read);
    for (int64_t i = 0, j = 0; i < levels_read; ++i) {
      if (def_levels[i]) {
        CHECK(j < values_read);
        auto& byte_array = parquet_data_ptr[j++];
        string_buffer_->appendElement(
            std::string{reinterpret_cast<const char*>(byte_array.ptr), byte_array.len});
      } else {
        string_buffer_->appendElement("");  // empty strings encode nulls
      }
    }
  }

 private:
  TypedParquetStorageBuffer<std::string>* string_buffer_;
};

}  // namespace foreign_storage
