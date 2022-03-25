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

#include "ForeignStorageException.h"
#include "LazyParquetChunkLoader.h"
#include "ParquetEncoder.h"
#include "TypedParquetDetectBuffer.h"

#include <parquet/schema.h>
#include <parquet/types.h>

namespace foreign_storage {

class ParquetDetectStringEncoder : public ParquetScalarEncoder {
 public:
  ParquetDetectStringEncoder(Data_Namespace::AbstractBuffer* buffer)
      : ParquetScalarEncoder(buffer)
      , detect_buffer_(dynamic_cast<TypedParquetDetectBuffer*>(buffer_)) {
    CHECK(detect_buffer_);
  }

  void setNull(int8_t* omnisci_data_bytes) override { UNREACHABLE(); }
  void copy(const int8_t* omnisci_data_bytes_source,
            int8_t* omnisci_data_bytes_destination) override {
    UNREACHABLE();
  }

  void encodeAndCopy(const int8_t* parquet_data_bytes,
                     int8_t* omnisci_data_bytes) override {
    UNREACHABLE();
  }

  void encodeAndCopyContiguous(const int8_t* parquet_data_bytes,
                               int8_t* omnisci_data_bytes,
                               const size_t num_elements) override {
    UNREACHABLE();
  }

  void validate(const int8_t* parquet_data,
                const int64_t j,
                const SQLTypeInfo& column_type) const override {
    auto parquet_data_ptr = reinterpret_cast<const parquet::ByteArray*>(parquet_data);
    auto& byte_array = parquet_data_ptr[j];
    if (byte_array.len > StringDictionary::MAX_STRLEN) {
      throw ForeignStorageException("String exceeeds max length allowed in dictionary");
    }
  }

  void validateUsingEncodersColumnType(const int8_t* parquet_data,
                                       const int64_t j) const override {
    validate(parquet_data, j, column_type_);
  }

  std::string encodedDataToString(const int8_t* bytes) const override {
    UNREACHABLE();
    return {};
  }

  void eraseInvalidIndicesInBuffer(
      const InvalidRowGroupIndices& invalid_indices) override {
    UNREACHABLE();
  }

  void validateAndAppendData(const int16_t* def_levels,
                             const int16_t* rep_levels,
                             const int64_t values_read,
                             const int64_t levels_read,
                             int8_t* values,
                             const SQLTypeInfo& column_type, /* may not be used */
                             InvalidRowGroupIndices& invalid_indices) override {
    UNREACHABLE();
  }

  void appendData(const int16_t* def_levels,
                  const int16_t* rep_levels,
                  const int64_t values_read,
                  const int64_t levels_read,
                  int8_t* values) override {
    CHECK(levels_read > 0);

    auto parquet_data_ptr = reinterpret_cast<const parquet::ByteArray*>(values);

    for (int64_t i = 0, j = 0; i < levels_read; ++i) {
      if (def_levels[i]) {
        CHECK(j < values_read);
        auto& byte_array = parquet_data_ptr[j++];
        if (is_error_tracking_enabled_ && byte_array.len > StringDictionary::MAX_STRLEN) {
          ParquetEncoder::invalid_indices_.insert(ParquetEncoder::current_chunk_offset_ +
                                                  i);
          detect_buffer_->appendValue({});  // add empty string
        } else {
          auto string_value =
              std::string{reinterpret_cast<const char*>(byte_array.ptr), byte_array.len};
          detect_buffer_->appendValue(string_value);
        }
      } else {
        detect_buffer_->appendValue("NULL");
      }
    }
    if (is_error_tracking_enabled_) {
      ParquetEncoder::current_chunk_offset_ += levels_read;
    }
  }

  void appendDataTrackErrors(const int16_t* def_levels,
                             const int16_t* rep_levels,
                             const int64_t values_read,
                             const int64_t levels_read,
                             int8_t* values) override {
    CHECK(is_error_tracking_enabled_);
    appendData(def_levels, rep_levels, values_read, levels_read, values);
  }

  TypedParquetDetectBuffer* detect_buffer_;
};

}  // namespace foreign_storage
