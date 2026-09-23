/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "ParquetEncoder.h"
#include "TypedParquetStorageBuffer.h"

#include <parquet/schema.h>
#include <parquet/types.h>

namespace foreign_storage {

class ParquetStringImportEncoder : public ParquetEncoder, public ParquetImportEncoder {
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

  void appendDataTrackErrors(const int16_t* def_levels,
                             const int16_t* rep_levels,
                             const int64_t values_read,
                             const int64_t levels_read,
                             int8_t* values) override {
    UNREACHABLE() << "unexpected call to appendDataTrackErrors from unsupported encoder";
  }

  void validateAndAppendData(const int16_t* def_levels,
                             const int16_t* rep_levels,
                             const int64_t values_read,
                             const int64_t levels_read,
                             int8_t* values,
                             const SQLTypeInfo& column_type, /* may not be used */
                             InvalidRowGroupIndices& invalid_indices) override {
    appendData(def_levels, rep_levels, values_read, levels_read, values);
  }

  void eraseInvalidIndicesInBuffer(
      const InvalidRowGroupIndices& invalid_indices) override {
    if (invalid_indices.empty()) {
      return;
    }
    string_buffer_->eraseInvalidData(invalid_indices);
  }

 private:
  TypedParquetStorageBuffer<std::string>* string_buffer_;
};

}  // namespace foreign_storage
