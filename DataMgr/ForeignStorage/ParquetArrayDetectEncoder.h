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

#include <parquet/types.h>

#include "ParquetArrayEncoder.h"
#include "ParquetDetectStringEncoder.h"
#include "Shared/StringTransform.h"
#include "TypedParquetDetectBuffer.h"

namespace foreign_storage {
class ParquetArrayDetectEncoder : public ParquetArrayEncoder {
 public:
  ParquetArrayDetectEncoder(Data_Namespace::AbstractBuffer* data_buffer,
                            std::shared_ptr<ParquetScalarEncoder> scalar_encoder,
                            const ColumnDescriptor* column_desciptor)
      : ParquetArrayEncoder(data_buffer, scalar_encoder, column_desciptor)
      , detect_buffer_(dynamic_cast<TypedParquetDetectBuffer*>(data_buffer))
      , is_string_array_(
            dynamic_cast<ParquetDetectStringEncoder*>(scalar_encoder_.get())) {
    CHECK(detect_buffer_);
  }

  void appendArrayItem(const int64_t encoded_index) override {
    if (!is_string_array_) {
      auto string_value =
          scalar_encoder_->encodedDataToString(encodedDataAtIndex(encoded_index));
      array_string_.emplace_back(string_value);
    } else {
      CHECK_GT(string_buffer_.size(), static_cast<size_t>(encoded_index));
      array_string_.emplace_back(string_buffer_[encoded_index]);
    }
    updateMetadataForAppendedArrayItem(encoded_index);
  }

 protected:
  void encodeAllValues(const int8_t* values, const int64_t values_read) override {
    if (!is_string_array_) {
      ParquetArrayEncoder::encodeAllValues(values, values_read);
    } else {  // string arrays are a special case that require special handling
      string_buffer_.clear();
      auto parquet_data_ptr = reinterpret_cast<const parquet::ByteArray*>(values);
      for (int64_t i = 0; i < values_read; ++i) {
        auto& byte_array = parquet_data_ptr[i];
        auto string_value =
            std::string{reinterpret_cast<const char*>(byte_array.ptr), byte_array.len};
        string_buffer_.push_back(string_value);
      }
    }
  }

  void appendArraysToBuffer() override {
    // no-op as data is already written to buffer in `processLastArray`
  }

  void processLastArray() override {
    appendToDetectBuffer();
    ParquetArrayEncoder::processLastArray();
  }

 private:
  void appendToDetectBuffer() {
    if (isLastArrayNull()) {
      detect_buffer_->appendValue("NULL");
    } else if (isLastArrayEmpty()) {
      detect_buffer_->appendValue("{}");
    } else {
      detect_buffer_->appendValue("{" + join(array_string_, ",") + "}");
      array_string_.clear();
    }
  }

  TypedParquetDetectBuffer* detect_buffer_;
  const bool is_string_array_;
  std::vector<std::string> array_string_;
  std::vector<std::string> string_buffer_;
};
}  // namespace foreign_storage
