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

#include <parquet/types.h>

#include "DataMgr/ArrayNoneEncoder.h"
#include "ParquetArrayEncoder.h"

namespace foreign_storage {
class ParquetVariableLengthArrayEncoder : public ParquetArrayEncoder {
 public:
  ParquetVariableLengthArrayEncoder(Data_Namespace::AbstractBuffer* data_buffer,
                                    Data_Namespace::AbstractBuffer* index_buffer,
                                    std::shared_ptr<ParquetScalarEncoder> scalar_encoder,
                                    const ColumnDescriptor* column_desciptor)
      : ParquetArrayEncoder(data_buffer, scalar_encoder, column_desciptor)
      , index_buffer_(index_buffer) {}

  void appendData(const int16_t* def_levels,
                  const int16_t* rep_levels,
                  const int64_t values_read,
                  const int64_t levels_read,
                  const bool is_last_batch,
                  int8_t* values) override {
    CHECK(levels_read > 0);
    setFirstOffsetForBuffer(def_levels[0]);
    ParquetArrayEncoder::appendData(
        def_levels, rep_levels, values_read, levels_read, is_last_batch, values);
  }

 protected:
  void appendArraysToBuffer() override {
    index_buffer_->append(reinterpret_cast<int8_t*>(offsets_.data()),
                          offsets_.size() * sizeof(ArrayOffsetT));
    offsets_.clear();
    ParquetArrayEncoder::appendArraysToBuffer();
  }

  void processLastArray() override { appendLastArrayOffset(); }

 private:
  void setFirstOffsetForBuffer(const int16_t def_level) {
    if (data_buffer_bytes_.size() == 0 && buffer_->size() == 0) {  // first  element
      if (def_level == ParquetArrayEncoder::list_null_def_level) {
        // OmniSci variable array types have a special encoding for chunks in
        // which the first array is null: the first `DEFAULT_NULL_PADDING_SIZE`
        // bytes of the chunk are filled and the offset is set appropriately.
        // Ostensibly, this is done to allow marking a null array by negating
        // a non-zero value.
        offsets_.push_back(ArrayNoneEncoder::DEFAULT_NULL_PADDING_SIZE);
        std::vector<int8_t> zero_bytes(ArrayNoneEncoder::DEFAULT_NULL_PADDING_SIZE, 0);
        data_buffer_bytes_.insert(
            data_buffer_bytes_.end(), zero_bytes.begin(), zero_bytes.end());
      } else {
        offsets_.push_back(0);
      }
    }
  }

  void appendLastArrayOffset() {
    int64_t last_offset = buffer_->size() + data_buffer_bytes_.size();
    if (!isLastArrayNull()) {
      // append array data offset
      offsets_.push_back(last_offset);
    } else {
      // append a null array offset
      offsets_.push_back(-last_offset);
    }
  }

  Data_Namespace::AbstractBuffer* index_buffer_;
  std::vector<ArrayOffsetT> offsets_;
};
}  // namespace foreign_storage
