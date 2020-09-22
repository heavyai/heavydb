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
#include "ParquetEncoder.h"

namespace foreign_storage {
class ParquetArrayEncoder : public ParquetEncoder {
 public:
  ParquetArrayEncoder(Data_Namespace::AbstractBuffer* data_buffer,
                      Data_Namespace::AbstractBuffer* index_buffer,
                      std::shared_ptr<ParquetScalarEncoder> scalar_encoder,
                      const ColumnDescriptor* column_desciptor)
      : ParquetEncoder(data_buffer)
      , index_buffer_(index_buffer)
      , omnisci_data_type_byte_size_(
            column_desciptor->columnType.get_elem_type().get_size())
      , scalar_encoder_(scalar_encoder)
      , has_assembly_started_(false)
      , is_null_array_(false) {}

  void appendData(const int16_t* def_levels,
                  const int16_t* rep_levels,
                  const int64_t values_read,
                  const int64_t levels_read,
                  const bool is_last_batch,
                  int8_t* values) override {
    CHECK(levels_read > 0);

    setFirstOffsetForBuffer(def_levels[0]);
    // Encode all values in the temporary in-memory `encode_buffer_`, doing
    // this encoding as a batch rather than element-wise exposes opportunities
    // for performance optimization for certain scalar types
    encodeAllValues(values, values_read);

    for (int64_t i = 0, j = 0; i < levels_read; ++i) {
      if (isNewArray(rep_levels[i])) {
        appendArrayOffsets();
      }
      processArrayItem(def_levels[i], j);
    }

    if (is_last_batch) {
      finalizeRowGroup();
    }
  }

 private:
  void finalizeRowGroup() {
    appendArrayOffsets();
    appendArrayToBuffer();
    has_assembly_started_ = false;
  }

  bool isNewArray(const int16_t rep_level) {
    return rep_level == 0 && has_assembly_started_;
  }

  void processArrayItem(const int16_t def_level, int64_t& encoded_index) {
    has_assembly_started_ = true;
    if (def_level == non_null_def_level) {
      // push back the element to in-mem data buffer
      appendArrayItem(encoded_index++);
    } else if (def_level == item_null_def_level) {
      // push back a null to in-mem data buffer
      appendNullArrayItem();
    } else if (def_level == list_null_def_level) {
      markArrayAsNull();
    } else {
      UNREACHABLE();
    }
  }

  void setFirstOffsetForBuffer(const int16_t def_level) {
    if (data_buffer_bytes_.size() == 0 && buffer_->size() == 0) {  // first  element
      if (def_level == list_null_def_level) {
        // OmniSci variable array types have a special encoding for chunks in
        // which the first array is null: the first 8 bytes of the chunk are
        // filled and the offset is set appropriately.  Ostensibly, this is
        // done to allow marking a null array by negating a non-zero value;
        // however, the choice of 8 appears arbitrary.
        offsets_.push_back(8);
        std::vector<int8_t> zero_bytes(8, 0);
        data_buffer_bytes_.insert(
            data_buffer_bytes_.end(), zero_bytes.begin(), zero_bytes.end());
      } else {
        offsets_.push_back(0);
      }
    }
  }

  void encodeAllValues(const int8_t* values, const int64_t values_read) {
    encode_buffer_.resize(values_read * omnisci_data_type_byte_size_);
    scalar_encoder_->encodeAndCopyContiguous(values, encode_buffer_.data(), values_read);
  }

  void markArrayAsNull() { is_null_array_ = true; }

  void appendArrayItem(const int64_t encoded_index) {
    auto current_data_byte_size = data_buffer_bytes_.size();
    data_buffer_bytes_.resize(current_data_byte_size + omnisci_data_type_byte_size_);
    auto omnisci_data_ptr = data_buffer_bytes_.data() + current_data_byte_size;
    scalar_encoder_->copy(
        encode_buffer_.data() + (encoded_index)*omnisci_data_type_byte_size_,
        omnisci_data_ptr);
  }

  void appendNullArrayItem() {
    auto current_data_byte_size = data_buffer_bytes_.size();
    data_buffer_bytes_.resize(current_data_byte_size + omnisci_data_type_byte_size_);
    auto omnisci_data_ptr = data_buffer_bytes_.data() + current_data_byte_size;
    scalar_encoder_->setNull(omnisci_data_ptr);
  }

  void appendArrayOffsets() {
    int64_t last_offset = buffer_->size() + data_buffer_bytes_.size();
    if (!is_null_array_) {
      // append array data offset
      offsets_.push_back(last_offset);
    } else {
      // append a null array offset
      offsets_.push_back(-last_offset);
      is_null_array_ = false;
    }
  }

  void appendArrayToBuffer() {
    index_buffer_->append(reinterpret_cast<int8_t*>(offsets_.data()),
                          offsets_.size() * sizeof(ArrayOffsetT));
    buffer_->append(data_buffer_bytes_.data(), data_buffer_bytes_.size());
    data_buffer_bytes_.clear();
    offsets_.clear();
  }

  Data_Namespace::AbstractBuffer* index_buffer_;
  size_t omnisci_data_type_byte_size_;
  std::shared_ptr<ParquetScalarEncoder> scalar_encoder_;

  std::vector<int8_t> encode_buffer_;
  bool has_assembly_started_;
  bool is_null_array_;

  std::vector<int8_t> data_buffer_bytes_;
  std::vector<ArrayOffsetT> offsets_;

  // constants used during Dremel encoding assembly
  const static int16_t non_null_def_level = 3;
  const static int16_t item_null_def_level = 2;
  const static int16_t list_null_def_level = 0;
};
}  // namespace foreign_storage
