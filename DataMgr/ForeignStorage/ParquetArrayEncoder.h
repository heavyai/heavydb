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
#include "ParquetEncoder.h"

namespace foreign_storage {
class ParquetArrayEncoder : public ParquetEncoder {
 public:
  ParquetArrayEncoder(Data_Namespace::AbstractBuffer* data_buffer,
                      std::shared_ptr<ParquetScalarEncoder> scalar_encoder,
                      const ColumnDescriptor* column_desciptor)
      : ParquetEncoder(data_buffer)
      , omnisci_data_type_byte_size_(
            column_desciptor->columnType.get_elem_type().get_size())
      , scalar_encoder_(scalar_encoder)
      , has_assembly_started_(false)
      , is_null_array_(false)
      , is_empty_array_(false)
      , num_elements_in_array_(0)
      , num_array_assembled_(0)
      , is_invalid_array_(false) {}

  void appendDataTrackErrors(const int16_t* def_levels,
                             const int16_t* rep_levels,
                             const int64_t values_read,
                             const int64_t levels_read,
                             int8_t* values) override {
    CHECK(is_error_tracking_enabled_);
    // validate all elements
    is_valid_item_.assign(values_read, true);
    for (int64_t j = 0; j < values_read; ++j) {
      try {
        scalar_encoder_->validateUsingEncodersColumnType(values, j);
      } catch (const std::runtime_error& error) {
        is_valid_item_[j] = false;
      }
    }
    appendData(def_levels, rep_levels, values_read, levels_read, values);
  }

  void appendData(const int16_t* def_levels,
                  const int16_t* rep_levels,
                  const int64_t values_read,
                  const int64_t levels_read,
                  int8_t* values) override {
    CHECK(levels_read > 0);

    // encode all values in the temporary in-memory `encode_buffer_`, doing
    // this encoding as a batch rather than element-wise exposes opportunities
    // for performance optimization for certain scalar types
    encodeAllValues(values, values_read);

    for (int64_t i = 0, j = 0; i < levels_read; ++i) {
      if (isNewArray(rep_levels[i])) {
        processLastArray();
        resetLastArrayMetadata();
      }
      processArrayItem(def_levels[i], j);
    }
  }

  void finalizeRowGroup() {
    processLastArray();
    resetLastArrayMetadata();
    appendArraysToBuffer();
    has_assembly_started_ = false;
  }

  std::shared_ptr<ChunkMetadata> getRowGroupMetadata(
      const parquet::RowGroupMetaData* group_metadata,
      const int parquet_column_index,
      const SQLTypeInfo& column_type) override {
    auto metadata = scalar_encoder_->getRowGroupMetadata(
        group_metadata, parquet_column_index, column_type);
    metadata->numBytes = 0;  // number of bytes is not known
    return metadata;
  }

  virtual void disableMetadataStatsValidation() override {
    ParquetEncoder::disableMetadataStatsValidation();
    scalar_encoder_->disableMetadataStatsValidation();
  }

  virtual void initializeErrorTracking() override {
    ParquetEncoder::initializeErrorTracking();
    scalar_encoder_->initializeErrorTracking();
  }

  virtual void initializeColumnType(const SQLTypeInfo& column_type) override {
    ParquetEncoder::initializeColumnType(column_type);
    scalar_encoder_->initializeColumnType(column_type.get_elem_type());
  }

 protected:
  virtual void processLastArray() {
    if (is_error_tracking_enabled_ &&
        (is_invalid_array_ || (isLastArrayNull() && column_type_.get_notnull()))) {
      invalid_indices_.insert(num_array_assembled_);
    }
    ++num_array_assembled_;
  }

  virtual void appendArraysToBuffer() {
    buffer_->append(data_buffer_bytes_.data(), data_buffer_bytes_.size());
    data_buffer_bytes_.clear();
  }

  bool isLastArrayNull() const { return is_null_array_; }

  bool isLastArrayEmpty() const { return is_empty_array_; }

  size_t sizeOfLastArray() const { return num_elements_in_array_; }

  int8_t* resizeArrayDataBytes(const size_t additional_num_elements) {
    auto current_data_byte_size = data_buffer_bytes_.size();
    data_buffer_bytes_.resize(current_data_byte_size +
                              additional_num_elements * omnisci_data_type_byte_size_);
    return data_buffer_bytes_.data() + current_data_byte_size;
  }

  size_t omnisci_data_type_byte_size_;
  std::shared_ptr<ParquetScalarEncoder> scalar_encoder_;
  std::vector<int8_t> data_buffer_bytes_;

  // constants used during Dremel encoding assembly
  const static int16_t non_null_def_level = 3;
  const static int16_t item_null_def_level = 2;
  const static int16_t empty_list_def_level = 1;
  const static int16_t list_null_def_level = 0;

  virtual void resetLastArrayMetadata() {
    is_empty_array_ = false;
    is_null_array_ = false;
    num_elements_in_array_ = 0;
    if (is_error_tracking_enabled_) {
      is_invalid_array_ = false;
    }
  }

  bool isNewArray(const int16_t rep_level) const {
    return rep_level == 0 && has_assembly_started_;
  }

  int8_t* encodedDataAtIndex(const size_t index) {
    return encode_buffer_.data() + (index)*omnisci_data_type_byte_size_;
  }

  void updateMetadataForAppendedArrayItem(const int64_t encoded_index) {
    num_elements_in_array_++;
    if (is_error_tracking_enabled_ && !is_valid_item_[encoded_index]) {
      is_invalid_array_ = true;
    }
  }

  virtual void appendArrayItem(const int64_t encoded_index) {
    auto omnisci_data_ptr = resizeArrayDataBytes(1);
    scalar_encoder_->copy(encodedDataAtIndex(encoded_index), omnisci_data_ptr);
    updateMetadataForAppendedArrayItem(encoded_index);
  }

  virtual void encodeAllValues(const int8_t* values, const int64_t values_read) {
    encode_buffer_.resize(values_read * omnisci_data_type_byte_size_);
    scalar_encoder_->encodeAndCopyContiguous(values, encode_buffer_.data(), values_read);
  }

 private:
  void processArrayItem(const int16_t def_level, int64_t& encoded_index) {
    has_assembly_started_ = true;
    if (def_level == non_null_def_level) {
      // push back a scalar element to in-memory data buffer
      appendArrayItem(encoded_index++);
    } else if (def_level == item_null_def_level) {
      // push back a scalar null to in-memory data buffer
      appendNullArrayItem();
    } else if (def_level == list_null_def_level) {
      markArrayAsNull();
    } else if (def_level == empty_list_def_level) {
      markArrayAsEmpty();
    } else {
      UNREACHABLE();
    }
  }

  void markArrayAsNull() { is_null_array_ = true; }

  void markArrayAsEmpty() { is_empty_array_ = true; }

  void appendNullArrayItem() {
    scalar_encoder_->setNull(resizeArrayDataBytes(1));
    num_elements_in_array_++;
  }

  std::vector<int8_t> encode_buffer_;
  bool has_assembly_started_;
  bool is_null_array_;
  bool is_empty_array_;
  size_t num_elements_in_array_;

  // error tracking related members
  size_t num_array_assembled_;
  bool is_invalid_array_;
  std::vector<bool> is_valid_item_;
};
}  // namespace foreign_storage
