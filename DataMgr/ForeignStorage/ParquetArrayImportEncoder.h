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

#include <parquet/types.h>

#include "DataMgr/ArrayNoneEncoder.h"
#include "ImportExport/Importer.h"
#include "ParquetArrayEncoder.h"
#include "TypedParquetStorageBuffer.h"

namespace foreign_storage {
class ParquetArrayImportEncoder : public ParquetArrayEncoder,
                                  public ParquetImportEncoder {
 public:
  ParquetArrayImportEncoder(Data_Namespace::AbstractBuffer* data_buffer,
                            std::shared_ptr<ParquetScalarEncoder> scalar_encoder,
                            const ColumnDescriptor* column_desciptor)
      : ParquetArrayEncoder(data_buffer, scalar_encoder, column_desciptor)
      , array_datum_buffer_(
            dynamic_cast<TypedParquetStorageBuffer<ArrayDatum>*>(data_buffer))
      , column_descriptor_(column_desciptor)
      , num_array_assembled_(0)
      , is_invalid_array_(false)
      , invalid_indices_(nullptr) {
    CHECK(array_datum_buffer_);
  }

  void appendArrayItem(const int64_t encoded_index) override {
    ParquetArrayEncoder::appendArrayItem(encoded_index);
    if (!is_valid_item_[encoded_index]) {
      is_invalid_array_ = true;
    }
  }

  void validateAndAppendData(const int16_t* def_levels,
                             const int16_t* rep_levels,
                             const int64_t values_read,
                             const int64_t levels_read,
                             int8_t* values,
                             const SQLTypeInfo& column_type, /* may not be used */
                             InvalidRowGroupIndices& invalid_indices) override {
    // validate all elements
    is_valid_item_.assign(levels_read, true);
    for (int64_t j = 0; j < values_read; ++j) {
      try {
        scalar_encoder_->validate(values, j, column_type);
      } catch (const std::runtime_error& error) {
        is_valid_item_[j] = false;
      }
    }
    invalid_indices_ = &invalid_indices;  // used in assembly algorithm
    appendData(def_levels, rep_levels, values_read, levels_read, values);
  }

  void resetLastArrayMetadata() override {
    ParquetArrayEncoder::resetLastArrayMetadata();
    is_invalid_array_ = false;
  }

 protected:
  void appendArraysToBuffer() override {
    // no-op as data is already written to buffer in `processLastArray`
  }

  void processLastArray() override {
    appendToArrayDatumBuffer();
    if (is_invalid_array_) {
      CHECK(invalid_indices_);
      invalid_indices_->insert(num_array_assembled_);
    }
    num_array_assembled_++;
  }

 private:
  ArrayDatum convertToArrayDatum(const int8_t* data, const size_t num_elements) {
    const size_t num_bytes = num_elements * omnisci_data_type_byte_size_;
    std::shared_ptr<int8_t> buffer(new int8_t[num_bytes],
                                   std::default_delete<int8_t[]>());
    memcpy(buffer.get(), data, num_bytes);
    return ArrayDatum(num_bytes, buffer, false);
  }

  ArrayDatum getNullArrayDatum() {
    return import_export::ImporterUtils::composeNullArray(column_descriptor_->columnType);
  }

  void appendToArrayDatumBuffer() {
    if (isLastArrayNull()) {
      // append a null array offset
      array_datum_buffer_->appendElement(getNullArrayDatum());
    } else if (isLastArrayEmpty()) {
      array_datum_buffer_->appendElement(convertToArrayDatum(nullptr, 0));
    } else {
      CHECK(data_buffer_bytes_.size() ==
            omnisci_data_type_byte_size_ * sizeOfLastArray());
      array_datum_buffer_->appendElement(
          convertToArrayDatum(data_buffer_bytes_.data(), sizeOfLastArray()));
      data_buffer_bytes_
          .clear();  // can clear immediately, only one array buffered at a time
    }
  }

  void eraseInvalidIndicesInBuffer(
      const InvalidRowGroupIndices& invalid_indices) override {
    if (invalid_indices.empty()) {
      return;
    }
    array_datum_buffer_->eraseInvalidData(invalid_indices);
  }

  std::vector<bool> is_valid_item_;
  TypedParquetStorageBuffer<ArrayDatum>* array_datum_buffer_;
  const ColumnDescriptor* column_descriptor_;
  size_t num_array_assembled_;
  bool is_invalid_array_;
  InvalidRowGroupIndices* invalid_indices_;
};
}  // namespace foreign_storage
