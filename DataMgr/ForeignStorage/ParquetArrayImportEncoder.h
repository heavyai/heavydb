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
class ParquetArrayImportEncoder : public ParquetArrayEncoder {
 public:
  ParquetArrayImportEncoder(Data_Namespace::AbstractBuffer* data_buffer,
                            std::shared_ptr<ParquetScalarEncoder> scalar_encoder,
                            const ColumnDescriptor* column_desciptor)
      : ParquetArrayEncoder(data_buffer, scalar_encoder, column_desciptor)
      , array_datum_buffer_(
            dynamic_cast<TypedParquetStorageBuffer<ArrayDatum>*>(data_buffer))
      , column_descriptor_(column_desciptor) {
    CHECK(array_datum_buffer_);
  }

 protected:
  void appendArraysToBuffer() override {
    // no-op as data is already written to buffer in `processLastArray`
  }

  void processLastArray() override { appendToArrayDatumBuffer(); }

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

  TypedParquetStorageBuffer<ArrayDatum>* array_datum_buffer_;
  const ColumnDescriptor* column_descriptor_;
};
}  // namespace foreign_storage
