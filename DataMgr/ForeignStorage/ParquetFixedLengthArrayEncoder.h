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

#include <stdexcept>

#include <parquet/types.h>
#include "ParquetArrayEncoder.h"
#include "Shared/InlineNullValues.h"

namespace foreign_storage {

class ParquetFixedLengthArrayEncoder : public ParquetArrayEncoder {
 public:
  ParquetFixedLengthArrayEncoder(Data_Namespace::AbstractBuffer* data_buffer,
                                 std::shared_ptr<ParquetScalarEncoder> scalar_encoder,
                                 const ColumnDescriptor* column_desciptor)
      : ParquetArrayEncoder(data_buffer, scalar_encoder, column_desciptor)
      , column_desciptor_(*column_desciptor)
      , array_size_(column_desciptor->columnType.get_size()) {
    CHECK(array_size_ % omnisci_data_type_byte_size_ == 0);
  }

 protected:
  void processLastArray() override { appendNullArrayOrCheckArraySize(); }

 private:
  void appendNullArray() {
    auto current_data_byte_size = data_buffer_bytes_.size();
    data_buffer_bytes_.resize(current_data_byte_size + array_size_);
    auto omnisci_data_ptr = data_buffer_bytes_.data() + current_data_byte_size;
    setNullArraySentinel(omnisci_data_ptr);
    size_t array_count = array_size_ / omnisci_data_type_byte_size_;
    for (size_t i = 1; i < array_count; ++i) {
      scalar_encoder_->setNull(omnisci_data_ptr + i * omnisci_data_type_byte_size_);
    }
  }

  void setNullArraySentinel(int8_t* omnisci_data_bytes) {
    auto ti = column_desciptor_.columnType.get_elem_type();
    if (ti.is_string()) {
      // TODO: after investigation as to why fixed length arrays with
      // strings can not represent null arrays, either fix this error
      // or erase this comment.
      throw std::runtime_error("Detected a null array being imported into OmniSci '" +
                               column_desciptor_.columnName +
                               "' column which has a fixed length array type of "
                               "dictionary encoded text. Currently "
                               "null arrays for this type of column are not allowed.");
    }
    const auto type = ti.get_type();
    switch (type) {
      case kBOOLEAN:
        reinterpret_cast<bool*>(omnisci_data_bytes)[0] =
            inline_fixed_encoding_null_array_val(ti);
        break;
      case kBIGINT:
      case kNUMERIC:
      case kDECIMAL:
        reinterpret_cast<int64_t*>(omnisci_data_bytes)[0] =
            inline_fixed_encoding_null_array_val(ti);
        break;
      case kINT:
        reinterpret_cast<int32_t*>(omnisci_data_bytes)[0] =
            inline_fixed_encoding_null_array_val(ti);
        break;
      case kSMALLINT:
        reinterpret_cast<int16_t*>(omnisci_data_bytes)[0] =
            inline_fixed_encoding_null_array_val(ti);
        break;
      case kTINYINT:
        reinterpret_cast<int8_t*>(omnisci_data_bytes)[0] =
            inline_fixed_encoding_null_array_val(ti);
        break;
      case kFLOAT:
        reinterpret_cast<float*>(omnisci_data_bytes)[0] = NULL_ARRAY_FLOAT;
        break;
      case kDOUBLE:
        reinterpret_cast<double*>(omnisci_data_bytes)[0] = NULL_ARRAY_DOUBLE;
        break;
      case kTIME:
      case kTIMESTAMP:
      case kDATE:
        reinterpret_cast<int64_t*>(omnisci_data_bytes)[0] =
            inline_fixed_encoding_null_array_val(ti);
        break;
      case kTEXT:
      case kVARCHAR:
      case kCHAR:
      case kPOINT:
      case kLINESTRING:
      case kPOLYGON:
      case kMULTIPOLYGON:
      default:
        UNREACHABLE();
    }
  }

  void appendNullArrayOrCheckArraySize() {
    auto size_of_last_array = sizeOfLastArray();
    if (!isLastArrayNull()) {
      if (size_of_last_array * omnisci_data_type_byte_size_ != array_size_) {
        throw std::runtime_error(
            "Detected a row with " + std::to_string(size_of_last_array) +
            " elements being loaded into"
            " OmniSci column '" +
            column_desciptor_.columnName +
            "' which has a fixed length array type,"
            " expecting " +
            std::to_string(array_size_ / omnisci_data_type_byte_size_) + " elements.");
      }
    } else {
      // append a null array sentinel
      CHECK(size_of_last_array == 0);
      appendNullArray();
    }
  }

  const ColumnDescriptor column_desciptor_;
  size_t array_size_;
};
}  // namespace foreign_storage
