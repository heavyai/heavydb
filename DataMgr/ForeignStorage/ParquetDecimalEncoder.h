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

#include <arrow/util/decimal.h>
#include "ParquetInPlaceEncoder.h"

namespace foreign_storage {
template <typename V, typename T>
class ParquetDecimalEncoder : public TypedParquetInPlaceEncoder<V, T> {
 public:
  ParquetDecimalEncoder(Data_Namespace::AbstractBuffer* buffer,
                        const ColumnDescriptor* column_desciptor,
                        const parquet::ColumnDescriptor* parquet_column_descriptor)
      : TypedParquetInPlaceEncoder<V, T>(buffer,
                                         column_desciptor,
                                         parquet_column_descriptor)
      , parquet_column_type_length_(parquet_column_descriptor->type_length())
      , decimal_overflow_validator_(column_desciptor->columnType) {}

  void encodeAndCopy(const int8_t* parquet_data_bytes,
                     int8_t* omnisci_data_bytes) override {
    const auto& parquet_data_value = reinterpret_cast<const T*>(parquet_data_bytes)[0];
    auto& omnisci_data_value = reinterpret_cast<V*>(omnisci_data_bytes)[0];
    omnisci_data_value = getDecimal(parquet_data_value);
  }

  void validate(const int8_t* parquet_data,
                const int64_t j,
                const SQLTypeInfo& column_type) const override {
    const auto& parquet_data_value = reinterpret_cast<const T*>(parquet_data)[j];
    int64_t omnisci_data_value = getDecimal(parquet_data_value);
    decimal_overflow_validator_.validate(omnisci_data_value);
  }

 protected:
  int64_t getDecimal(const int32_t& parquet_data_value) const {
    return parquet_data_value;
  }

  int64_t getDecimal(const int64_t& parquet_data_value) const {
    return parquet_data_value;
  }

  int64_t getDecimal(const parquet::FixedLenByteArray& parquet_data_value) const {
    return convertDecimalByteArrayToInt(parquet_data_value.ptr,
                                        parquet_column_type_length_);
  }

  int64_t getDecimal(const parquet::ByteArray& parquet_data_value) const {
    return convertDecimalByteArrayToInt(parquet_data_value.ptr, parquet_data_value.len);
  }

  bool encodingIsIdentityForSameTypes() const override { return true; }

 private:
  int64_t convertDecimalByteArrayToInt(const uint8_t* byte_array,
                                       const int byte_array_size) const {
    auto result = arrow::Decimal128::FromBigEndian(byte_array, byte_array_size);
    CHECK(result.ok()) << result.status().message();
    auto& decimal = result.ValueOrDie();
    return static_cast<int64_t>(decimal);
  }

  const size_t parquet_column_type_length_;
  const DecimalOverflowValidator decimal_overflow_validator_;
};
}  // namespace foreign_storage
