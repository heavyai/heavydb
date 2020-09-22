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
      , parquet_column_type_length_(parquet_column_descriptor->type_length()) {}

  void encodeAndCopy(const int8_t* parquet_data_bytes,
                     int8_t* omnisci_data_bytes) override {
    const auto& parquet_data_value = reinterpret_cast<const T*>(parquet_data_bytes)[0];
    auto& omnisci_data_value = reinterpret_cast<V*>(omnisci_data_bytes)[0];
    encodeAndCopy(parquet_data_value, omnisci_data_value);
  }

 protected:
  void encodeAndCopy(const int32_t& parquet_data_value, V& omnisci_data_value) {
    omnisci_data_value = parquet_data_value;
  }

  void encodeAndCopy(const int64_t& parquet_data_value, V& omnisci_data_value) {
    omnisci_data_value = parquet_data_value;
  }

  void encodeAndCopy(const parquet::FixedLenByteArray& parquet_data_value,
                     V& omnisci_data_value) {
    omnisci_data_value =
        convertDecimalByteArrayToInt(parquet_data_value.ptr, parquet_column_type_length_);
  }

  void encodeAndCopy(const parquet::ByteArray& parquet_data_value,
                     V& omnisci_data_value) {
    omnisci_data_value =
        convertDecimalByteArrayToInt(parquet_data_value.ptr, parquet_data_value.len);
  }

  bool encodingIsIdentityForSameTypes() const override { return true; }

 private:
  int64_t convertDecimalByteArrayToInt(const uint8_t* byte_array,
                                       const int byte_array_size) {
    auto result = arrow::Decimal128::FromBigEndian(byte_array, byte_array_size);
    CHECK(result.ok()) << result.status().message();
    auto& decimal = result.ValueOrDie();
    return static_cast<int64_t>(decimal);
  }

  const size_t parquet_column_type_length_;
};
}  // namespace foreign_storage
