/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "ParquetInPlaceEncoder.h"

namespace foreign_storage {

// The following semantics apply to the templated types below.
//
// V - type of omnisci data
// T - physical type of parquet data
// conversion_denominator - the denominator constant used in converting parquet to omnisci
// data
// NullType - type of encoded null
//
// The `conversion_denominator` template is used instead of a class member to
// specify it at compile-time versus run-time. In testing this has a major
// impact on the runtime of the conversion performed by this encoder since the
// compiler can significantly optimize if this is known at compile time.
template <typename V, typename T, T conversion_denominator, typename NullType = V>
class ParquetTimeEncoder : public TypedParquetInPlaceEncoder<V, T, NullType> {
 public:
  ParquetTimeEncoder(Data_Namespace::AbstractBuffer* buffer,
                     const ColumnDescriptor* column_desciptor,
                     const parquet::ColumnDescriptor* parquet_column_descriptor)
      : TypedParquetInPlaceEncoder<V, T, NullType>(buffer,
                                                   column_desciptor,
                                                   parquet_column_descriptor) {
    CHECK(parquet_column_descriptor->logical_type()->is_time());
  }

  void encodeAndCopy(const int8_t* parquet_data_bytes,
                     int8_t* omnisci_data_bytes) override {
    const auto& parquet_data_value = reinterpret_cast<const T*>(parquet_data_bytes)[0];
    auto& omnisci_data_value = reinterpret_cast<V*>(omnisci_data_bytes)[0];
    omnisci_data_value = parquet_data_value / conversion_denominator;
  }
};

}  // namespace foreign_storage
