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

#include "ParquetTimestampEncoder.h"

namespace foreign_storage {

// The following semantics apply to the templated types below.
//
// V - type of omnisci data
// T - physical type of parquet data
// conversion_denominator - the denominator constant used in converting parquet to omnisci
// data
//
// The `conversion_denominator` template is used instead of a class member to
// specify it at compile-time versus run-time. In testing this has a major
// impact on the runtime of the conversion performed by this encoder since the
// compiler can significantly optimize if this is known at compile time.
template <typename V, typename T, T conversion_denominator, typename NullType = V>
class ParquetDateInDaysFromTimestampEncoder
    : public ParquetTimestampEncoder<V,
                                     T,
                                     conversion_denominator * kSecsPerDay,
                                     NullType> {
 public:
  ParquetDateInDaysFromTimestampEncoder(
      Data_Namespace::AbstractBuffer* buffer,
      const ColumnDescriptor* column_desciptor,
      const parquet::ColumnDescriptor* parquet_column_descriptor)
      : ParquetTimestampEncoder<V, T, conversion_denominator * kSecsPerDay, NullType>(
            buffer,
            column_desciptor,
            parquet_column_descriptor) {}

  void validate(const int8_t* parquet_data,
                const int64_t j,
                const SQLTypeInfo& column_type) const override {
    const auto& parquet_data_value = reinterpret_cast<const T*>(parquet_data)[j];
    CHECK(column_type.is_date());
    DateInDaysBoundsValidator<T>::validateValue(this->convert(parquet_data_value),
                                                column_type);
  }

  void validate(std::shared_ptr<parquet::Statistics> stats,
                const SQLTypeInfo& column_type) const override {
    UNREACHABLE() << "ParquetDateInDaysFromTimestampEncoder should never be used during "
                     "metadata scan"
                  << std::endl;
  }
};
}  // namespace foreign_storage
