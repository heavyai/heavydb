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

#include "ParquetInPlaceEncoder.h"

namespace foreign_storage {

// ParquetFixedLengthEncoder is used in two separate use cases: metadata
// scanning & chunk loading.  During metadata scan the type of metadata (& in
// some cases data) must be known, while during chunk loading only the type of
// data needs to be known.
//
// The following semantics apply to the templated types below.
//
// At metadata scan:
// V - type of metadata (for loading metadata)
// T - physical type of parquet data
//
// At chunk load:
// V - type of data (to load data)
// T - physical type of parquet data
// NullType - the type to use for encoding nulls
template <typename V, typename T, typename NullType = V>
class ParquetFixedLengthEncoder : public TypedParquetInPlaceEncoder<V, T, NullType>,
                                  public ParquetMetadataValidator {
 public:
  ParquetFixedLengthEncoder(Data_Namespace::AbstractBuffer* buffer,
                            const ColumnDescriptor* column_desciptor,
                            const parquet::ColumnDescriptor* parquet_column_descriptor)
      : TypedParquetInPlaceEncoder<V, T, NullType>(buffer,
                                                   column_desciptor,
                                                   parquet_column_descriptor) {}

  ParquetFixedLengthEncoder(Data_Namespace::AbstractBuffer* buffer,
                            const size_t omnisci_data_type_byte_size,
                            const size_t parquet_data_type_byte_size)
      : TypedParquetInPlaceEncoder<V, T, NullType>(buffer,
                                                   omnisci_data_type_byte_size,
                                                   parquet_data_type_byte_size) {}

  void encodeAndCopy(const int8_t* parquet_data_bytes,
                     int8_t* omnisci_data_bytes) override {
    const auto& parquet_data_value = reinterpret_cast<const T*>(parquet_data_bytes)[0];
    auto& omnisci_data_value = reinterpret_cast<V*>(omnisci_data_bytes)[0];
    omnisci_data_value = parquet_data_value;
  }

  void validate(std::shared_ptr<parquet::Statistics> stats,
                const SQLTypeInfo& column_type) const override {
    validateIntegralOrFloatingPointMetadata(stats, column_type);
  }

  void validate(const int8_t* parquet_data,
                const int64_t j,
                const SQLTypeInfo& column_type) const override {
    const auto& parquet_data_value = reinterpret_cast<const T*>(parquet_data)[j];
    validateIntegralOrFloatingPointValue(parquet_data_value, column_type);
  }

  bool encodingIsIdentityForSameTypes() const override { return true; }

 private:
  template <
      typename TT = T,
      std::enable_if_t<(!std::is_integral<TT>::value || std::is_same<TT, bool>::value) &&
                           !std::is_floating_point<TT>::value,
                       int> = 0>
  void validateIntegralOrFloatingPointValue(const T& value,
                                            const SQLTypeInfo& column_type) const {
    // do nothing when type `T` is non-integral and non-floating-point (case
    // for which this can happen are when `T` is bool)
  }

  template <typename TT = T, std::enable_if_t<std::is_floating_point<TT>::value, int> = 0>
  void validateIntegralOrFloatingPointValue(const T& value,
                                            const SQLTypeInfo& column_type) const {
    if (column_type.is_fp()) {
      FloatPointValidator<T>::validateValue(value, column_type);
    } else {
      UNREACHABLE();
    }
  }

  template <
      typename TT = T,
      std::enable_if_t<std::is_integral<TT>::value && !std::is_same<TT, bool>::value,
                       int> = 0>
  void validateIntegralOrFloatingPointValue(const T& value,
                                            const SQLTypeInfo& column_type) const {
    if (column_type.is_integer()) {
      IntegralFixedLengthBoundsValidator<T>::validateValue(value, column_type);
    } else if (column_type.is_timestamp()) {
      TimestampBoundsValidator<T>::validateValue(value, column_type);
    } else if (column_type.is_date()) {
      DateInDaysBoundsValidator<T>::validateValue(value, column_type);
    }
  }

  void validateIntegralOrFloatingPointMetadata(std::shared_ptr<parquet::Statistics> stats,
                                               const SQLTypeInfo& column_type) const {
    if (!column_type.is_integer() && !column_type.is_timestamp() &&
        !column_type.is_fp()) {
      return;
    }
    auto [unencoded_stats_min, unencoded_stats_max] =
        TypedParquetInPlaceEncoder<V, T, NullType>::getUnencodedStats(stats);
    validateIntegralOrFloatingPointValue(unencoded_stats_min, column_type);
    validateIntegralOrFloatingPointValue(unencoded_stats_max, column_type);
  }
};

// ParquetUnsignedFixedLengthEncoder is used in two separate use cases:
// metadata scanning & chunk loading.  During metadata scan the type of
// metadata (& in some cases data) must be known, while during chunk loading
// only the type of data needs to be known.
//
// The following semantics apply to the templated types below.
//
// At metadata scan:
// V - type of metadata (for loading metadata)
// T - physical type of parquet data
// U - unsigned type that the parquet data represents
//
// At chunk load:
// V - type of data (to load data)
// T - physical type of parquet data
// U - unsigned type that the parquet data represents
// NullType - the type to use for encoding nulls
template <typename V, typename T, typename U, typename NullType = V>
class ParquetUnsignedFixedLengthEncoder
    : public TypedParquetInPlaceEncoder<V, T, NullType>,
      public ParquetMetadataValidator {
 public:
  ParquetUnsignedFixedLengthEncoder(
      Data_Namespace::AbstractBuffer* buffer,
      const ColumnDescriptor* column_desciptor,
      const parquet::ColumnDescriptor* parquet_column_descriptor)
      : TypedParquetInPlaceEncoder<V, T, NullType>(buffer,
                                                   column_desciptor,
                                                   parquet_column_descriptor) {}

  ParquetUnsignedFixedLengthEncoder(Data_Namespace::AbstractBuffer* buffer,
                                    const size_t omnisci_data_type_byte_size,
                                    const size_t parquet_data_type_byte_size)
      : TypedParquetInPlaceEncoder<V, T, NullType>(buffer,
                                                   omnisci_data_type_byte_size,
                                                   parquet_data_type_byte_size) {}

  void encodeAndCopy(const int8_t* parquet_data_bytes,
                     int8_t* omnisci_data_bytes) override {
    const auto& parquet_data_value = reinterpret_cast<const T*>(parquet_data_bytes)[0];
    auto& omnisci_data_value = reinterpret_cast<V*>(omnisci_data_bytes)[0];
    omnisci_data_value = static_cast<U>(parquet_data_value);
  }

  void validate(std::shared_ptr<parquet::Statistics> stats,
                const SQLTypeInfo& column_type) const override {
    if (!column_type.is_integer()) {  // do not validate non-integral types
      return;
    }
    auto [unencoded_stats_min, unencoded_stats_max] =
        TypedParquetInPlaceEncoder<V, T, NullType>::getUnencodedStats(stats);
    IntegralFixedLengthBoundsValidator<U>::validateValue(unencoded_stats_max,
                                                         column_type);
    IntegralFixedLengthBoundsValidator<U>::validateValue(unencoded_stats_min,
                                                         column_type);
  }

  void validate(const int8_t* parquet_data,
                const int64_t j,
                const SQLTypeInfo& column_type) const override {
    const auto& parquet_data_value = reinterpret_cast<const T*>(parquet_data)[j];
    IntegralFixedLengthBoundsValidator<U>::validateValue(parquet_data_value, column_type);
  }
};

}  // namespace foreign_storage
