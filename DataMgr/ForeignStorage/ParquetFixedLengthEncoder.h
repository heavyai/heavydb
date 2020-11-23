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

#include "ParquetInPlaceEncoder.h"

namespace foreign_storage {

template <typename T>
class IntegralFixedLengthBoundsValidator {
 public:
  static bool valueWithinBounds(const T& value, const SQLTypeInfo& column_type) {
    CHECK(column_type.is_integer() && std::is_integral<T>::value);
    switch (column_type.get_size()) {
      case 1:
        return checkBounds<int8_t>(value);
      case 2:
        return checkBounds<int16_t>(value);
      case 4:
        return checkBounds<int32_t>(value);
      case 8:
        return checkBounds<int64_t>(value);
      default:
        UNREACHABLE();
    }
    return {};
  }

  static std::pair<std::string, std::string> getMinMaxBoundsAsStrings(
      const SQLTypeInfo& column_type) {
    CHECK(column_type.is_integer() && std::is_integral<T>::value);
    switch (column_type.get_size()) {
      case 1:
        return getMinMaxBoundsAsStrings<int8_t>();
      case 2:
        return getMinMaxBoundsAsStrings<int16_t>();
      case 4:
        return getMinMaxBoundsAsStrings<int32_t>();
      case 8:
        return getMinMaxBoundsAsStrings<int64_t>();
      default:
        UNREACHABLE();
    }
    return {};
  }

 private:
  /**
   * @brief Check bounds for value in _signed_ case
   *
   * @param value - value to check
   *
   * @return true if value within bounds
   */
  template <typename D,
            typename TT = T,
            std::enable_if_t<std::is_signed<TT>::value, int> = 0>
  static bool checkBounds(const T& value) {
    auto [min_value, max_value] = getMinMaxBounds<D>();
    return value >= min_value && value <= max_value;
  }

  /**
   * @brief Check bounds for value in _unsigned_ case
   *
   * @param value - value to check
   *
   * @return true if value within bounds
   */
  template <typename D,
            typename TT = T,
            std::enable_if_t<!std::is_signed<TT>::value, int> = 0>
  static bool checkBounds(const T& value) {
    auto [min_value, max_value] = getMinMaxBounds<D>();
    auto signed_value = static_cast<D>(value);
    return signed_value >= 0 && signed_value <= max_value;
  }

  template <typename D>
  static std::pair<D, D> getMinMaxBounds() {
    return {get_null_value<D>() + 1, std::numeric_limits<D>::max()};
  }

  template <typename D>
  static std::pair<std::string, std::string> getMinMaxBoundsAsStrings() {
    auto [min_value, max_value] = getMinMaxBounds<D>();
    return {std::to_string(+min_value), std::to_string(+max_value)};
  }
};

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
template <typename V, typename T>
class ParquetFixedLengthEncoder : public TypedParquetInPlaceEncoder<V, T>,
                                  public ParquetMetadataValidator {
 public:
  ParquetFixedLengthEncoder(Data_Namespace::AbstractBuffer* buffer,
                            const ColumnDescriptor* column_desciptor,
                            const parquet::ColumnDescriptor* parquet_column_descriptor)
      : TypedParquetInPlaceEncoder<V, T>(buffer,
                                         column_desciptor,
                                         parquet_column_descriptor) {}

  ParquetFixedLengthEncoder(Data_Namespace::AbstractBuffer* buffer,
                            const size_t omnisci_data_type_byte_size,
                            const size_t parquet_data_type_byte_size)
      : TypedParquetInPlaceEncoder<V, T>(buffer,
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
    if (!column_type.is_integer()) {  // do not validate non-integral types
      return;
    }
    auto [unencoded_stats_min, unencoded_stats_max] =
        TypedParquetInPlaceEncoder<V, T>::getUnencodedStats(stats);
    validateValue(unencoded_stats_max, column_type);
    validateValue(unencoded_stats_min, column_type);
  }

  bool encodingIsIdentityForSameTypes() const override { return true; }

 private:
  void validateValue(const T& parquet_data_value, const SQLTypeInfo& column_type) const {
    if (!IntegralFixedLengthBoundsValidator<T>::valueWithinBounds(parquet_data_value,
                                                                  column_type)) {
      auto [min_allowed_value, max_allowed_value] =
          IntegralFixedLengthBoundsValidator<T>::getMinMaxBoundsAsStrings(column_type);
      std::stringstream error_message;
      error_message << "Parquet column contains values that are outside the range of the "
                       "OmniSci column "
                       "type. Consider using a wider column type. Min allowed value: "
                    << min_allowed_value << ". Max allowed value: " << max_allowed_value
                    << ". Encountered value: " << +parquet_data_value << ".";
      throw std::runtime_error(error_message.str());
    }
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
template <typename V, typename T, typename U>
class ParquetUnsignedFixedLengthEncoder : public TypedParquetInPlaceEncoder<V, T>,
                                          public ParquetMetadataValidator {
 public:
  ParquetUnsignedFixedLengthEncoder(
      Data_Namespace::AbstractBuffer* buffer,
      const ColumnDescriptor* column_desciptor,
      const parquet::ColumnDescriptor* parquet_column_descriptor)
      : TypedParquetInPlaceEncoder<V, T>(buffer,
                                         column_desciptor,
                                         parquet_column_descriptor) {}

  ParquetUnsignedFixedLengthEncoder(Data_Namespace::AbstractBuffer* buffer,
                                    const size_t omnisci_data_type_byte_size,
                                    const size_t parquet_data_type_byte_size)
      : TypedParquetInPlaceEncoder<V, T>(buffer,
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
        TypedParquetInPlaceEncoder<V, T>::getUnencodedStats(stats);
    validateValue(unencoded_stats_max, column_type);
    validateValue(unencoded_stats_min, column_type);
  }

 private:
  void validateValue(const T& parquet_data_value, const SQLTypeInfo& column_type) const {
    U unsigned_parquet_data_value = static_cast<U>(parquet_data_value);
    if (!IntegralFixedLengthBoundsValidator<U>::valueWithinBounds(parquet_data_value,
                                                                  column_type)) {
      auto [min_allowed_value, max_allowed_value] =
          IntegralFixedLengthBoundsValidator<U>::getMinMaxBoundsAsStrings(column_type);
      std::stringstream error_message;
      error_message << "Parquet column contains values that are outside the range of the "
                       "OmniSci column "
                       "type. Consider using a wider column type. Min allowed value: "
                    << min_allowed_value << ". Max allowed value: " << max_allowed_value
                    << ". Encountered value: " << +unsigned_parquet_data_value << ".";
      throw std::runtime_error(error_message.str());
    }
  }
};

}  // namespace foreign_storage
