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

#include "ParquetEncoder.h"
#include "SharedMetadataValidator.h"

namespace foreign_storage {
class ParquetMetadataValidator {
 public:
  virtual ~ParquetMetadataValidator() = default;

  virtual void validate(std::shared_ptr<parquet::Statistics> stats,
                        const SQLTypeInfo& column_type) const = 0;
};

template <typename D, typename T>
inline bool check_bounds(const T& value) {
  auto [min_value, max_value] = get_min_max_bounds<D>();
  return value >= min_value && value <= max_value;
}

template <typename D>
inline std::string datetime_to_string(const D& timestamp,
                                      const SQLTypeInfo& column_type) {
  CHECK(column_type.is_timestamp() || column_type.is_date());
  Datum d;
  d.bigintval = timestamp;
  return DatumToString(d, column_type);
}

inline void throw_parquet_metadata_out_of_bounds_error(
    const std::string& min_value,
    const std::string& max_value,
    const std::string& encountered_value) {
  std::stringstream error_message;
  error_message << "Parquet column contains values that are outside the range of the "
                   "OmniSci column "
                   "type. Consider using a wider column type. Min allowed value: "
                << min_value << ". Max allowed value: " << max_value
                << ". Encountered value: " << encountered_value << ".";
  throw std::runtime_error(error_message.str());
}

template <typename T>
class TimestampBoundsValidator {
  static_assert(std::is_integral<T>::value && std::is_signed<T>::value,
                "TimestampBoundsValidator is only defined for signed integral types.");

 public:
  template <typename D>
  static void validateValue(const D& data_value, const SQLTypeInfo& column_type) {
    if (!valueWithinBounds(data_value, column_type)) {
      auto [min_allowed_value, max_allowed_value] = getMinMaxBoundsAsStrings(column_type);
      throw_parquet_metadata_out_of_bounds_error(
          min_allowed_value,
          max_allowed_value,
          datetime_to_string(data_value, column_type));
    }
  }

 private:
  static bool valueWithinBounds(const T& value, const SQLTypeInfo& column_type) {
    CHECK(column_type.is_timestamp());
    switch (column_type.get_size()) {
      case 4:
        return check_bounds<int32_t>(value);
      case 8:
        return check_bounds<int64_t>(value);
      default:
        UNREACHABLE();
    }
    return {};
  }

  static std::pair<std::string, std::string> getMinMaxBoundsAsStrings(
      const SQLTypeInfo& column_type) {
    CHECK(column_type.is_timestamp());
    switch (column_type.get_size()) {
      case 4:
        return getMinMaxBoundsAsStrings<int32_t>(column_type);
      case 8:
        return getMinMaxBoundsAsStrings<int64_t>(column_type);
      default:
        UNREACHABLE();
    }
    return {};
  }

  template <typename D>
  static std::pair<std::string, std::string> getMinMaxBoundsAsStrings(
      const SQLTypeInfo& column_type) {
    auto [min_value, max_value] = get_min_max_bounds<D>();
    return {datetime_to_string(min_value, column_type),
            datetime_to_string(max_value, column_type)};
  }
};

template <typename T>
class IntegralFixedLengthBoundsValidator {
  static_assert(std::is_integral<T>::value,
                "IntegralFixedLengthBoundsValidator is only defined for integral types.");

 public:
  template <typename D>
  static void validateValue(const D& data_value, const SQLTypeInfo& column_type) {
    if (!valueWithinBounds(data_value, column_type)) {
      auto [min_allowed_value, max_allowed_value] = getMinMaxBoundsAsStrings(column_type);
      if (std::is_signed<T>::value) {
        throw_parquet_metadata_out_of_bounds_error(
            min_allowed_value, max_allowed_value, std::to_string(data_value));
      } else {
        throw_parquet_metadata_out_of_bounds_error(
            min_allowed_value,
            max_allowed_value,
            std::to_string(static_cast<T>(data_value)));
      }
    }
  }

 private:
  static bool valueWithinBounds(const T& value, const SQLTypeInfo& column_type) {
    CHECK(column_type.is_integer());
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
    CHECK(column_type.is_integer());
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
    return check_bounds<D>(value);
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
    auto [min_value, max_value] = get_min_max_bounds<D>();
    auto signed_value = static_cast<D>(value);
    return signed_value >= 0 && signed_value <= max_value;
  }

  template <typename D>
  static std::pair<std::string, std::string> getMinMaxBoundsAsStrings() {
    auto [min_value, max_value] = get_min_max_bounds<D>();
    return {std::to_string(min_value), std::to_string(max_value)};
  }
};

template <typename T>
class DateInSecondsBoundsValidator {
  static_assert(
      std::is_integral<T>::value && std::is_signed<T>::value,
      "DateInSecondsBoundsValidator is only defined for signed integral types.");

 public:
  template <typename D>
  static void validateValue(const D& data_value, const SQLTypeInfo& column_type) {
    if (!valueWithinBounds(data_value, column_type)) {
      auto [min_allowed_value, max_allowed_value] = getMinMaxBoundsAsStrings(column_type);
      throw_parquet_metadata_out_of_bounds_error(
          min_allowed_value,
          max_allowed_value,
          datetime_to_string(data_value, column_type));
    }
  }

 private:
  static bool valueWithinBounds(const T& value, const SQLTypeInfo& column_type) {
    CHECK(column_type.is_date());
    switch (column_type.get_size()) {
      case 4:
        return checkBounds<int32_t>(value);
      case 2:
        return checkBounds<int16_t>(value);
      default:
        UNREACHABLE();
    }
    return {};
  }

  static std::pair<std::string, std::string> getMinMaxBoundsAsStrings(
      const SQLTypeInfo& column_type) {
    CHECK(column_type.is_date());
    switch (column_type.get_size()) {
      case 4:
        return getMinMaxBoundsAsStrings<int32_t>(column_type);
      case 2:
        return getMinMaxBoundsAsStrings<int16_t>(column_type);
      default:
        UNREACHABLE();
    }
    return {};
  }

  template <typename D>
  static bool checkBounds(const T& value) {
    auto [min_value, max_value] = get_min_max_bounds<D>();
    return value >= kSecsPerDay * min_value && value <= kSecsPerDay * max_value;
  }

  template <typename D>
  static std::pair<std::string, std::string> getMinMaxBoundsAsStrings(
      const SQLTypeInfo& column_type) {
    auto [min_value, max_value] = get_min_max_bounds<D>();
    return {datetime_to_string(kSecsPerDay * min_value, column_type),
            datetime_to_string(kSecsPerDay * max_value, column_type)};
  }
};

template <typename T>
class FloatPointValidator {
  static_assert(std::is_floating_point<T>::value,
                "FloatPointValidator is only defined for floating point types.");

 public:
  template <typename D>
  static void validateValue(const D& data_value, const SQLTypeInfo& column_type) {
    if (!valueWithinBounds(data_value, column_type)) {
      auto [min_allowed_value, max_allowed_value] = getMinMaxBoundsAsStrings(column_type);
      throw_parquet_metadata_out_of_bounds_error(
          min_allowed_value, max_allowed_value, std::to_string(data_value));
    }
  }

 private:
  static bool valueWithinBounds(const T& value, const SQLTypeInfo& column_type) {
    CHECK(column_type.is_fp());
    switch (column_type.get_size()) {
      case 4:
        return checkBounds<float>(value);
      case 8:
        return checkBounds<double>(value);
      default:
        UNREACHABLE();
    }
    return {};
  }

  static std::pair<std::string, std::string> getMinMaxBoundsAsStrings(
      const SQLTypeInfo& column_type) {
    CHECK(column_type.is_fp());
    switch (column_type.get_size()) {
      case 4:
        return getMinMaxBoundsAsStrings<float>();
      case 8:
        return getMinMaxBoundsAsStrings<double>();
      default:
        UNREACHABLE();
    }
    return {};
  }

  template <typename D>
  static bool checkBounds(const T& value) {
    return check_bounds<D>(value);
  }

  template <typename D>
  static std::pair<std::string, std::string> getMinMaxBoundsAsStrings() {
    auto [min_value, max_value] = get_min_max_bounds<D>();
    return {std::to_string(min_value), std::to_string(max_value)};
  }
};

}  // namespace foreign_storage
