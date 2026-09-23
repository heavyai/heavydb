/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <any>

#include "QueryRenderer/Data/Types.h"
#include "QueryRenderer/Utils/NumericUtils.h"

namespace QueryRenderer {

template <typename T>
T convertType(const QueryDataType type,
              const std::any& value,
              const bool ignore_null = false) {
  switch (type) {
    case QueryDataType::UINT: {
      unsigned int val = std::any_cast<unsigned int>(value);
      return !ignore_null && isNullValue(val) ? getNullValue<T>() : static_cast<T>(val);
    }
    case QueryDataType::INT: {
      int val = std::any_cast<int>(value);
      return !ignore_null && isNullValue(val) ? getNullValue<T>() : static_cast<T>(val);
    }
    case QueryDataType::FLOAT: {
      float val = std::any_cast<float>(value);
      return !ignore_null && isNullValue(val) ? getNullValue<T>() : static_cast<T>(val);
    }
    case QueryDataType::DOUBLE: {
      double val = std::any_cast<double>(value);
      return !ignore_null && isNullValue(val) ? getNullValue<T>() : static_cast<T>(val);
    }
    case QueryDataType::UINT64: {
      uint64_t val = std::any_cast<uint64_t>(value);
      return !ignore_null && isNullValue(val) ? getNullValue<T>() : static_cast<T>(val);
    }
    case QueryDataType::INT64: {
      int64_t val = std::any_cast<int64_t>(value);
      return !ignore_null && isNullValue(val) ? getNullValue<T>() : static_cast<T>(val);
    }
    case QueryDataType::BOOL: {
      bool val = std::any_cast<bool>(value);
      return static_cast<T>(val);
    }
    case QueryDataType::LINE_JOIN_ENUM:
    case QueryDataType::SYMBOL_SHAPE_ENUM:
    case QueryDataType::ANGLE_UNIT_ENUM: {
      int val;
      try {
        val = std::any_cast<unsigned int>(value);
      } catch (const std::bad_any_cast&) {
        val = std::any_cast<int>(value);
      }
      return static_cast<T>(val);
    }
    case QueryDataType::STRING:
    case QueryDataType::COLOR:
    case QueryDataType::POLYGON_DOUBLE:
    case QueryDataType::LINE_DOUBLE:
      THROW_RUNTIME_EX("Converting " + to_string(type) + " to " + typeid(T).name() +
                       " is currently unsupported.");
  }

  return T();
}

namespace details {
template <typename T>
std::vector<T> convertVectorType(const QueryDataType type,
                                 const std::any& value,
                                 const bool ignore_null = false) {
  std::vector<T> rtn;
  switch (type) {
    case QueryDataType::UINT: {
      auto val = std::any_cast<std::vector<unsigned int>>(value);
      rtn.resize(val.size());
      for (size_t i = 0; i < val.size(); ++i) {
        rtn[i] = !ignore_null && isNullValue(val[i]) ? getNullValue<T>()
                                                     : static_cast<T>(val[i]);
      }
      break;
    }
    case QueryDataType::INT: {
      auto val = std::any_cast<std::vector<int>>(value);
      rtn.resize(val.size());
      for (size_t i = 0; i < val.size(); ++i) {
        rtn[i] = !ignore_null && isNullValue(val[i]) ? getNullValue<T>()
                                                     : static_cast<T>(val[i]);
      }
      break;
    }
    case QueryDataType::FLOAT: {
      auto val = std::any_cast<std::vector<float>>(value);
      rtn.resize(val.size());
      for (size_t i = 0; i < val.size(); ++i) {
        rtn[i] = !ignore_null && isNullValue(val[i]) ? getNullValue<T>()
                                                     : static_cast<T>(val[i]);
      }
      break;
    }
    case QueryDataType::DOUBLE: {
      auto val = std::any_cast<std::vector<double>>(value);
      rtn.resize(val.size());
      for (size_t i = 0; i < val.size(); ++i) {
        rtn[i] = !ignore_null && isNullValue(val[i]) ? getNullValue<T>()
                                                     : static_cast<T>(val[i]);
      }
      break;
    }
    case QueryDataType::UINT64: {
      auto val = std::any_cast<std::vector<uint64_t>>(value);
      rtn.resize(val.size());
      for (size_t i = 0; i < val.size(); ++i) {
        rtn[i] = !ignore_null && isNullValue(val[i]) ? getNullValue<T>()
                                                     : static_cast<T>(val[i]);
      }
      break;
    }
    case QueryDataType::INT64: {
      auto val = std::any_cast<std::vector<int64_t>>(value);
      rtn.resize(val.size());
      for (size_t i = 0; i < val.size(); ++i) {
        rtn[i] = !ignore_null && isNullValue(val[i]) ? getNullValue<T>()
                                                     : static_cast<T>(val[i]);
      }
      break;
    }
    case QueryDataType::BOOL: {
      auto val = std::any_cast<std::vector<bool>>(value);
      rtn.resize(val.size());
      for (size_t i = 0; i < val.size(); ++i) {
        rtn[i] = static_cast<T>(val[i]);
      }
      break;
    }
    case QueryDataType::LINE_JOIN_ENUM:
    case QueryDataType::SYMBOL_SHAPE_ENUM:
    case QueryDataType::ANGLE_UNIT_ENUM: {
      auto val = std::any_cast<std::vector<unsigned int>>(value);
      rtn.resize(val.size());
      for (size_t i = 0; i < val.size(); ++i) {
        rtn[i] = static_cast<T>(val[i]);
      }
      break;
    }
    case QueryDataType::STRING:
    case QueryDataType::COLOR:
    case QueryDataType::POLYGON_DOUBLE:
    case QueryDataType::LINE_DOUBLE:
      THROW_RUNTIME_EX("Converting an vector of type " + to_string(type) + " to " +
                       typeid(T).name() + " is currently unsupported.");
  }
  return rtn;
}

template <typename T>
T convertTypeAtIndex(const QueryDataType type,
                     const std::any& value,
                     const size_t idx,
                     const bool ignore_null = false) {
  switch (type) {
    case QueryDataType::UINT: {
      auto val = std::any_cast<std::vector<unsigned int>>(value);
      CHECK_LE(idx, val.size());
      return !ignore_null && isNullValue(val[idx]) ? getNullValue<T>()
                                                   : static_cast<T>(val[idx]);
      break;
    }
    case QueryDataType::INT: {
      auto val = std::any_cast<std::vector<int>>(value);
      CHECK_LE(idx, val.size());
      return !ignore_null && isNullValue(val[idx]) ? getNullValue<T>()
                                                   : static_cast<T>(val[idx]);
      break;
    }
    case QueryDataType::FLOAT: {
      auto val = std::any_cast<std::vector<float>>(value);
      CHECK_LE(idx, val.size());
      return !ignore_null && isNullValue(val[idx]) ? getNullValue<T>()
                                                   : static_cast<T>(val[idx]);
      break;
    }
    case QueryDataType::DOUBLE: {
      auto val = std::any_cast<std::vector<double>>(value);
      CHECK_LE(idx, val.size());
      return !ignore_null && isNullValue(val[idx]) ? getNullValue<T>()
                                                   : static_cast<T>(val[idx]);
      break;
    }
    case QueryDataType::UINT64: {
      auto val = std::any_cast<std::vector<uint64_t>>(value);
      CHECK_LE(idx, val.size());
      return !ignore_null && isNullValue(val[idx]) ? getNullValue<T>()
                                                   : static_cast<T>(val[idx]);
      break;
    }
    case QueryDataType::INT64: {
      auto val = std::any_cast<std::vector<int64_t>>(value);
      CHECK_LE(idx, val.size());
      return !ignore_null && isNullValue(val[idx]) ? getNullValue<T>()
                                                   : static_cast<T>(val[idx]);
      break;
    }
    case QueryDataType::BOOL: {
      auto val = std::any_cast<std::vector<bool>>(value);
      CHECK_LE(idx, val.size());
      return static_cast<T>(val[idx]);
      break;
    }
    case QueryDataType::LINE_JOIN_ENUM:
    case QueryDataType::SYMBOL_SHAPE_ENUM:
    case QueryDataType::ANGLE_UNIT_ENUM: {
      auto val = std::any_cast<std::vector<unsigned int>>(value);
      CHECK_LE(idx, val.size());
      return static_cast<T>(val[idx]);
      break;
    }
    case QueryDataType::STRING:
    case QueryDataType::COLOR:
    case QueryDataType::POLYGON_DOUBLE:
    case QueryDataType::LINE_DOUBLE:
      THROW_RUNTIME_EX("Converting an vector of type " + to_string(type) + " to " +
                       typeid(T).name() + " is currently unsupported.");
  }
  return T();
}
}  // namespace details

class AnyDataType {
 private:
  using SizeType = int64_t;
  QueryDataType type_;
  std::any value_;
  SizeType size_;

 public:
  AnyDataType() : type_{QueryDataType::INT}, value_{int(0)}, size_{-1} {}
  AnyDataType(const QueryDataType type, const std::any& value)
      : type_{type}, value_{value}, size_{-1} {}

  template <typename T,
            typename std::enable_if_t<std::is_arithmetic<T>::value>* = nullptr>
  AnyDataType(const std::vector<T>& values)
      : type_{TypeToQueryDataTypeSelector<T>::getQueryDataType()}
      , value_{values}
      , size_{static_cast<SizeType>(values.size())} {}

  template <typename T,
            typename std::enable_if_t<std::is_arithmetic<T>::value>* = nullptr>
  AnyDataType(const std::vector<T>&& values)
      : type_{TypeToQueryDataTypeSelector<T>::getQueryDataType()}
      , value_{std::move(values)}
      , size_{static_cast<SizeType>(values.size())} {}

  ~AnyDataType() {}

  void set(const QueryDataType type, const std::any& val);

  template <typename T,
            typename std::enable_if_t<std::is_arithmetic<T>::value>* = nullptr>
  void set(const std::vector<T>& values) {
    type_ = TypeToQueryDataTypeSelector<T>::getQueryDataType();
    value_ = values;
    size_ = values.size();
  }

  template <typename T,
            typename std::enable_if_t<std::is_arithmetic<T>::value>* = nullptr>
  void set(const std::vector<T>&& values) {
    type_ = TypeToQueryDataTypeSelector<T>::getQueryDataType();
    value_ = std::move(values);
    size_ = values.size();
  }

  QueryDataType getType() const { return type_; }
  size_t size() const { return (size_ == -1 ? 1 : size_); }
  bool isVector() const { return size_ > -1; }

  template <typename T,
            typename std::enable_if_t<std::is_arithmetic<T>::value>* = nullptr>
  void convertToType() {
    // TODO(croot): check that when value_ is converted that it is lossless?
    if (size_ == -1) {
      value_ = convertType<T>(type_, value_);
    } else {
      value_ = details::convertVectorType<T>(type_, value_);
    }

    type_ = TypeToQueryDataTypeSelector<T>::getQueryDataType();
  }

  void convertToType(const QueryDataType new_type);

  template <typename T>
  T getVal() const {
    RUNTIME_EX_ASSERT(size_ == -1,
                      "Can only call getVal() on a singular value AnyDataType. This "
                      "particular type is a vector of size " +
                          std::to_string(size_) + " of type " + to_string(type_) + ".");

    if (!value_.has_value()) {
      return T();
    }

    try {
      return convertType<T>(type_, value_);
    } catch (std::bad_any_cast& err) {
      THROW_RUNTIME_EX(
          "Cannot get the value of the AnyDataType. Its internals have gotten out of "
          "sync. It's labeled as a " +
          to_string(type_) + " but its value is not. The value is a " +
          value_.type().name());
    }

    return T();
  }

  template <typename T>
  T getValAtIndex(const size_t idx) const {
    RUNTIME_EX_ASSERT(size_ != -1,
                      "Can only call getValAtIndex() on a vector valued AnyDataType. "
                      "This AnyDataType is a " +
                          std::string(*this) + ".");
    RUNTIME_EX_ASSERT(value_.has_value(),
                      "The AnyDataType is empty. Cannot get vector data.");
    RUNTIME_EX_ASSERT(idx < static_cast<size_t>(size_),
                      "Invalid index " + std::to_string(idx) +
                          ". It exceeds the array size of " + std::to_string(size_) +
                          ".");

    try {
      return details::convertTypeAtIndex<T>(type_, value_, idx);
    } catch (std::bad_any_cast& err) {
      THROW_RUNTIME_EX(
          "Cannot get the value of the AnyDataType. Its internals have gotten out of "
          "sync. It's labeled as a " +
          to_string(type_) + " but its value is not. The value is a " +
          value_.type().name());
    }

    return T();
  }

  template <typename T>
  std::vector<T> getVectorVal() const {
    RUNTIME_EX_ASSERT(size_ != -1,
                      "Can only call getVectorVal() on an vector value AnyDataType. This "
                      "AnyDataType is a " +
                          std::string(*this) + ".");

    if (!value_.has_value()) {
      return std::vector<T>();
    }

    try {
      return details::convertVectorType<T>(type_, value_);
    } catch (std::bad_any_cast& err) {
      THROW_RUNTIME_EX(
          "Cannot get the value of the AnyDataType. Its internals have gotten out of "
          "sync. It's labeled as a vector of "
          "type " +
          to_string(type_) + " with size " + std::to_string(size_) +
          " but its value is not. The value is a " + value_.type().name());
    }

    return std::vector<T>();
  }

  template <typename T>
  const std::vector<T>& getVectorRef() const {
    RUNTIME_EX_ASSERT(size_ != -1,
                      "Can only call getVectorRef() on an vector value AnyDataType. This "
                      "AnyDataType is a " +
                          std::string(*this) + ".");

    RUNTIME_EX_ASSERT(value_.has_value(),
                      "The AnyDataType is empty. Cannot get vector reference.");

    auto data_type = TypeToQueryDataTypeSelector<T>::getQueryDataType();
    RUNTIME_EX_ASSERT(data_type == type_,
                      "Cannot retrieve the vector reference. Attempting to get a vector "
                      "reference of type " +
                          to_string(data_type) +
                          ", but the AnyDataType is storing a vector of type " +
                          to_string(type_) + ". These must match.");
    try {
      return std::any_cast<const std::vector<T>&>(value_);
    } catch (std::bad_any_cast& err) {
      THROW_RUNTIME_EX(
          "Cannot get the vector reference of the AnyDataType. Its internals have gotten "
          "out of sync. It's labeled as "
          "a vector of "
          "type " +
          to_string(type_) + " with size " + std::to_string(size_) +
          " but its value is not. The value is a " + value_.type().name());
    }
  }

  template <typename T,
            typename std::enable_if_t<gfx::is_color_union<T>::value>* = nullptr>
  const T& getColorRef() const {
    RUNTIME_EX_ASSERT(size_ == -1,
                      "Can only call getColorRef() on a singular value AnyDataType. This "
                      "particular type is a vector of size " +
                          std::to_string(size_) + " of type " + to_string(type_) + ".");

    RUNTIME_EX_ASSERT(type_ == QueryDataType::COLOR,
                      "Cannot get reference to a color. The AnyDataType is marked as a " +
                          to_string(type_) + ".");

    RUNTIME_EX_ASSERT(
        value_.has_value(),
        "Cannot get reference to a color. The AnyDataType object is not initialized");

    try {
      return std::any_cast<const gfx::ColorUnion&>(value_);
    } catch (std::bad_any_cast& err) {
      THROW_RUNTIME_EX(
          "Cannot get the value of the AnyDataType. Its internals have gotten out of "
          "sync. It's labeled as a " +
          to_string(type_) + " but its value is not.");
    }
  }

  template <typename T,
            typename std::enable_if<gfx::is_color_union<T>::value>::type* = nullptr>
  T& getColorRef() {
    return const_cast<T&>(static_cast<const AnyDataType&>(*this).getColorRef<T>());
  }

  std::string getStringVal() const;

  bool operator==(const AnyDataType& other) const;
  bool operator!=(const AnyDataType& other) const { return !operator==(other); }

  AnyDataType& operator+=(const AnyDataType& other);

  operator std::string() const;

  // used as the compare function for std::min() calls
  static bool minCompare(const AnyDataType& lhs, const AnyDataType& rhs);

  // used as the compare function for std::max() calls. This is still a less-than
  // operator, but NULLs are handled differently than minCompare
  static bool maxCompare(const AnyDataType& lhs, const AnyDataType& rhs);

  friend bool minCompare(const AnyDataType& lhs, const AnyDataType& rhs);
  friend bool maxCompare(const AnyDataType& lhs, const AnyDataType& rhs);
};

}  // namespace QueryRenderer
