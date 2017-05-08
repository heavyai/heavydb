/*
 * Copyright 2017 MapD Technologies, Inc.
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

/*
 * @file    NullableValue.h
 * @author  Dmitri Shtilman <d@mapd.com>
 *
 * Copyright (c) 2016 MapD Technologies, Inc.  All rights reserved.
 */

#ifndef NULLABLEVALUE_H
#define NULLABLEVALUE_H

template <typename T>
class NullableValue {
 public:
  NullableValue() : state_(Invalid), value_(getDefaultValue()) {}
  NullableValue(T v) : state_(Valid), value_(v) {}

  bool isValid() const { return state_ == Valid; }
  bool isInvalid() const { return state_ == Invalid; }
  T getValue() const { return value_; }

  NullableValue<T> operator+(T v) const {
    if (isInvalid())
      return NullableValue<T>();
    return NullableValue<T>(value_ + v);
  }
  NullableValue<T> operator-(T v) const {
    if (isInvalid())
      return NullableValue<T>();
    return NullableValue<T>(value_ - v);
  }
  NullableValue<T> operator*(T v) const {
    if (isInvalid())
      return NullableValue<T>();
    return NullableValue<T>(value_ * v);
  }
  NullableValue<T> operator/(T v) const {
    if (isInvalid())
      return NullableValue<T>();
    return NullableValue<T>(value_ / v);
  }
  NullableValue<T> operator+(const NullableValue<T>& other) const {
    if (isInvalid() && other.isInvalid())
      return NullableValue<T>();
    return NullableValue<T>(value_ + other.getValue());
  }
  NullableValue<T> operator-(const NullableValue<T>& other) const {
    if (isInvalid() && other.isInvalid())
      return NullableValue<T>();
    return NullableValue<T>(value_ - other.getValue());
  }
  NullableValue<T> operator*(const NullableValue<T>& other) const {
    if (isInvalid() && other.isInvalid())
      return NullableValue<T>();
    return NullableValue<T>(value_ * other.getValue());
  }
  NullableValue<T> operator/(const NullableValue<T>& other) const {
    if (isInvalid() && other.isInvalid())
      return NullableValue<T>();
    return NullableValue<T>(value_ / other.getValue());
  }
  bool operator==(const T v) const { return isValid() ? (value_ == v) : false; }
  bool operator!=(const T v) const { return isValid() ? (value_ != v) : false; }
  bool operator>(const T v) const { return isValid() ? (value_ > v) : false; }
  bool operator>=(const T v) const { return isValid() ? (value_ >= v) : false; }
  bool operator<(const T v) const { return isValid() ? (value_ < v) : false; }
  bool operator<=(const T v) const { return isValid() ? (value_ <= v) : false; }

  static T getDefaultValue();

 private:
  enum State { Invalid, Valid };
  State state_;
  T value_;
};

typedef NullableValue<float> Likelihood;
template <>
float Likelihood::getDefaultValue() {
  return 0.5;
};

typedef NullableValue<uint64_t> Weight;
template <>
uint64_t Weight::getDefaultValue() {
  return 1;
}

#endif  // NULLABLEVALUE_H
