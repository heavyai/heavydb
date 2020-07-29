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

#include <type_traits>

/**
 * Template operators to add typesafe bitwise operators to enum classes.
 * These are intended for use with mask types, so shift operators are
 * not provided as they can lead to invalid bits being set.
 *
 * To avoid nest namespace issues, ENABLE_BITMASK_OPS must be used outside any namespaces
 *
 * Example Usage:
 * namespace my_namespace {
 * enum class MyEnum : uint8_t {
 *   kEmpty = 0x00,
 *   kSomething = 0x01,
 *   kAnother = 0x02
 * };
 * } // namespace my_namespace
 *
 * ENABLE_BITMASK_OPS(::my_namespace::MyEnum)
 * MyEnum e = MyEnum::kSomething & MyEnum::kAnother;
 * */

template <typename T>
struct EnableBitmaskOps {
  static const bool enable = false;
};

template <typename T>
typename std::enable_if_t<EnableBitmaskOps<T>::enable, T> operator&(T lhs, T rhs) {
  using type = typename std::underlying_type_t<T>;
  return static_cast<T>(static_cast<type>(lhs) & static_cast<type>(rhs));
}
template <typename T>
typename std::enable_if_t<EnableBitmaskOps<T>::enable, T> operator|(T lhs, T rhs) {
  using type = typename std::underlying_type_t<T>;
  return static_cast<T>(static_cast<type>(lhs) | static_cast<type>(rhs));
}
template <typename T>
typename std::enable_if_t<EnableBitmaskOps<T>::enable, T> operator~(T t) {
  return static_cast<T>(~static_cast<std::underlying_type_t<T>>(t));
}
template <typename T>
typename std::enable_if_t<EnableBitmaskOps<T>::enable, T> operator|=(T& lhs, T rhs) {
  lhs = lhs | rhs;
  return lhs;
}
template <typename T>
typename std::enable_if_t<EnableBitmaskOps<T>::enable, T> operator&=(T& lhs, T rhs) {
  lhs = lhs & rhs;
  return lhs;
}

#define ENABLE_BITMASK_OPS(x)        \
  template <>                        \
  struct EnableBitmaskOps<x> {       \
    static const bool enable = true; \
  };
