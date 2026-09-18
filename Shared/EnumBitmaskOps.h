/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
  static constexpr bool enable = false;
};

template <typename T>
typename std::enable_if_t<EnableBitmaskOps<T>::enable, T> constexpr operator&(T lhs,
                                                                              T rhs) {
  using type = typename std::underlying_type_t<T>;
  return static_cast<T>(static_cast<type>(lhs) & static_cast<type>(rhs));
}
template <typename T>
typename std::enable_if_t<EnableBitmaskOps<T>::enable, T> constexpr operator|(T lhs,
                                                                              T rhs) {
  using type = typename std::underlying_type_t<T>;
  return static_cast<T>(static_cast<type>(lhs) | static_cast<type>(rhs));
}
template <typename T>
typename std::enable_if_t<EnableBitmaskOps<T>::enable, T> constexpr operator~(T t) {
  return static_cast<T>(~static_cast<std::underlying_type_t<T>>(t));
}
template <typename T>
typename std::enable_if_t<EnableBitmaskOps<T>::enable, T> constexpr operator|=(T& lhs,
                                                                               T rhs) {
  lhs = lhs | rhs;
  return lhs;
}
template <typename T>
typename std::enable_if_t<EnableBitmaskOps<T>::enable, T> constexpr operator&=(T& lhs,
                                                                               T rhs) {
  lhs = lhs & rhs;
  return lhs;
}

template <typename T>
typename std::enable_if_t<EnableBitmaskOps<T>::enable, bool> constexpr any_bits_set(T t) {
  using type = typename std::underlying_type_t<T>;
  constexpr type zero{};
  return static_cast<type>(t) != zero;
}

#define ENABLE_BITMASK_OPS(x)            \
  template <>                            \
  struct EnableBitmaskOps<x> {           \
    static constexpr bool enable = true; \
  };
