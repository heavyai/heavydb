/*
 * Copyright 2023 HEAVY.AI, Inc.
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

/**
 * @file    define_enum_class.h
 * @brief   Macros/templates for defining enum classes and related utilities.
 *          Place macro calls in the heavyai namespace so that functions like to_string()
 *          can be found by the compiler via ADL.
 *
 *          Example: HEAVYAI_DEFINE_ENUM_CLASS(Color, Red, Green, Blue)
 *          Defines:
 *          1. enum class Color { Red, Green, Blue };
 *          2. constexpr char const* to_string(Color const);
 *          3. inline std::ostream &operator<<(std::ostream&, Color const);
 *
 *          The macro HEAVYAI_DEFINE_ENUM_CLASS_WITH_DESCRIPTIONS() additionally defines
 *          4. constexpr char const* to_description(Color const);
 *
 *          template <typename Enum>
 *          constexpr std::optional<Enum> to_enum(std::string_view const name);
 *          returns the Enum if found by its string representation in O(log(N)) time.
 *
 */

#pragma once

#include <boost/preprocessor.hpp>

#include <algorithm>
#include <array>
#include <optional>
#include <ostream>
#include <string_view>

#define HEAVYAI_DEFINE_ENUM_CLASS(enum_class, ...)                        \
  enum class enum_class { __VA_ARGS__, N_ };                              \
                                                                          \
  constexpr char const* to_string(enum_class const e) {                   \
    constexpr char const* strings[]{HEAVYAI_QUOTE_EACH(__VA_ARGS__)};     \
    constexpr size_t nstrings = sizeof(strings) / sizeof(*strings);       \
    static_assert(nstrings == size_t(enum_class::N_));                    \
    return strings[size_t(e)];                                            \
  }                                                                       \
                                                                          \
  inline std::ostream& operator<<(std::ostream& os, enum_class const e) { \
    return os << to_string(e);                                            \
  }

#define HEAVYAI_DEFINE_ENUM_CLASS_WITH_DESCRIPTIONS(enum_class, ...)   \
  HEAVYAI_DEFINE_ENUM_CLASS(enum_class, HEAVYAI_PLUCK(0, __VA_ARGS__)) \
                                                                       \
  constexpr char const* to_description(enum_class const e) {           \
    constexpr char const* strings[]{HEAVYAI_PLUCK(1, __VA_ARGS__)};    \
    constexpr size_t nstrings = sizeof(strings) / sizeof(*strings);    \
    static_assert(nstrings == size_t(enum_class::N_));                 \
    return strings[size_t(e)];                                         \
  }

// Helper macros
#define HEAVYAI_QUOTE(r, data, i, elem) BOOST_PP_COMMA_IF(i) BOOST_PP_STRINGIZE(elem)
#define HEAVYAI_QUOTE_EACH(...) \
  BOOST_PP_SEQ_FOR_EACH_I(HEAVYAI_QUOTE, , BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))
#define HEAVYAI_PLUCK_ONE(r, j, i, pair) \
  BOOST_PP_COMMA_IF(i) BOOST_PP_TUPLE_ELEM(2, j, pair)
#define HEAVYAI_PLUCK(j, ...) \
  BOOST_PP_SEQ_FOR_EACH_I(HEAVYAI_PLUCK_ONE, j, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))

namespace heavyai {

// Helper function and struct templates
template <typename Enum>
struct StringEnum {
  std::string_view name;
  Enum value;

  bool operator<(std::string_view const name) const { return this->name < name; }
};

template <typename Enum, size_t... I>
constexpr auto enum_to_array(std::index_sequence<I...>) {
  return std::array<StringEnum<Enum>, sizeof...(I)>{
      StringEnum<Enum>{to_string(static_cast<Enum>(I)), static_cast<Enum>(I)}...};
}

template <typename T, size_t N>
constexpr void insertion_sort(std::array<T, N>& arr) {
  for (size_t i = 1; i < N; ++i) {
    auto key = arr[i];
    size_t j = i;
    for (; j && key.name < arr[j - 1].name; --j) {
      arr[j] = arr[j - 1];
    }
    arr[j] = key;
  }
}

template <typename Enum>
constexpr std::array<StringEnum<Enum>, size_t(Enum::N_)> sort_by_name() {
  auto arr = enum_to_array<Enum>(std::make_index_sequence<size_t(Enum::N_)>());
  insertion_sort(arr);
  return arr;
}

// Return std::optional<Enum> given string name in O(log(Enum::N_)) time and stack space.
template <typename Enum>
std::optional<Enum> to_enum(std::string_view const name) {
  constexpr std::array<StringEnum<Enum>, size_t(Enum::N_)> arr = sort_by_name<Enum>();
  auto const itr = std::lower_bound(arr.begin(), arr.end(), name);
  bool const found = itr != arr.end() && itr->name == name;
  return found ? std::make_optional(itr->value) : std::nullopt;
}

// Example: IsAny<Color::Red, Color::Green, Color::Blue>::check(Color::Blue);
template <auto... Values>
struct IsAny {
  template <typename T>
  static bool check(T const value) {
    // Casting to T allows for safe comparison against out-of-range value.
    // Example: IsAny<Color::Red, Color::Green, Color::Blue>::check(-1);
    return ((static_cast<T>(Values) == value) || ...);
  }
};

}  // namespace heavyai
