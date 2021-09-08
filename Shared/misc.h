/*
 * Copyright 2019 OmniSci, Inc.
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

#ifndef SHARED_MISC_H
#define SHARED_MISC_H

#include <cstdint>
#include <cstring>
#include <deque>
#include <iterator>
#include <list>
#include <map>
#include <set>
#include <unordered_set>
#include <vector>

namespace {

template <typename T>
constexpr T power(T const a, T const n) {
  return n ? a * power(a, n - 1) : static_cast<T>(1);
}

template <typename T, size_t... Indices>
constexpr std::array<T, sizeof...(Indices)> powersOfImpl(
    T const a,
    std::index_sequence<Indices...>) {
  return {power(a, static_cast<T>(Indices))...};
}

template <size_t... Indices>
constexpr std::array<double, sizeof...(Indices)> inversePowersOfImpl(
    double const a,
    std::index_sequence<Indices...>) {
  return {(1.0 / power(a, static_cast<double>(Indices)))...};
}

}  // namespace

namespace shared {

template <typename K, typename V>
V& get_from_map(std::map<K, V>& map, const K& key) {
  auto find_it = map.find(key);
  CHECK(find_it != map.end());
  return find_it->second;
}

template <typename K, typename V>
const V& get_from_map(const std::map<K, V>& map, const K& key) {
  auto find_it = map.find(key);
  CHECK(find_it != map.end());
  return find_it->second;
}

// source is destructively appended to the back of destination.
// source.empty() is true after call. Return number of elements appended.
template <typename T>
size_t appendMove(std::vector<T>& destination, std::vector<T>& source) {
  if (source.empty()) {
    return 0;
  } else if (destination.empty()) {
    destination = std::move(source);
    return destination.size();
  } else {
    size_t const source_size = source.size();
    destination.reserve(destination.size() + source_size);
    std::move(std::begin(source), std::end(source), std::back_inserter(destination));
    source.clear();
    return source_size;
  }
}

template <typename... Ts, typename T>
bool dynamic_castable_to_any(T const* ptr) {
  return (... || dynamic_cast<Ts const*>(ptr));
}

// Helper to print out contents of simple containers (e.g. vector, list, deque)
// including nested containers, e.g. 2d vectors, list of vectors, etc.
// Base value_type must be a std::is_scalar_v type, though you can add custom
// objects below with a new `else if constexpr` block.
// Example: VLOG(1) << "container=" << shared::printContainer(container);
template <typename CONTAINER>
struct PrintContainer {
  CONTAINER& container;
};

template <typename CONTAINER>
PrintContainer<CONTAINER> printContainer(CONTAINER& container) {
  return {container};
}

template <typename CONTAINER>
struct is_std_container : std::false_type {};
template <typename T, typename A>
struct is_std_container<std::deque<T, A> > : std::true_type {};
template <typename T, typename A>
struct is_std_container<std::list<T, A> > : std::true_type {};
template <typename T, typename A>
struct is_std_container<std::set<T, A> > : std::true_type {};
template <typename T, typename A>
struct is_std_container<std::unordered_set<T, A> > : std::true_type {};
template <typename T, typename A>
struct is_std_container<std::vector<T, A> > : std::true_type {};

template <typename OSTREAM, typename CONTAINER>
OSTREAM& operator<<(OSTREAM& os, PrintContainer<CONTAINER> pc) {
  if (pc.container.empty()) {
    return os << "()";
  } else {
    if constexpr (is_std_container<typename CONTAINER::value_type>::value) {
      os << '(';
      for (auto& container : pc.container) {
        os << printContainer(container);
      }
    } else {
      for (auto itr = pc.container.begin(); itr != pc.container.end(); ++itr) {
        if constexpr (std::is_pointer_v<typename CONTAINER::value_type>) {
          os << (itr == pc.container.begin() ? '(' : ' ') << (void const*)*itr;
        } else {
          os << (itr == pc.container.begin() ? '(' : ' ') << *itr;
        }
      }
    }
    return os << ')';
  }
}

// Same as strftime(buf, max, "%F", tm) but guarantees that the year is
// zero-padded to a minimum length of 4. Return the number of characters
// written, not including null byte. If max is not large enough, return 0.
size_t formatDate(char* buf, size_t const max, int64_t const unixtime);

// Same as strftime(buf, max, "%F %T", tm) but guarantees that the year is
// zero-padded to a minimum length of 4. Return the number of characters
// written, not including null byte. If max is not large enough, return 0.
// Requirement: 0 <= dimension <= 9.
size_t formatDateTime(char* buf,
                      size_t const max,
                      int64_t const timestamp,
                      int const dimension);

// Write unixtime in seconds since epoch as "HH:MM:SS" format.
size_t formatHMS(char* buf, size_t const max, int64_t const unixtime);

// Result of division where quot is floored and rem is unsigned.
struct DivUMod {
  int64_t quot;
  int64_t rem;
};

// Requirement: 0 < den
inline DivUMod divUMod(int64_t num, int64_t den) {
  DivUMod div{num / den, num % den};
  if (div.rem < 0) {
    --div.quot;
    div.rem += den;
  }
  return div;
}

// Requirement: 0 < den.
inline uint64_t unsignedMod(int64_t num, int64_t den) {
  int64_t mod = num % den;
  if (mod < 0) {
    mod += den;
  }
  return mod;
}

template <typename T, typename U>
inline bool contains(const T& container, const U& element) {
  if (std::find(container.begin(), container.end(), element) == container.end()) {
    return false;
  } else {
    return true;
  }
}

// Calculate polynomial c0 + c1*x + c2*x^2 + ... + cn*x^n using Horner's method.
template <typename... COEFFICIENTS>
constexpr double horner(double const x, double const c0, COEFFICIENTS... c) {
  if constexpr (sizeof...(COEFFICIENTS) == 0) {
    return c0;
  } else {
    return horner(x, c...) * x + c0;
  }
  return {};  // quiet nvcc warning https://stackoverflow.com/a/64561686/2700898
}

// OK for -0.15 <= x <= 0.15
inline double fastAtanh(double const x) {
  // Mathematica: CoefficientList[Normal@Series[ArcTanh[x],{x,0,16}],x] // InputForm
  return x * horner(x * x, 1, 1 / 3., 1 / 5., 1 / 7., 1 / 9., 1 / 11., 1 / 13., 1 / 15.);
}

// OK for -1 <= x <= 1
inline double fastCos(double const x) {
  // Mathematica: CoefficientList[Normal@Series[Cos[x],{x,0,16}],x] // InputForm
  // clang-format off
  return horner(x * x, 1, -1/2., 1/24., -1/720., 1/40320., -1/3628800.,
                1/479001600., -1/87178291200., 1/20922789888000.);
  // clang-format on
}

// OK for -1 <= x <= 1
inline double fastCosh(double const x) {
  // Mathematica: CoefficientList[Normal@Series[Cosh[x],{x,0,16}],x] // InputForm
  // clang-format off
  return horner(x * x, 1, 1/2., 1/24., 1/720., 1/40320., 1/3628800.,
                1/479001600., 1/87178291200., 1/20922789888000.);
  // clang-format on
}

// OK for -1 <= x <= 1
inline double fastSin(double const x) {
  // Mathematica: CoefficientList[Normal@Series[Sin[x],{x,0,16}],x] // InputForm
  // clang-format off
  return x * horner(x * x, 1, -1/6., 1/120., -1/5040., 1/362880.,
                    -1/39916800., 1/6227020800., -1/1307674368000.);
  // clang-format on
}

// OK for -1 <= x <= 1
inline double fastSinh(double const x) {
  // Mathematica: CoefficientList[Normal@Series[Sinh[x],{x,0,16}],x] // InputForm
  // clang-format off
  return x * horner(x * x, 1, 1/6., 1/120., 1/5040., 1/362880.,
                    1/39916800., 1/6227020800., 1/1307674368000.);
  // clang-format on
}

// Return constexpr std::array<T, N> of {1, a, a^2, a^3, ..., a^(N-1)}.
template <typename T, size_t N>
constexpr std::array<T, N> powersOf(T const a) {
  return powersOfImpl<T>(a, std::make_index_sequence<N>{});
}

// Return constexpr std::array<double, N> of {1, 1/a, 1/a^2, 1/a^3, ..., 1/a^(N-1)}.
template <size_t N>
constexpr std::array<double, N> inversePowersOf(double const a) {
  return inversePowersOfImpl(a, std::make_index_sequence<N>{});
}

// May be constexpr in C++20.
template <typename TO, typename FROM>
inline TO reinterpret_bits(FROM const from) {
  TO to{0};
  memcpy(&to, &from, sizeof(TO) < sizeof(FROM) ? sizeof(TO) : sizeof(FROM));
  return to;
}

}  // namespace shared

#endif  // SHARED_MISC_H
