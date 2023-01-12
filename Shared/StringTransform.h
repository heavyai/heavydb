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

#ifndef SHARED_STRINGTRANSFORM_H
#define SHARED_STRINGTRANSFORM_H

#ifndef __CUDACC__
#include <boost/config.hpp>
#include <optional>
#include <string_view>

#include "Shared/clean_boost_regex.hpp"
#endif  // __CUDACC__

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#ifndef __CUDACC__
void apply_shim(std::string& result,
                const boost::regex& reg_expr,
                const std::function<void(std::string&, const boost::smatch&)>& shim_fn);

// cat - Concatenate values of arbitrary types into a string.
template <typename... Ts>
std::string cat(Ts&&... args) {
  std::ostringstream oss;
  (oss << ... << std::forward<Ts>(args));
  return oss.str();
}
#endif  // __CUDACC__

std::vector<std::pair<size_t, size_t>> find_string_literals(const std::string& query);

// Replace passwords, keys, etc. in a sql query with 'XXXXXXXX'.
std::string hide_sensitive_data_from_query(std::string const& query_str);

std::string format_num_bytes(const size_t bytes);

#ifndef __CUDACC__
std::optional<size_t> inside_string_literal(
    const size_t start,
    const size_t length,
    const std::vector<std::pair<size_t, size_t>>& literal_positions);
#endif  // __CUDACC__

template <typename T>
std::string join(T const& container, std::string const& delim) {
  std::stringstream ss;
  if (!container.empty()) {
    ss << *container.cbegin();
    for (auto itr = std::next(container.cbegin()); itr != container.cend(); ++itr) {
      ss << delim << *itr;
    }
  }
  return ss.str();
}

template <typename T>
std::string to_string(T&& v) {
  std::ostringstream oss;
  oss << v;
  return oss.str();
}

template <>
std::string to_string(char const*&& v);

template <>
std::string to_string(std::string&& v);

// NOTE(sy): to_upper/to_lower: As of Feb 2020, this is a solution recommended by Stack
// Overflow. Boost's to_upper_copy() is many times slower, maybe because it uses
// locale-aware std::toupper. Probably don't bother converting the input parameters to
// std::string_view because testing gave a small slowdown for std::string inputs and a
// small speedup for c-style string inputs for this usage.

inline std::string to_upper(const std::string& str) {
  auto str_uc = str;
  std::transform(str_uc.begin(), str_uc.end(), str_uc.begin(), ::toupper);
  return str_uc;
}

inline std::string to_lower(const std::string& str) {
  auto str_lc = str;
  std::transform(str_lc.begin(), str_lc.end(), str_lc.begin(), ::tolower);
  return str_lc;
}

std::string generate_random_string(const size_t len);

#ifndef __CUDACC__
//! split apart a string into a vector of substrings
std::vector<std::string> split(std::string_view str,
                               std::string_view delim = {},
                               std::optional<size_t> maxsplit = std::nullopt);

//! return trimmed string_view
std::string_view sv_strip(std::string_view str);

//! trim any whitespace from the left and right ends of a string
std::string strip(std::string_view str);

//! return substring of str with postfix if str.size() > substr_length
std::pair<std::string_view, const char*> substring(const std::string& str,
                                                   size_t substr_length);
#endif  // __CUDACC__

//! sanitize an SQL string
bool remove_unquoted_newlines_linefeeds_and_tabs_from_sql_string(
    std::string& str) noexcept;

//! simple sanitize string (replace control characters with space)
#ifndef __CUDACC__
std::string simple_sanitize(const std::string& str);
#endif  // __CUDACC__

#ifndef __CUDACC__
//! Quote a string while escaping any existing quotes in the string.
std::string get_quoted_string(const std::string& filename,
                              char quote = '"',
                              char escape = '\\');
#endif  // __CUDACC__

#ifndef __CUDACC__
namespace {

template <typename T>
inline decltype(auto) stringlike(T&& parm) {
  // String.
  if constexpr (std::is_base_of_v<std::string,
                                  std::remove_reference_t<decltype(parm)>>) {  // NOLINT
    return std::forward<T>(parm);

    // Char Array.

  } else if constexpr (std::is_array_v<
                           std::remove_reference_t<decltype(parm)>>) {  // NOLINT
    return std::forward<T>(parm);

    // Char String.

  } else if constexpr (std::is_same_v<std::remove_reference_t<decltype(parm)>,
                                      const char*> ||
                       std::is_same_v<std::remove_reference_t<decltype(parm)>,
                                      char*>) {  // NOLINT
    return std::forward<T>(parm);

    // Integer or Floating Point.

  } else if constexpr (std::is_integral_v<std::remove_reference_t<decltype(parm)>> ||
                       std::is_floating_point_v<
                           std::remove_reference_t<decltype(parm)>>) {  // NOLINT
    return std::to_string(std::forward<T>(parm));
  }

  // Unsupported type that will fail at compile-time.
  else {
    static_assert(std::is_base_of_v<void, decltype(parm)>);
    return std::string();  // unreachable, but needed to avoid extra error messages
  }
}

}  // anonymous namespace

template <typename... Types>
std::string concat(Types&&... parms) {
  struct Joiner {
    Joiner() {}

    std::string txt;

    void append(std::string_view moretxt) { txt += moretxt; }
  };  // struct Joiner
  Joiner j{};
  (j.append(stringlike(std::forward<Types>(parms))), ...);
  return std::move(j.txt);
}

template <typename... Types>
std::string concat_with(std::string_view with, Types&&... parms) {
  struct JoinerWith {
    JoinerWith(std::string_view join) : join(join), first(true) {}

    std::string_view join;
    bool first;
    std::string txt;

    void append(std::string_view moretxt) {
      if (!first) {
        txt += join;
      } else {
        first = false;
      }
      txt += moretxt;
    }
  };  // struct JoinerWith
  JoinerWith j{with};
  (j.append(stringlike(std::forward<Types>(parms))), ...);
  return std::move(j.txt);
}
#endif  // __CUDACC__

#endif  // SHARED_STRINGTRANSFORM_H
