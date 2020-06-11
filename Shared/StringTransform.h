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

#ifndef SHARED_STRINGTRANSFORM_H
#define SHARED_STRINGTRANSFORM_H

#include "Logger.h"

#ifndef __CUDACC__
#include <boost/config.hpp>
#include <boost/regex.hpp>
#include <optional>
#include <string_view>
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
#ifdef BOOST_NO_CXX17_FOLD_EXPRESSIONS
  (void)(int[]){0, ((void)(oss << std::forward<Ts>(args)), 0)...};
#else
  (oss << ... << std::forward<Ts>(args));
#endif
  return oss.str();
}
#endif  // __CUDACC__

std::vector<std::pair<size_t, size_t>> find_string_literals(const std::string& query);

// Replace passwords, keys, etc. in a sql query with 'XXXXXXXX'.
std::string hide_sensitive_data_from_query(std::string const& query_str);

ssize_t inside_string_literal(
    const size_t start,
    const size_t length,
    const std::vector<std::pair<size_t, size_t>>& literal_positions);

template <typename T>
std::string join(T const& container, std::string const& delim) {
  std::stringstream ss;
  if (!container.empty()) {
    ss << container.front();
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

//! trim any whitespace from the left and right ends of a string
std::string strip(std::string_view str);
#endif  // __CUDACC__

//! sanitize an SQL string
bool remove_unquoted_newlines_linefeeds_and_tabs_from_sql_string(
    std::string& str) noexcept;

// Remove quotes if they match from beginning and end of string.
// Return true if string was changed, false if not.
// Does not check for escaped quotes within string.
bool unquote(std::string&);

#ifndef __CUDACC__
//! Quote a string while escaping any existing quotes in the string.
std::string get_quoted_string(const std::string& filename,
                              char quote = '"',
                              char escape = '\\');

//! Throws exception if security problems found. Returns quoted filename.
void filename_security_check(const std::string& filename);
#endif  // __CUDACC__

#ifndef __CUDACC__
namespace {

template <typename T>
inline decltype(auto) stringlike(T&& parm) {
  // String.
  if constexpr (std::is_base_of_v<std::string, std::remove_reference_t<decltype(parm)>>) {
    return std::forward<T>(parm);
  }

  // Char Array.
  else if constexpr (std::is_array_v<std::remove_reference_t<decltype(parm)>>) {
    return std::forward<T>(parm);
  }

  // Char String.
  else if constexpr (std::is_same_v<std::remove_reference_t<decltype(parm)>,
                                    const char*> ||
                     std::is_same_v<std::remove_reference_t<decltype(parm)>, char*>) {
    return std::forward<T>(parm);
  }

  // Integer or Floating Point.
  else if constexpr (std::is_integral_v<std::remove_reference_t<decltype(parm)>> ||
                     std::is_floating_point_v<std::remove_reference_t<decltype(parm)>>) {
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
