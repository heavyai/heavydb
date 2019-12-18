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
#include <boost/regex.hpp>
#endif

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

// NOTE: check for C++17 is required here because Parser/ is pinned to C++14.
#if __cplusplus >= 201703L
#include <string_view>
#endif  // __cplusplus >= 201703L

#ifndef __CUDACC__
void apply_shim(std::string& result,
                const boost::regex& reg_expr,
                const std::function<void(std::string&, const boost::smatch&)>& shim_fn);
#endif

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

inline std::string to_upper(const std::string& str) {
  auto str_uc = str;
  std::transform(str_uc.begin(), str_uc.end(), str_uc.begin(), ::toupper);
  return str_uc;
}

std::string generate_random_string(const size_t len);

//! split apart a string into a vector of substrings
std::vector<std::string> split(const std::string& str, const std::string& delim);

//! trim any whitespace from the left and right ends of a string
std::string strip(const std::string& str);

//! sanitize an SQL string
bool remove_unquoted_newlines_linefeeds_and_tabs_from_sql_string(
    std::string& str) noexcept;

// Remove quotes if they match from beginning and end of string.
// Return true if string was changed, false if not.
// Does not check for escaped quotes within string.
bool unquote(std::string&);

// NOTE: this check for C++17 is required because Parser/ is pinned to C++14.
#if __cplusplus >= 201703L
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
#endif  // __cplusplus >= 201703L

#endif  // SHARED_STRINGTRANSFORM_H
