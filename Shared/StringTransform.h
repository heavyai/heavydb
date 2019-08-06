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

#include <boost/regex.hpp>

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <string>

void apply_shim(std::string& result,
                const boost::regex& reg_expr,
                const std::function<void(std::string&, const boost::smatch&)>& shim_fn);

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

// split apart a string into a vector of substrings
std::vector<std::string> split(const std::string& str, const std::string& delim);

// trim any whitespace from the left and right ends of a string
std::string strip(const std::string& str);

// sanitize an SQL string
bool remove_unquoted_newlines_linefeeds_and_tabs_from_sql_string(
    std::string& str) noexcept;

#endif  // SHARED_STRINGTRANSFORM_H
