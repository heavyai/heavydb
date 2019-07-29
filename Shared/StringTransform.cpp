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

#include "StringTransform.h"

#include <numeric>
#include <random>
#include <regex>

void apply_shim(std::string& result,
                const boost::regex& reg_expr,
                const std::function<void(std::string&, const boost::smatch&)>& shim_fn) {
  boost::smatch what;
  std::vector<std::pair<size_t, size_t>> lit_pos = find_string_literals(result);
  auto start_it = result.cbegin();
  auto end_it = result.cend();
  while (true) {
    if (!boost::regex_search(start_it, end_it, what, reg_expr)) {
      break;
    }
    const auto next_start =
        inside_string_literal(what.position(), what.length(), lit_pos);
    if (next_start >= 0) {
      start_it = result.cbegin() + next_start;
    } else {
      shim_fn(result, what);
      lit_pos = find_string_literals(result);
      start_it = result.cbegin();
      end_it = result.cend();
    }
  }
}

std::vector<std::pair<size_t, size_t>> find_string_literals(const std::string& query) {
  boost::regex literal_string_regex{R"(([^']+)('(?:[^']+|'')+'))", boost::regex::perl};
  boost::smatch what;
  auto it = query.begin();
  auto prev_it = it;
  std::vector<std::pair<size_t, size_t>> positions;
  while (true) {
    if (!boost::regex_search(it, query.end(), what, literal_string_regex)) {
      break;
    }
    CHECK_GT(what[1].length(), 0);
    prev_it = it;
    it += what.length();
    positions.emplace_back(prev_it + what[1].length() - query.begin(),
                           it - query.begin());
  }
  return positions;
}

std::string hide_sensitive_data_from_query(std::string const& query_str) {
  constexpr std::regex::flag_type flags =
      std::regex::ECMAScript | std::regex::icase | std::regex::optimize;
  static const std::initializer_list<std::pair<std::regex, std::string>> rules{
      {std::regex(R"(\b((?:password|s3_access_key|s3_secret_key)\s*=\s*)'.+?')", flags),
       "$1'XXXXXXXX'"},
      {std::regex(R"((\\set_license\s+)\S+)", flags), "$1XXXXXXXX"}};
  return std::accumulate(
      rules.begin(), rules.end(), query_str, [](auto& str, auto& rule) {
        return std::regex_replace(str, rule.first, rule.second);
      });
}

ssize_t inside_string_literal(
    const size_t start,
    const size_t length,
    const std::vector<std::pair<size_t, size_t>>& literal_positions) {
  const auto end = start + length;
  for (const auto& literal_position : literal_positions) {
    if (literal_position.first <= start && end <= literal_position.second) {
      return literal_position.second;
    }
  }
  return -1;
}

template <>
std::string to_string(char const*&& v) {
  return std::string(v);
}

template <>
std::string to_string(std::string&& v) {
  return std::move(v);
}

std::string generate_random_string(const size_t len) {
  static char charset[] =
      "0123456789"
      "abcdefghijklmnopqrstuvwxyz"
      "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

  static std::mt19937 prng{std::random_device{}()};
  static std::uniform_int_distribution<size_t> dist(0, strlen(charset) - 1);

  std::string str;
  str.reserve(len);
  for (size_t i = 0; i < len; i++) {
    str += charset[dist(prng)];
  }
  return str;
}

std::vector<std::string> split(const std::string& str, const std::string& delim) {
  CHECK(!delim.empty());
  std::vector<std::string> result;
  std::string::size_type i = 0, j = 0;
  while ((i = str.find(delim, i)) != std::string::npos) {
    result.emplace_back(str, j, i - j);
    i += delim.size();
    j = i;
  }
  result.emplace_back(str, j);
  return result;
}

std::string strip(const std::string& str) {
  std::string::size_type i, j;
  for (i = 0; i < str.size() && std::isspace(str[i]); ++i) {
  }
  for (j = str.size(); j > i && std::isspace(str[j - 1]); --j) {
  }
  return str.substr(i, j - i);
}

bool remove_unquoted_newlines_linefeeds_and_tabs_from_sql_string(
    std::string& str) noexcept {
  char inside_quote = 0;
  bool previous_c_was_backslash = false;
  for (auto& c : str) {
    // if this character is a quote of either type
    if (c == '\'' || c == '\"') {
      // ignore if previous character was a backslash
      if (!previous_c_was_backslash) {
        // start or end of a quoted region
        if (inside_quote == c) {
          // end region
          inside_quote = 0;
        } else if (inside_quote == 0) {
          // start region
          inside_quote = c;
        }
      }
    } else if (inside_quote == 0) {
      // outside quoted region
      if (c == '\n' || c == '\t' || c == '\r') {
        // replace these with space
        c = ' ';
      }
      // otherwise leave alone, including quotes of a different type
    }
    // handle backslashes, except for double backslashes
    if (c == '\\') {
      previous_c_was_backslash = !previous_c_was_backslash;
    } else {
      previous_c_was_backslash = false;
    }
  }
  // if we didn't end a region, there were unclosed or mixed-nested quotes
  // accounting for backslashes should mean that this should only be the
  // case with truly malformed strings which Calcite will barf on anyway
  return (inside_quote == 0);
}
