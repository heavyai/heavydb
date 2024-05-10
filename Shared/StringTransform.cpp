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

#include "StringTransform.h"
#include "Logger/Logger.h"

#include <numeric>
#include <random>
#include <regex>

#include <cmath>  // format_bytes round call

#ifndef __CUDACC__
#include <boost/filesystem.hpp>
#include <iomanip>
#endif

void apply_shim(std::string& result,
                const boost::regex& reg_expr,
                const std::function<void(std::string&, const boost::smatch&)>& shim_fn) {
  boost::smatch what;
  auto lit_pos = find_string_literals(result);
  auto start_it = result.cbegin();
  auto end_it = result.cend();
  while (true) {
    if (!boost::regex_search(start_it, end_it, what, reg_expr)) {
      break;
    }
    const auto next_start =
        inside_string_literal(what.position(), what.length(), lit_pos);
    if (next_start) {
      start_it = result.cbegin() + *next_start;
    } else {
      shim_fn(result, what);
      lit_pos = find_string_literals(result);
      start_it = result.cbegin();
      end_it = result.cend();
    }
  }
}

// Scan query and save all single-quoted string literals as [begin,end) index pairs into
// lit_pos, including the surrounding quotes.
std::vector<std::pair<size_t, size_t>> find_string_literals(const std::string& query) {
  boost::regex literal_string_regex{R"(([^']+)('(?:[^']+|'')*'))", boost::regex::perl};
  boost::smatch what;
  auto it = query.begin();
  auto prev_it = it;
  std::vector<std::pair<size_t, size_t>> positions;
  while (true) {
    try {
      if (!boost::regex_search(it, query.end(), what, literal_string_regex)) {
        break;
      }
    } catch (const std::exception& e) {
      // boost::regex throws an exception about the complexity of matching when
      // the wrong type of quotes are used or they're mismatched. Let the query
      // through unmodified, the parser will throw a much more informative error.
      // This can also throw on very long queries
      std::ostringstream oss;
      oss << "Detecting an error while processing string literal regex search: "
          << e.what();
      throw std::runtime_error(oss.str());
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
      {std::regex(
           R"(\b((?:password|s3_access_key|s3_secret_key|s3_session_token|username|credential_string)\s*=\s*)'.+?')",
           flags),
       "$1'XXXXXXXX'"},
      {std::regex(R"((\\set_license\s+)\S+)", flags), "$1XXXXXXXX"}};
  return std::accumulate(
      rules.begin(), rules.end(), query_str, [](auto& str, auto& rule) {
        return std::regex_replace(str, rule.first, rule.second);
      });
}

std::string format_num_bytes(const size_t bytes) {
  const size_t units_per_k_unit{1024};
  const std::vector<std::string> byte_units = {" bytes", "KB", "MB", "GB", "TB", "PB"};
  const std::vector<size_t> bytes_per_scale_unit = {size_t(1),
                                                    size_t(1) << 10,
                                                    size_t(1) << 20,
                                                    size_t(1) << 30,
                                                    size_t(1) << 40,
                                                    size_t(1) << 50,
                                                    size_t(1) << 60};
  if (bytes < units_per_k_unit) {
    return std::to_string(bytes) + " bytes";
  }
  CHECK_GE(bytes, units_per_k_unit);
  const size_t byte_scale = log(bytes) / log(units_per_k_unit);
  CHECK_GE(byte_scale, size_t(1));
  CHECK_LE(byte_scale, size_t(5));
  const size_t scaled_bytes_left_of_decimal = bytes / bytes_per_scale_unit[byte_scale];
  const size_t scaled_bytes_right_of_decimal = bytes % bytes_per_scale_unit[byte_scale];
  const size_t fractional_digits = static_cast<double>(scaled_bytes_right_of_decimal) /
                                   bytes_per_scale_unit[byte_scale] * 100.;
  return std::to_string(scaled_bytes_left_of_decimal) + "." +
         std::to_string(fractional_digits) + " " + byte_units[byte_scale];
}

template <>
std::string to_string(char const*&& v) {
  return std::string(v);
}

template <>
std::string to_string(std::string&& v) {
  return std::move(v);
}

std::pair<std::string_view, const char*> substring(const std::string& str,
                                                   size_t substr_length) {
  // return substring with a post_fix
  // assume input str is valid and we perform substring starting from str's initial pos
  // (=0)
  const auto str_size = str.size();
  if (substr_length >= str_size) {
    return {str, ""};
  }
  std::string_view substr(str.c_str(), substr_length);
  return {substr, "..."};
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

#ifndef __CUDACC__
// This version of split works almost exactly like Python's split,
// which is very convienently-designed.
// See also: https://docs.python.org/3.8/library/stdtypes.html#str.split
std::vector<std::string> split(std::string_view str,
                               std::string_view delim,
                               std::optional<size_t> maxsplit) {
  std::vector<std::string> result;

  // Use an explicit delimiter.
  if (!delim.empty()) {
    std::string::size_type i = 0, j = 0;
    while ((i = str.find(delim, i)) != std::string::npos &&
           (!maxsplit || result.size() < maxsplit.value())) {
      result.emplace_back(str, j, i - j);
      i += delim.size();
      j = i;
    }
    result.emplace_back(str, j, std::string::npos);
    return result;

    // Treat any number of consecutive whitespace characters as a delimiter.
  } else {
    bool prev_ws = true;
    std::string::size_type i = 0, j = 0;
    for (; i < str.size(); ++i) {
      if (prev_ws) {
        if (!isspace(str[i])) {
          // start of word
          prev_ws = false;
          j = i;
        }
      } else {
        if (isspace(str[i])) {
          // start of space
          result.emplace_back(str, j, i - j);
          prev_ws = true;
          j = i;
          if ((maxsplit && result.size() == maxsplit.value())) {
            // stop early if maxsplit was reached
            result.emplace_back(str, j, std::string::npos);
            return result;
          }
        }
      }
    }
    if (!prev_ws) {
      result.emplace_back(str, j, std::string::npos);
    }
    return result;
  }
}

std::string_view sv_strip(std::string_view str) {
  std::string::size_type i, j;
  for (i = 0; i < str.size() && std::isspace(str[i]); ++i) {
  }
  for (j = str.size(); j > i && std::isspace(str[j - 1]); --j) {
  }
  return str.substr(i, j - i);
}

std::string strip(std::string_view str) {
  return std::string(sv_strip(str));
}

std::optional<size_t> inside_string_literal(
    const size_t start,
    const size_t length,
    std::vector<std::pair<size_t, size_t>> const& literal_positions) {
  const auto end = start + length;
  for (const auto& literal_position : literal_positions) {
    if (literal_position.first <= start && end <= literal_position.second) {
      return literal_position.second;
    }
  }
  return std::nullopt;
}

#endif  // __CUDACC__

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

#ifndef __CUDACC__
std::string get_quoted_string(const std::string& filename, char quote, char escape) {
  std::stringstream ss;
  ss << std::quoted(filename, quote, escape);  // TODO: prevents string_view Jun 2020
  return ss.str();
}
#endif  // __CUDACC__

#ifndef __CUDACC__
std::string simple_sanitize(const std::string& str) {
  auto sanitized_str{str};
  for (auto& c : sanitized_str) {
    c = (c < 32) ? ' ' : c;
  }
  return sanitized_str;
}
#endif  // __CUDACC__
