/*
 * Copyright (c) 2020 OmniSci, Inc.
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

#include "sqltypes.h"

#include <boost/regex.hpp>

#include <optional>
#include <ostream>
#include <string_view>
#include <unordered_map>
#include <vector>

template <SQLTypes SQL_TYPE>
int64_t dateTimeParse(std::string_view, unsigned const dim);

template <>
int64_t dateTimeParse<kDATE>(std::string_view, unsigned const dim);

template <>
int64_t dateTimeParse<kTIME>(std::string_view, unsigned const dim);

template <>
int64_t dateTimeParse<kTIMESTAMP>(std::string_view, unsigned const dim);

// Set format and parse date/time/timestamp strings into (s,ms,us,ns) since the epoch
// based on given dim in (0,3,6,9) respectively.  Basic idea is to transform given format
// ("%Y-%m-%d") into a regular expression with capturing groups corresponding to each
// %-field.  The regex is then matched to the string and the DateTime dt_ struct is set,
// from which the final epoch-based int64_t value is calculated.
class DateTimeParser {
 public:
  void setFormat(std::string_view const&);
  std::optional<int64_t> parse(std::string_view const&, unsigned dim);
  std::string_view unparsed() const;

  struct DateTime {
    int64_t Y{1970};        // Full year
    unsigned m{1};          // month (1-12)
    unsigned d{1};          // day of month (1-31)
    unsigned H{0};          // hour (0-23)
    unsigned M{0};          // minute (0-59)
    unsigned S{0};          // second (0-61)
    unsigned n{0};          // fraction of a second in nanoseconds (0-999999999)
    int z{0};               // timezone offset in seconds
    std::optional<bool> p;  // true if pm, false if am, nullopt if unspecified

    int64_t getTime(unsigned const dim) const;
    friend std::ostream& operator<<(std::ostream&, DateTime const&);
  };

 private:
  std::vector<char> fields_;
  boost::regex regex_;
  std::string_view unparsed_;
  DateTime dt_;

  static const boost::regex field_regex;
  static const std::unordered_map<char, char const*> field_to_regex;
  static const std::unordered_map<int, unsigned> month_name_lookup;

  void resetDateTime();
  void updateDateTime(char field, std::string_view const&);
};
