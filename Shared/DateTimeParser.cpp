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

#include "DateTimeParser.h"
#include "StringTransform.h"

#include <boost/algorithm/string/predicate.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <limits>
#include <sstream>
#include <vector>

namespace {

constexpr std::array<int, 12> month_prefixes{{int('j') << 16 | int('a') << 8 | int('n'),
                                              int('f') << 16 | int('e') << 8 | int('b'),
                                              int('m') << 16 | int('a') << 8 | int('r'),
                                              int('a') << 16 | int('p') << 8 | int('r'),
                                              int('m') << 16 | int('a') << 8 | int('y'),
                                              int('j') << 16 | int('u') << 8 | int('n'),
                                              int('j') << 16 | int('u') << 8 | int('l'),
                                              int('a') << 16 | int('u') << 8 | int('g'),
                                              int('s') << 16 | int('e') << 8 | int('p'),
                                              int('o') << 16 | int('c') << 8 | int('t'),
                                              int('n') << 16 | int('o') << 8 | int('v'),
                                              int('d') << 16 | int('e') << 8 | int('c')}};

constexpr std::array<std::string_view, 13> month_suffixes{
    {""
     "uary",
     "ruary",
     "ch",
     "il",
     "",
     "e",
     "y",
     "ust",
     "tember",
     "ober",
     "ember",
     "ember"}};

constexpr unsigned
    pow_10[10]{1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000};

// Return y-m-d minus 1970-01-01 in days according to Gregorian calendar.
// Credit: http://howardhinnant.github.io/date_algorithms.html#days_from_civil
int64_t daysFromCivil(int64_t y, unsigned const m, unsigned const d) {
  y -= m <= 2;
  int64_t const era = (y < 0 ? y - 399 : y) / 400;
  unsigned const yoe = static_cast<unsigned>(y - era * 400);             // [0, 399]
  unsigned const doy = (153 * (m + (m <= 2 ? 9 : -3)) + 2) / 5 + d - 1;  // [0, 365]
  unsigned const doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;            // [0, 146096]
  return era * 146097 + static_cast<int64_t>(doe) - 719468;
}

// Order of entries correspond to enum class FormatType { Date, Time, Timezone }.
std::vector<std::vector<std::string_view>> formatViews() {
  return {{{"%Y-%m-%d", "%m/%d/%y", "%m/%d/%Y", "%Y/%m/%d", "%d-%b-%y", "%d/%b/%Y"},
           {"%I:%M:%S %p",
            "%H:%M:%S",
            "%I:%M %p",
            "%H:%M",
            "%H%M%S",
            "%I . %M . %S %p",
            "%I %p"},
           {"%z"}}};
}

// Optionally eat month name after first 3 letters.  Assume first 3 letters are correct.
void eatMonth(unsigned const month, std::string_view& str) {
  str.remove_prefix(3);
  std::string_view const suffix = month_suffixes[month];
  if (boost::algorithm::istarts_with(str, suffix)) {
    str.remove_prefix(suffix.size());
  }
}

void eatSpace(std::string_view& str) {
  while (!str.empty() && isspace(str.front())) {
    str.remove_prefix(1);
  }
}

// Parse str as a number of maxlen and type T.
// Return value and consume from str on success,
// otherwise return std::nullopt and do not change str.
template <typename T>
std::optional<T> fromChars(std::string_view& str,
                           size_t maxlen = std::numeric_limits<size_t>::max()) {
  T retval;
  maxlen = std::min(maxlen, str.size());
  auto const result = std::from_chars(str.data(), str.data() + maxlen, retval);
  if (result.ec == std::errc()) {
    str.remove_prefix(result.ptr - str.data());
    return retval;
  } else {
    return std::nullopt;
  }
}

std::optional<int64_t> unixTime(std::string_view const str) {
  int64_t time{0};
  auto const result = std::from_chars(str.data(), str.data() + str.size(), time);
  // is_valid = str =~ /^-?\d+(\.\d*)$/
  bool const is_valid = result.ec == std::errc() &&
                        (result.ptr == str.data() + str.size() ||
                         (*result.ptr == '.' &&
                          std::all_of(result.ptr + 1, str.data() + str.size(), isdigit)));
  return is_valid ? std::make_optional(time) : std::nullopt;
}

}  // namespace

// Interpret str according to DateTimeParser::FormatType::Time.
// Return number of (s,ms,us,ns) since midnight based on dim in (0,3,6,9) resp.
template <>
std::optional<int64_t> dateTimeParseOptional<kTIME>(std::string_view str,
                                                    unsigned const dim) {
  if (!str.empty() && str.front() == 'T') {
    str.remove_prefix(1);
  }
  DateTimeParser parser;
  parser.setFormatType(DateTimeParser::FormatType::Time);
  std::optional<int64_t> time = parser.parse(str, dim);
  if (!time) {
    return std::nullopt;
  }
  // Parse optional timezone
  std::string_view timezone = parser.unparsed();
  parser.setFormatType(DateTimeParser::FormatType::Timezone);
  std::optional<int64_t> tz = parser.parse(timezone, dim);
  if (!parser.unparsed().empty()) {
    return std::nullopt;
  }
  return *time + tz.value_or(0);
}

// Interpret str according to DateTimeParser::FormatType::Date and Time.
// Return number of (s,ms,us,ns) since epoch based on dim in (0,3,6,9) resp.
template <>
std::optional<int64_t> dateTimeParseOptional<kTIMESTAMP>(std::string_view str,
                                                         unsigned const dim) {
  if (!str.empty() && str.front() == 'T') {
    str.remove_prefix(1);
  }
  DateTimeParser parser;
  // Parse date
  parser.setFormatType(DateTimeParser::FormatType::Date);
  std::optional<int64_t> date = parser.parse(str, dim);
  if (!date) {
    return unixTime(str);
  }
  // Parse time-of-day
  std::string_view time_of_day = parser.unparsed();
  if (time_of_day.empty()) {
    return std::nullopt;
  } else if (time_of_day.front() == 'T' || time_of_day.front() == ':') {
    time_of_day.remove_prefix(1);
  }
  parser.setFormatType(DateTimeParser::FormatType::Time);
  std::optional<int64_t> time = parser.parse(time_of_day, dim);
  // Parse optional timezone
  std::string_view timezone = parser.unparsed();
  parser.setFormatType(DateTimeParser::FormatType::Timezone);
  std::optional<int64_t> tz = parser.parse(timezone, dim);
  return *date + time.value_or(0) + tz.value_or(0);
}

// Interpret str according to DateTimeParser::FormatType::Date.
// Return number of (s,ms,us,ns) since epoch based on dim in (0,3,6,9) resp.
template <>
std::optional<int64_t> dateTimeParseOptional<kDATE>(std::string_view str,
                                                    unsigned const dim) {
  DateTimeParser parser;
  // Parse date
  parser.setFormatType(DateTimeParser::FormatType::Date);
  std::optional<int64_t> date = parser.parse(str, dim);
  if (!date) {
    return unixTime(str);
  }
  // Parse optional timezone
  std::string_view timezone = parser.unparsed();
  parser.setFormatType(DateTimeParser::FormatType::Timezone);
  std::optional<int64_t> tz = parser.parse(timezone, dim);
  return *date + tz.value_or(0);
}

// Return number of (s,ms,us,ns) since epoch based on dim in (0,3,6,9) resp.
int64_t DateTimeParser::DateTime::getTime(unsigned const dim) const {
  int64_t const days = daysFromCivil(Y, m, d);
  int const seconds = static_cast<int>(3600 * H + 60 * M + S) - z +
                      (p ? *p && H != 12    ? 12 * 3600
                           : !*p && H == 12 ? -12 * 3600
                                            : 0
                         : 0);
  return (24 * 3600 * days + seconds) * pow_10[dim] + n / pow_10[9 - dim];
}

// Return true if successful parse, false otherwise.  Update dt_ and str.
// OK to be destructive to str on failed match.
bool DateTimeParser::parseWithFormat(std::string_view format, std::string_view& str) {
  while (!format.empty()) {
    if (format.front() == '%') {
      eatSpace(str);
      if (!updateDateTimeAndStr(format[1], str)) {
        return false;
      }
      format.remove_prefix(2);
    } else if (isspace(format.front())) {
      eatSpace(format);
      eatSpace(str);
    } else if (!str.empty() && format.front() == str.front()) {
      format.remove_prefix(1);
      str.remove_prefix(1);
    } else {
      return false;
    }
  }
  return true;
}

// Update dt_ based on given str and current value of format_type_.
// Return number of (s,ms,us,ns) since epoch based on dim in (0,3,6,9) resp.
// or std::nullopt if no format matches str.
// In either case, update unparsed_ to the remaining part of str that was not matched.
std::optional<int64_t> DateTimeParser::parse(std::string_view const str, unsigned dim) {
  static std::vector<std::vector<std::string_view>> const& format_views = formatViews();
  auto const& formats = format_views.at(static_cast<int>(format_type_));
  for (std::string_view const format : formats) {
    std::string_view str_unparsed = str;
    if (parseWithFormat(format, str_unparsed)) {
      unparsed_ = str_unparsed;
      return dt_.getTime(dim);
    }
  }
  unparsed_ = str;
  return std::nullopt;
}

void DateTimeParser::resetDateTime() {
  dt_ = DateTime();
}

void DateTimeParser::setFormatType(FormatType format_type) {
  resetDateTime();
  format_type_ = format_type;
}

std::string_view DateTimeParser::unparsed() const {
  return unparsed_;
}

// Return true if successful parse, false otherwise.  Update dt_ and str on success.
// OK to be destructive to str on failed parse.
bool DateTimeParser::updateDateTimeAndStr(char const field, std::string_view& str) {
  switch (field) {
    case 'Y':
      if (auto const year = fromChars<int64_t>(str)) {
        dt_.Y = *year;
        return true;
      }
      return false;
    case 'y':
      // %y matches 1 or 2 digits. If 3 or more digits are provided,
      // then it is considered an unsuccessful parse.
      if (auto const year = fromChars<unsigned>(str)) {
        if (*year < 69) {
          dt_.Y = 2000 + *year;
          return true;
        } else if (*year < 100) {
          dt_.Y = 1900 + *year;
          return true;
        }
      }
      return false;
    case 'm':
      if (auto const month = fromChars<unsigned>(str, 2)) {
        if (1 <= *month && *month <= 12) {
          dt_.m = *month;
          return true;
        }
      }
      return false;
    case 'b':
      if (3 <= str.size()) {
        int const key =
            std::tolower(str[0]) << 16 | std::tolower(str[1]) << 8 | std::tolower(str[2]);
        constexpr auto end = month_prefixes.data() + month_prefixes.size();
        // This is faster than a lookup into a std::unordered_map.
        auto const ptr = std::find(month_prefixes.data(), end, key);
        if (ptr != end) {
          dt_.m = ptr - month_prefixes.data() + 1;
          eatMonth(dt_.m, str);
          return true;
        }
      }
      return false;
    case 'd':
      if (auto const day = fromChars<unsigned>(str, 2)) {
        if (1 <= *day && *day <= 31) {
          dt_.d = *day;
          return true;
        }
      }
      return false;
    case 'H':
      if (auto const hour = fromChars<unsigned>(str, 2)) {
        if (*hour <= 23) {
          dt_.H = *hour;
          return true;
        }
      }
      return false;
    case 'I':
      if (auto const hour = fromChars<unsigned>(str, 2)) {
        if (1 <= *hour && *hour <= 12) {
          dt_.H = *hour;
          return true;
        }
      }
      return false;
    case 'M':
      if (auto const minute = fromChars<unsigned>(str, 2)) {
        if (*minute <= 59) {
          dt_.M = *minute;
          return true;
        }
      }
      return false;
    case 'S':
      if (auto const second = fromChars<unsigned>(str, 2)) {
        if (*second <= 61) {
          dt_.S = *second;
          if (!str.empty() && str.front() == '.') {
            str.remove_prefix(1);
            size_t len = str.size();
            if (auto const ns = fromChars<unsigned>(str, 9)) {
              len -= str.size();
              dt_.n = *ns * pow_10[9 - len];
            } else {
              return false;  // Reject period not followed by a digit
            }
          }
          return true;
        }
      }
      return false;
    case 'z':
      // [-+]\d\d:?\d\d
      if (5 <= str.size() && (str.front() == '-' || str.front() == '+') &&
          isdigit(str[1]) && isdigit(str[2]) && isdigit(str[4]) &&
          (str[3] == ':' ? 6 <= str.size() && isdigit(str[5]) : isdigit(str[3]))) {
        char const* sep = &str[3];
        int hours{0}, minutes{0};
        std::from_chars(str.data() + 1, sep, hours);
        sep += *sep == ':';
        std::from_chars(sep, sep + 2, minutes);
        dt_.z = (str.front() == '-' ? -60 : 60) * (60 * hours + minutes);
        str.remove_prefix(sep - str.data() + 2);
        return true;
      }
      return false;
    case 'p':
      // %p implies optional, so never return false
      if (boost::algorithm::istarts_with(str, "am") ||
          boost::algorithm::istarts_with(str, "pm") ||
          boost::algorithm::istarts_with(str, "a.m.") ||
          boost::algorithm::istarts_with(str, "p.m.")) {
        dt_.p = std::tolower(str.front()) == 'p';
        str.remove_prefix(std::tolower(str[1]) == 'm' ? 2 : 4);
      } else {
        dt_.p.reset();
      }
      return true;
    default:
      throw std::runtime_error(cat("Unrecognized format: %", field));
  }
}

std::ostream& operator<<(std::ostream& out, DateTimeParser::DateTime const& dt) {
  return out << dt.Y << '-' << dt.m << '-' << dt.d << ' ' << dt.H << ':' << dt.M << ':'
             << dt.S << '.' << dt.n << " p("
             << (dt.p ? *dt.p ? "true" : "false" : "unset") << ") z(" << dt.z << ')';
}
