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

#include <algorithm>
#include <cctype>
#include <charconv>
#include <sstream>

namespace {
constexpr unsigned
    pow_10[10]{1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000};
constexpr std::array<char const*, 5> date_formats{
    {"%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d", "%d-%b-%y", "%d/%b/%Y"}};
constexpr std::array<char const*, 7> time_formats{{"%I:%M:%S %p",
                                                   "%H:%M:%S",
                                                   "%I:%M %p",
                                                   "%H:%M",
                                                   "%H%M%S",
                                                   "%I %p",
                                                   "%I . %M . %S %p"}};

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

// Parse str into int64_t or throw exception.
int64_t unixTime(std::string_view const& str) {
  int64_t time;
  auto const result = std::from_chars(str.data(), str.data() + str.size(), time);
  if (result.ec == std::errc()) {
    return time;
  } else {
    throw std::runtime_error(cat("Invalid DATE/TIMESTAMP string (", str, ')'));
  }
}
}  // namespace

// Interpret str according to first matched pattern in time_formats.
// Return number of (s,ms,us,ns) since midnight based on dim in (0,3,6,9) resp.
template <>
int64_t dateTimeParse<kTIME>(std::string_view str, unsigned const dim) {
  if (str.front() == 'T') {
    str.remove_prefix(1);
  }
  DateTimeParser parser;
  std::optional<int64_t> time;
  for (char const* time_format : time_formats) {
    parser.setFormat(time_format);
    if ((time = parser.parse(str, dim))) {
      break;
    }
  }
  if (!time) {
    throw std::runtime_error(cat("Invalid TIME string (", str, ')'));
  }
  // Parse optional timezone
  std::string_view timezone = parser.unparsed();
  parser.setFormat("%z");
  std::optional<int64_t> tz = parser.parse(timezone, dim);
  if (!parser.unparsed().empty()) {
    throw std::runtime_error(cat("Invalid TIME string (", str, ')'));
  }
  return *time + tz.value_or(0);
}

// Interpret str according to first matched pattern in date_formats and time_formats.
// Return number of (s,ms,us,ns) since epoch based on dim in (0,3,6,9) resp.
template <>
int64_t dateTimeParse<kTIMESTAMP>(std::string_view str, unsigned const dim) {
  if (str.front() == 'T') {
    str.remove_prefix(1);
  }
  DateTimeParser parser;
  // Parse date
  std::optional<int64_t> date;
  for (char const* date_format : date_formats) {
    parser.setFormat(date_format);
    if ((date = parser.parse(str, dim))) {
      break;
    }
  }
  if (!date) {
    return unixTime(str);
  }
  // Parse optional time-of-day
  std::string_view time_of_day = parser.unparsed();
  if (time_of_day.empty()) {
    throw std::runtime_error(cat("TIMESTAMP requires a time-of-day (", str, ')'));
  }
  if (time_of_day.front() == 'T' || time_of_day.front() == ':') {
    time_of_day.remove_prefix(1);
  }
  std::optional<int64_t> time;
  for (char const* time_format : time_formats) {
    parser.setFormat(time_format);
    if ((time = parser.parse(time_of_day, dim))) {
      break;
    }
  }
  // Parse optional timezone
  std::string_view timezone = parser.unparsed();
  parser.setFormat("%z");
  std::optional<int64_t> tz = parser.parse(timezone, dim);
  return *date + time.value_or(0) + tz.value_or(0);
}

// Interpret str according to first matched pattern in date_formats.
// Return number of (s,ms,us,ns) since epoch based on dim in (0,3,6,9) resp.
template <>
int64_t dateTimeParse<kDATE>(std::string_view str, unsigned const dim) {
  DateTimeParser parser;
  // Parse date
  std::optional<int64_t> date;
  for (char const* date_format : date_formats) {
    parser.setFormat(date_format);
    if ((date = parser.parse(str, dim))) {
      break;
    }
  }
  if (!date) {
    return unixTime(str);
  }
  // Parse optional timezone
  std::string_view timezone = parser.unparsed();
  parser.setFormat("%z");
  std::optional<int64_t> tz = parser.parse(timezone, dim);
  return *date + tz.value_or(0);
}

// Return number of (s,ms,us,ns) since epoch based on dim in (0,3,6,9) resp.
int64_t DateTimeParser::DateTime::getTime(unsigned const dim) const {
  int64_t const days = daysFromCivil(Y, m, d);
  int const seconds =
      static_cast<int>(3600 * H + 60 * M + S) - z +
      (p ? *p && H != 12 ? 12 * 3600 : !*p && H == 12 ? -12 * 3600 : 0 : 0);
  return (24 * 3600 * days + seconds) * pow_10[dim] + n / pow_10[9 - dim];
}

// Translate given format into regex and save in regex_.
// Save format character (e.g. 'Y' for "%Y") into fields_.
// Characters in fields_ will correspond to capturing groups in regex_.
void DateTimeParser::setFormat(std::string_view const& format) {
  resetDateTime();
  fields_.clear();
  std::ostringstream oss;
  oss << '^';
  std::ostream_iterator<char, char> oitr(oss);
  boost::regex_replace(
      oitr, format.begin(), format.end(), field_regex, [this](boost::cmatch const& md) {
        if (*md[0].first == '%') {  // matched /%(.)/
          auto itr = field_to_regex.find(*md[1].first);
          if (itr == field_to_regex.end()) {
            throw std::runtime_error(cat("Unrecognized format: %", *md[1].first));
          }
          if (itr->first != '%') {
            fields_.push_back(itr->first);
          }
          return itr->second;
        } else if (*md[0].first == '.') {  // matched /\./
          return "\\.";                    // escape regex wilcard character.
        } else {                           // matched /\s+/
          // whitespace "matches zero or more whitespace characters in the input string."
          return "\\s*";
        }
      });
  regex_.assign(oss.str(), boost::regex::icase);
}

// Parse given str and update dt_ based on current values of regex_ and fields_.
// Return number of (s,ms,us,ns) since epoch based on dim in (0,3,6,9) resp.
// or std::nullopt if regex_ does not match str.
// In either case, update unparsed_ to the remaining part of str that was not matched.
std::optional<int64_t> DateTimeParser::parse(std::string_view const& str, unsigned dim) {
  boost::cmatch md;
  if (boost::regex_search(str.begin(), str.end(), md, regex_)) {
    for (unsigned i = 1; i < md.size(); ++i) {
      updateDateTime(fields_[i - 1], std::string_view(md[i].first, md[i].length()));
    }
    unparsed_ = str.substr(md[0].length());
    return dt_.getTime(dim);
  }
  unparsed_ = str;
  return std::nullopt;
}

void DateTimeParser::resetDateTime() {
  dt_ = DateTime();
}

std::string_view DateTimeParser::unparsed() const {
  return unparsed_;
}

// Update dt_ based on given field and matched str.
// It is assumed that str matches the capture group in field_to_regex[field].
void DateTimeParser::updateDateTime(char const field, std::string_view const& str) {
  switch (field) {
    case 'Y':
      std::from_chars(str.data(), str.data() + str.size(), dt_.Y);
      break;
    case 'y':
      std::from_chars(str.data(), str.data() + str.size(), dt_.Y);
      dt_.Y += dt_.Y < 69 ? 2000 : 1900;
      break;
    case 'm':
      std::from_chars(str.data(), str.data() + str.size(), dt_.m);
      break;
    case 'b': {
      int const key =
          std::tolower(str[0]) << 16 | std::tolower(str[1]) << 8 | std::tolower(str[2]);
      dt_.m = month_name_lookup.at(key);
    }
    case 'd':
      std::from_chars(str.data(), str.data() + str.size(), dt_.d);
      break;
    case 'H':
    case 'I':
      std::from_chars(str.data(), str.data() + str.size(), dt_.H);
      break;
    case 'M':
      std::from_chars(str.data(), str.data() + str.size(), dt_.M);
      break;
    case 'S': {
      size_t const radix_pos = str.find('.');
      if (radix_pos == std::string_view::npos) {
        std::from_chars(str.data(), str.data() + str.size(), dt_.S);
      } else {
        char const* const radix = &str[radix_pos];
        std::from_chars(str.data(), radix, dt_.S);
        size_t const frac_len = std::min(size_t(9), str.size() - (radix_pos + 1));
        char const* const end = radix + (frac_len + 1);
        std::from_chars(radix + 1, end, dt_.n);
        dt_.n *= pow_10[9 - frac_len];
      }
    } break;
    case 'z': {
      char const* sep = &str[3];
      int hours, minutes;
      std::from_chars(str.data() + 1, sep, hours);
      sep += str.size() == 6;  // if str has a colon, increment sep to next char.
      std::from_chars(sep, sep + 2, minutes);
      dt_.z = (str.front() == '-' ? -60 : 60) * (60 * hours + minutes);
    } break;
    case 'p':
      dt_.p = str.empty() ? std::nullopt
                          : std::optional<bool>(std::tolower(str.front()) == 'p');
      break;
    default:
      throw std::runtime_error(cat("Unrecognized format: %", field));
  }
}

// Used to search and replace the strings in date_formats and time_formats to
// transform them into regular expressions:
// %(.) -> string value in field_to_regex.
// \.   -> match period in regex
// \s+  -> match \s* in regex (0 or more whitespace). This matches strptime rules.
const boost::regex DateTimeParser::field_regex("%(.)|\\.|\\s+");

// Whitespace may always precede a token, but not proceed it, unless it's in the input.
const std::unordered_map<char, char const*> DateTimeParser::field_to_regex{
    {'Y', "\\s*(-?\\d+)"},
    {'y', "\\s*(\\d\\d)"},
    {'m', "\\s*(1[012]|0?[1-9])"},
    {'b',
     "\\s*(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?"
     "|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"},
    {'d', "\\s*(3[01]|[12]\\d|0?[1-9])"},
    {'I', "\\s*(1[012]|0?[1-9])"},
    {'H', "\\s*(2[0-3]|1\\d|0?\\d)"},
    {'M', "\\s*([1-5]\\d|0?\\d)"},
    {'S', "\\s*((?:6[01]|[1-5]\\d|0?\\d)(?:\\.\\d*)?)"},
    {'z', "\\s*([-+]\\d\\d:?\\d\\d)"},
    {'p', "\\s*([ap]\\.?m\\.?|)"},
    {'%', "%"}};

const std::unordered_map<int, unsigned> DateTimeParser::month_name_lookup{
    {int('j') << 16 | int('a') << 8 | int('n'), 1},
    {int('f') << 16 | int('e') << 8 | int('b'), 2},
    {int('m') << 16 | int('a') << 8 | int('r'), 3},
    {int('a') << 16 | int('p') << 8 | int('r'), 4},
    {int('m') << 16 | int('a') << 8 | int('y'), 5},
    {int('j') << 16 | int('u') << 8 | int('n'), 6},
    {int('j') << 16 | int('u') << 8 | int('l'), 7},
    {int('a') << 16 | int('u') << 8 | int('g'), 8},
    {int('s') << 16 | int('e') << 8 | int('p'), 9},
    {int('o') << 16 | int('c') << 8 | int('t'), 10},
    {int('n') << 16 | int('o') << 8 | int('v'), 11},
    {int('d') << 16 | int('e') << 8 | int('c'), 12}};

std::ostream& operator<<(std::ostream& out, DateTimeParser::DateTime const& dt) {
  return out << dt.Y << '-' << dt.m << '-' << dt.d << ' ' << dt.H << ':' << dt.M << ':'
             << dt.S << '.' << dt.n << " p("
             << (dt.p ? *dt.p ? "true" : "false" : "unset") << ") z(" << dt.z << ')';
}
