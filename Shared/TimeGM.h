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
 * @brief		Functions to handle date/time types
 **/

#ifndef TIMEGM_H
#define TIMEGM_H
#include <cmath>
#include <cstring>
#include <ctime>
#include <sstream>
#include <type_traits>
#include "sqltypes.h"

#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#endif
#include <cinttypes>

class TimeGM {
 public:
  time_t my_timegm(const struct tm* tm);
  time_t my_timegm(const struct tm* tm, const time_t fsc, const int32_t dimen);
  time_t my_timegm_days(const struct tm* tm);
  time_t parse_fractional_seconds(uint64_t sfrac,
                                  const int32_t ntotal,
                                  const int32_t dimen);
  time_t parse_meridians(const time_t& timeval,
                         const char* p,
                         const uint32_t hour,
                         const int32_t dimen);
  static TimeGM& instance() {
    static TimeGM timegm{};
    return timegm;
  }

 private:
  /* Number of days per month (except for February in leap years). */
  const int monoff[12] = {0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334};
  int is_leap_year(int year);
  int leap_days(int y1, int y2);
  TimeGM(){};
  virtual ~TimeGM(){};
};

template <SQLTypes TYPE>
class DateTimeStringValidate {
 public:
  constexpr int64_t operator()(const std::string& datetime_str, const int32_t dimen) {
    static_assert(TYPE == kDATE || TYPE == kTIME || TYPE == kTIMESTAMP);
    str_ = datetime_str;
    dimen_ = dimen;
    return parseDateTimeString();
  };

 private:
  constexpr int64_t parseDateTimeString() {
    char* tp = nullptr;
    if constexpr (TYPE == kTIME) {
      tm_.tm_hour = tm_.tm_min = tm_.tm_sec = -1;
      parseTimePart<const char*>(str_.c_str(), tp);
      tm_.tm_mday = 1;
      tm_.tm_mon = 0;
      tm_.tm_year = 70;
      return static_cast<int64_t>(TimeGM::instance().my_timegm(&tm_));
    }
    parseDatePart(str_, tp);
    if (!tp) {
      return getEpochValue();
    }
    char* p = nullptr;
    parseTimePart<char*&>(tp, p);
    if constexpr (TYPE == kDATE) {
      return static_cast<int64_t>(TimeGM::instance().my_timegm(&tm_));
    }
    // handle fractional seconds
    int64_t timeval = 0;
    if (dimen_ > 0) {
      const int64_t frac = parseFractionalSeconds(p);
      timeval = static_cast<int64_t>(TimeGM::instance().my_timegm(&tm_, frac, dimen_));
    } else {
      timeval = static_cast<int64_t>(TimeGM::instance().my_timegm(&tm_));
      if (*p == '.') {
        p++;
      }
    }
    if (*p != '\0') {
      uint32_t hour = 0;
      sscanf(tp, "%u", &hour);
      timeval = static_cast<int64_t>(TimeGM::instance().parse_meridians(
          static_cast<time_t>(timeval), p, hour, dimen_));
    }
    return timeval;
  }

  template <typename T>
  constexpr void parseTimePart(T s, char*& p) {
    if constexpr (TYPE == kDATE) {
      if (*s == 'T' || *s == ' ') {
        ++s;
      }
      p = strptime(s, "%z", &tm_);
      return;
    } else if constexpr (TYPE == kTIMESTAMP) {
      if (*s == 'T' || *s == ' ' || *s == ':') {
        ++s;
      } else {
        throw std::runtime_error("Invalid TIMESTAMP break string " + std::string(s));
      }
    }
    p = strptime(s, "%T %z", &tm_);
    if (!p) {
      p = strptime(s, "%T", &tm_);
    }
    if (!p) {
      p = strptime(s, "%H%M%S", &tm_);
    }
    if (!p) {
      p = strptime(s, "%R", &tm_);
    }
    if constexpr (TYPE == kTIME) {
      if (tm_.tm_hour == -1 && tm_.tm_min == -1 && tm_.tm_sec == -1) {
        throw std::runtime_error("Invalid TIME string " + str_);
      }
      return;
    }
    if constexpr (TYPE == kTIMESTAMP) {
      if (!p) {
        // check for weird customer format and remove decimal seconds from string if there
        // is a period followed by a number
        char* startptr = nullptr;
        char* endptr;
        // find last decimal in string
        int loop = strlen(s);
        while (loop > 0) {
          if (s[loop] == '.') {
            // found last period
            startptr = &s[loop];
            break;
          }
          loop--;
        }
        if (startptr) {
          // look for space
          endptr = strchr(startptr, ' ');
          if (endptr) {
            // ok we found a start and and end remove the decimal portion will need to
            // capture this for later
            memmove(startptr, endptr, strlen(endptr) + 1);
          }
        }
        p = strptime(s, "%I . %M . %S %p", &tm_);  // customers weird '.' separated date
      }
      if (!p) {
        throw std::runtime_error("Invalid TIMESTAMP time string " + std::string(s));
      }
    }
  }

  constexpr void parseDatePart(const std::string& s, char*& p) {
    // try ISO8601 date first
    p = strptime(s.c_str(), "%Y-%m-%d", &tm_);
    if (!p) {
      p = strptime(s.c_str(), "%m/%d/%Y", &tm_);  // accept American date
    }
    if (!p) {
      p = strptime(s.c_str(), "%d-%b-%y", &tm_);  // accept 03-Sep-15
    }
    if (!p) {
      p = strptime(s.c_str(), "%d/%b/%Y", &tm_);  // accept 03/Sep/2015
    }
  }

  constexpr int64_t parseFractionalSeconds(char*& p) {
    if (*p == '.') {
      p++;
      uint64_t frac_num = 0;
      int ntotal = 0;
      sscanf(p, "%" SCNu64 "%n", &frac_num, &ntotal);
      return static_cast<int64_t>(
          TimeGM::instance().parse_fractional_seconds(frac_num, ntotal, dimen_));
    } else if (*p == '\0') {
      return 0;
    } else {  // check for misleading/unclear syntax
      throw std::runtime_error("Unclear syntax for leading fractional seconds: " +
                               std::string(p));
    }
  }

  const int64_t getEpochValue() const {
    // Check for unix time format
    try {
      return static_cast<int64_t>(std::stoll(str_));
    } catch (const std::invalid_argument& ia) {
      throw std::runtime_error("Invalid DATE/TIMESTAMP string " + str_);
    }
  }

  std::string str_;
  int32_t dimen_;
  std::tm tm_{0};
};

#endif  // TIMEGM_H
