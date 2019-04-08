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
#include <string.h>
#include <cmath>
#include <ctime>
#include <sstream>
#include "sqltypes.h"

class TimeGM {
 public:
  time_t my_timegm(const struct tm* tm);
  time_t my_timegm(const struct tm* tm, const time_t& fsc, const SQLTypeInfo& ti);
  time_t my_timegm_days(const struct tm* tm);
  time_t parse_fractional_seconds(uint64_t sfrac,
                                  const int ntotal,
                                  const SQLTypeInfo& ti);
  time_t parse_meridians(const time_t& timeval,
                         const char* p,
                         const uint32_t& hour,
                         const SQLTypeInfo& ti);
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

#endif  // TIMEGM_H
