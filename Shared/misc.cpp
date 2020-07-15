/*
 * Copyright 2020 OmniSci, Inc.
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

#include "misc.h"

#include <cstdio>

namespace shared {

size_t formatDate(char* buf, size_t const max, std::tm const* tm) {
  // 10 = strlen("YYYY-MM-DD")
  if (10 < max) {
    size_t const year_len = formatYear(buf, max, tm);
    if (0 < year_len) {
      size_t const mon_day_len = strftime(buf + year_len, max - year_len, "-%m-%d", tm);
      if (0 < mon_day_len) {
        return year_len + mon_day_len;
      }
    }
  }
  return 0;
}

size_t formatDateTime(char* buf, size_t const max, std::tm const* tm) {
  // 19 = strlen("YYYY-MM-DD HH:MM:SS")
  if (19 < max) {
    size_t const year_len = formatYear(buf, max, tm);
    if (0 < year_len) {
      size_t const time_len = strftime(buf + year_len, max - year_len, "-%m-%d %T", tm);
      if (0 < time_len) {
        return year_len + time_len;
      }
    }
  }
  return 0;
}

size_t formatYear(char* buf, size_t const max, std::tm const* tm) {
  int const year = 1900 + tm->tm_year;  // man gmtime
  int const len = snprintf(buf, max, "%04d", year);
  if (4 <= len && static_cast<size_t>(len) < max) {
    return static_cast<size_t>(len);
  }
  return 0;
}

}  // namespace shared
