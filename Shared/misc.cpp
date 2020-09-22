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

// Credits: Howard Hinnant for open source date calculations.

#include "misc.h"

#include <cstdio>

namespace shared {

size_t formatDate(char* buf, size_t const max, int64_t const unixtime) {
  DivUMod const div_day = divUMod(unixtime, 24 * 60 * 60);
  DivUMod const div_era = divUMod(div_day.quot - 11017, 146097);
  unsigned const doe = static_cast<unsigned>(div_era.rem);
  unsigned const yoe = (doe - doe / 1460 + doe / 36524 - (doe == 146096)) / 365;
  unsigned const doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
  unsigned const moy = (5 * doy + 2) / 153;
  static_assert(8 <= sizeof(long long));  // long long needed for snprintf()
  long long const y = 2000 + div_era.quot * 400 + yoe + (9 < moy);
  unsigned const m = moy + (9 < moy ? -9 : 3);
  unsigned const d = doy - (153 * moy + 2) / 5 + 1;
  int const len = snprintf(buf, max, "%04lld-%02u-%02u", y, m, d);
  if (0 <= len && static_cast<size_t>(len) < max) {
    return static_cast<size_t>(len);
  }
  return 0;
}

size_t formatDateTime(char* buf,
                      size_t const max,
                      int64_t const timestamp,
                      int const dimension) {
  constexpr int pow10[10]{
      1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000};
  DivUMod const div_hip = divUMod(timestamp, pow10[dimension]);
  DivUMod const div_day = divUMod(div_hip.quot, 24 * 60 * 60);
  DivUMod const div_era = divUMod(div_day.quot - 11017, 146097);
  unsigned const doe = static_cast<unsigned>(div_era.rem);
  unsigned const yoe = (doe - doe / 1460 + doe / 36524 - (doe == 146096)) / 365;
  unsigned const doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
  unsigned const moy = (5 * doy + 2) / 153;
  static_assert(8 <= sizeof(long long));  // long long needed for snprintf()
  long long const y = 2000 + div_era.quot * 400 + yoe + (9 < moy);
  unsigned const m = moy + (9 < moy ? -9 : 3);
  unsigned const d = doy - (153 * moy + 2) / 5 + 1;
  unsigned const minutes = static_cast<unsigned>(div_day.rem) / 60;
  unsigned const ss = div_day.rem % 60;
  unsigned const hh = minutes / 60;
  unsigned const mm = minutes % 60;
  int const len =
      snprintf(buf, max, "%04lld-%02u-%02u %02u:%02u:%02u", y, m, d, hh, mm, ss);
  if (0 <= len && static_cast<size_t>(len) < max) {
    if (dimension) {
      int const len_frac = snprintf(
          buf + len, max - len, ".%0*d", dimension, static_cast<int>(div_hip.rem));
      if (0 <= len_frac && static_cast<size_t>(len + len_frac) < max) {
        return static_cast<size_t>(len + len_frac);
      }
    } else {
      return static_cast<size_t>(len);
    }
  }
  return 0;
}

size_t formatHMS(char* buf, size_t const max, int64_t const unixtime) {
  unsigned const seconds = static_cast<unsigned>(unsignedMod(unixtime, 24 * 60 * 60));
  unsigned const minutes = seconds / 60;
  unsigned const ss = seconds % 60;
  unsigned const hh = minutes / 60;
  unsigned const mm = minutes % 60;
  int const len = snprintf(buf, max, "%02u:%02u:%02u", hh, mm, ss);
  if (0 <= len && static_cast<size_t>(len) < max) {
    return static_cast<size_t>(len);
  }
  return 0;
}

}  // namespace shared
