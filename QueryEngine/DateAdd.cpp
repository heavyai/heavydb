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

#include "DateAdd.h"
#include "ExtractFromTime.h"

#ifdef EXECUTE_INCLUDE

#ifndef __CUDACC__
#include <cstdlib>  // abort()
#endif

namespace {

// Represent time as number of months + day-of-month + seconds since 2000 March 1.
class MonthDaySecond {
  int64_t months;  // Number of months since 2000 March 1.
  unsigned dom;    // day-of-month (0-based)
  unsigned sod;    // second-of-day

  // Clamp day-of-month to max day of the month. E.g. April 31 -> 30.
  DEVICE static unsigned clampDom(unsigned yoe, unsigned moy, unsigned dom) {
    constexpr unsigned max_days[11]{30, 29, 30, 29, 30, 30, 29, 30, 29, 30, 30};
    if (dom < 28) {
      return dom;
    } else {
      unsigned const max_day =
          moy == 11 ? 27 + (++yoe % 4 == 0 && (yoe % 100 != 0 || yoe == 400))
                    : max_days[moy];
      return dom < max_day ? dom : max_day;
    }
  }

 public:
  DEVICE MonthDaySecond(int64_t const timeval) {
    int64_t const day = floor_div(timeval, kSecsPerDay);
    int64_t const era = floor_div(day - kEpochAdjustedDays, kDaysPer400Years);
    sod = timeval - day * kSecsPerDay;
    unsigned const doe = day - kEpochAdjustedDays - era * kDaysPer400Years;
    unsigned const yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    unsigned const doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    unsigned const moy = (5 * doy + 2) / 153;
    dom = doy - (153 * moy + 2) / 5;
    months = (era * 400 + yoe) * 12 + moy;
  }

  DEVICE MonthDaySecond const& addMonths(int64_t const months) {
    this->months += months;
    return *this;
  }

  // Return number of seconds since 1 January 1970.
  DEVICE int64_t unixtime() const {
    int64_t const era = floor_div(months, 12 * 400);
    unsigned const moe = months - era * (12 * 400);
    unsigned const yoe = moe / 12;
    unsigned const moy = moe % 12;
    unsigned const doy = (153 * moy + 2) / 5 + clampDom(yoe, moy, dom);
    unsigned const doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return (kEpochAdjustedDays + era * kDaysPer400Years + doe) * kSecsPerDay + sod;
  }
};

}  // namespace

extern "C" NEVER_INLINE DEVICE int64_t DateAdd(DateaddField field,
                                               const int64_t number,
                                               const int64_t timeval) {
  switch (field) {
    case daNANOSECOND:
    case daMICROSECOND:
    case daMILLISECOND:
    case daSECOND:
      /* this is the limit of current granularity */
      return timeval + number;
    case daMINUTE:
      return timeval + number * kSecsPerMin;
    case daHOUR:
      return timeval + number * kSecsPerHour;
    case daWEEKDAY:
    case daDAYOFYEAR:
    case daDAY:
      return timeval + number * kSecsPerDay;
    case daWEEK:
      return timeval + number * (7 * kSecsPerDay);
    case daMONTH:
      return MonthDaySecond(timeval).addMonths(number).unixtime();
    case daQUARTER:
      return MonthDaySecond(timeval).addMonths(number * 3).unixtime();
    case daYEAR:
      return MonthDaySecond(timeval).addMonths(number * 12).unixtime();
    case daDECADE:
      return MonthDaySecond(timeval).addMonths(number * 120).unixtime();
    case daCENTURY:
      return MonthDaySecond(timeval).addMonths(number * 1200).unixtime();
    case daMILLENNIUM:
      return MonthDaySecond(timeval).addMonths(number * 12000).unixtime();
    default:
#ifdef __CUDACC__
      return -1;
#else
      abort();
#endif
  }
}

extern "C" NEVER_INLINE DEVICE int64_t DateAddHighPrecision(DateaddField field,
                                                            const int64_t number,
                                                            const int64_t timeval,
                                                            const int64_t scale) {
  switch (field) {
    case daNANOSECOND:
    case daMICROSECOND:
    case daMILLISECOND:
      /* Since number is constant, it is being adjusted according to the dimension
      of type and field in RelAlgTranslator. Therefore, here would skip math and
      just add the value.*/
      return timeval + number;
    default:
      return DateAdd(field, number, floor_div(timeval, scale)) * scale +
             unsigned_mod(timeval, scale);
  }
}

extern "C" DEVICE int64_t DateAddNullable(const DateaddField field,
                                          const int64_t number,
                                          const int64_t timeval,
                                          const int64_t null_val) {
  if (timeval == null_val) {
    return null_val;
  }
  return DateAdd(field, number, timeval);
}

extern "C" DEVICE int64_t DateAddHighPrecisionNullable(const DateaddField field,
                                                       const int64_t number,
                                                       const int64_t timeval,
                                                       const int64_t scale,
                                                       const int64_t null_val) {
  if (timeval == null_val) {
    return null_val;
  }
  return DateAddHighPrecision(field, number, timeval, scale);
}

#endif  // EXECUTE_INCLUDE
