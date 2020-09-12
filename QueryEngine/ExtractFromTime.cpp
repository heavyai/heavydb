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

/*
 * Thank you Howard Hinnant for public domain date algorithms
 * http://howardhinnant.github.io/date_algorithms.html
 */

#include "ExtractFromTime.h"
#include "../Shared/funcannotations.h"

#ifndef __CUDACC__
#include <cstdlib>  // abort()
#endif

namespace {

// Return day-of-era of the Monday of ISO 8601 week 1 in the given yoe.
// Week 1 always contains Jan 4.
DEVICE unsigned iso_week_start_from_yoe(unsigned const yoe) {
  unsigned const march1 = yoe * 365 + yoe / 4 - yoe / 100;
  unsigned const jan4 = march1 + (31 + 30 + 31 + 30 + 31 + 31 + 30 + 31 + 30 + 31 + 3);
  unsigned const jan4dow = (jan4 + 2) % 7;  // 2000-03-01 is Wed so + 2 to get Mon = 0.
  return jan4 - jan4dow;
}

}  // namespace

extern "C" ALWAYS_INLINE DEVICE int64_t extract_hour(const int64_t lcltime) {
  return unsigned_mod(lcltime, kSecsPerDay) / kSecsPerHour;
}

extern "C" ALWAYS_INLINE DEVICE int64_t extract_minute(const int64_t lcltime) {
  return unsigned_mod(lcltime, kSecsPerHour) / kSecsPerMin;
}

extern "C" ALWAYS_INLINE DEVICE int64_t extract_second(const int64_t lcltime) {
  return unsigned_mod(lcltime, kSecsPerMin);
}

extern "C" ALWAYS_INLINE DEVICE int64_t extract_millisecond(const int64_t lcltime) {
  return unsigned_mod(lcltime, kSecsPerMin * kMilliSecsPerSec);
}

extern "C" ALWAYS_INLINE DEVICE int64_t extract_microsecond(const int64_t lcltime) {
  return unsigned_mod(lcltime, kSecsPerMin * kMicroSecsPerSec);
}

extern "C" ALWAYS_INLINE DEVICE int64_t extract_nanosecond(const int64_t lcltime) {
  return unsigned_mod(lcltime, kSecsPerMin * kNanoSecsPerSec);
}

// First day of epoch is Thursday, so + 4 to have Sunday=0.
extern "C" ALWAYS_INLINE DEVICE int64_t extract_dow(const int64_t lcltime) {
  int64_t const days_past_epoch = floor_div(lcltime, kSecsPerDay);
  return unsigned_mod(days_past_epoch + 4, kDaysPerWeek);
}

extern "C" ALWAYS_INLINE DEVICE int64_t extract_quarterday(const int64_t lcltime) {
  return unsigned_mod(lcltime, kSecsPerDay) / kSecsPerQuarterDay + 1;
}

DEVICE int32_t extract_month_fast(const int64_t lcltime) {
  STATIC_QUAL const uint32_t cumulative_month_epoch_starts[kMonsPerYear] = {0,
                                                                            2678400,
                                                                            5270400,
                                                                            7948800,
                                                                            10540800,
                                                                            13219200,
                                                                            15897600,
                                                                            18489600,
                                                                            21168000,
                                                                            23760000,
                                                                            26438400,
                                                                            29116800};
  uint32_t seconds_march_1900 = lcltime + kEpochOffsetYear1900 - kSecsJanToMar1900;
  uint32_t seconds_past_4year_period = seconds_march_1900 % kSecondsPer4YearCycle;
  uint32_t year_seconds_past_4year_period =
      (seconds_past_4year_period / kSecondsPerNonLeapYear) * kSecondsPerNonLeapYear;
  if (seconds_past_4year_period >=
      kSecondsPer4YearCycle - kUSecsPerDay) {  // if we are in Feb 29th
    year_seconds_past_4year_period -= kSecondsPerNonLeapYear;
  }
  uint32_t seconds_past_march =
      seconds_past_4year_period - year_seconds_past_4year_period;
  uint32_t month =
      seconds_past_march / (30 * kUSecsPerDay);  // Will make the correct month either be
                                                 // the guessed month or month before
  month = month <= 11 ? month : 11;
  if (cumulative_month_epoch_starts[month] > seconds_past_march) {
    month--;
  }
  return (month + 2) % 12 + 1;
}

DEVICE int32_t extract_quarter_fast(const int64_t lcltime) {
  STATIC_QUAL const uint32_t cumulative_quarter_epoch_starts[4] = {
      0, 7776000, 15638400, 23587200};
  STATIC_QUAL const uint32_t cumulative_quarter_epoch_starts_leap_year[4] = {
      0, 7862400, 15724800, 23673600};
  uint32_t seconds_1900 = lcltime + kEpochOffsetYear1900;
  uint32_t leap_years = (seconds_1900 - kSecsJanToMar1900) / kSecondsPer4YearCycle;
  uint32_t year = (seconds_1900 - leap_years * kSecsPerDay) / kSecondsPerNonLeapYear;
  uint32_t base_year_leap_years = (year - 1) / 4;
  uint32_t base_year_seconds =
      year * kSecondsPerNonLeapYear + base_year_leap_years * kUSecsPerDay;
  bool is_leap_year = year % 4 == 0 && year != 0;
  const uint32_t* quarter_offsets = is_leap_year
                                        ? cumulative_quarter_epoch_starts_leap_year
                                        : cumulative_quarter_epoch_starts;
  uint32_t partial_year_seconds = seconds_1900 % base_year_seconds;
  uint32_t quarter = partial_year_seconds / (90 * kUSecsPerDay);
  quarter = quarter <= 3 ? quarter : 3;
  if (quarter_offsets[quarter] > partial_year_seconds) {
    quarter--;
  }
  return quarter + 1;
}

DEVICE int32_t extract_year_fast(const int64_t lcltime) {
  const uint32_t seconds_1900 = lcltime + kEpochOffsetYear1900;
  const uint32_t leap_years = (seconds_1900 - kSecsJanToMar1900) / kSecondsPer4YearCycle;
  const uint32_t year =
      (seconds_1900 - leap_years * kUSecsPerDay) / kSecondsPerNonLeapYear + 1900;
  return year;
}

extern "C" ALWAYS_INLINE DEVICE int64_t extract_epoch(const int64_t timeval) {
  return timeval;
}

extern "C" ALWAYS_INLINE DEVICE int64_t extract_dateepoch(const int64_t timeval) {
  return timeval - unsigned_mod(timeval, kSecsPerDay);
}

// First day of epoch is Thursday, so + 3 to have Monday=0, then + 1 at the end.
extern "C" ALWAYS_INLINE DEVICE int64_t extract_isodow(const int64_t timeval) {
  int64_t const days_past_epoch = floor_div(timeval, kSecsPerDay);
  return unsigned_mod(days_past_epoch + 3, kDaysPerWeek) + 1;
}

extern "C" ALWAYS_INLINE DEVICE int64_t extract_day(const int64_t timeval) {
  int64_t const day = floor_div(timeval, kSecsPerDay);
  unsigned const doe = unsigned_mod(day - kEpochAdjustedDays, kDaysPer400Years);
  unsigned const yoe = (doe - doe / 1460 + doe / 36524 - (doe == 146096)) / 365;
  unsigned const doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
  unsigned const moy = (5 * doy + 2) / 153;
  return doy - (153 * moy + 2) / 5 + 1;
}

extern "C" ALWAYS_INLINE DEVICE int64_t extract_day_of_year(const int64_t timeval) {
  int64_t const day = floor_div(timeval, kSecsPerDay);
  unsigned const doe = unsigned_mod(day - kEpochAdjustedDays, kDaysPer400Years);
  unsigned const yoe = (doe - doe / 1460 + doe / 36524 - (doe == 146096)) / 365;
  unsigned const doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
  return doy + (doy < MARJAN ? 1 + JANMAR + (yoe % 4 == 0 && (yoe % 100 != 0 || yoe == 0))
                             : 1 - MARJAN);
}

extern "C" ALWAYS_INLINE DEVICE int64_t extract_week(const int64_t timeval) {
  int64_t const day = floor_div(timeval, kSecsPerDay);
  unsigned const doe = unsigned_mod(day - kEpochAdjustedDays, kDaysPer400Years);
  unsigned const yoe = (doe - doe / 1460 + doe / 36524 - (doe == 146096)) / 365;
  unsigned iso_week_start = iso_week_start_from_yoe(yoe);
  if (doe < iso_week_start) {
    if (yoe == 0) {
      return (doe + 2) / 7 + 9;  // 2000-03-01 is +2 days from Mon, week +9.
    } else {
      iso_week_start = iso_week_start_from_yoe(yoe - 1);
    }
  }
  return (doe - iso_week_start) / 7 + 1;
}

extern "C" ALWAYS_INLINE DEVICE int64_t extract_month(const int64_t timeval) {
  if (timeval >= 0L && timeval <= UINT32_MAX - kEpochOffsetYear1900) {
    return extract_month_fast(timeval);
  }
  int64_t const day = floor_div(timeval, kSecsPerDay);
  unsigned const doe = unsigned_mod(day - kEpochAdjustedDays, kDaysPer400Years);
  unsigned const yoe = (doe - doe / 1460 + doe / 36524 - (doe == 146096)) / 365;
  unsigned const doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
  unsigned const moy = (5 * doy + 2) / 153;
  return moy + (moy < 10 ? 3 : -9);
}

extern "C" ALWAYS_INLINE DEVICE int64_t extract_quarter(const int64_t timeval) {
  if (timeval >= 0L && timeval <= UINT32_MAX - kEpochOffsetYear1900) {
    return extract_quarter_fast(timeval);
  }
  constexpr int64_t quarter[12]{1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 1, 1};
  int64_t const day = floor_div(timeval, kSecsPerDay);
  unsigned const doe = unsigned_mod(day - kEpochAdjustedDays, kDaysPer400Years);
  unsigned const yoe = (doe - doe / 1460 + doe / 36524 - (doe == 146096)) / 365;
  unsigned const doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
  unsigned const moy = (5 * doy + 2) / 153;
  return quarter[moy];
}

extern "C" ALWAYS_INLINE DEVICE int64_t extract_year(const int64_t timeval) {
  if (timeval >= 0L && timeval <= UINT32_MAX - kEpochOffsetYear1900) {
    return extract_year_fast(timeval);
  }
  int64_t const day = floor_div(timeval, kSecsPerDay);
  int64_t const era = floor_div(day - kEpochAdjustedDays, kDaysPer400Years);
  unsigned const doe = day - kEpochAdjustedDays - era * kDaysPer400Years;
  unsigned const yoe = (doe - doe / 1460 + doe / 36524 - (doe == 146096)) / 365;
  unsigned const doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
  return 2000 + era * 400 + yoe + (MARJAN <= doy);
}

/*
 * @brief support the SQL EXTRACT function
 */
DEVICE int64_t ExtractFromTime(ExtractField field, const int64_t timeval) {
  switch (field) {
    case kEPOCH:
      return extract_epoch(timeval);
    case kDATEEPOCH:
      return extract_dateepoch(timeval);
    case kQUARTERDAY:
      return extract_quarterday(timeval);
    case kHOUR:
      return extract_hour(timeval);
    case kMINUTE:
      return extract_minute(timeval);
    case kSECOND:
      return extract_second(timeval);
    case kMILLISECOND:
      return extract_millisecond(timeval);
    case kMICROSECOND:
      return extract_microsecond(timeval);
    case kNANOSECOND:
      return extract_nanosecond(timeval);
    case kDOW:
      return extract_dow(timeval);
    case kISODOW:
      return extract_isodow(timeval);
    case kDAY:
      return extract_day(timeval);
    case kWEEK:
      return extract_week(timeval);
    case kDOY:
      return extract_day_of_year(timeval);
    case kMONTH:
      return extract_month(timeval);
    case kQUARTER:
      return extract_quarter(timeval);
    case kYEAR:
      return extract_year(timeval);
  }

#ifdef __CUDACC__
  return -1;
#else
  abort();
#endif
}
