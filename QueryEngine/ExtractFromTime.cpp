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

#include "ExtractFromTime.h"
#include "../Shared/funcannotations.h"

#ifndef __CUDACC__
#include <glog/logging.h>
#endif

extern "C" NEVER_INLINE DEVICE int32_t extract_hour(const time_t* tim_p) {
  int64_t days, rem;
  const time_t lcltime = *tim_p;
  days = static_cast<int64_t>(lcltime) / SECSPERDAY - EPOCH_ADJUSTMENT_DAYS;
  rem = static_cast<int64_t>(lcltime) % SECSPERDAY;
  if (rem < 0) {
    rem += SECSPERDAY;
    --days;
  }
  return static_cast<int32_t>(rem / SECSPERHOUR);
}

DEVICE int32_t extract_minute(const time_t* tim_p) {
  int64_t days, rem;
  const time_t lcltime = *tim_p;
  days = static_cast<int64_t>(lcltime) / SECSPERDAY - EPOCH_ADJUSTMENT_DAYS;
  rem = static_cast<int64_t>(lcltime) % SECSPERDAY;
  if (rem < 0) {
    rem += SECSPERDAY;
    --days;
  }
  rem %= SECSPERHOUR;
  return static_cast<int32_t>(rem / SECSPERMIN);
}

DEVICE int32_t extract_second(const time_t* tim_p) {
  const time_t lcltime = *tim_p;
  return static_cast<int32_t>(static_cast<int64_t>(lcltime) % SECSPERMIN);
}

DEVICE int32_t extract_millisecond(const time_t* tim_p) {
  const time_t lcltime = *tim_p;
  return static_cast<int32_t>(static_cast<int64_t>(lcltime) % MILLISECSPERSEC);
}

DEVICE int32_t extract_microsecond(const time_t* tim_p) {
  const time_t lcltime = *tim_p;
  return static_cast<int32_t>(static_cast<int64_t>(lcltime) % MICROSECSPERSEC);
}

DEVICE int32_t extract_nanosecond(const time_t* tim_p) {
  const time_t lcltime = *tim_p;
  return static_cast<int32_t>(static_cast<int64_t>(lcltime) % NANOSECSPERSEC);
}

DEVICE int32_t extract_dow(const time_t* tim_p) {
  int64_t days, rem;
  int32_t weekday;
  const time_t lcltime = *tim_p;
  days = static_cast<int64_t>(lcltime) / SECSPERDAY - EPOCH_ADJUSTMENT_DAYS;
  rem = static_cast<int64_t>(lcltime) % SECSPERDAY;
  if (rem < 0) {
    rem += SECSPERDAY;
    --days;
  }

  if ((weekday = ((ADJUSTED_EPOCH_WDAY + days) % DAYSPERWEEK)) < 0) {
    weekday += DAYSPERWEEK;
  }
  return weekday;
}

DEVICE int32_t extract_quarterday(const time_t* tim_p) {
  int64_t quarterdays;
  const time_t lcltime = *tim_p;
  quarterdays = static_cast<int64_t>(lcltime) / SECSPERQUARTERDAY;
  return static_cast<int32_t>(quarterdays % 4) + 1;
}

DEVICE int32_t extract_month_fast(const time_t* tim_p) {
  STATIC_QUAL const uint32_t cumulative_month_epoch_starts[MONSPERYEAR] = {0,
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
  const time_t lcltime = *tim_p;
  uint32_t seconds_march_1900 = static_cast<int64_t>(lcltime) + EPOCH_OFFSET_YEAR_1900 -
                                SECONDS_FROM_JAN_1900_TO_MARCH_1900;
  uint32_t seconds_past_4year_period = seconds_march_1900 % SECONDS_PER_4_YEAR_CYCLE;
  uint32_t year_seconds_past_4year_period =
      (seconds_past_4year_period / SECONDS_PER_NON_LEAP_YEAR) * SECONDS_PER_NON_LEAP_YEAR;
  if (seconds_past_4year_period >=
      SECONDS_PER_4_YEAR_CYCLE - SECONDS_PER_DAY) {  // if we are in Feb 29th
    year_seconds_past_4year_period -= SECONDS_PER_NON_LEAP_YEAR;
  }
  uint32_t seconds_past_march =
      seconds_past_4year_period - year_seconds_past_4year_period;
  uint32_t month = seconds_past_march /
                   (30 * SECONDS_PER_DAY);  // Will make the correct month either be the
                                            // guessed month or month before
  month = month <= 11 ? month : 11;
  if (cumulative_month_epoch_starts[month] > seconds_past_march) {
    month--;
  }
  return (month + 2) % 12 + 1;
}

DEVICE int32_t extract_quarter_fast(const time_t* tim_p) {
  STATIC_QUAL const uint32_t cumulative_quarter_epoch_starts[4] = {
      0, 7776000, 15638400, 23587200};
  STATIC_QUAL const uint32_t cumulative_quarter_epoch_starts_leap_year[4] = {
      0, 7862400, 15724800, 23673600};
  const time_t lcltime = *tim_p;
  uint32_t seconds_1900 = static_cast<int64_t>(lcltime) + EPOCH_OFFSET_YEAR_1900;
  uint32_t leap_years =
      (seconds_1900 - SECONDS_FROM_JAN_1900_TO_MARCH_1900) / SECONDS_PER_4_YEAR_CYCLE;
  uint32_t year =
      (seconds_1900 - leap_years * SECONDS_PER_DAY) / SECONDS_PER_NON_LEAP_YEAR;
  uint32_t base_year_leap_years = (year - 1) / 4;
  uint32_t base_year_seconds =
      year * SECONDS_PER_NON_LEAP_YEAR + base_year_leap_years * SECONDS_PER_DAY;
  bool is_leap_year = year % 4 == 0 && year != 0;
  const uint32_t* quarter_offsets = is_leap_year
                                        ? cumulative_quarter_epoch_starts_leap_year
                                        : cumulative_quarter_epoch_starts;
  uint32_t partial_year_seconds = seconds_1900 % base_year_seconds;
  uint32_t quarter = partial_year_seconds / (90 * SECONDS_PER_DAY);
  quarter = quarter <= 3 ? quarter : 3;
  if (quarter_offsets[quarter] > partial_year_seconds) {
    quarter--;
  }
  return quarter + 1;
}

DEVICE int32_t extract_year_fast(const time_t* tim_p) {
  const time_t lcltime = *tim_p;
  uint32_t seconds_1900 = static_cast<int64_t>(lcltime) + EPOCH_OFFSET_YEAR_1900;
  uint32_t leap_years =
      (seconds_1900 - SECONDS_FROM_JAN_1900_TO_MARCH_1900) / SECONDS_PER_4_YEAR_CYCLE;
  uint32_t year =
      (seconds_1900 - leap_years * SECONDS_PER_DAY) / SECONDS_PER_NON_LEAP_YEAR + 1900;
  return year;
}

DEVICE tm* gmtime_r_newlib(const time_t* tim_p, tm* res) {
  const int32_t month_lengths[2][MONSPERYEAR] = {
      {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31},
      {31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31}};
  int64_t days, rem;
  const time_t lcltime = *tim_p;
  int32_t year, month, yearday, weekday;
  int32_t years400, years100, years4, remainingyears;
  int32_t yearleap;
  const int32_t* ip;

  days = (static_cast<int64_t>(lcltime) / SECSPERDAY) - EPOCH_ADJUSTMENT_DAYS;
  rem = static_cast<int64_t>(lcltime) % SECSPERDAY;
  if (rem < 0) {
    rem += SECSPERDAY;
    --days;
  }

  /* compute hour, min, and sec */
  res->tm_hour = static_cast<int32_t>(rem / SECSPERHOUR);
  rem %= SECSPERHOUR;
  res->tm_min = static_cast<int32_t>(rem / SECSPERMIN);
  res->tm_sec = static_cast<int32_t>(rem % SECSPERMIN);

  /* compute day of week */
  if ((weekday = ((ADJUSTED_EPOCH_WDAY + days) % DAYSPERWEEK)) < 0) {
    weekday += DAYSPERWEEK;
  }
  res->tm_wday = weekday;

  /* compute year & day of year */
  years400 = days / DAYS_PER_400_YEARS;
  days -= years400 * DAYS_PER_400_YEARS;
  /* simplify by making the values positive */
  if (days < 0) {
    days += DAYS_PER_400_YEARS;
    --years400;
  }

  years100 = days / DAYS_PER_100_YEARS;
  if (years100 == 4) { /* required for proper day of year calculation */
    --years100;
  }
  days -= years100 * DAYS_PER_100_YEARS;
  years4 = days / DAYS_PER_4_YEARS;
  days -= years4 * DAYS_PER_4_YEARS;
  remainingyears = days / DAYS_PER_YEAR;
  if (remainingyears == 4) { /* required for proper day of year calculation */
    --remainingyears;
  }
  days -= remainingyears * DAYS_PER_YEAR;

  year =
      ADJUSTED_EPOCH_YEAR + years400 * 400 + years100 * 100 + years4 * 4 + remainingyears;

  /* If remainingyears is zero, it means that the years were completely
   * "consumed" by modulo calculations by 400, 100 and 4, so the year is:
   * 1. a multiple of 4, but not a multiple of 100 or 400 - it's a leap year,
   * 2. a multiple of 4 and 100, but not a multiple of 400 - it's not a leap
   * year,
   * 3. a multiple of 4, 100 and 400 - it's a leap year.
   * If years4 is non-zero, it means that the year is not a multiple of 100 or
   * 400 (case 1), so it's a leap year. If years100 is zero (and years4 is zero
   * - due to short-circuiting), it means that the year is a multiple of 400
   * (case 3), so it's also a leap year. */
  yearleap = remainingyears == 0 && (years4 != 0 || years100 == 0);

  /* adjust back to 1st January */
  yearday = days + DAYS_IN_JANUARY + DAYS_IN_FEBRUARY + yearleap;
  if (yearday >= DAYS_PER_YEAR + yearleap) {
    yearday -= DAYS_PER_YEAR + yearleap;
    ++year;
  }
  res->tm_yday = yearday;
  res->tm_year = year - YEAR_BASE;

  /* Because "days" is the number of days since 1st March, the additional leap
   * day (29th of February) is the last possible day, so it doesn't matter much
   * whether the year is actually leap or not. */
  ip = month_lengths[1];
  month = 2;
  while (days >= ip[month]) {
    days -= ip[month];
    if (++month >= MONSPERYEAR) {
      month = 0;
    }
  }
  res->tm_mon = month;
  res->tm_mday = days + 1;

  res->tm_isdst = 0;

  return (res);
}

/*
 * @brief support the SQL EXTRACT function
 */
extern "C" NEVER_INLINE DEVICE int64_t ExtractFromTime(ExtractField field,
                                                       time_t timeval) {
  // We have fast paths for the 5 fields below - do not need to do full gmtime
  switch (field) {
    case kEPOCH:
      return timeval;
    case kQUARTERDAY:
      return extract_quarterday(&timeval);
    case kHOUR:
      return extract_hour(&timeval);
    case kMINUTE:
      return extract_minute(&timeval);
    case kSECOND:
      return extract_second(&timeval);
    case kMILLISECOND:
    case kMICROSECOND:
    case kNANOSECOND:
      return 0;
    case kDOW:
      return extract_dow(&timeval);
    case kISODOW: {
      int64_t dow = extract_dow(&timeval);
      return (dow == 0 ? 7 : dow);
    }
    case kMONTH: {
      if (timeval >= 0L && timeval <= UINT32_MAX - EPOCH_OFFSET_YEAR_1900) {
        return extract_month_fast(&timeval);
      }
      break;
    }
    case kQUARTER: {
      if (timeval >= 0L && timeval <= UINT32_MAX - EPOCH_OFFSET_YEAR_1900) {
        return extract_quarter_fast(&timeval);
      }
      break;
    }
    case kYEAR: {
      if (timeval >= 0L && timeval <= UINT32_MAX - EPOCH_OFFSET_YEAR_1900) {
        return extract_year_fast(&timeval);
      }
      break;
    }
    default:
      break;
  }

  tm tm_struct;
  gmtime_r_newlib(&timeval, &tm_struct);
  switch (field) {
    case kYEAR:
      return 1900 + tm_struct.tm_year;
    case kQUARTER:
      return (tm_struct.tm_mon) / 3 + 1;
    case kMONTH:
      return tm_struct.tm_mon + 1;
    case kDAY:
      return tm_struct.tm_mday;
    case kDOY:
      return tm_struct.tm_yday + 1;
    case kWEEK: {
      int32_t doy = tm_struct.tm_yday;          // numbered from 0
      int32_t dow = extract_dow(&timeval) + 1;  // use Sunday 1 - Saturday 7
      int32_t week = (doy / 7) + 1;
      // now adjust for offset at start of year
      //      S M T W T F S
      // doy      0 1 2 3 4
      // doy  5 6
      // mod  5 6 0 1 2 3 4
      // dow  1 2 3 4 5 6 7
      // week 2 2 1 1 1 1 1
      if (dow > (doy % 7)) {
        return week;
      }
      return week + 1;
    }
    default:
#ifdef __CUDACC__
      return -1;
#else
      abort();
#endif
  }
}

extern "C" DEVICE int64_t ExtractFromTimeHighPrecision(ExtractField field,
                                                       time_t timeval,
                                                       int64_t scale) {
  switch (field) {
    case kMILLISECOND: {
      time_t mtime = timeval;
      if (scale == MICROSECSPERSEC) {
        mtime = static_cast<int64_t>(timeval) / MILLISECSPERSEC;
      } else if (scale == NANOSECSPERSEC) {
        mtime = static_cast<int64_t>(timeval) / MICROSECSPERSEC;
      }
      return extract_millisecond(&mtime);
    }
    case kMICROSECOND: {
      time_t mtime = timeval;
      if (scale == NANOSECSPERSEC) {
        mtime = static_cast<int64_t>(timeval) / MILLISECSPERSEC;
      } else if (scale == MILLISECSPERSEC) {
        return 0;
      }
      return extract_microsecond(&mtime);
    }
    case kNANOSECOND: {
      if (scale == MILLISECSPERSEC || scale == MICROSECSPERSEC) {
        return 0;
      } else {
        return extract_nanosecond(&timeval);
      }
    }
    default:
      break;
  }
  const time_t stimeval = static_cast<int64_t>(timeval) / scale;
  return ExtractFromTime(field, stimeval);
}

extern "C" DEVICE int64_t ExtractFromTimeNullable(ExtractField field,
                                                  time_t timeval,
                                                  const int64_t null_val) {
  if (timeval == null_val) {
    return null_val;
  }
  return ExtractFromTime(field, timeval);
}

extern "C" DEVICE int64_t ExtractFromTimeHighPrecisionNullable(ExtractField field,
                                                               time_t timeval,
                                                               int64_t scale,
                                                               const int64_t null_val) {
  if (timeval == null_val) {
    return null_val;
  }
  return ExtractFromTimeHighPrecision(field, timeval, scale);
}
