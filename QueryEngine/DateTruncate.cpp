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

#include "DateTruncate.h"
#include "ExtractFromTime.h"

#ifndef __CUDACC__
#include <cstdlib>  // abort()
#endif

#include <cmath>
#include <ctime>
#include <iostream>

/*
 * @brief support the SQL DATE_TRUNC function
 */
extern "C" NEVER_INLINE DEVICE int64_t DateTruncate(DatetruncField field,
                                                    const int64_t timeval) {
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
  STATIC_QUAL const uint32_t cumulative_quarter_epoch_starts[4] = {
      0, 7776000, 15638400, 23587200};
  STATIC_QUAL const uint32_t cumulative_quarter_epoch_starts_leap_year[4] = {
      0, 7862400, 15724800, 23673600};
  // Number of days from March 1 to Jan 1.
  constexpr unsigned marjan = 31 + 30 + 31 + 30 + 31 + 31 + 30 + 31 + 30 + 31;
  constexpr unsigned janmar = 31 + 28;  // leap day handled separately
  switch (field) {
    case dtNANOSECOND:
    case dtMICROSECOND:
    case dtMILLISECOND:
    case dtSECOND:
      return timeval;
    case dtMINUTE:
      return timeval - unsigned_mod(timeval, kSecsPerMin);
    case dtHOUR:
      return timeval - unsigned_mod(timeval, kSecsPerHour);
    case dtQUARTERDAY:
      return timeval - unsigned_mod(timeval, kSecsPerQuarterDay);
    case dtDAY:
      return timeval - unsigned_mod(timeval, kSecsPerDay);
    case dtWEEK:
      // Truncate to Monday. 1 Jan 1970 is a Thursday (+3*kSecsPerDay).
      return timeval - unsigned_mod(timeval + 3 * kSecsPerDay, 7 * kSecsPerDay);
    case dtMONTH: {
      if (timeval >= 0L && timeval <= UINT32_MAX - (kEpochOffsetYear1900)) {
        // Handles times from Thu 01 Jan 1970 00:00:00 - Thu 07 Feb 2036 06:28:15.
        uint32_t seconds_march_1900 = timeval + kEpochOffsetYear1900 - kSecsJanToMar1900;
        uint32_t seconds_past_4year_period = seconds_march_1900 % kSecondsPer4YearCycle;
        uint32_t four_year_period_seconds =
            (seconds_march_1900 / kSecondsPer4YearCycle) * kSecondsPer4YearCycle;
        uint32_t year_seconds_past_4year_period =
            (seconds_past_4year_period / kSecondsPerNonLeapYear) * kSecondsPerNonLeapYear;
        if (seconds_past_4year_period >=
            kSecondsPer4YearCycle - kUSecsPerDay) {  // if we are in Feb 29th
          year_seconds_past_4year_period -= kSecondsPerNonLeapYear;
        }
        uint32_t seconds_past_march =
            seconds_past_4year_period - year_seconds_past_4year_period;
        uint32_t month = seconds_past_march /
                         (30 * kUSecsPerDay);  // Will make the correct month either be
                                               // the guessed month or month before
        month = month <= 11 ? month : 11;
        if (cumulative_month_epoch_starts[month] > seconds_past_march) {
          month--;
        }
        return (static_cast<int64_t>(four_year_period_seconds) +
                year_seconds_past_4year_period + cumulative_month_epoch_starts[month] -
                kEpochOffsetYear1900 + kSecsJanToMar1900);
      } else {
        int64_t const day = floor_div(timeval, kSecsPerDay);
        unsigned const doe = unsigned_mod(day - kEpochAdjustedDays, kDaysPer400Years);
        unsigned const yoe = (doe - doe / 1460 + doe / 36524 - (doe == 146096)) / 365;
        unsigned const doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        unsigned const moe = (5 * doy + 2) / 153;
        unsigned const dom = doy - (153 * moe + 2) / 5;
        return (day - dom) * kSecsPerDay;
      }
    }
    case dtQUARTER: {
      if (timeval >= 0L && timeval <= UINT32_MAX - kEpochOffsetYear1900) {
        // Handles times from Thu 01 Jan 1970 00:00:00 - Thu 07 Feb 2036 06:28:15.
        uint32_t seconds_1900 = timeval + kEpochOffsetYear1900;
        uint32_t leap_years = (seconds_1900 - kSecsJanToMar1900) / kSecondsPer4YearCycle;
        uint32_t year =
            (seconds_1900 - leap_years * kUSecsPerDay) / kSecondsPerNonLeapYear;
        uint32_t base_year_leap_years = (year - 1) / 4;
        uint32_t base_year_seconds =
            year * kSecondsPerNonLeapYear + base_year_leap_years * kUSecsPerDay;
        const bool is_leap_year = year % 4 == 0 && year != 0;
        const uint32_t* quarter_offsets = is_leap_year
                                              ? cumulative_quarter_epoch_starts_leap_year
                                              : cumulative_quarter_epoch_starts;
        uint32_t partial_year_seconds = seconds_1900 % base_year_seconds;
        uint32_t quarter = partial_year_seconds / (90 * kUSecsPerDay);
        quarter = quarter <= 3 ? quarter : 3;
        if (quarter_offsets[quarter] > partial_year_seconds) {
          quarter--;
        }
        return (static_cast<int64_t>(base_year_seconds) + quarter_offsets[quarter] -
                kEpochOffsetYear1900);
      } else {
        int64_t const day = floor_div(timeval, kSecsPerDay);
        unsigned const doe = unsigned_mod(day - kEpochAdjustedDays, kDaysPer400Years);
        unsigned const yoe = (doe - doe / 1460 + doe / 36524 - (doe == 146096)) / 365;
        unsigned const doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        constexpr unsigned apr1 = 31;  // Days in march
        unsigned doq;  // Day-of-quarter = Days since last Apr1, Jul1, Oct1, Jan1.
        if (doy < apr1) {
          bool const leap = yoe % 4 == 0 && (yoe % 100 != 0 || yoe == 0);
          doq = janmar + leap + doy;  // Q1
        } else {
          unsigned const q = (3 * (doy - apr1) + 2) / 275;  // quarter = 0, 1, 2
          doq = doy - (apr1 + q * 92 - (q != 0));           // Q2, Q3, Q4
        }
        return (day - doq) * kSecsPerDay;
      }
    }
    case dtYEAR: {
      if (timeval >= 0L && timeval <= UINT32_MAX - kEpochOffsetYear1900) {
        // Handles times from Thu 01 Jan 1970 00:00:00 - Thu 07 Feb 2036 06:28:15.
        uint32_t seconds_1900 = static_cast<uint32_t>(timeval) + kEpochOffsetYear1900;
        uint32_t leap_years = (seconds_1900 - kSecsJanToMar1900) / kSecondsPer4YearCycle;
        uint32_t year =
            (seconds_1900 - leap_years * kUSecsPerDay) / kSecondsPerNonLeapYear;
        uint32_t base_year_leap_years = (year - 1) / 4;
        return (static_cast<int64_t>(year) * kSecondsPerNonLeapYear +
                base_year_leap_years * kUSecsPerDay - kEpochOffsetYear1900);
      } else {
        int64_t const day = floor_div(timeval, kSecsPerDay);
        unsigned const doe = unsigned_mod(day - kEpochAdjustedDays, kDaysPer400Years);
        unsigned const yoe = (doe - doe / 1460 + doe / 36524 - (doe == 146096)) / 365;
        unsigned const doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        unsigned docy;  // Day-of-calendar-year = Days since last Jan1.
        if (doy < marjan) {
          bool const leap = yoe % 4 == 0 && (yoe == 0 || yoe % 100 != 0);
          docy = janmar + leap + doy;
        } else {
          docy = doy - marjan;
        }
        return (day - docy) * kSecsPerDay;
      }
    }
    case dtDECADE: {
      // Number of days from x00301 to (x+1)00101. Always includes exactly two leap days.
      constexpr unsigned decmarjan = marjan + 9 * 365 + 2;
      int64_t const day = floor_div(timeval, kSecsPerDay);
      unsigned const doe = unsigned_mod(day - kEpochAdjustedDays, kDaysPer400Years);
      unsigned const yoe = (doe - doe / 1460 + doe / 36524 - (doe == 146096)) / 365;
      unsigned const decoe = yoe - yoe % 10;  // Decade-of-era
      // Initialize to days after mar1 of decade, then adjust to after jan1 below.
      unsigned days_after_decade = doe - (365 * decoe + decoe / 4 - decoe / 100);
      if (days_after_decade < decmarjan) {
        bool const leap = decoe % 4 == 0 && (decoe == 0 || decoe % 100 != 0);
        days_after_decade += janmar + leap;
      } else {
        days_after_decade -= decmarjan;
      }
      return (day - days_after_decade) * kSecsPerDay;
    }
    case dtCENTURY: {
      int64_t const day = floor_div(timeval, kSecsPerDay);
      unsigned const doe = unsigned_mod(day - kEpochAdjustedDays, kDaysPer400Years);
      // Day-of-century = Days since last 010101 (Jan 1 1901, 2001, 2101, etc.)
      unsigned const doc = doe < marjan ? doe + (36525 - marjan) : (doe - marjan) % 36524;
      return (day - doc) * kSecsPerDay;
    }
    case dtMILLENNIUM: {
      constexpr unsigned millennium2001 = 365242;  // Days from Jan 1 2001 to 3001.
      int64_t const day = floor_div(timeval, kSecsPerDay);
      // lcm(400, 1000) = 2000 so use 5*400-year eras at a time.
      unsigned dom = unsigned_mod(day - kEpochAdjustedDays, 5 * kDaysPer400Years);
      if (dom < marjan) {
        dom += millennium2001 + 1 - marjan;
      } else if (dom < marjan + millennium2001) {
        dom -= marjan;
      } else {
        dom -= marjan + millennium2001;
      }
      return (day - dom) * kSecsPerDay;
    }
    default:
#ifdef __CUDACC__
      return -1;
#else
      abort();
#endif
  }
}

extern "C" DEVICE int64_t DateTruncateNullable(DatetruncField field,
                                               const int64_t timeval,
                                               const int64_t null_val) {
  if (timeval == null_val) {
    return null_val;
  }
  return DateTruncate(field, timeval);
}

// scale is 10^{3,6,9}
extern "C" DEVICE int64_t DateTruncateHighPrecisionToDate(const int64_t timeval,
                                                          const int64_t scale) {
  return floor_div(timeval, scale * kSecsPerDay) * kSecsPerDay;
}

extern "C" DEVICE int64_t
DateTruncateHighPrecisionToDateNullable(const int64_t timeval,
                                        const int64_t scale,
                                        const int64_t null_val) {
  if (timeval == null_val) {
    return null_val;
  }
  return DateTruncateHighPrecisionToDate(timeval, scale);
}

extern "C" DEVICE int64_t DateDiff(const DatetruncField datepart,
                                   const int64_t startdate,
                                   const int64_t enddate) {
  int64_t res = enddate - startdate;
  switch (datepart) {
    case dtNANOSECOND:
      return res * kNanoSecsPerSec;
    case dtMICROSECOND:
      return res * kMicroSecsPerSec;
    case dtMILLISECOND:
      return res * kMilliSecsPerSec;
    case dtSECOND:
      return res;
    case dtMINUTE:
      return res / kSecsPerMin;
    case dtHOUR:
      return res / kSecsPerHour;
    case dtQUARTERDAY:
      return res / kSecsPerQuarterDay;
    case dtDAY:
      return res / kSecsPerDay;
    case dtWEEK:
      return res / (kSecsPerDay * kDaysPerWeek);
    default:
      break;
  }

  auto future_date = (res > 0);
  auto end = future_date ? enddate : startdate;
  auto start = future_date ? startdate : enddate;
  res = 0;
  int64_t crt = end;
  while (crt > start) {
    const int64_t dt = DateTruncate(datepart, crt);
    if (dt <= start) {
      break;
    }
    ++res;
    crt = dt - 1;
  }
  return future_date ? res : -res;
}

extern "C" DEVICE int64_t DateDiffHighPrecision(const DatetruncField datepart,
                                                const int64_t startdate,
                                                const int64_t enddate,
                                                const int32_t adj_dimen,
                                                const int64_t adj_scale,
                                                const int64_t sml_scale,
                                                const int64_t scale) {
  /* TODO(wamsi): When adj_dimen is 1 i.e. both precisions are same,
     this code is really not required. We cam direcly do enddate-startdate here.
     Need to address this in refactoring focussed subsequent PR.*/
  int64_t res = (adj_dimen > 0) ? (enddate - (startdate * adj_scale))
                                : ((enddate * adj_scale) - startdate);
  switch (datepart) {
    case dtNANOSECOND:
      // limit of current granularity
      return res;
    case dtMICROSECOND: {
      if (scale == kNanoSecsPerSec) {
        return res / kMilliSecsPerSec;
      } else {
        { return res; }
      }
    }
    case dtMILLISECOND: {
      if (scale == kNanoSecsPerSec) {
        return res / kMicroSecsPerSec;
      } else if (scale == kMicroSecsPerSec) {
        return res / kMilliSecsPerSec;
      } else {
        { return res; }
      }
    }
    default:
      break;
  }
  const int64_t nstartdate = adj_dimen > 0 ? startdate / sml_scale : startdate / scale;
  const int64_t nenddate = adj_dimen < 0 ? enddate / sml_scale : enddate / scale;
  return DateDiff(datepart, nstartdate, nenddate);
}

extern "C" DEVICE int64_t DateDiffNullable(const DatetruncField datepart,
                                           const int64_t startdate,
                                           const int64_t enddate,
                                           const int64_t null_val) {
  if (startdate == null_val || enddate == null_val) {
    return null_val;
  }
  return DateDiff(datepart, startdate, enddate);
}

extern "C" DEVICE int64_t DateDiffHighPrecisionNullable(const DatetruncField datepart,
                                                        const int64_t startdate,
                                                        const int64_t enddate,
                                                        const int32_t adj_dimen,
                                                        const int64_t adj_scale,
                                                        const int64_t sml_scale,
                                                        const int64_t scale,
                                                        const int64_t null_val) {
  if (startdate == null_val || enddate == null_val) {
    return null_val;
  }
  return DateDiffHighPrecision(
      datepart, startdate, enddate, adj_dimen, adj_scale, sml_scale, scale);
}
