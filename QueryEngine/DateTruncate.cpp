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
#include <limits>

extern "C" ALWAYS_INLINE DEVICE int64_t datetrunc_minute(int64_t timeval) {
  return timeval - unsigned_mod(timeval, kSecsPerMin);
}

extern "C" ALWAYS_INLINE DEVICE int64_t datetrunc_hour(int64_t timeval) {
  return timeval - unsigned_mod(timeval, kSecsPerHour);
}

extern "C" ALWAYS_INLINE DEVICE int64_t datetrunc_quarterday(int64_t timeval) {
  return timeval - unsigned_mod(timeval, kSecsPerQuarterDay);
}

extern "C" ALWAYS_INLINE DEVICE int64_t datetrunc_day(int64_t timeval) {
  return timeval - unsigned_mod(timeval, kSecsPerDay);
}

extern "C" ALWAYS_INLINE DEVICE int64_t datetrunc_week(int64_t timeval) {
  // Truncate to Monday. 1 Jan 1970 is a Thursday (+3*kSecsPerDay).
  return timeval - unsigned_mod(timeval + 3 * kSecsPerDay, 7 * kSecsPerDay);
}

extern "C" ALWAYS_INLINE DEVICE int64_t datetrunc_month(int64_t timeval) {
  if (timeval >= 0L && timeval <= UINT32_MAX - (kEpochOffsetYear1900)) {
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
    uint32_t month =
        seconds_past_march / (30 * kUSecsPerDay);  // Will make the correct month either
                                                   // be the guessed month or month before
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
    unsigned const moy = (5 * doy + 2) / 153;
    unsigned const dom = doy - (153 * moy + 2) / 5;
    return (day - dom) * kSecsPerDay;
  }
}

extern "C" ALWAYS_INLINE DEVICE int64_t datetrunc_quarter(int64_t timeval) {
  if (timeval >= 0L && timeval <= UINT32_MAX - kEpochOffsetYear1900) {
    STATIC_QUAL const uint32_t cumulative_quarter_epoch_starts[4] = {
        0, 7776000, 15638400, 23587200};
    STATIC_QUAL const uint32_t cumulative_quarter_epoch_starts_leap_year[4] = {
        0, 7862400, 15724800, 23673600};
    // Handles times from Thu 01 Jan 1970 00:00:00 - Thu 07 Feb 2036 06:28:15.
    uint32_t seconds_1900 = timeval + kEpochOffsetYear1900;
    uint32_t leap_years = (seconds_1900 - kSecsJanToMar1900) / kSecondsPer4YearCycle;
    uint32_t year = (seconds_1900 - leap_years * kUSecsPerDay) / kSecondsPerNonLeapYear;
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
      doq = JANMAR + leap + doy;  // Q1
    } else {
      unsigned const q = (3 * (doy - apr1) + 2) / 275;  // quarter = 0, 1, 2
      doq = doy - (apr1 + q * 92 - (q != 0));           // Q2, Q3, Q4
    }
    return (day - doq) * kSecsPerDay;
  }
}

extern "C" ALWAYS_INLINE DEVICE int64_t datetrunc_year(int64_t timeval) {
  if (timeval >= 0L && timeval <= UINT32_MAX - kEpochOffsetYear1900) {
    // Handles times from Thu 01 Jan 1970 00:00:00 - Thu 07 Feb 2036 06:28:15.
    uint32_t seconds_1900 = static_cast<uint32_t>(timeval) + kEpochOffsetYear1900;
    uint32_t leap_years = (seconds_1900 - kSecsJanToMar1900) / kSecondsPer4YearCycle;
    uint32_t year = (seconds_1900 - leap_years * kUSecsPerDay) / kSecondsPerNonLeapYear;
    uint32_t base_year_leap_years = (year - 1) / 4;
    return (static_cast<int64_t>(year) * kSecondsPerNonLeapYear +
            base_year_leap_years * kUSecsPerDay - kEpochOffsetYear1900);
  } else {
    int64_t const day = floor_div(timeval, kSecsPerDay);
    unsigned const doe = unsigned_mod(day - kEpochAdjustedDays, kDaysPer400Years);
    unsigned const yoe = (doe - doe / 1460 + doe / 36524 - (doe == 146096)) / 365;
    unsigned const doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    unsigned docy;  // Day-of-calendar-year = Days since last Jan1.
    if (doy < MARJAN) {
      bool const leap = yoe % 4 == 0 && (yoe == 0 || yoe % 100 != 0);
      docy = JANMAR + leap + doy;
    } else {
      docy = doy - MARJAN;
    }
    return (day - docy) * kSecsPerDay;
  }
}

extern "C" ALWAYS_INLINE DEVICE int64_t datetrunc_decade(int64_t timeval) {
  // Number of days from x00301 to (x+1)00101. Always includes exactly two leap days.
  constexpr unsigned decmarjan = MARJAN + 9 * 365 + 2;
  int64_t const day = floor_div(timeval, kSecsPerDay);
  unsigned const doe = unsigned_mod(day - kEpochAdjustedDays, kDaysPer400Years);
  unsigned const yoe = (doe - doe / 1460 + doe / 36524 - (doe == 146096)) / 365;
  unsigned const decoe = yoe - yoe % 10;  // Decade-of-era
  // Initialize to days after mar1 of decade, then adjust to after jan1 below.
  unsigned days_after_decade = doe - (365 * decoe + decoe / 4 - decoe / 100);
  if (days_after_decade < decmarjan) {
    bool const leap = decoe % 4 == 0 && (decoe == 0 || decoe % 100 != 0);
    days_after_decade += JANMAR + leap;
  } else {
    days_after_decade -= decmarjan;
  }
  return (day - days_after_decade) * kSecsPerDay;
}

extern "C" ALWAYS_INLINE DEVICE int64_t datetrunc_century(int64_t timeval) {
  int64_t const day = floor_div(timeval, kSecsPerDay);
  unsigned const doe = unsigned_mod(day - kEpochAdjustedDays, kDaysPer400Years);
  // Day-of-century = Days since last 010101 (Jan 1 1901, 2001, 2101, etc.)
  unsigned const doc = doe < MARJAN ? doe + (36525 - MARJAN) : (doe - MARJAN) % 36524;
  return (day - doc) * kSecsPerDay;
}

extern "C" ALWAYS_INLINE DEVICE int64_t datetrunc_millennium(int64_t timeval) {
  constexpr unsigned millennium2001 = 365242;  // Days from Jan 1 2001 to 3001.
  int64_t const day = floor_div(timeval, kSecsPerDay);
  // lcm(400, 1000) = 2000 so use 5*400-year eras at a time.
  unsigned dom = unsigned_mod(day - kEpochAdjustedDays, 5 * kDaysPer400Years);
  if (dom < MARJAN) {
    dom += millennium2001 + 1 - MARJAN;
  } else if (dom < MARJAN + millennium2001) {
    dom -= MARJAN;
  } else {
    dom -= MARJAN + millennium2001;
  }
  return (day - dom) * kSecsPerDay;
}

/*
 * @brief support the SQL DATE_TRUNC function
 */
int64_t DateTruncate(DatetruncField field, const int64_t timeval) {
  switch (field) {
    case dtNANOSECOND:
    case dtMICROSECOND:
    case dtMILLISECOND:
    case dtSECOND:
      return timeval;
    case dtMINUTE:
      return datetrunc_minute(timeval);
    case dtHOUR:
      return datetrunc_hour(timeval);
    case dtQUARTERDAY:
      return datetrunc_quarterday(timeval);
    case dtDAY:
      return datetrunc_day(timeval);
    case dtWEEK:
      return datetrunc_week(timeval);
    case dtMONTH:
      return datetrunc_month(timeval);
    case dtQUARTER:
      return datetrunc_quarter(timeval);
    case dtYEAR:
      return datetrunc_year(timeval);
    case dtDECADE:
      return datetrunc_decade(timeval);
    case dtCENTURY:
      return datetrunc_century(timeval);
    case dtMILLENNIUM:
      return datetrunc_millennium(timeval);
    default:
#ifdef __CUDACC__
      return std::numeric_limits<int64_t>::min();
#else
      abort();
#endif
  }
}

// scale is 10^{3,6,9}
extern "C" ALWAYS_INLINE DEVICE int64_t
DateTruncateHighPrecisionToDate(const int64_t timeval, const int64_t scale) {
  return floor_div(timeval, scale * kSecsPerDay) * kSecsPerDay;
}

extern "C" ALWAYS_INLINE DEVICE int64_t
DateTruncateHighPrecisionToDateNullable(const int64_t timeval,
                                        const int64_t scale,
                                        const int64_t null_val) {
  if (timeval == null_val) {
    return null_val;
  }
  return DateTruncateHighPrecisionToDate(timeval, scale);
}

namespace {

struct EraTime {
  int64_t const era;
  int const yoe;  // year-of-era
  int const moy;  // month-of-year (March = 0)
  int const dom;  // day-of-month
  int const sod;  // second-of-day

  DEVICE static EraTime make(int64_t const time) {
    int64_t const day = floor_div(time, kSecsPerDay);
    int64_t const era = floor_div(day - kEpochAdjustedDays, kDaysPer400Years);
    int const sod = time - day * kSecsPerDay;
    int const doe = day - kEpochAdjustedDays - era * kDaysPer400Years;
    int const yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    int const doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    int const moy = (5 * doy + 2) / 153;
    int const dom = doy - (153 * moy + 2) / 5;
    return {era, yoe, moy, dom, sod};
  }

  DEVICE EraTime operator-() const { return {-era, -yoe, -moy, -dom, -sod}; }

  DEVICE EraTime operator-(EraTime const& rhs) {
    return {era - rhs.era, yoe - rhs.yoe, moy - rhs.moy, dom - rhs.dom, sod - rhs.sod};
  }

  enum Field { ERA, YOE, MOY, DOM, SOD };
  // Sign of EraTime starting at field.
  DEVICE int sign(Field const field) const {
    switch (field) {
      case ERA:
        if (era != 0) {
          return era < 0 ? -1 : 1;
        }
      case YOE:
        if (yoe != 0) {
          return yoe < 0 ? -1 : 1;
        }
      case MOY:
        if (moy != 0) {
          return moy < 0 ? -1 : 1;
        }
      case DOM:
        if (dom != 0) {
          return dom < 0 ? -1 : 1;
        }
      case SOD:
        if (sod != 0) {
          return sod < 0 ? -1 : 1;
        }
      default:
        return 0;
    }
  }

  DEVICE int64_t count(DatetruncField const field) const {
    int const sgn = sign(ERA);
    EraTime const ut = sgn == -1 ? -*this : *this;  // Unsigned time
    switch (field) {
      case dtMONTH:
        return sgn * (12 * (400 * ut.era + ut.yoe) + ut.moy - (ut.sign(DOM) == -1));
      case dtQUARTER: {
        int const quarters = ut.moy / 3;
        int const rem = ut.moy % 3;
        return sgn * (4 * (400 * ut.era + ut.yoe) + quarters -
                      (rem < 0 || (rem == 0 && ut.sign(DOM) == -1)));
      }
      case dtYEAR:
        return sgn * (400 * ut.era + ut.yoe - (ut.sign(MOY) == -1));
      case dtDECADE: {
        uint64_t const decades = (400 * ut.era + ut.yoe) / 10;
        unsigned const rem = (400 * ut.era + ut.yoe) % 10;
        return sgn * (decades - (rem == 0 && ut.sign(MOY) == -1));
      }
      case dtCENTURY: {
        uint64_t const centuries = (400 * ut.era + ut.yoe) / 100;
        unsigned const rem = (400 * ut.era + ut.yoe) % 100;
        return sgn * (centuries - (rem == 0 && ut.sign(MOY) == -1));
      }
      case dtMILLENNIUM: {
        uint64_t const millennia = (400 * ut.era + ut.yoe) / 1000;
        unsigned const rem = (400 * ut.era + ut.yoe) % 1000;
        return sgn * (millennia - (rem == 0 && ut.sign(MOY) == -1));
      }
      default:
#ifdef __CUDACC__
        return std::numeric_limits<int64_t>::min();
#else
        abort();
#endif
    }
  }
};

}  // namespace

extern "C" DEVICE int64_t DateDiff(const DatetruncField datepart,
                                   const int64_t startdate,
                                   const int64_t enddate) {
  switch (datepart) {
    case dtNANOSECOND:
      return (enddate - startdate) * kNanoSecsPerSec;
    case dtMICROSECOND:
      return (enddate - startdate) * kMicroSecsPerSec;
    case dtMILLISECOND:
      return (enddate - startdate) * kMilliSecsPerSec;
    case dtSECOND:
      return enddate - startdate;
    case dtMINUTE:
      return (enddate - startdate) / kSecsPerMin;
    case dtHOUR:
      return (enddate - startdate) / kSecsPerHour;
    case dtQUARTERDAY:
      return (enddate - startdate) / (kSecsPerDay / 4);
    case dtDAY:
      return (enddate - startdate) / kSecsPerDay;
    case dtWEEK:
      return (enddate - startdate) / (7 * kSecsPerDay);
    default:
      return (EraTime::make(enddate) - EraTime::make(startdate)).count(datepart);
  }
}

extern "C" DEVICE int64_t DateDiffHighPrecision(const DatetruncField datepart,
                                                const int64_t startdate,
                                                const int64_t enddate,
                                                const int32_t start_dim,
                                                const int32_t end_dim) {
  // Return pow(10,i). Only valid for i = 0, 3, 6, 9.
  constexpr int pow10[10]{1, 0, 0, 1000, 0, 0, 1000 * 1000, 0, 0, 1000 * 1000 * 1000};
  switch (datepart) {
    case dtNANOSECOND:
    case dtMICROSECOND:
    case dtMILLISECOND: {
      static_assert(dtMILLISECOND + 1 == dtMICROSECOND, "Please keep these consecutive.");
      static_assert(dtMICROSECOND + 1 == dtNANOSECOND, "Please keep these consecutive.");
      int const target_dim = (datepart - (dtMILLISECOND - 1)) * 3;  // 3, 6, or 9.
      int const delta_dim = end_dim - start_dim;  // in [-9,9] multiple of 3
      int const adj_dim = target_dim - (delta_dim < 0 ? start_dim : end_dim);
      int64_t const numerator = delta_dim < 0 ? enddate * pow10[-delta_dim] - startdate
                                              : enddate - startdate * pow10[delta_dim];
      return adj_dim < 0 ? numerator / pow10[-adj_dim] : numerator * pow10[adj_dim];
    }
    default:
      int64_t const end_seconds = floor_div(enddate, pow10[end_dim]);
      int delta_ns = (enddate - end_seconds * pow10[end_dim]) * pow10[9 - end_dim];
      int64_t const start_seconds = floor_div(startdate, pow10[start_dim]);
      delta_ns -= (startdate - start_seconds * pow10[start_dim]) * pow10[9 - start_dim];
      int64_t const delta_s = end_seconds - start_seconds;
      // sub-second values must be accounted for when calling DateDiff. Examples:
      // 2000-02-15 12:00:00.006 to 2000-03-15 12:00:00.005 is 0 months.
      // 2000-02-15 12:00:00.006 to 2000-03-15 12:00:00.006 is 1 month.
      int const adj_sec =
          0 < delta_s && delta_ns < 0 ? -1 : delta_s < 0 && 0 < delta_ns ? 1 : 0;
      return DateDiff(datepart, start_seconds, end_seconds + adj_sec);
  }
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
                                                        const int32_t start_dim,
                                                        const int32_t end_dim,
                                                        const int64_t null_val) {
  if (startdate == null_val || enddate == null_val) {
    return null_val;
  }
  return DateDiffHighPrecision(datepart, startdate, enddate, start_dim, end_dim);
}
