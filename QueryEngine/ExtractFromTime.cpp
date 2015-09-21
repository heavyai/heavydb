#include "ExtractFromTime.h"

#ifndef __CUDACC__
#include <glog/logging.h>
#endif

#define SECSPERMIN 60L
#define MINSPERHOUR 60L
#define HOURSPERDAY 24L
#define SECSPERHOUR (SECSPERMIN * MINSPERHOUR)
#define SECSPERDAY (SECSPERHOUR * HOURSPERDAY)
#define DAYSPERWEEK 7
#define MONSPERYEAR 12

#define YEAR_BASE 1900

/* move epoch from 01.01.1970 to 01.03.2000 - this is the first day of new
 * 400-year long cycle, right after additional day of leap year. This adjustment
 * is required only for date calculation, so instead of modifying time_t value
 * (which would require 64-bit operations to work correctly) it's enough to
 * adjust the calculated number of days since epoch. */
#define EPOCH_ADJUSTMENT_DAYS 11017
/* year to which the adjustment was made */
#define ADJUSTED_EPOCH_YEAR 2000
/* 1st March of 2000 is Wednesday */
#define ADJUSTED_EPOCH_WDAY 3
/* there are 97 leap years in 400-year periods. ((400 - 97) * 365 + 97 * 366) */
#define DAYS_PER_400_YEARS 146097L
/* there are 24 leap years in 100-year periods. ((100 - 24) * 365 + 24 * 366) */
#define DAYS_PER_100_YEARS 36524L
/* there is one leap year every 4 years */
#define DAYS_PER_4_YEARS (3 * 365 + 366)
/* number of days in a non-leap year */
#define DAYS_PER_YEAR 365
/* number of days in January */
#define DAYS_IN_JANUARY 31
/* number of days in non-leap February */
#define DAYS_IN_FEBRUARY 28

extern "C" __attribute__((noinline))

#ifdef __CUDACC__
__device__
#endif
  int extract_hour(const time_t* tim_p) {
    long days, rem;
    const time_t lcltime = *tim_p;
    days = ((long)lcltime) / SECSPERDAY - EPOCH_ADJUSTMENT_DAYS;
    rem = ((long)lcltime) % SECSPERDAY;
    if (rem < 0) {
      rem += SECSPERDAY;
      --days;
    }
    return (int)(rem / SECSPERHOUR);
  }

#ifdef __CUDACC__
  __device__
#endif
  int extract_minute(const time_t* tim_p) {
    long days, rem;
    const time_t lcltime = *tim_p;
    days = ((long)lcltime) / SECSPERDAY - EPOCH_ADJUSTMENT_DAYS;
    rem = ((long)lcltime) % SECSPERDAY;
    if (rem < 0) {
      rem += SECSPERDAY;
      --days;
    }
    rem %= SECSPERHOUR;
    return (int)(rem / SECSPERMIN);
  }

#ifdef __CUDACC__
  __device__
#endif
  int extract_second(const time_t* tim_p) {
    const time_t lcltime = *tim_p;
    return (int) ((long)lcltime % SECSPERMIN);
  }

#ifdef __CUDACC__
  __device__
#endif
  int extract_dow(const time_t* tim_p) {
    long days, rem;
    int weekday;
    const time_t lcltime = *tim_p;
    days = ((long)lcltime) / SECSPERDAY - EPOCH_ADJUSTMENT_DAYS;
    rem = ((long)lcltime) % SECSPERDAY;
    if (rem < 0) {
      rem += SECSPERDAY;
      --days;
    }

    if ((weekday = ((ADJUSTED_EPOCH_WDAY + days) % DAYSPERWEEK)) < 0)
      weekday += DAYSPERWEEK;
    return weekday;
  }



#ifdef __CUDACC__
__device__
#endif
  tm* gmtime_r_newlib(const time_t* tim_p, tm* res) {
  const int month_lengths[2][MONSPERYEAR] = {{31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31},
                                             {31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31}};
  long days, rem;
  const time_t lcltime = *tim_p;
  int year, month, yearday, weekday;
  int years400, years100, years4, remainingyears;
  int yearleap;
  const int* ip;

  days = ((long)lcltime) / SECSPERDAY - EPOCH_ADJUSTMENT_DAYS;
  rem = ((long)lcltime) % SECSPERDAY;
  if (rem < 0) {
    rem += SECSPERDAY;
    --days;
  }

  /* compute hour, min, and sec */
  res->tm_hour = (int)(rem / SECSPERHOUR);
  rem %= SECSPERHOUR;
  res->tm_min = (int)(rem / SECSPERMIN);
  res->tm_sec = (int)(rem % SECSPERMIN);

  /* compute day of week */
  if ((weekday = ((ADJUSTED_EPOCH_WDAY + days) % DAYSPERWEEK)) < 0)
    weekday += DAYSPERWEEK;
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
  if (years100 == 4) /* required for proper day of year calculation */
    --years100;
  days -= years100 * DAYS_PER_100_YEARS;
  years4 = days / DAYS_PER_4_YEARS;
  days -= years4 * DAYS_PER_4_YEARS;
  remainingyears = days / DAYS_PER_YEAR;
  if (remainingyears == 4) /* required for proper day of year calculation */
    --remainingyears;
  days -= remainingyears * DAYS_PER_YEAR;

  year = ADJUSTED_EPOCH_YEAR + years400 * 400 + years100 * 100 + years4 * 4 + remainingyears;

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
    if (++month >= MONSPERYEAR)
      month = 0;
  }
  res->tm_mon = month;
  res->tm_mday = days + 1;

  res->tm_isdst = 0;

  return (res);
}

/*
 * @brief support the SQL EXTRACT function
 */
extern "C" __attribute__((noinline))
#ifdef __CUDACC__
__device__
#endif
int64_t ExtractFromTime(ExtractField field, time_t timeval) {

  // We have fast paths for the 5 fields below - do not need to do full gmtime
  switch (field) {
    case kEPOCH:
      return timeval;
    case kHOUR:
      return extract_hour(&timeval);
    case kMINUTE:
      return extract_minute(&timeval);
    case kSECOND:
      return extract_second(&timeval);
    case kDOW:
      return extract_dow(&timeval);
    default:
      break;
  }

  tm tm_struct;
  gmtime_r_newlib(&timeval, &tm_struct);
  switch (field) {
    case kYEAR:
      return 1900 + tm_struct.tm_year;
    case kMONTH:
      return tm_struct.tm_mon + 1;
    case kDAY:
      return tm_struct.tm_mday;
    case kDOY:
      return tm_struct.tm_yday + 1;
    default:
#ifdef __CUDACC__
      return -1;
#else
      CHECK(false);
#endif
  }
}

extern "C"
#ifdef __CUDACC__
    __device__
#endif
        int64_t ExtractFromTimeNullable(ExtractField field, time_t timeval, const int64_t null_val) {
  if (timeval == null_val) {
    return null_val;
  }
  return ExtractFromTime(field, timeval);
}
