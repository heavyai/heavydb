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

#ifndef QUERYENGINE_EXTRACTFROMTIME_H
#define QUERYENGINE_EXTRACTFROMTIME_H

#include <stdint.h>
#include <time.h>
#include "../Shared/funcannotations.h"

#define SECSPERMIN 60L
#define MINSPERHOUR 60L
#define HOURSPERDAY 24L
#define SECSPERHOUR (SECSPERMIN * MINSPERHOUR)
#define SECSPERDAY (SECSPERHOUR * HOURSPERDAY)
#define SECSPERQUARTERDAY (SECSPERHOUR * 6)
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

#define SECONDS_PER_NON_LEAP_YEAR 31536000
#define SECONDS_PER_4_YEAR_CYCLE 126230400
#define SECONDS_PER_DAY 86400
#define EPOCH_OFFSET_YEAR_1900 2208988800
#define SECONDS_FROM_JAN_1900_TO_MARCH_1900 5097600

enum ExtractField {
  kYEAR,
  kQUARTER,
  kMONTH,
  kDAY,
  kHOUR,
  kMINUTE,
  kSECOND,
  kDOW,
  kISODOW,
  kDOY,
  kEPOCH,
  kQUARTERDAY,
  kWEEK
};

// Shared by DateTruncate
#ifdef __CUDACC__
__device__
#endif
    int
    extract_dow(const time_t* tim_p);

#ifdef __CUDACC__
__device__
#endif
    tm*
    gmtime_r_newlib(const time_t* tim_p, tm* res);

extern "C" __attribute__((noinline))
#ifdef __CUDACC__
__device__
#endif
    int64_t
    ExtractFromTime(ExtractField field, time_t timeval);

#endif  // QUERYENGINE_EXTRACTFROMTIME_H
