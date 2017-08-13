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
#include <glog/logging.h>
#endif

DEVICE
int is_leap(int64_t year) {
  return (((year % 400) == 0) || ((year % 4) == 0 && ((year % 100) != 0))) ? 1 : 0;
}

DEVICE
time_t skip_months(time_t timeval, int64_t months_to_go) {
  const int month_lengths[2][MONSPERYEAR] = {{31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31},
                                             {31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31}};
  tm tm_struct;
  gmtime_r_newlib(&timeval, &tm_struct);
  auto tod = (timeval % SECSPERDAY);
  auto day = (timeval / SECSPERDAY) * SECSPERDAY;
  // calculate the day of month offset
  int dom = tm_struct.tm_mday;
  time_t month = day - (dom * SECSPERDAY);
  // find what month we are
  int mon = tm_struct.tm_mon;
  int64_t months_covered = 0;
  while (months_to_go != 0) {
    int leap_year = 0;
    if (months_to_go >= 48) {
      month += (months_to_go / 48) * DAYS_PER_4_YEARS * SECSPERDAY;
      months_to_go = months_to_go % 48;
      months_covered += 48 * (months_to_go / 48);
      continue;
    }
    if (months_to_go > 0) {
      auto m = (mon + months_covered) % MONSPERYEAR;
      if (m == 1)
        leap_year = is_leap(ExtractFromTime(kYEAR, month));
      month += (month_lengths[0 + leap_year][m] * SECSPERDAY);
      months_to_go--;
      months_covered++;
      continue;
    }
    if (months_to_go <= -48) {
      month -= ((-months_to_go) / 48) * DAYS_PER_4_YEARS * SECSPERDAY;
      months_to_go = -((-months_to_go) % 48);
      months_covered += 48 * ((-months_to_go) / 48);
      continue;
    }
    if (months_to_go < 0) {
      auto m = (((mon - 1 - months_covered) % MONSPERYEAR) + MONSPERYEAR) % MONSPERYEAR;
      if (m == 1)
        leap_year = is_leap(ExtractFromTime(kYEAR, month));
      month -= (month_lengths[0 + leap_year][m] * SECSPERDAY);
      months_to_go++;
      months_covered++;
    }
  }

  auto new_timeval = month + dom * SECSPERDAY + tod;
  tm new_tm_struct;
  gmtime_r_newlib(&new_timeval, &new_tm_struct);
  int new_dom = new_tm_struct.tm_mday;
  if (dom > new_dom) {
    // Landed on a month with fewer days, overshot by a few days,
    // e.g. 2008-1-31 + INTERVAL '1' MONTH should yield 2008-2-29 and
    // e.g. 2009-1-31 + INTERVAL '1' MONTH should yield 2008-2-28
    // Go to the last day of preceeding month
    new_timeval -= new_dom * SECSPERDAY;
  }
  return new_timeval;
}

extern "C" NEVER_INLINE DEVICE time_t DateAdd(DateaddField field, int64_t number, time_t timeval) {
  switch (field) {
    case daSECOND:
      /* this is the limit of current granularity */
      return timeval + number;
    case daMINUTE:
      return timeval + number * SECSPERMIN;
    case daHOUR:
      return timeval + number * SECSPERHOUR;
    case daDAY:
      return timeval + number * SECSPERDAY;
    case daWEEK:
      return timeval + number * DAYSPERWEEK * SECSPERDAY;
    default:
      break;
  }

  int64_t months_to_go;
  switch (field) {
    case daMONTH:
      months_to_go = 1;
      break;
    case daQUARTER:
      months_to_go = 3;
      break;
    case daYEAR:
      months_to_go = 12;
      break;
    case daDECADE:
      months_to_go = 10 * 12;
      break;
    case daCENTURY:
      months_to_go = 100 * 12;
      break;
    case daMILLENNIUM:
      months_to_go = 1000 * 12;
      break;
    default:
#ifdef __CUDACC__
      return -1;
#else
      abort();
#endif
  }
  months_to_go *= number;
  return skip_months(timeval, months_to_go);
}

extern "C" DEVICE time_t DateAddNullable(const DateaddField field,
                                         int64_t number,
                                         time_t timeval,
                                         const time_t null_val) {
  if (timeval == null_val) {
    return null_val;
  }
  return DateAdd(field, number, timeval);
}

#endif  // EXECUTE_INCLUDE
