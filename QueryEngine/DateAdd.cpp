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

#include <math.h>
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
  long dimen = 1;  // placeholder
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
        leap_year = is_leap(ExtractFromTime(kYEAR, month, dimen));
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
        leap_year = is_leap(ExtractFromTime(kYEAR, month, dimen));
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

extern "C" NEVER_INLINE DEVICE time_t DateAdd(DateaddField field, int64_t number, time_t timeval, const int32_t dimen) {
  long scale_ret = 1;
  if (dimen == 3) {
    scale_ret = MILLISECSPERSEC;
  } else if (dimen == 6) {
    scale_ret = MICROSECSPERSEC;
  } else if (dimen == 9) {
    scale_ret = NANOSECSPERSEC;
  }
  switch (field) {
    case daNANOSECOND: {
      // highest level of granularity
      if (dimen < 9) {
        return timeval;
      } else
        return timeval + number;
    }
    case daMICROSECOND: {
      int64_t mutimeval = 0;
      if (dimen == 9) {
        mutimeval = timeval + number * MILLISECSPERSEC;
      } else if (dimen == 6) {
        mutimeval = timeval + number;
      } else
        mutimeval = timeval;
      return mutimeval;
    }
    case daMILLISECOND: {
      int64_t mitimeval = 0;
      if (dimen == 9) {
        mitimeval = timeval + number * MICROSECSPERSEC;
      } else if (dimen == 6) {
        mitimeval = timeval + number * MILLISECSPERSEC;
      } else
        mitimeval = timeval + number;
      return mitimeval;
    }
    case daSECOND:
      return timeval + number * scale_ret;
    case daMINUTE:
      return timeval + number * SECSPERMIN * scale_ret;
    case daHOUR:
      return timeval + number * SECSPERHOUR * scale_ret;
    case daDAY:
      return timeval + number * SECSPERDAY * scale_ret;
    case daWEEK:
      return timeval + number * DAYSPERWEEK * SECSPERDAY * scale_ret;
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
  const time_t stimeval = (int64_t)(timeval / scale_ret);
  const time_t nfrac = (int)((long)timeval % scale_ret);
  return (skip_months(stimeval, months_to_go) * scale_ret) + nfrac;
}

extern "C" DEVICE time_t
DateAddNullable(const DateaddField field, int64_t number, time_t timeval, const int32_t dimen, const time_t null_val) {
  if (timeval == null_val) {
    return null_val;
  }
  return DateAdd(field, number, timeval, dimen);
}

#endif  // EXECUTE_INCLUDE
