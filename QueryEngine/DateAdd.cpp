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
int32_t is_leap(int64_t year) {
  return (((year % 400) == 0) || ((year % 4) == 0 && ((year % 100) != 0))) ? 1 : 0;
}

DEVICE
int64_t skip_months(int64_t timeval, int64_t months_to_go) {
  const int32_t month_lengths[2][kMonsPerYear] = {
      {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31},
      {31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31}};
  tm tm_struct;
  gmtime_r_newlib(&timeval, &tm_struct);
  auto tod = timeval % kSecsPerDay;
  auto day = (timeval / kSecsPerDay) * kSecsPerDay;
  // calculate the day of month offset
  int32_t dom = tm_struct.tm_mday;
  int64_t month = day - (dom * kSecsPerDay);
  // find what month we are
  int32_t mon = tm_struct.tm_mon;
  int64_t months_covered = 0;
  while (months_to_go != 0) {
    int32_t leap_year = 0;
    if (months_to_go >= 48) {
      month += (months_to_go / 48) * kDaysPer4Years * kSecsPerDay;
      months_to_go = months_to_go % 48;
      months_covered += 48 * (months_to_go / 48);
      continue;
    }
    if (months_to_go > 0) {
      auto m = (mon + months_covered) % kMonsPerYear;
      if (m == 1)
        leap_year = is_leap(ExtractFromTime(kYEAR, month));
      month += (month_lengths[0 + leap_year][m] * kSecsPerDay);
      months_to_go--;
      months_covered++;
      continue;
    }
    if (months_to_go <= -48) {
      month -= ((-months_to_go) / 48) * kDaysPer4Years * kSecsPerDay;
      months_to_go = -((-months_to_go) % 48);
      months_covered += 48 * ((-months_to_go) / 48);
      continue;
    }
    if (months_to_go < 0) {
      auto m =
          (((mon - 1 - months_covered) % kMonsPerYear) + kMonsPerYear) % kMonsPerYear;
      if (m == 1)
        leap_year = is_leap(ExtractFromTime(kYEAR, month));
      month -= (month_lengths[0 + leap_year][m] * kSecsPerDay);
      months_to_go++;
      months_covered++;
    }
  }

  int64_t new_timeval = month + dom * kSecsPerDay + tod;
  tm new_tm_struct;
  gmtime_r_newlib(&new_timeval, &new_tm_struct);
  int32_t new_dom = new_tm_struct.tm_mday;
  if (dom > new_dom) {
    // Landed on a month with fewer days, overshot by a few days,
    // e.g. 2008-1-31 + INTERVAL '1' MONTH should yield 2008-2-29 and
    // e.g. 2009-1-31 + INTERVAL '1' MONTH should yield 2008-2-28
    // Go to the last day of preceeding month
    new_timeval -= new_dom * kSecsPerDay;
  }
  return new_timeval;
}

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
      return timeval + number * kSecPerHour;
    case daWEEKDAY:
    case daDAYOFYEAR:
    case daDAY:
      return timeval + number * kSecsPerDay;
    case daWEEK:
      return timeval + number * kDaysPerWeek * kSecsPerDay;
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
      break;
  }
  return (DateAdd(field, number, timeval / scale) * scale) + (timeval % scale);
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
