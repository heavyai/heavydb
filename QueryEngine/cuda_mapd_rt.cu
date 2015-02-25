#include <stdint.h>
#include <limits>

extern "C"
__device__ int32_t pos_start_impl() {
  return blockIdx.x * blockDim.x + threadIdx.x;
}

extern "C"
__device__ int32_t pos_step_impl() {
  return blockDim.x * gridDim.x;
}

#define EMPTY_KEY -9223372036854775808L

extern "C" __attribute__((noinline))
__device__ int64_t* get_matching_group_value(int64_t* groups_buffer,
                                  const int32_t h,
                                  const int64_t* key,
                                  const int32_t key_qw_count,
                                  const int32_t agg_col_count) {
  int64_t off = h * (key_qw_count + agg_col_count);
  if (groups_buffer[off] == EMPTY_KEY) {
    memcpy(groups_buffer + off, key, key_qw_count * sizeof(*key));
    return groups_buffer + off + key_qw_count;
  }
  bool match = true;
  for (int64_t i = 0; i < key_qw_count; ++i) {
    if (groups_buffer[off + i] != key[i]) {
      match = false;
      break;
    }
  }
  return match ? groups_buffer + off + key_qw_count : NULL;
}

extern "C" __attribute__((noinline))
__device__ int32_t key_hash(const int64_t* key, const int32_t key_qw_count,
                            const int32_t groups_buffer_entry_count) {
  int32_t hash = 0;
  for (int32_t i = 0; i < key_qw_count; ++i) {
    hash = ((hash << 5) - hash + key[i]) % groups_buffer_entry_count;
  }
  return hash;
}

extern "C" __attribute__((noinline))
__device__ int64_t* get_group_value(int64_t* groups_buffer,
                                    const int32_t groups_buffer_entry_count,
                                    const int64_t* key,
                                    const int32_t key_qw_count,
                                    const int32_t agg_col_count) {
  int64_t h = key_hash(key, key_qw_count, groups_buffer_entry_count);
  int64_t* matching_group = get_matching_group_value(groups_buffer, h, key, key_qw_count, agg_col_count);
  if (matching_group) {
    return matching_group;
  }
  int64_t h_probe = h + 1;
  while (h_probe != h) {
    matching_group = get_matching_group_value(groups_buffer, h_probe, key, key_qw_count, agg_col_count);
    if (matching_group) {
      return matching_group;
    }
    h_probe = (h_probe + 1) % groups_buffer_entry_count;
  }
  // TODO(alex): handle error by resizing?
  return NULL;
}

#define SECSPERMIN	60L
#define MINSPERHOUR	60L
#define HOURSPERDAY	24L
#define SECSPERHOUR	(SECSPERMIN * MINSPERHOUR)
#define SECSPERDAY	(SECSPERHOUR * HOURSPERDAY)
#define DAYSPERWEEK	7
#define MONSPERYEAR	12

#define YEAR_BASE	1900

/* move epoch from 01.01.1970 to 01.03.2000 - this is the first day of new
 * 400-year long cycle, right after additional day of leap year. This adjustment
 * is required only for date calculation, so instead of modifying time_t value
 * (which would require 64-bit operations to work correctly) it's enough to
 * adjust the calculated number of days since epoch. */
#define EPOCH_ADJUSTMENT_DAYS	11017
/* year to which the adjustment was made */
#define ADJUSTED_EPOCH_YEAR	2000
/* 1st March of 2000 is Wednesday */
#define ADJUSTED_EPOCH_WDAY	3
/* there are 97 leap years in 400-year periods. ((400 - 97) * 365 + 97 * 366) */
#define DAYS_PER_400_YEARS	146097L
/* there are 24 leap years in 100-year periods. ((100 - 24) * 365 + 24 * 366) */
#define DAYS_PER_100_YEARS	36524L
/* there is one leap year every 4 years */
#define DAYS_PER_4_YEARS	(3 * 365 + 366)
/* number of days in a non-leap year */
#define DAYS_PER_YEAR		365
/* number of days in January */
#define DAYS_IN_JANUARY		31
/* number of days in non-leap February */
#define DAYS_IN_FEBRUARY	28

extern "C"
__device__  tm* gmtime_r_cuda(const time_t *tim_p, tm* res) {
  const int month_lengths[2][MONSPERYEAR] = {
    {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31},
    {31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31}
  };
  long days, rem;
  const time_t lcltime = *tim_p;
  int year, month, yearday, weekday;
  int years400, years100, years4, remainingyears;
  int yearleap;
  const int *ip;

  days = ((long)lcltime) / SECSPERDAY - EPOCH_ADJUSTMENT_DAYS;
  rem = ((long)lcltime) % SECSPERDAY;
  if (rem < 0)
    {
      rem += SECSPERDAY;
      --days;
    }

  /* compute hour, min, and sec */
  res->tm_hour = (int) (rem / SECSPERHOUR);
  rem %= SECSPERHOUR;
  res->tm_min = (int) (rem / SECSPERMIN);
  res->tm_sec = (int) (rem % SECSPERMIN);

  /* compute day of week */
  if ((weekday = ((ADJUSTED_EPOCH_WDAY + days) % DAYSPERWEEK)) < 0)
    weekday += DAYSPERWEEK;
  res->tm_wday = weekday;

  /* compute year & day of year */
  years400 = days / DAYS_PER_400_YEARS;
  days -= years400 * DAYS_PER_400_YEARS;
  /* simplify by making the values positive */
  if (days < 0)
    {
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

  year = ADJUSTED_EPOCH_YEAR + years400 * 400 + years100 * 100 + years4 * 4 +
      remainingyears;

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
  if (yearday >= DAYS_PER_YEAR + yearleap)
    {
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
  while (days >= ip[month])
    {
      days -= ip[month];
      if (++month >= MONSPERYEAR)
        month = 0;
    }
  res->tm_mon = month;
  res->tm_mday = days + 1;

  res->tm_isdst = 0;

  return (res);
}

#include "ExtractFromTime.cpp"
