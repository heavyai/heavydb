/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Credits: Howard Hinnant for open source date calculations.

#include "Shared/timedate.h"

#include <cctype>
#include <cstdint>
#include <cstdio>

#include "Shared/misc.h"
#include "Shared/sqltypes.h"

namespace shared {

void computeDate(int64_t const unixtime,
                 uint32_t& y_out,
                 uint32_t& m_out,
                 uint32_t& d_out) {
  // original code
  DivUMod const div_day = divUMod(unixtime, 24 * 60 * 60);
  DivUMod const div_era = divUMod(div_day.quot - 11017, 146097);
  unsigned const doe = static_cast<unsigned>(div_era.rem);
  unsigned const yoe = (doe - doe / 1460 + doe / 36524 - (doe == 146096)) / 365;
  unsigned const doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
  unsigned const moy = (5 * doy + 2) / 153;
  static_assert(8 <= sizeof(long long));  // long long needed for snprintf()
  long long const y = 2000 + div_era.quot * 400 + yoe + (9 < moy);
  unsigned const m = moy + (9 < moy ? -9 : 3);
  unsigned const d = doy - (153 * moy + 2) / 5 + 1;
  // convert for output
  y_out = static_cast<uint32_t>(y);
  m_out = static_cast<uint32_t>(m);
  d_out = static_cast<uint32_t>(d);
}

size_t formatDate(char* buf, size_t const max, int64_t const unixtime) {
  uint32_t y, m, d;
  computeDate(unixtime, y, m, d);
  int const len = snprintf(buf, max, "%04u-%02u-%02u", y, m, d);
  if (0 <= len && static_cast<size_t>(len) < max) {
    return static_cast<size_t>(len);
  }
  return 0;
}

void computeDateTime(const int64_t timestamp,
                     const int32_t dimension,
                     uint32_t& y_out,
                     uint32_t& m_out,
                     uint32_t& d_out,
                     uint32_t& hh_out,
                     uint32_t& mm_out,
                     uint32_t& ss_out,
                     uint32_t& frac_out) {
  // original code
  constexpr int pow10[10]{
      1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000};
  DivUMod const div_hip = divUMod(timestamp, pow10[dimension]);
  DivUMod const div_day = divUMod(div_hip.quot, 24 * 60 * 60);
  DivUMod const div_era = divUMod(div_day.quot - 11017, 146097);
  unsigned const doe = static_cast<unsigned>(div_era.rem);
  unsigned const yoe = (doe - doe / 1460 + doe / 36524 - (doe == 146096)) / 365;
  unsigned const doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
  unsigned const moy = (5 * doy + 2) / 153;
  static_assert(8 <= sizeof(long long));  // long long needed for snprintf()
  long long const y = 2000 + div_era.quot * 400 + yoe + (9 < moy);
  unsigned const m = moy + (9 < moy ? -9 : 3);
  unsigned const d = doy - (153 * moy + 2) / 5 + 1;
  unsigned const minutes = static_cast<unsigned>(div_day.rem) / 60;
  unsigned const ss = div_day.rem % 60;
  unsigned const hh = minutes / 60;
  unsigned const mm = minutes % 60;
  int const frac = static_cast<int>(div_hip.rem);
  // convert for output
  y_out = static_cast<uint32_t>(y);
  m_out = static_cast<uint32_t>(m);
  d_out = static_cast<uint32_t>(d);
  hh_out = static_cast<uint32_t>(hh);
  mm_out = static_cast<uint32_t>(mm);
  ss_out = static_cast<uint32_t>(ss);
  frac_out = static_cast<uint32_t>(frac);
}

size_t formatDateTime(char* buf,
                      size_t const max,
                      int64_t const timestamp,
                      int const dimension,
                      bool use_iso_format) {
  uint32_t y, m, d, hh, mm, ss, frac;
  computeDateTime(timestamp, dimension, y, m, d, hh, mm, ss, frac);
  const char* date_time_format;
  if (use_iso_format) {
    if (dimension) {
      date_time_format = "%04u-%02u-%02uT%02u:%02u:%02u";
    } else {
      date_time_format = "%04u-%02u-%02uT%02u:%02u:%02uZ";
    }
  } else {
    date_time_format = "%04u-%02u-%02u %02u:%02u:%02u";
  }
  int const len = snprintf(buf, max, date_time_format, y, m, d, hh, mm, ss);
  if (0 <= len && static_cast<size_t>(len) < max) {
    if (dimension) {
      auto precision_format = use_iso_format ? ".%0*uZ" : ".%0*u";
      int const len_frac =
          snprintf(buf + len, max - len, precision_format, dimension, frac);
      if (0 <= len_frac && static_cast<size_t>(len + len_frac) < max) {
        return static_cast<size_t>(len + len_frac);
      }
    } else {
      return static_cast<size_t>(len);
    }
  }
  return 0;
}

void computeHMS(const int64_t unixtime,
                uint32_t& hh_out,
                uint32_t& mm_out,
                uint32_t& ss_out) {
  // original code
  unsigned const seconds = static_cast<unsigned>(unsignedMod(unixtime, 24 * 60 * 60));
  unsigned const minutes = seconds / 60;
  unsigned const ss = seconds % 60;
  unsigned const hh = minutes / 60;
  unsigned const mm = minutes % 60;
  // convert for output
  hh_out = static_cast<uint32_t>(hh);
  mm_out = static_cast<uint32_t>(mm);
  ss_out = static_cast<uint32_t>(ss);
}

size_t formatHMS(char* buf, size_t const max, int64_t const unixtime) {
  uint32_t hh, mm, ss;
  computeHMS(unixtime, hh, mm, ss);
  int const len = snprintf(buf, max, "%02u:%02u:%02u", hh, mm, ss);
  if (0 <= len && static_cast<size_t>(len) < max) {
    return static_cast<size_t>(len);
  }
  return 0;
}

std::string convert_temporal_to_iso_format(const int64_t unix_time,
                                           const SQLTypes sql_type,
                                           const int precision) {
  std::string iso_str;
  if (sql_type == kTIME) {
    // Set a buffer size that can contain HH:MM:SS
    iso_str.resize(8);
    const auto len = shared::formatHMS(iso_str.data(), iso_str.length() + 1, unix_time);
    CHECK_EQ(len, iso_str.length());
  } else if (sql_type == kDATE) {
    // Set a buffer size that can contain YYYYYYYYYYYY-mm-dd (int64_t can represent up to
    // 12 digit years)
    iso_str.resize(18);
    const size_t len =
        shared::formatDate(iso_str.data(), iso_str.length() + 1, unix_time);
    CHECK_GT(len, static_cast<size_t>(0));
    iso_str.resize(len);
  } else if (sql_type == kTIMESTAMP) {
    // Set a buffer size that can contain the specified timestamp precision
    // YYYYYYYYYYYY-mm-dd(18) T(1) HH:MM:SS(8) .(precision?) nnnnnnnnn(precision) Z(1)
    // (int64_t can represent up to 12 digit years with seconds precision)
    iso_str.resize(18 + 1 + 8 + bool(precision) + precision + 1);
    const size_t len = shared::formatDateTime(
        iso_str.data(), iso_str.length() + 1, unix_time, precision, true);
    CHECK_GT(len, static_cast<size_t>(0));
    iso_str.resize(len);
  } else {
    UNREACHABLE() << "Unexpected column type: " << toString(sql_type);
  }
  return iso_str;
}

}  // namespace shared
