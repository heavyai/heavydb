/*
 * Copyright 2022 HEAVY.AI, Inc.
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

// Credits: Howard Hinnant for open source date calculations.

#include "misc.h"
#include "sqltypes.h"

#include <cctype>
#include <cstdio>
#include <fstream>
#include <iomanip>

namespace shared {

size_t formatDate(char* buf, size_t const max, int64_t const unixtime) {
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
  int const len = snprintf(buf, max, "%04lld-%02u-%02u", y, m, d);
  if (0 <= len && static_cast<size_t>(len) < max) {
    return static_cast<size_t>(len);
  }
  return 0;
}

size_t formatDateTime(char* buf,
                      size_t const max,
                      int64_t const timestamp,
                      int const dimension,
                      bool use_iso_format) {
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
  const char* date_time_format;
  if (use_iso_format) {
    if (dimension) {
      date_time_format = "%04lld-%02u-%02uT%02u:%02u:%02u";
    } else {
      date_time_format = "%04lld-%02u-%02uT%02u:%02u:%02uZ";
    }
  } else {
    date_time_format = "%04lld-%02u-%02u %02u:%02u:%02u";
  }
  int const len = snprintf(buf, max, date_time_format, y, m, d, hh, mm, ss);
  if (0 <= len && static_cast<size_t>(len) < max) {
    if (dimension) {
      auto precision_format = use_iso_format ? ".%0*dZ" : ".%0*d";
      int const len_frac = snprintf(buf + len,
                                    max - len,
                                    precision_format,
                                    dimension,
                                    static_cast<int>(div_hip.rem));
      if (0 <= len_frac && static_cast<size_t>(len + len_frac) < max) {
        return static_cast<size_t>(len + len_frac);
      }
    } else {
      return static_cast<size_t>(len);
    }
  }
  return 0;
}

size_t formatHMS(char* buf, size_t const max, int64_t const unixtime) {
  unsigned const seconds = static_cast<unsigned>(unsignedMod(unixtime, 24 * 60 * 60));
  unsigned const minutes = seconds / 60;
  unsigned const ss = seconds % 60;
  unsigned const hh = minutes / 60;
  unsigned const mm = minutes % 60;
  int const len = snprintf(buf, max, "%02u:%02u:%02u", hh, mm, ss);
  if (0 <= len && static_cast<size_t>(len) < max) {
    return static_cast<size_t>(len);
  }
  return 0;
}

std::string convert_temporal_to_iso_format(const SQLTypeInfo& type_info,
                                           int64_t unix_time) {
  std::string iso_str;
  if (type_info.get_type() == kTIME) {
    // Set a buffer size that can contain HH:MM:SS
    iso_str.resize(8);
    const auto len = shared::formatHMS(iso_str.data(), iso_str.length() + 1, unix_time);
    CHECK_EQ(len, iso_str.length());
  } else if (type_info.get_type() == kDATE) {
    // Set a buffer size that can contain YYYYYYYYYYYY-mm-dd (int64_t can represent up to
    // 12 digit years)
    iso_str.resize(18);
    const size_t len =
        shared::formatDate(iso_str.data(), iso_str.length() + 1, unix_time);
    CHECK_GT(len, static_cast<size_t>(0));
    iso_str.resize(len);
  } else if (type_info.get_type() == kTIMESTAMP) {
    auto precision = type_info.get_precision();
    // Set a buffer size that can contain the specified timestamp precision
    // YYYYYYYYYYYY-mm-dd(18) T(1) HH:MM:SS(8) .(precision?) nnnnnnnnn(precision) Z(1)
    // (int64_t can represent up to 12 digit years with seconds precision)
    iso_str.resize(18 + 1 + 8 + bool(precision) + precision + 1);
    const size_t len = shared::formatDateTime(
        iso_str.data(), iso_str.length() + 1, unix_time, precision, true);
    CHECK_GT(len, static_cast<size_t>(0));
    iso_str.resize(len);
  } else {
    UNREACHABLE() << "Unexpected column type: " << type_info.toString();
  }
  return iso_str;
}

size_t compute_hash(int32_t item_1, int32_t item_2) {
  static_assert(sizeof(item_1) + sizeof(item_2) <= sizeof(size_t));
  return (static_cast<size_t>(item_1) << (8 * sizeof(item_2))) |
         (static_cast<size_t>(item_2));
}

// Escape and quote contents of filename as a json string and output to os.
// Q: Why not just return the file contents as a string?
// A: Constructing a string may unnecessarily contribute to memory fragmentation,
//    and is probably less performant due to the extra heap allocations.
void FileContentsEscaper::quoteAndPrint(std::ostream& os) const {
  std::ifstream file(filename);
  if (!file.is_open()) {
    os << "\"Unable to open " << filename << '"';
    return;
  }
  char ch;
  std::ios orig_os_state(nullptr);
  orig_os_state.copyfmt(os);
  os << '"';
  while (file.get(ch)) {
    if (ch == '"') {
      os << "\\\"";
    } else if (ch == '\\') {
      os << "\\\\";
    } else if (std::isprint(ch) || ch == ' ') {
      os << ch;
    } else {
      switch (ch) {
        // clang-format off
        case '\b': os << "\\b"; break;
        case '\f': os << "\\f"; break;
        case '\n': os << "\\n"; break;
        case '\r': os << "\\r"; break;
        case '\t': os << "\\t"; break;
        // clang-format on
        default:
          os << "\\u" << std::hex << std::setw(4) << std::setfill('0')
             << static_cast<unsigned>(static_cast<unsigned char>(ch));
          break;
      }
    }
  }
  os << '"';
  os.copyfmt(orig_os_state);
}

std::ostream& operator<<(std::ostream& os, FileContentsEscaper const& fce) {
  fce.quoteAndPrint(os);
  return os;
}

}  // namespace shared
