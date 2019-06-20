/*
 * Copyright 2018 OmniSci, Inc.
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

#pragma once

#include "DateAdd.h"
#include "DateTruncate.h"

#include <cstdint>
#include <ctime>
#include <map>
#include <string>

#include "../Shared/sqldefs.h"

namespace {

static const std::map<std::pair<int32_t, ExtractField>, std::pair<SQLOps, int64_t>>
    extract_precision_lookup = {{{3, kMICROSECOND}, {kMULTIPLY, kMilliSecsPerSec}},
                                {{3, kNANOSECOND}, {kMULTIPLY, kMicroSecsPerSec}},
                                {{6, kMILLISECOND}, {kDIVIDE, kMilliSecsPerSec}},
                                {{6, kNANOSECOND}, {kMULTIPLY, kMilliSecsPerSec}},
                                {{9, kMILLISECOND}, {kDIVIDE, kMicroSecsPerSec}},
                                {{9, kMICROSECOND}, {kDIVIDE, kMilliSecsPerSec}}};

static const std::map<std::pair<int32_t, DatetruncField>, int64_t>
    datetrunc_precision_lookup = {{{6, dtMILLISECOND}, kMilliSecsPerSec},
                                  {{9, dtMICROSECOND}, kMilliSecsPerSec},
                                  {{9, dtMILLISECOND}, kMicroSecsPerSec}};

}  // namespace

namespace DateTimeUtils {

constexpr inline int64_t get_timestamp_precision_scale(const int32_t dimen) {
  switch (dimen) {
    case 0:
      return 1;
    case 3:
      return kMilliSecsPerSec;
    case 6:
      return kMicroSecsPerSec;
    case 9:
      return kNanoSecsPerSec;
    default:
      throw std::runtime_error("Unknown dimen = " + std::to_string(dimen));
  }
  return -1;
}

constexpr inline int64_t get_dateadd_timestamp_precision_scale(const DateaddField field) {
  switch (field) {
    case daMILLISECOND:
      return kMilliSecsPerSec;
    case daMICROSECOND:
      return kMicroSecsPerSec;
    case daNANOSECOND:
      return kNanoSecsPerSec;
    default:
      throw std::runtime_error("Unknown field = " + std::to_string(field));
  }
  return -1;
}

constexpr inline int64_t get_extract_timestamp_precision_scale(const ExtractField field) {
  switch (field) {
    case kMILLISECOND:
      return kMilliSecsPerSec;
    case kMICROSECOND:
      return kMicroSecsPerSec;
    case kNANOSECOND:
      return kNanoSecsPerSec;
    default:
      throw std::runtime_error("Unknown field = " + std::to_string(field));
  }
  return -1;
}

constexpr inline bool is_subsecond_extract_field(const ExtractField& field) {
  return field == kMILLISECOND || field == kMICROSECOND || field == kNANOSECOND;
}

constexpr inline bool is_subsecond_dateadd_field(const DateaddField field) {
  return field == daMILLISECOND || field == daMICROSECOND || field == daNANOSECOND;
}

constexpr inline bool is_subsecond_datetrunc_field(const DatetruncField field) {
  return field == dtMILLISECOND || field == dtMICROSECOND || field == dtNANOSECOND;
}

const inline std::pair<SQLOps, int64_t> get_dateadd_high_precision_adjusted_scale(
    const DateaddField field,
    int32_t dimen) {
  switch (field) {
    case daNANOSECOND:
      switch (dimen) {
        case 9:
          return {};
        case 6:
          return {kDIVIDE, kMilliSecsPerSec};
        case 3:
          return {kDIVIDE, kMicroSecsPerSec};
        default:
          throw std::runtime_error("Unknown dimen = " + std::to_string(dimen));
      }
    case daMICROSECOND:
      switch (dimen) {
        case 9:
          return {kMULTIPLY, kMilliSecsPerSec};
        case 6:
          return {};
        case 3:
          return {kDIVIDE, kMilliSecsPerSec};
        default:
          throw std::runtime_error("Unknown dimen = " + std::to_string(dimen));
      }
    case daMILLISECOND:
      switch (dimen) {
        case 9:
          return {kMULTIPLY, kMicroSecsPerSec};
        case 6:
          return {kMULTIPLY, kMilliSecsPerSec};
        case 3:
          return {};
        default:
          throw std::runtime_error("Unknown dimen = " + std::to_string(dimen));
      }
    default:
      throw std::runtime_error("Unknown field = " + std::to_string(field));
  }
  return {};
}

const inline std::pair<SQLOps, int64_t> get_extract_high_precision_adjusted_scale(
    const ExtractField& field,
    const int32_t dimen) {
  const auto result = extract_precision_lookup.find(std::make_pair(dimen, field));
  if (result != extract_precision_lookup.end()) {
    return result->second;
  }
  return {};
}

const inline int64_t get_datetrunc_high_precision_scale(const DatetruncField& field,
                                                        const int32_t dimen) {
  const auto result = datetrunc_precision_lookup.find(std::make_pair(dimen, field));
  if (result != datetrunc_precision_lookup.end()) {
    return result->second;
  }
  return -1;
}

}  // namespace DateTimeUtils
