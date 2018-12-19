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

#ifndef QUERYENGINE_DATETIMEUTILS_H
#define QUERYENGINE_DATETIMEUTILS_H

#include "DateAdd.h"
#include "DateTruncate.h"

#include <cstdint>
#include <ctime>
#include <map>

#include "../Shared/sqldefs.h"
#include "../Shared/unreachable.h"

static const std::map<int, DatetruncField> timestamp_precisions_lookup_{
    {0, DatetruncField::dtSECOND},
    {3, DatetruncField::dtMILLISECOND},
    {6, DatetruncField::dtMICROSECOND},
    {9, DatetruncField::dtNANOSECOND}};

inline int64_t get_timestamp_precision_scale(const int32_t dimen) {
  switch (dimen) {
    case 0:
      return 1;
    case 3:
      return MILLISECSPERSEC;
    case 6:
      return MICROSECSPERSEC;
    case 9:
      return NANOSECSPERSEC;
    default:
      UNREACHABLE();
  }
  return -1;
}

inline int64_t get_dateadd_timestamp_precision_scale(const DateaddField field) {
  switch (field) {
    case daMILLISECOND:
      return MILLISECSPERSEC;
    case daMICROSECOND:
      return MICROSECSPERSEC;
    case daNANOSECOND:
      return NANOSECSPERSEC;
    default:
      UNREACHABLE();
  }
  return -1;
}

inline bool is_subsecond_dateadd_field(const DateaddField field) {
  return field == daMILLISECOND || field == daMICROSECOND || field == daNANOSECOND;
}

inline bool is_subsecond_datetrunc_field(const DatetruncField field) {
  return field == dtMILLISECOND || field == dtMICROSECOND || field == dtNANOSECOND;
}

inline std::pair<SQLOps, int64_t> get_dateadd_high_precision_adjusted_scale(
    const DateaddField field,
    int32_t dimen) {
  switch (field) {
    case daNANOSECOND:
      switch (dimen) {
        case 9:
          return {};
        case 6:
          return {kDIVIDE, MILLISECSPERSEC};
        case 3:
          return {kDIVIDE, MICROSECSPERSEC};
        default:
          UNREACHABLE();
      }
    case daMICROSECOND:
      switch (dimen) {
        case 9:
          return {kMULTIPLY, MILLISECSPERSEC};
        case 6:
          return {};
        case 3:
          return {kDIVIDE, MILLISECSPERSEC};
        default:
          UNREACHABLE();
      }
    case daMILLISECOND:
      switch (dimen) {
        case 9:
          return {kMULTIPLY, MICROSECSPERSEC};
        case 6:
          return {kMULTIPLY, MILLISECSPERSEC};
        case 3:
          return {};
        default:
          UNREACHABLE();
      }
    default:
      UNREACHABLE();
  }
  return {};
}

#endif  // QUERYENGINE_DATETIMEUTILS_H
