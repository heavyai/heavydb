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

#ifndef QUERYENGINE_DATETRUNCATE_H
#define QUERYENGINE_DATETRUNCATE_H

#include <stdint.h>

#include "../Shared/funcannotations.h"
#include "ExtractFromTime.h"

/*
 * year
 * month
 * day
 * hour
 * minute
 * second
 *
 * millennium
 * century
 * decade
 * milliseconds
 * microseconds
 * nanoseconds
 * week
 * quarterday
 */
enum DatetruncField {
  dtYEAR,
  dtQUARTER,
  dtMONTH,
  dtDAY,
  dtHOUR,
  dtMINUTE,
  dtSECOND,
  dtMILLENNIUM,
  dtCENTURY,
  dtDECADE,
  dtMILLISECOND,
  dtMICROSECOND,
  dtNANOSECOND,
  dtWEEK,
  dtQUARTERDAY,
  dtINVALID
};

extern "C" NEVER_INLINE DEVICE time_t DateTruncate(DatetruncField field, time_t timeval);

extern "C" NEVER_INLINE DEVICE time_t DateTruncateHighPrecision(DatetruncField field,
                                                                time_t timeval,
                                                                const int64_t scale);

extern "C" ALWAYS_INLINE inline DEVICE int64_t
DateTruncateAlterPrecisionScaleUp(DatetruncField field,
                                  time_t timeval,
                                  const int64_t scale) {
  return timeval * scale;
}

extern "C" ALWAYS_INLINE inline DEVICE int64_t
DateTruncateAlterPrecisionScaleDown(DatetruncField field,
                                    time_t timeval,
                                    const int64_t scale) {
  return timeval / scale;
}

#endif  // QUERYENGINE_DATETRUNCATE_H
