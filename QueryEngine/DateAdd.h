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

#ifndef QUERYENGINE_DATEADD_H
#define QUERYENGINE_DATEADD_H

#include <stdint.h>
#include <time.h>

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
 * week
 * quarterday
 */
enum DateaddField {
  daYEAR,
  daQUARTER,
  daMONTH,
  daDAY,
  daHOUR,
  daMINUTE,
  daSECOND,
  daMILLENNIUM,
  daCENTURY,
  daDECADE,
  daMILLISECOND,
  daMICROSECOND,
  daNANOSECOND,
  daWEEK,
  daQUARTERDAY,
  daWEEKDAY,
  daDAYOFYEAR,
  daINVALID
};

extern "C" NEVER_INLINE DEVICE time_t DateAdd(DateaddField field,
                                              int64_t number,
                                              time_t timeval);

extern "C" NEVER_INLINE DEVICE time_t DateAddHighPrecision(DateaddField field,
                                                           const int64_t number,
                                                           time_t timeval,
                                                           const int64_t scale);

#endif  // QUERYENGINE_DATEADD_H
