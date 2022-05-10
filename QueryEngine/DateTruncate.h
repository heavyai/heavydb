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

#include <array>
#include <cstdint>

#include "../Shared/funcannotations.h"
#include "ExtractFromTime.h"

// DatetruncField must be synced with datetrunc_fname
enum DatetruncField {
  dtYEAR = 0,
  dtQUARTER,
  dtMONTH,
  dtDAY,
  dtHOUR,
  dtMINUTE,
  dtSECOND,
  dtMILLISECOND,
  dtMICROSECOND,
  dtNANOSECOND,
  dtMILLENNIUM,
  dtCENTURY,
  dtDECADE,
  dtWEEK,
  dtWEEK_SUNDAY,
  dtWEEK_SATURDAY,
  dtQUARTERDAY,
  dtINVALID
};

constexpr std::array<char const*, dtINVALID> datetrunc_fname_lookup{
    {"datetrunc_year",
     "datetrunc_quarter",
     "datetrunc_month",
     "datetrunc_day",
     "datetrunc_hour",
     "datetrunc_minute",
     "datetrunc_second",       // not used
     "datetrunc_millisecond",  // not used
     "datetrunc_microsecond",  // not used
     "datetrunc_nanosecond",   // not used
     "datetrunc_millennium",
     "datetrunc_century",
     "datetrunc_decade",
     "datetrunc_week_monday",
     "datetrunc_week_sunday",
     "datetrunc_week_saturday",
     "datetrunc_quarterday"}};

// Arithmetic which relies on these enums being consecutive is used elsewhere.
static_assert(dtSECOND + 1 == dtMILLISECOND, "Please keep these consecutive.");
static_assert(dtMILLISECOND + 1 == dtMICROSECOND, "Please keep these consecutive.");
static_assert(dtMICROSECOND + 1 == dtNANOSECOND, "Please keep these consecutive.");

int64_t DateTruncate(DatetruncField field, const int64_t timeval);

// for usage in compiled and linked modules in the binary
int64_t truncate_high_precision_timestamp_to_date(const int64_t timeval,
                                                  const int64_t scale);

#endif  // QUERYENGINE_DATETRUNCATE_H
