/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

DEVICE int64_t DateTruncate(DatetruncField field, const int64_t timeval);

extern "C" RUNTIME_EXPORT DEVICE int64_t
DateTruncateHighPrecisionToDate(const int64_t timeval, const int64_t scale);

#endif  // QUERYENGINE_DATETRUNCATE_H
