/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef QUERYENGINE_DATEADD_H
#define QUERYENGINE_DATEADD_H

#include <cstdint>
#include <ctime>

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

extern "C" RUNTIME_EXPORT DEVICE int64_t
DateAddHighPrecisionNullable(const DateaddField field,
                             const int64_t number,
                             const int64_t timeval,
                             const int32_t dim,
                             const int64_t null_val);

#endif  // QUERYENGINE_DATEADD_H
