/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include "../QueryEngine/ExtractFromTime.h"

namespace DateConverters {

inline int64_t get_epoch_days_from_seconds(const int64_t seconds) {
  return (seconds < 0 && seconds % kSecsPerDay != 0) ? (seconds / kSecsPerDay) - 1
                                                     : seconds / kSecsPerDay;
}

inline int64_t get_epoch_seconds_from_days(const int64_t days) {
  return days * kSecsPerDay;
}

}  // namespace DateConverters
