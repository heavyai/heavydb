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
