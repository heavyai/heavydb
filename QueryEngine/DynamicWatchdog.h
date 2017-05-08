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

#ifndef QUERYENGINE_DYNAMICWATCHDOG_H
#define QUERYENGINE_DYNAMICWATCHDOG_H

#include <cstdint>

enum DynamicWatchdogFlags { DW_DEADLINE = 0, DW_ABORT = -1, DW_RESET = -2 };

extern "C" uint64_t dynamic_watchdog_init(unsigned ms_budget);

extern "C" bool dynamic_watchdog();

#endif  // QUERYENGINE_DYNAMICWATCHDOG_H
