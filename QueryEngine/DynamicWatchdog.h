/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef QUERYENGINE_DYNAMICWATCHDOG_H
#define QUERYENGINE_DYNAMICWATCHDOG_H

#include "Shared/funcannotations.h"

#include <cstdint>

enum DynamicWatchdogFlags { DW_DEADLINE = 0, DW_ABORT = -1, DW_RESET = -2 };

extern "C" RUNTIME_EXPORT uint64_t dynamic_watchdog_init(unsigned ms_budget);

extern "C" RUNTIME_EXPORT bool dynamic_watchdog();

#endif  // QUERYENGINE_DYNAMICWATCHDOG_H
