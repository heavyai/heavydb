/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef THREADING_H
#define THREADING_H

#include "threading_serial.h"  // includes threading_std.h, useful to include unconditionally for debugging purposes
#if DISABLE_CONCURRENCY
namespace threading = threading_serial;
#elif ENABLE_TBB
#include "threading_tbb.h"
namespace threading = threading_tbb;
#else
namespace threading = threading_std;
#endif

#endif  // THREADING_H
