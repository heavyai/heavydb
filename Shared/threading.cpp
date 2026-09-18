/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "threading.h"
#include "thread_count.h"
#if DISABLE_CONCURRENCY
#elif ENABLE_TBB
namespace threading_tbb {
::tbb::task_arena g_tbb_arena;
}
#endif