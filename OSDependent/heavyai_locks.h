/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "OSDependent/os/heavyai_locks.h"

inline bool g_read_only{false};
inline bool g_multi_instance{
    false};  // TODO(sy): set true after internal testing is complete
inline size_t g_lockfile_lock_extension_milliseconds{1000};
inline bool g_verbose_lock_logging{false};
