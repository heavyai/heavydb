/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Use if constexpr to ensure that log messages still compile, but are elided
#define DEBUG_OUTPUT_STREAM(enable_flag, stream) \
  if constexpr (!enable_flag)                    \
    ;                                            \
  else                                           \
    stream
