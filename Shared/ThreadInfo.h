/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>

struct ThreadInfo {
  int64_t num_threads{0};
  int64_t num_elems_per_thread;

  ThreadInfo(const int64_t max_thread_count,
             const int64_t num_elems,
             const int64_t target_elems_per_thread) {
    num_threads =
        std::min(std::max(max_thread_count, int64_t(1)),
                 ((num_elems + target_elems_per_thread - 1) / target_elems_per_thread));
    num_elems_per_thread =
        std::max((num_elems + num_threads - 1) / num_threads, int64_t(1));
  }
};
