/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <thread>

extern unsigned g_cpu_threads_override;
extern size_t g_max_import_threads;

inline int cpu_threads() {
  auto ov = g_cpu_threads_override;
  return (ov <= 0) ? std::max(2 * std::thread::hardware_concurrency(), 1U) : ov;
}

namespace import_export {
inline size_t num_import_threads(const int32_t copy_params_threads) {
  if (copy_params_threads > 0) {
    return static_cast<size_t>(copy_params_threads);
  }
  return std::min(static_cast<size_t>(std::thread::hardware_concurrency()),
                  g_max_import_threads);
}
}  // namespace import_export
