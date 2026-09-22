/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace import_export {

struct ParquetImportThreadPlan {
  size_t threads_per_fragment;
  size_t concurrent_fragments;
};

inline ParquetImportThreadPlan plan_parquet_import_threads(
    const size_t max_threads,
    const size_t max_threads_per_fragment,
    const size_t max_concurrent_fragments) {
  if (max_threads == 0 || max_threads_per_fragment == 0 ||
      max_concurrent_fragments == 0) {
    throw std::invalid_argument("Parquet import thread limits must be positive");
  }

  const auto balanced_thread_count =
      std::max<size_t>(1, static_cast<size_t>(std::sqrt(max_threads)));
  auto threads_per_fragment =
      std::min({max_threads, max_threads_per_fragment, balanced_thread_count});
  auto concurrent_fragments =
      std::min(max_concurrent_fragments, max_threads / threads_per_fragment);

  // Return budget that cannot be spent on more fragments to each fragment's
  // column workers.
  threads_per_fragment = std::min(
      max_threads_per_fragment, std::max<size_t>(1, max_threads / concurrent_fragments));
  concurrent_fragments = std::min(
      max_concurrent_fragments, std::max<size_t>(1, max_threads / threads_per_fragment));

  return {threads_per_fragment, concurrent_fragments};
}

}  // namespace import_export
