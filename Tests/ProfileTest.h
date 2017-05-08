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

/**
 * @file    ProfileTest.h
 * @author  Minggang Yu <miyu@mapd.com>
 * @brief   Unit tests for microbenchmark.
 *
 * Copyright (c) 2016 MapD Technologies, Inc.  All rights reserved.
 */

#ifndef PROFILETEST_H
#define PROFILETEST_H

#include "../QueryEngine/GpuRtConstants.h"

#ifndef __CUDACC__
#include <glog/logging.h>
#else
#include "../Shared/always_assert.h"
#endif  // __CUDACC__

#ifdef HAVE_CUDA
#include <cuda_runtime_api.h>
#endif

#if defined(HAVE_CUDA) && CUDA_VERSION >= 8000
#include <cuda.h>

#define TRY_COLUMNAR
#define TRY_MASH
#define TRY_MASH_COLUMNAR
#if defined(TRY_MASH) || defined(TRY_MASH_COLUMNAR)
#define SAVE_MASH_BUF
#endif
#endif

#include <vector>

#ifndef __CUDACC__
#include <algorithm>
#include <unistd.h>  // sysconf
inline long cpu_threads() {
  // could use std::thread::hardware_concurrency(), but some
  // slightly out-of-date compilers (gcc 4.7) implement it as always 0.
  // Play it POSIX.1 safe instead.
  return std::max(2 * sysconf(_SC_NPROCESSORS_CONF), 1L);
}
#endif

enum DEV_KIND { CPU, GPU };

enum DIST_KIND { INVALID, NRM, EXP1, EXP2, UNI, POI };

enum OP_KIND { OP_COUNT, OP_SUM, OP_MIN, OP_MAX };

#if defined(HAVE_CUDA) && CUDA_VERSION >= 8000
void init_groups_on_device(int8_t* groups,
                           const size_t group_count,
                           const size_t col_count,
                           const std::vector<size_t>& col_widths,
                           const std::vector<size_t>& init_vals,
                           const bool is_columnar);

void run_query_on_device(int8_t* groups_buffer,
                         const size_t group_count,
                         const int8_t* row_buffer,
                         const size_t row_count,
                         const size_t key_count,
                         const size_t val_count,
                         const std::vector<size_t>& col_widths,
                         const std::vector<OP_KIND>& agg_ops,
                         const bool is_columnar);
#if defined(TRY_MASH) || defined(TRY_MASH_COLUMNAR)
void mash_run_query_on_device(int8_t* groups_buffer,
                              const size_t group_count,
                              const int8_t* row_buffer,
                              const size_t row_count,
                              const size_t key_count,
                              const size_t val_count,
                              const std::vector<size_t>& col_widths,
                              const std::vector<OP_KIND>& agg_ops,
                              const bool is_columnar);
#endif
bool generate_columns_on_device(int8_t* buffers,
                                const size_t row_count,
                                const size_t col_count,
                                const std::vector<size_t>& col_widths,
                                const std::vector<std::pair<int64_t, int64_t>>& ranges,
                                const bool is_columnar,
                                const std::vector<DIST_KIND>& dists);

void columnarize_groups_on_device(int8_t* columnar_buffer,
                                  const int8_t* rowwise_buffer,
                                  const size_t group_count,
                                  const std::vector<size_t>& col_widths);

size_t deduplicate_rows_on_device(int8_t* row_buffer,
                                  const size_t row_count,
                                  const size_t key_count,
                                  const std::vector<size_t>& col_widths,
                                  const bool is_columnar);

int8_t* get_hashed_copy(int8_t* dev_buffer,
                        const size_t entry_count,
                        const size_t new_entry_count,
                        const std::vector<size_t>& col_widths,
                        const std::vector<OP_KIND>& agg_ops,
                        const std::vector<size_t>& init_vals,
                        const bool is_columnar);

size_t drop_rows(int8_t* row_buffer,
                 const size_t entry_count,
                 const size_t entry_size,
                 const size_t row_count,
                 const float fill_rate,
                 const bool is_columnar);

void reduce_on_device(int8_t*& this_dev_buffer,
                      const size_t this_dev_id,
                      size_t& this_entry_count,
                      int8_t* that_dev_buffer,
                      const size_t that_dev_id,
                      const size_t that_entry_count,
                      const size_t that_actual_row_count,
                      const std::vector<size_t>& col_widths,
                      const std::vector<OP_KIND>& agg_ops,
                      const std::vector<size_t>& init_vals,
                      const bool is_columnar);

int8_t* fetch_segs_from_others(std::vector<int8_t*>& dev_reduced_buffers,
                               const size_t entry_count,
                               const size_t dev_id,
                               const size_t dev_count,
                               const std::vector<size_t>& col_widths,
                               const bool is_columnar,
                               const size_t start,
                               const size_t end);

std::pair<int8_t*, size_t> get_perfect_hashed_copy(int8_t* dev_buffer,
                                                   const size_t entry_count,
                                                   const std::vector<size_t>& col_widths,
                                                   const std::vector<std::pair<int64_t, int64_t>>& ranges,
                                                   const std::vector<OP_KIND>& agg_ops,
                                                   const std::vector<size_t>& init_vals,
                                                   const bool is_columnar);

void reduce_segment_on_device(int8_t* dev_seg_buf,
                              const int8_t* dev_other_segs,
                              const size_t entry_count,
                              const size_t seg_count,
                              const std::vector<size_t>& col_widths,
                              const std::vector<OP_KIND>& agg_ops,
                              const bool is_columnar,
                              const size_t start,
                              const size_t end);
#endif

#endif /* PROFILETEST_H */
