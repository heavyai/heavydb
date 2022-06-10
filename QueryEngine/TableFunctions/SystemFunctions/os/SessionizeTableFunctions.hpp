/*
 * Copyright 2022 HEAVY.AI, Inc.
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

#pragma once

#ifndef __CUDACC__
#ifdef HAVE_TBB

#include "QueryEngine/heavydbTypes.h"
#include "Shared/ThreadInfo.h"

#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_sort.h>
#include <algorithm>
#include <vector>

inline int32_t get_elapsed_seconds(const Timestamp& start, const Timestamp& end) {
  return (end.time - start.time) / 1000000000;
}

// clang-format off
/*
  UDTF: tf_compute_dwell_times__cpu_template(TableFunctionManager,
  Cursor<Column<I> entity_id, Column<S> session_id, Column<Timestamp> ts> data,
  int64_t min_dwell_points | require="min_dwell_points >= 0",
  int64_t min_dwell_seconds | require="min_dwell_seconds >= 0",
  int64_t max_inactive_seconds | require="max_inactive_seconds >= 0") | filter_table_function_transpose=on ->
  Column<I> entity_id | input_id=args<0>, Column<S> site_id | input_id=args<1>, Column<S> prev_site_id | input_id=args<1>, 
  Column<S> next_site_id | input_id=args<1>, Column<int32_t> session_id, Column<int32_t> start_seq_id,
  Column<Timestamp> ts, Column<int32_t> dwell_time_sec, Column<int32_t> num_dwell_points, 
  I=[TextEncodingDict, int64_t], S=[TextEncodingDict, int64_t]
 */
// clang-format on

template <typename I, typename S>
NEVER_INLINE HOST int32_t
tf_compute_dwell_times__cpu_template(TableFunctionManager& mgr,
                                     const Column<I>& input_id,
                                     const Column<S>& input_site_id,
                                     const Column<Timestamp>& input_ts,
                                     const int64_t min_dwell_points,
                                     const int64_t min_dwell_seconds,
                                     const int64_t max_inactive_seconds,
                                     Column<I>& output_id,
                                     Column<S>& output_site_id,
                                     Column<S>& output_prev_site_id,
                                     Column<S>& output_next_site_id,
                                     Column<int32_t>& output_session_id,
                                     Column<int32_t>& output_start_seq_id,
                                     Column<Timestamp>& output_start_ts,
                                     Column<int32_t>& output_dwell_time_sec,
                                     Column<int32_t>& output_dwell_points) {
  auto func_timer = DEBUG_TIMER(__func__);

  const I id_null_val = inline_null_value<I>();
  const S site_id_null_val = inline_null_value<S>();

  const int32_t num_input_rows = input_id.size();

  // Short circuit early both to avoid unneeded computation and
  // also eliminate the need to ifguard against empty input sets
  // below

  if (num_input_rows == 0) {
    return 0;
  }

  std::vector<int32_t> permutation_idxs(num_input_rows);
  {
    auto permute_creation_timer = DEBUG_TIMER("Create permutation index");
    tbb::parallel_for(tbb::blocked_range<int32_t>(0, num_input_rows),
                      [&](const tbb::blocked_range<int32_t>& r) {
                        const int32_t r_end = r.end();
                        for (int32_t p = r.begin(); p < r_end; ++p) {
                          permutation_idxs[p] = p;
                        }
                      });
  }

  {
    auto permute_sort_timer = DEBUG_TIMER("Sort permutation index");
    // Sort permutation_idx in ascending order
    tbb::parallel_sort(permutation_idxs.begin(),
                       permutation_idxs.begin() + num_input_rows,
                       [&](const int32_t& a, const int32_t& b) {
                         return input_id[a] == input_id[b] ? input_ts[a] < input_ts[b]
                                                           : input_id[a] < input_id[b];
                       });
  }

  constexpr int64_t target_rows_per_thread{20000};
  const ThreadInfo thread_info(
      std::thread::hardware_concurrency(), num_input_rows, target_rows_per_thread);
  CHECK_GT(thread_info.num_threads, 0);
  std::vector<int32_t> per_thread_session_counts(thread_info.num_threads, 0);
  std::vector<std::pair<int32_t, int32_t>> per_thread_actual_idx_offsets(
      thread_info.num_threads);
  std::vector<std::future<void>> worker_threads;
  int32_t start_row_idx = 0;
  {
    // First we count number of dwell sessions found and start and end input and
    // output offsets per thread
    auto pre_flight_dwell_count_timer = DEBUG_TIMER("Pre-flight Dwell Count");
    for (int32_t thread_idx = 0; thread_idx < thread_info.num_threads; ++thread_idx) {
      const int32_t end_row_idx =
          std::min(start_row_idx + thread_info.num_elems_per_thread,
                   static_cast<int64_t>(num_input_rows));
      worker_threads.emplace_back(std::async(
          std::launch::async,
          [&, min_dwell_points, min_dwell_seconds, num_input_rows, max_inactive_seconds](
              const auto start_idx, const auto end_idx, const auto thread_idx) {
            int32_t thread_session_count = per_thread_session_counts[thread_idx];
            // First find first new index
            int32_t first_valid_idx = start_idx;
            // First partition reads from beginning
            if (start_idx > 0) {
              I first_id = input_id[permutation_idxs[first_valid_idx]];
              for (; first_valid_idx < end_idx; ++first_valid_idx) {
                const int32_t permuted_idx = permutation_idxs[first_valid_idx];
                if (!input_id.isNull(permuted_idx) &&
                    input_id[permuted_idx] != first_id) {
                  break;
                }
              }
            }
            per_thread_actual_idx_offsets[thread_idx].first = first_valid_idx;

            auto last_id = input_id[permutation_idxs[first_valid_idx]];
            auto last_site_id = input_site_id[permutation_idxs[first_valid_idx]];

            int32_t i = first_valid_idx;
            int32_t session_num_points = 0;
            auto session_start_ts = input_ts[permutation_idxs[i]];
            auto last_ts = input_ts[permutation_idxs[i]];
            for (; i < end_idx; ++i) {
              const auto permuted_idx = permutation_idxs[i];
              const auto& id = input_id[permuted_idx];
              const auto& site_id = input_site_id[permuted_idx];
              const auto& current_ts = input_ts[permuted_idx];
              if (id != last_id || site_id != last_site_id ||
                  get_elapsed_seconds(last_ts, current_ts) > max_inactive_seconds) {
                if (last_id != id_null_val && last_site_id != site_id_null_val) {
                  if (session_num_points >= min_dwell_points &&
                      get_elapsed_seconds(session_start_ts, last_ts) >=
                          min_dwell_seconds) {
                    thread_session_count++;
                  }
                }
                session_num_points = 1;
                session_start_ts = current_ts;
              } else {
                session_num_points++;
              }

              last_id = id;
              last_site_id = site_id;
              last_ts = current_ts;
            }

            CHECK_EQ(i, end_idx);

            if (end_idx < num_input_rows) {
              const int32_t max_transitions = (input_id[permutation_idxs[end_idx]] !=
                                               input_id[permutation_idxs[end_idx - 1]])
                                                  ? 2
                                                  : 1;
              int32_t transition_count = 0;
              for (; i < num_input_rows; ++i) {
                const auto permuted_idx = permutation_idxs[i];
                const auto& id = input_id[permuted_idx];
                const auto& site_id = input_site_id[permuted_idx];
                const auto& current_ts = input_ts[permuted_idx];

                if (id != last_id || site_id != last_site_id ||
                    get_elapsed_seconds(last_ts, current_ts) > max_inactive_seconds) {
                  if (id != last_id) {
                    transition_count++;
                    if (transition_count == max_transitions) {
                      break;
                    }
                  }
                  if (last_id != id_null_val && last_site_id != site_id_null_val) {
                    if (session_num_points >= min_dwell_points &&
                        get_elapsed_seconds(session_start_ts, last_ts) >=
                            min_dwell_seconds) {
                      thread_session_count++;
                    }
                  }
                  last_id = id;
                  last_site_id = site_id;
                  session_num_points = 1;
                  session_start_ts = current_ts;
                } else {
                  session_num_points++;
                }
                last_ts = current_ts;
              }
            }
            if (last_id != id_null_val && last_site_id != site_id_null_val) {
              if (session_num_points >= min_dwell_points &&
                  get_elapsed_seconds(session_start_ts, last_ts) >= min_dwell_seconds) {
                thread_session_count++;
              }
            }
            per_thread_actual_idx_offsets[thread_idx].second = i;
            per_thread_session_counts[thread_idx] = thread_session_count;
          },
          start_row_idx,
          end_row_idx,
          thread_idx));

      start_row_idx += thread_info.num_elems_per_thread;
    }
  }
  for (auto& worker_thread : worker_threads) {
    worker_thread.wait();
  }
  worker_threads.clear();

  // Now compute a prefix_sum
  std::vector<int32_t> session_counts_prefix_sums(thread_info.num_threads + 1);
  session_counts_prefix_sums[0] = 0;
  for (int32_t thread_idx = 0; thread_idx < thread_info.num_threads; ++thread_idx) {
    session_counts_prefix_sums[thread_idx + 1] =
        session_counts_prefix_sums[thread_idx] + per_thread_session_counts[thread_idx];
  }
  const auto num_output_rows = session_counts_prefix_sums[thread_info.num_threads];
  mgr.set_output_row_size(num_output_rows);
  if (num_output_rows == 0) {
    return num_output_rows;
  }

  {
    // Now actually compute the dwell times and other attributes, using the per-thread
    // computed input and output offsets computed above
    auto dwell_calc_timer = DEBUG_TIMER("Dwell Calc");
    for (int32_t thread_idx = 0; thread_idx < thread_info.num_threads; ++thread_idx) {
      const int32_t start_row_idx = per_thread_actual_idx_offsets[thread_idx].first;
      const int32_t end_row_idx = per_thread_actual_idx_offsets[thread_idx].second;
      const int32_t output_start_offset = session_counts_prefix_sums[thread_idx];
      const int32_t num_sessions = session_counts_prefix_sums[thread_idx + 1] -
                                   session_counts_prefix_sums[thread_idx];
      worker_threads.emplace_back(std::async(
          std::launch::async,
          [&input_id,
           &input_site_id,
           &input_ts,
           &output_id,
           &output_site_id,
           &output_session_id,
           &output_start_seq_id,
           &output_start_ts,
           &output_dwell_time_sec,
           &output_dwell_points,
           &permutation_idxs,
           &id_null_val,
           &site_id_null_val,
           min_dwell_points,
           min_dwell_seconds,
           max_inactive_seconds](const auto start_row_idx,
                                 const auto end_row_idx,
                                 const auto output_start_offset,
                                 const auto num_sessions) {
            if (!(end_row_idx > start_row_idx)) {
              return;
            }
            int32_t output_offset = output_start_offset;
            int32_t session_start_seq_id = 1;
            int32_t session_seq_id = 1;
            int32_t session_id = 1;
            auto last_id = input_id[permutation_idxs[start_row_idx]];
            auto last_site_id = input_site_id[permutation_idxs[start_row_idx]];
            auto session_start_ts = input_ts[permutation_idxs[start_row_idx]];
            auto last_ts = input_ts[permutation_idxs[start_row_idx]];
            for (int32_t idx = start_row_idx; idx < end_row_idx; ++idx) {
              const auto permuted_idx = permutation_idxs[idx];
              const auto& id = input_id[permuted_idx];
              const auto& site_id = input_site_id[permuted_idx];
              const auto& current_ts = input_ts[permuted_idx];
              if (id != last_id || site_id != last_site_id ||
                  get_elapsed_seconds(last_ts, current_ts) > max_inactive_seconds) {
                if (last_id != id_null_val && last_site_id != site_id_null_val) {
                  const int32_t num_dwell_points = session_seq_id - session_start_seq_id;
                  const int32_t num_dwell_seconds =
                      get_elapsed_seconds(session_start_ts, last_ts);
                  if (num_dwell_points >= min_dwell_points &&
                      num_dwell_seconds >= min_dwell_seconds) {
                    output_id[output_offset] = last_id;
                    output_site_id[output_offset] = last_site_id;
                    output_session_id[output_offset] = session_id++;
                    output_start_seq_id[output_offset] = session_start_seq_id;
                    output_start_ts[output_offset] = session_start_ts;
                    output_dwell_time_sec[output_offset] = num_dwell_seconds;
                    output_dwell_points[output_offset] = num_dwell_points;
                    output_offset++;
                  }
                }
                last_site_id = site_id;
                session_start_ts = input_ts[permuted_idx];
                if (id != last_id) {
                  last_id = id;
                  session_start_seq_id = 1;
                  session_seq_id = 1;
                  session_id = 1;
                } else {
                  session_start_seq_id = session_seq_id;
                }
              }
              session_seq_id++;
              last_ts = current_ts;
            }
            if (last_id != id_null_val && last_site_id != site_id_null_val) {
              const int32_t num_dwell_points = session_seq_id - session_start_seq_id;
              const int32_t num_dwell_seconds =
                  get_elapsed_seconds(session_start_ts, last_ts);
              if (num_dwell_points >= min_dwell_points &&
                  num_dwell_seconds >= min_dwell_seconds) {
                output_id[output_offset] = last_id;
                output_site_id[output_offset] = last_site_id;
                output_session_id[output_offset] = session_id++;
                output_start_seq_id[output_offset] = session_start_seq_id;
                output_start_ts[output_offset] = session_start_ts;
                output_dwell_time_sec[output_offset] = num_dwell_seconds;
                output_dwell_points[output_offset] = num_dwell_points;
                output_offset++;
              }
            }
            CHECK_EQ(output_offset - output_start_offset, num_sessions);
          },
          start_row_idx,
          end_row_idx,
          output_start_offset,
          num_sessions));
    }
  }
  for (auto& worker_thread : worker_threads) {
    worker_thread.wait();
  }

  {
    output_prev_site_id[0] = site_id_null_val;
    output_next_site_id[0] = num_output_rows > 1 && output_id[0] == output_id[1]
                                 ? output_site_id[1]
                                 : site_id_null_val;
    output_prev_site_id[num_output_rows - 1] =
        num_output_rows > 1 &&
                output_id[num_output_rows - 1] == output_id[num_output_rows - 2]
            ? output_site_id[num_output_rows - 2]
            : site_id_null_val;
    output_next_site_id[num_output_rows - 1] = site_id_null_val;

    auto permute_creation_timer = DEBUG_TIMER("Fill lagged and lead site ids");
    tbb::parallel_for(tbb::blocked_range<int32_t>(1, num_output_rows - 1),
                      [&](const tbb::blocked_range<int32_t>& r) {
                        const int32_t r_end = r.end();
                        for (int32_t p = r.begin(); p < r_end; ++p) {
                          output_prev_site_id[p] = output_id[p] == output_id[p - 1]
                                                       ? output_site_id[p - 1]
                                                       : site_id_null_val;
                          output_next_site_id[p] = output_id[p] == output_id[p + 1]
                                                       ? output_site_id[p + 1]
                                                       : site_id_null_val;
                        }
                      });
  }

  return num_output_rows;
}

#endif  // #ifdef HAVE_TBB
#endif  // #ifndef __CUDACC__
