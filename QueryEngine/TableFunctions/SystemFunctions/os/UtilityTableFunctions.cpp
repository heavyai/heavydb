/*
 * Copyright 2021 OmniSci, Inc.
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

#ifndef __CUDACC__

#include <string>

#ifdef HAVE_TBB
#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>
#endif

#include "Shared/ThreadInfo.h"
#include "UtilityTableFunctions.h"

EXTENSION_NOINLINE_HOST
#ifdef _WIN32
#pragma comment(linker "/INCLUDE:generate_series_parallel")
#else
__attribute__((__used__))
#endif
int32_t generate_series_parallel(const int64_t start,
                                 const int64_t stop,
                                 const int64_t step,
                                 Column<int64_t>& series_output) {
  const int64_t num_rows = ((stop - start) / step) + 1;

  tbb::parallel_for(tbb::blocked_range<int64_t>(0, num_rows),
                    [&](const tbb::blocked_range<int64_t>& r) {
                      const int64_t start_out_idx = r.begin();
                      const int64_t end_out_idx = r.end();
                      for (int64_t out_idx = start_out_idx; out_idx != end_out_idx;
                           ++out_idx) {
                        series_output[out_idx] = start + out_idx * step;
                      }
                    });
  return num_rows;
}

EXTENSION_NOINLINE_HOST
#ifdef _WIN32
#pragma comment(linker "/INCLUDE:generate_series__cpu_1")
#else
__attribute__((__used__))
#endif
int32_t generate_series__cpu_1(TableFunctionManager& mgr,
                               const int64_t start,
                               const int64_t stop,
                               const int64_t step,
                               Column<int64_t>& series_output) {
  const int64_t MAX_ROWS{1L << 30};
  const int64_t PARALLEL_THRESHOLD{10000L};
  const int64_t num_rows = ((stop - start) / step) + 1;
  if (num_rows <= 0) {
    mgr.set_output_row_size(0);
    return 0;
  }
  mgr.set_output_row_size(num_rows);

  if (num_rows > MAX_ROWS) {
    return mgr.ERROR_MESSAGE(
        "Invocation of generate_series would result in " + std::to_string(num_rows) +
        " rows, which exceeds the max limit of " + std::to_string(MAX_ROWS) + " rows.");
  }

#ifdef HAVE_TBB
  if (num_rows > PARALLEL_THRESHOLD) {
    return generate_series_parallel(start, stop, step, series_output);
  }
#endif

  for (int64_t out_idx = 0; out_idx != num_rows; ++out_idx) {
    series_output[out_idx] = start + out_idx * step;
  }
  return num_rows;
}

EXTENSION_NOINLINE_HOST
#ifdef _WIN32
#pragma comment(linker "/INCLUDE:generate_series__cpu_2")
#else
__attribute__((__used__))
#endif
int32_t generate_series__cpu_2(TableFunctionManager& mgr,
                               const int64_t start,
                               const int64_t stop,
                               Column<int64_t>& series_output) {
  return generate_series__cpu_1(mgr, start, stop, 1, series_output);
}

#include <chrono>
#include <random>
#include <thread>

HOST std::string gen_random_str(std::mt19937& generator, const int64_t str_len) {
  constexpr char alphanum_lookup_table[] =
      "0123456789"
      "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
      "abcdefghijklmnopqrstuvwxyz";
  constexpr size_t char_mod = sizeof(alphanum_lookup_table) - 1;
  std::uniform_int_distribution<int32_t> rand_distribution(0, char_mod);

  std::string tmp_s;
  tmp_s.reserve(str_len);
  for (int i = 0; i < str_len; ++i) {
    tmp_s += alphanum_lookup_table[rand_distribution(generator)];
  }
  return tmp_s;
}

EXTENSION_NOINLINE_HOST
#ifdef _WIN32
#pragma comment(linker "/INCLUDE:generate_random_strings__cpu_")
#else
__attribute__((__used__))
#endif
int32_t generate_random_strings__cpu_(TableFunctionManager& mgr,
                                      const int64_t num_strings,
                                      const int64_t string_length,
                                      Column<int64_t>& output_id,
                                      Column<TextEncodingDict>& output_strings) {
  auto timer = DEBUG_TIMER(__func__);
  // Check for out-of-range errors for the input parameters
  // in the function instead of with require due to issue encountered
  // with require over multiple variables
  constexpr int64_t max_strings{10000000L};
  constexpr int64_t max_str_len{10000L};
  if (num_strings > max_strings) {
    return mgr.ERROR_MESSAGE(
        "generate_random_strings: num_strings must be between 0 and 10,000,000.");
  }
  if (string_length > max_str_len) {
    return mgr.ERROR_MESSAGE(
        "generate_random_strings: string_length must be between 1 and 10,000.");
  }
  if (num_strings == 0L) {
    // Bail early as there is no work to be done
    return 0;
  }

  mgr.set_output_row_size(num_strings);
  constexpr int64_t target_strings_per_thread{5000};
  const ThreadInfo thread_info(
      std::thread::hardware_concurrency(), num_strings, target_strings_per_thread);
  std::vector<std::mt19937> per_thread_rand_generators;
  per_thread_rand_generators.reserve(thread_info.num_threads);
  for (int64_t thread_idx = 0; thread_idx < thread_info.num_threads; ++thread_idx) {
    const uint64_t seed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                              std::chrono::system_clock::now().time_since_epoch())
                              .count() +
                          thread_idx * 971;
    per_thread_rand_generators.emplace_back(seed);
  }
  std::vector<std::string> rand_strings(num_strings);
  tbb::task_arena limited_arena(thread_info.num_threads);
  limited_arena.execute([&] {
    CHECK_LE(tbb::this_task_arena::max_concurrency(), thread_info.num_threads);
    tbb::parallel_for(
        tbb::blocked_range<int64_t>(0, num_strings, thread_info.num_elems_per_thread),
        [&](const tbb::blocked_range<int64_t>& r) {
          const int64_t tbb_thread_idx = tbb::this_task_arena::current_thread_index();
          const int64_t start_out_idx = r.begin();
          const int64_t end_out_idx = r.end();
          for (int64_t out_idx = start_out_idx; out_idx != end_out_idx; ++out_idx) {
            rand_strings[out_idx] =
                gen_random_str(per_thread_rand_generators[tbb_thread_idx], string_length);
          }
        },
        tbb::simple_partitioner());
  });
  const std::vector<int32_t> rand_string_ids =
      output_strings.string_dict_proxy_->getOrAddTransientBulk(rand_strings);
  for (int64_t row_idx = 0; row_idx < num_strings; row_idx++) {
    output_id[row_idx] = row_idx;
    output_strings[row_idx] = rand_string_ids[row_idx];
  }
  return num_strings;
}

#endif  //__CUDACC__
