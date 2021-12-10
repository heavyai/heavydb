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
#endif

#include "UtilityTableFunctions.h"

EXTENSION_NOINLINE
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

EXTENSION_NOINLINE
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

EXTENSION_NOINLINE
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

#endif  //__CUDACC__
