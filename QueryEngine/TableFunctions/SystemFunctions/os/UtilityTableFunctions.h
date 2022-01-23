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

#pragma once

#include "QueryEngine/OmniSciTypes.h"

#ifndef __CUDACC__

EXTENSION_NOINLINE_HOST
int32_t generate_series_parallel(const int64_t start,
                                 const int64_t stop,
                                 const int64_t step,
                                 Column<int64_t>& series_output);
// clang-format off
/*
  UDTF: generate_series__cpu_1(TableFunctionManager, int64_t start, int64_t stop, int64_t step | require="step != 0") -> Column<int64_t> generate_series
  UDTF: generate_series__cpu_2(TableFunctionManager, int64_t start, int64_t stop) -> Column<int64_t> generate_series
*/
// clang-format on

EXTENSION_NOINLINE_HOST
int32_t generate_series__cpu_1(TableFunctionManager& mgr,
                               const int64_t start,
                               const int64_t stop,
                               const int64_t step,
                               Column<int64_t>& series_output);

EXTENSION_NOINLINE_HOST
int32_t generate_series__cpu_2(TableFunctionManager& mgr,
                               const int64_t start,
                               const int64_t stop,
                               Column<int64_t>& series_output);

// clang-format off
/*
  UDTF: generate_random_strings__cpu_(TableFunctionManager, int64_t num_strings, int64_t string_length) -> Column<int64_t> id, Column<TextEncodingDict> rand_str | input_id=args<>
*/
// clang-format on

EXTENSION_NOINLINE_HOST
int32_t generate_random_strings__cpu_(TableFunctionManager& mgr,
                                      const int64_t num_strings,
                                      const int64_t string_length,
                                      Column<int64_t>& output_id,
                                      Column<TextEncodingDict>& output_strings);

#endif  // __CUDACC__
