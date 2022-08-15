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

#include "QueryEngine/heavydbTypes.h"

#ifndef __CUDACC__

// Note that "start" is a reserved keyword in SQL, so to allow usage as a named
// parameter without quoting we added the "series_" prefix to all args

// clang-format off
/*
  UDTF: generate_series__cpu_template(TableFunctionManager, int64_t series_start, int64_t series_stop, int64_t series_step | require="series_step != 0") -> Column<int64_t> generate_series
  UDTF: generate_series__cpu_template(TableFunctionManager, T series_start, T series_stop) -> Column<T> generate_series, T=[int64_t]
  UDTF: generate_series__cpu_template(TableFunctionManager, Timestamp series_start, Timestamp series_stop, T series_step | require="series_step != 0") -> Column<Timestamp> generate_series, T=[DayTimeInterval, YearMonthTimeInterval]
*/
// clang-format on

template <typename T, typename K>
NEVER_INLINE HOST int32_t generate_series__cpu_template(TableFunctionManager& mgr,
                                                        const T start,
                                                        const T stop,
                                                        const K step,
                                                        Column<T>& series_output);

template <typename T>
NEVER_INLINE HOST int32_t generate_series__cpu_template(TableFunctionManager& mgr,
                                                        const T start,
                                                        const T stop,
                                                        Column<T>& series_output);

// clang-format off
/*
  UDTF: generate_random_strings__cpu_(TableFunctionManager,
                                      int64_t num_strings | require="num_strings >= 0",
                                      int64_t string_length | require="string_length > 0") ->
                                    Column<int64_t> id, Column<TextEncodingDict> rand_str | input_id=args<>
*/
// clang-format on

EXTENSION_NOINLINE_HOST
int32_t generate_random_strings__cpu_(TableFunctionManager& mgr,
                                      const int64_t num_strings,
                                      const int64_t string_length,
                                      Column<int64_t>& output_id,
                                      Column<TextEncodingDict>& output_strings);

#endif  // __CUDACC__
