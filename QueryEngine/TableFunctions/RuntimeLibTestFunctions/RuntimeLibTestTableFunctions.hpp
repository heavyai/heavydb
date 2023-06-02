/*
 * Copyright 2023 HEAVY.AI, Inc.
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

#include "../../../QueryEngine/heavydbTypes.h"
#include "TestRuntimeLib.h"

// clang-format off
/*
  UDTF: ct_test_runtime_libs_add__cpu_template_(TableFunctionManager, Column<T>, Column<T>) -> Column<T>, T=[int64_t, double]
  UDTF: ct_test_runtime_libs_sub__cpu_template_(TableFunctionManager, Column<T>, Column<T>) -> Column<T>, T=[int64_t, double]
*/
// clang-format on
template <typename T>
TEMPLATE_NOINLINE int32_t
ct_test_runtime_libs_add__cpu_template_(TableFunctionManager& mgr,
                                        const Column<T>& input1,
                                        const Column<T>& input2,
                                        Column<T>& out) {
  int64_t size = input1.size();
  mgr.set_output_row_size(size);
  for (int64_t i = 0; i < size; ++i) {
    out[i] = _test_runtime_add(input1[i], input2[i]);
  }
  return size;
}

template <typename T>
TEMPLATE_NOINLINE int32_t
ct_test_runtime_libs_sub__cpu_template_(TableFunctionManager& mgr,
                                        const Column<T>& input1,
                                        const Column<T>& input2,
                                        Column<T>& out) {
  int64_t size = input1.size();
  mgr.set_output_row_size(size);
  for (int64_t i = 0; i < size; ++i) {
    out[i] = _test_runtime_sub(input1[i], input2[i]);
  }
  return size;
}