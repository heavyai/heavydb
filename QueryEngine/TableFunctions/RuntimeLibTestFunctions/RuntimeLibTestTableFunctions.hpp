/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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