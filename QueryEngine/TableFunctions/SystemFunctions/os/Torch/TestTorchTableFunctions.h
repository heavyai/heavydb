/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "QueryEngine/heavydbTypes.h"

#ifndef __CUDACC__

// clang-format off
/*
  UDTF: tf_test_runtime_torch(TableFunctionManager, Column<int64_t>) -> Column<int64_t>
  UDTF: tf_test_runtime_torch_template__template(TableFunctionManager, Column<T>) -> Column<T>, T=[int64_t, double]
  UDTF: tf_test_torch_regression(TableFunctionManager, ColumnList<double> features,
                                 int32_t batch_size | default=32,
                                 bool use_gpu | default=true, bool save_model | default=true,
                                 TextEncodingNone model_filename | default="test.pt" | require = "!save_model || model_filename.size() != 0") -> Column<double> output
  UDTF: tf_test_torch_generate_random_column(TableFunctionManager, int32_t num_elements) -> Column<double> output
  UDTF: tf_test_torch_load_model(TableFunctionManager, TextEncodingNone model_filename) -> Column<bool>
*/

// clang-format on

EXTENSION_NOINLINE
int32_t tf_test_runtime_torch(TableFunctionManager& mgr,
                              Column<int64_t>& input,
                              Column<int64_t>& output);

template <typename T>
TEMPLATE_NOINLINE int32_t
tf_test_runtime_torch_template__template(TableFunctionManager& mgr,
                                         const Column<T>& input,
                                         Column<T>& output);

EXTENSION_NOINLINE int32_t
tf_test_torch_regression(TableFunctionManager& mgr,
                         const ColumnList<double>& features,
                         int32_t batch_size,
                         bool use_gpu,
                         bool save_model,
                         const TextEncodingNone& model_filename,
                         Column<double>& output);

EXTENSION_NOINLINE int32_t tf_test_torch_generate_random_column(TableFunctionManager& mgr,
                                                                int32_t num_elements,
                                                                Column<double>& output);

EXTENSION_NOINLINE int32_t
tf_test_torch_load_model(TableFunctionManager& mgr,
                         const TextEncodingNone& model_filename,
                         Column<bool>& output);

#endif  // __CUDACC__
