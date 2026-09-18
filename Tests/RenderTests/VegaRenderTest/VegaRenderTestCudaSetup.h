/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef VEGARENDERTESTCUDASETUP_H_
#define VEGARENDERTESTCUDASETUP_H_

#include <curand.h>
#include <curand_kernel.h>
#include <cstdint>
#include <memory>

namespace QueryRenderer {
class QueryDataLayout;
}

struct SimplePointRow {
  int64_t key;
  double x;
  double y;
  double val;
  int64_t party;
  int64_t rowid;

  static std::shared_ptr<::QueryRenderer::QueryDataLayout> getQueryDataLayout();
  static void setup_kernel(curandState* state,
                           const size_t block_size_x,
                           const size_t grid_size_x,
                           int seed);
  static void get_random_data(curandState* state,
                              SimplePointRow* row,
                              int numPts,
                              const std::array<double, 2>& xExtents,
                              const std::array<double, 2>& yExtents,
                              const std::array<double, 2>& valExtents,
                              const std::array<int64_t, 2>& partyExtents,
                              const size_t block_size_x,
                              const size_t grid_size_x);
};

#endif  // VEGARENDERTESTCUDASETUP_H_
