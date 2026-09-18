/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <QueryRenderer/QueryDataLayout.h>
#include "VegaRenderTestCudaSetup.h"

__global__ void setup_kernel_gpu_wrapper(curandState* state, int seed) {
  int idx = threadIdx.x;
  curand_init(seed, idx, 0, &state[idx]);
}

__global__ void get_random_simple_point_row_data_gpu_wrapper(curandState* state,
                                                             SimplePointRow* row,
                                                             int numPts,
                                                             double xmin,
                                                             double xmax,
                                                             double ymin,
                                                             double ymax,
                                                             double valmin,
                                                             double valmax,
                                                             int64_t partymin,
                                                             int64_t partymax) {
  int idx = threadIdx.x;

  curandState localState = state[idx];

  row[idx].key = idx;
  row[idx].x = xmin + curand_uniform(&localState) * (xmax - xmin);
  row[idx].y = ymin + curand_uniform(&localState) * (ymax - ymin);
  row[idx].val = valmin + curand_uniform(&localState) * (valmax - valmin);
  row[idx].party =
      partymin +
      int64_t(floor(curand_uniform(&localState) * float(partymax - partymin + 1) -
                    0.0000001));
  row[idx].rowid = idx;

  state[idx] = localState;
}

__global__ void get_random_numbers_gpu_wrapper(curandState* state, float* results) {
  int idx = threadIdx.x;
  curandState localState = state[idx];

  results[idx] = curand_uniform(&localState);

  // copy local state back to global state
  state[idx] = localState;
}

std::shared_ptr<::QueryRenderer::QueryDataLayout> SimplePointRow::getQueryDataLayout() {
  std::shared_ptr<::QueryRenderer::QueryDataLayout> rtn(
      new ::QueryRenderer::QueryDataLayout(
          {::QueryRenderer::QueryDataLayout::AttrAliasInfo("key",
                                                           SQLTypeInfo(kBIGINT, true)),
           ::QueryRenderer::QueryDataLayout::AttrAliasInfo("x",
                                                           SQLTypeInfo(kDOUBLE, true)),
           ::QueryRenderer::QueryDataLayout::AttrAliasInfo("y",
                                                           SQLTypeInfo(kDOUBLE, true)),
           ::QueryRenderer::QueryDataLayout::AttrAliasInfo("val",
                                                           SQLTypeInfo(kDOUBLE, true)),
           ::QueryRenderer::QueryDataLayout::AttrAliasInfo("party",
                                                           SQLTypeInfo(kBIGINT, true)),
           ::QueryRenderer::QueryDataLayout::AttrAliasInfo("rowid",
                                                           SQLTypeInfo(kBIGINT, true))}));
  return rtn;
}

void SimplePointRow::setup_kernel(curandState* state,
                                  const size_t block_size_x,
                                  const size_t grid_size_x,
                                  int seed) {
  setup_kernel_gpu_wrapper<<<grid_size_x, block_size_x>>>(state, seed);
}

void SimplePointRow::get_random_data(curandState* state,
                                     SimplePointRow* row,
                                     int numPts,
                                     const std::array<double, 2>& xExtents,
                                     const std::array<double, 2>& yExtents,
                                     const std::array<double, 2>& valExtents,
                                     const std::array<int64_t, 2>& partyExtents,
                                     const size_t block_size_x,
                                     const size_t grid_size_x) {
  get_random_simple_point_row_data_gpu_wrapper<<<grid_size_x, block_size_x>>>(
      state,
      row,
      numPts,
      xExtents[0],
      xExtents[1],
      yExtents[0],
      yExtents[1],
      valExtents[0],
      valExtents[1],
      partyExtents[0],
      partyExtents[1]);
}
