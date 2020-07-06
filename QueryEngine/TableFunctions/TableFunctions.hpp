/*
 * Copyright 2019 OmniSci, Inc.
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

#include "../../QueryEngine/OmniSciTypes.h"
#include "../../Shared/funcannotations.h"

#define EXTENSION_INLINE extern "C" ALWAYS_INLINE DEVICE
#define EXTENSION_NOINLINE extern "C" NEVER_INLINE DEVICE

EXTENSION_NOINLINE int32_t row_copier(Column<double> input_col,
                                      int copy_multiplier,
                                      Column<double> output_col) {
  int32_t output_row_count = copy_multiplier * input_col.sz;
  if (output_row_count > 100) {
    // Test failure propagation.
    return -1;
  }
  // Set the output columne size for consistency. The output column
  // size will be effective only here, it will not propagate back to
  // the caller because the output_col is passed in by value.
  output_col.sz = output_row_count;

#ifdef __CUDACC__
  int32_t start = threadIdx.x + blockDim.x * blockIdx.x;
  int32_t stop = static_cast<int32_t>(input_col.sz);
  int32_t step = blockDim.x * gridDim.x;
#else
  auto start = 0;
  auto stop = input_col.sz;
  auto step = 1;
#endif

  for (auto i = start; i < stop; i += step) {
    for (int c = 0; c < copy_multiplier; c++) {
      output_col.ptr[i + (c * input_col.sz)] = input_col.ptr[i];
    }
  }

  return output_row_count;
}
