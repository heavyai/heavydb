/*
 * Copyright 2017 MapD Technologies, Inc.
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

#include "ThrustAllocator.h"

#include <cstdint>
#include <glog/logging.h>

#ifdef HAVE_CUDA
#include "InPlaceSortImpl.h"
#endif

void sort_groups_gpu(int64_t* val_buff,
                     int32_t* idx_buff,
                     const uint64_t entry_count,
                     const bool desc,
                     const uint32_t chosen_bytes,
                     ThrustAllocator& alloc) {
#ifdef HAVE_CUDA
  switch (chosen_bytes) {
    case 4:
    case 8:
      sort_on_gpu(val_buff, idx_buff, entry_count, desc, chosen_bytes, alloc);
      break;
    default:
      CHECK(false);
  }
#endif
}

void sort_groups_cpu(int64_t* val_buff,
                     int32_t* idx_buff,
                     const uint64_t entry_count,
                     const bool desc,
                     const uint32_t chosen_bytes) {
#ifdef HAVE_CUDA
  switch (chosen_bytes) {
    case 4:
    case 8:
      sort_on_cpu(val_buff, idx_buff, entry_count, desc, chosen_bytes);
      break;
    default:
      CHECK(false);
  }
#endif
}

void apply_permutation_gpu(int64_t* val_buff,
                           int32_t* idx_buff,
                           const uint64_t entry_count,
                           const uint32_t chosen_bytes,
                           ThrustAllocator& alloc) {
#ifdef HAVE_CUDA
  switch (chosen_bytes) {
    case 4:
    case 8:
      apply_permutation_on_gpu(val_buff, idx_buff, entry_count, chosen_bytes, alloc);
      break;
    default:
      CHECK(false);
  }
#endif
}

void apply_permutation_cpu(int64_t* val_buff,
                           int32_t* idx_buff,
                           const uint64_t entry_count,
                           int64_t* tmp_buff,
                           const uint32_t chosen_bytes) {
#ifdef HAVE_CUDA
  switch (chosen_bytes) {
    case 4:
    case 8:
      apply_permutation_on_cpu(val_buff, idx_buff, entry_count, tmp_buff, chosen_bytes);
      break;
    default:
      CHECK(false);
  }
#endif
}
