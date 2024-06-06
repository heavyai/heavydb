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

#include "InPlaceSort.h"
#include "InPlaceSortImpl.h"

#include <Analyzer/Analyzer.h>
#include "DataMgr/Allocators/ThrustAllocator.h"
#include "Descriptors/QueryMemoryDescriptor.h"
#include "Logger/Logger.h"

#include <cstdint>

#ifdef HAVE_CUDA
#include <cuda.h>
#else
#include <Shared/nocuda.h>
#endif

void sort_groups_cpu(int64_t* val_buff,
                     int32_t* idx_buff,
                     const uint64_t entry_count,
                     const bool desc,
                     const uint32_t chosen_bytes) {
#ifdef HAVE_CUDA
  switch (chosen_bytes) {
    case 1:
    case 2:
    case 4:
    case 8:
      sort_on_cpu(val_buff, idx_buff, entry_count, desc, chosen_bytes);
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
    case 1:
    case 2:
    case 4:
    case 8:
      apply_permutation_on_cpu(val_buff, idx_buff, entry_count, tmp_buff, chosen_bytes);
      break;
    default:
      CHECK(false);
  }
#endif
}

namespace {

void sort_groups_gpu(int64_t* val_buff,
                     int32_t* idx_buff,
                     const uint64_t entry_count,
                     const bool desc,
                     const uint32_t chosen_bytes,
                     ThrustAllocator& alloc,
                     CUstream cuda_stream) {
#ifdef HAVE_CUDA
  switch (chosen_bytes) {
    case 1:
    case 2:
    case 4:
    case 8:
      sort_on_gpu(
          val_buff, idx_buff, entry_count, desc, chosen_bytes, alloc, cuda_stream);
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
                           ThrustAllocator& alloc,
                           CUstream cuda_stream) {
#ifdef HAVE_CUDA
  switch (chosen_bytes) {
    case 1:
    case 2:
    case 4:
    case 8:
      apply_permutation_on_gpu(
          val_buff, idx_buff, entry_count, chosen_bytes, alloc, cuda_stream);
      break;
    default:
      CHECK(false);
  }
#endif
}

}  // namespace

void inplace_sort_gpu(const std::list<Analyzer::OrderEntry>& order_entries,
                      const QueryMemoryDescriptor& query_mem_desc,
                      const GpuGroupByBuffers& group_by_buffers,
                      CudaAllocator* cuda_allocator,
                      CUstream cuda_stream) {
  CHECK(cuda_allocator);
  CHECK(cuda_allocator->getDataMgr());
  ThrustAllocator alloc(cuda_allocator->getDataMgr(), cuda_allocator->getDeviceId());
  CHECK_EQ(size_t(1), order_entries.size());
  const auto idx_buff = group_by_buffers.data -
                        align_to_int64(query_mem_desc.getEntryCount() * sizeof(int32_t));
  for (const auto& order_entry : order_entries) {
    const auto target_idx = order_entry.tle_no - 1;
    const auto val_buff =
        group_by_buffers.data + query_mem_desc.getColOffInBytes(target_idx);
    const auto chosen_bytes = query_mem_desc.getPaddedSlotWidthBytes(target_idx);
    sort_groups_gpu(reinterpret_cast<int64_t*>(val_buff),
                    reinterpret_cast<int32_t*>(idx_buff),
                    query_mem_desc.getEntryCount(),
                    order_entry.is_desc,
                    chosen_bytes,
                    alloc,
                    cuda_stream);
    if (!query_mem_desc.hasKeylessHash()) {
      apply_permutation_gpu(reinterpret_cast<int64_t*>(group_by_buffers.data),
                            reinterpret_cast<int32_t*>(idx_buff),
                            query_mem_desc.getEntryCount(),
                            sizeof(int64_t),
                            alloc,
                            cuda_stream);
    }
    for (size_t target_idx = 0; target_idx < query_mem_desc.getSlotCount();
         ++target_idx) {
      if (static_cast<int>(target_idx) == order_entry.tle_no - 1) {
        continue;
      }
      const auto chosen_bytes = query_mem_desc.getPaddedSlotWidthBytes(target_idx);
      const auto val_buff =
          group_by_buffers.data + query_mem_desc.getColOffInBytes(target_idx);
      apply_permutation_gpu(reinterpret_cast<int64_t*>(val_buff),
                            reinterpret_cast<int32_t*>(idx_buff),
                            query_mem_desc.getEntryCount(),
                            chosen_bytes,
                            alloc,
                            cuda_stream);
    }
  }
}
