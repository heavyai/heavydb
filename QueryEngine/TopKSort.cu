/* copyright 2017 MapD Technologies, Inc.
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

/*
 * @file    TopKSort.cu
 * @author  Minggang Yu <miyu@mapd.com>
 * @brief   Top-k sorting on streaming top-k heaps on VRAM
 *
 * Copyright (c) 2017 MapD Technologies, Inc.  All rights reserved.
 */
#include "TopKSort.h"
#include "BufferEntryUtils.h"
#include "GpuMemUtils.h"
#include "ResultSetBufferAccessors.h"
#include "StreamingTopN.h"
#include "SortUtils.cuh"

#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/partition.h>
#include <thrust/sort.h>

#include <iostream>

template <class K, class I = int32_t>
struct is_taken_entry {
  is_taken_entry(const int8_t* buff, const size_t stride) : buff_ptr(buff), key_stride(stride) {}
  __host__ __device__ bool operator()(const I index) {
    return !is_empty_entry<K>(static_cast<size_t>(index), buff_ptr, key_stride);
  }
  const int8_t* buff_ptr;
  const size_t key_stride;
};

template <class K, class I = int32_t>
struct is_null_order_entry {
  typedef I argument_type;
  is_null_order_entry(const int8_t* base, const size_t stride, const int64_t nul)
      : oe_base(base), oe_stride(stride), null_val(nul) {}
  __host__ __device__ bool operator()(const I index) {
    const auto oe_val = *reinterpret_cast<const K*>(oe_base + index * oe_stride);
    switch (sizeof(K)) {
      case 4:
        return *reinterpret_cast<const int32_t*>(&oe_val) == static_cast<int32_t>(null_val);
      case 8:
        return *reinterpret_cast<const int64_t*>(&oe_val) == null_val;
      default:
        return false;
    }
  }
  const int8_t* oe_base;
  const size_t oe_stride;
  const int64_t null_val;
};

template <typename ForwardIterator>
ForwardIterator partition_by_null(ForwardIterator first,
                                  ForwardIterator last,
                                  const int64_t null_val,
                                  const bool nulls_first,
                                  const int8_t* rows_ptr,
                                  const GroupByBufferLayoutInfo& layout) {
  if (nulls_first) {
    return (layout.col_bytes == 4)
               ? thrust::partition(
                     first, last, is_null_order_entry<int32_t>(rows_ptr + layout.col_off, layout.row_bytes, null_val))
               : thrust::partition(
                     first, last, is_null_order_entry<int64_t>(rows_ptr + layout.col_off, layout.row_bytes, null_val));
  } else {
    return (layout.col_bytes == 4)
               ? thrust::partition(
                     first,
                     last,
                     thrust::not1(is_null_order_entry<int32_t>(rows_ptr + layout.col_off, layout.row_bytes, null_val)))
               : thrust::partition(
                     first,
                     last,
                     thrust::not1(is_null_order_entry<int64_t>(rows_ptr + layout.col_off, layout.row_bytes, null_val)));
  }
}

template <class K, class I>
struct KeyFetcher {
  KeyFetcher(K* out_base, const int8_t* src_oe_base, const size_t stride, const I* indices)
      : key_base(out_base), oe_base(src_oe_base), oe_stride(stride), idx_base(indices) {}
  __host__ __device__ void operator()(const I index) {
    key_base[index] = *reinterpret_cast<const K*>(oe_base + idx_base[index] * oe_stride);
  }

  K* key_base;
  const int8_t* oe_base;
  const size_t oe_stride;
  const I* idx_base;
};

template <class K>
struct KeyReseter {
  KeyReseter(int8_t* out_base, const size_t stride, const K emp_key)
      : rows_base(out_base), key_stride(stride), empty_key(emp_key) {}
  __host__ __device__ void operator()(const size_t index) {
    K* key_ptr = reinterpret_cast<K*>(rows_base + index * key_stride);
    *key_ptr = empty_key;
  }

  int8_t* rows_base;
  const size_t key_stride;
  const K empty_key;
};

// TODO(miyu) : switch to shared version in ResultSetSortImpl.cu.
template <class K, class I>
void collect_order_entry_column(thrust::device_ptr<K>& d_oe_col_buffer,
                                const int8_t* d_src_buffer,
                                const thrust::device_ptr<I>& d_idx_first,
                                const size_t idx_count,
                                const size_t oe_offset,
                                const size_t oe_stride) {
  thrust::for_each(thrust::make_counting_iterator(size_t(0)),
                   thrust::make_counting_iterator(idx_count),
                   KeyFetcher<K, I>(thrust::raw_pointer_cast(d_oe_col_buffer),
                                    d_src_buffer + oe_offset,
                                    oe_stride,
                                    thrust::raw_pointer_cast(d_idx_first)));
}

template <class K, class I>
void sort_indices_by_key(thrust::device_ptr<I> d_idx_first,
                         const size_t idx_count,
                         const thrust::device_ptr<K>& d_key_buffer,
                         const bool desc,
                         ThrustAllocator& allocator) {
  if (desc) {
    thrust::sort_by_key(
        thrust::device(allocator), d_key_buffer, d_key_buffer + idx_count, d_idx_first, thrust::greater<K>());
  } else {
    thrust::sort_by_key(thrust::device(allocator), d_key_buffer, d_key_buffer + idx_count, d_idx_first);
  }
}

template <class I = int32_t>
void do_radix_sort(thrust::device_ptr<I> d_idx_first,
                   const size_t idx_count,
                   const int8_t* d_src_buffer,
                   const PodOrderEntry& oe,
                   const GroupByBufferLayoutInfo& layout,
                   ThrustAllocator& allocator) {
  const auto& oe_type = layout.oe_target_info.sql_type;
  if (oe_type.is_fp()) {
    switch (layout.col_bytes) {
      case 4: {
        auto d_oe_buffer = get_device_ptr<float>(idx_count, allocator);
        collect_order_entry_column(d_oe_buffer, d_src_buffer, d_idx_first, idx_count, layout.col_off, layout.row_bytes);
        sort_indices_by_key(d_idx_first, idx_count, d_oe_buffer, oe.is_desc, allocator);
      } break;
      case 8: {
        auto d_oe_buffer = get_device_ptr<double>(idx_count, allocator);
        collect_order_entry_column(d_oe_buffer, d_src_buffer, d_idx_first, idx_count, layout.col_off, layout.row_bytes);
        sort_indices_by_key(d_idx_first, idx_count, d_oe_buffer, oe.is_desc, allocator);
      } break;
      default:
        CHECK(false);
    }
    return;
  }
  CHECK(oe_type.is_number() || oe_type.is_time());
  switch (layout.col_bytes) {
    case 4: {
      auto d_oe_buffer = get_device_ptr<int32_t>(idx_count, allocator);
      collect_order_entry_column(d_oe_buffer, d_src_buffer, d_idx_first, idx_count, layout.col_off, layout.row_bytes);
      sort_indices_by_key(d_idx_first, idx_count, d_oe_buffer, oe.is_desc, allocator);
    } break;
    case 8: {
      auto d_oe_buffer = get_device_ptr<int64_t>(idx_count, allocator);
      collect_order_entry_column(d_oe_buffer, d_src_buffer, d_idx_first, idx_count, layout.col_off, layout.row_bytes);
      sort_indices_by_key(d_idx_first, idx_count, d_oe_buffer, oe.is_desc, allocator);
    } break;
    default:
      CHECK(false);
  }
}

template <class I>
struct RowFetcher {
  RowFetcher(int8_t* out_base, const int8_t* in_base, const I* indices, const size_t row_sz)
      : dst_base(out_base), src_base(in_base), idx_base(indices), row_size(row_sz) {}
  __host__ __device__ void operator()(const I index) {
    memcpy(dst_base + index * row_size, src_base + idx_base[index] * row_size, row_size);
  }

  int8_t* dst_base;
  const int8_t* src_base;
  const I* idx_base;
  const size_t row_size;
};

template <typename DerivedPolicy>
void reset_keys_in_row_buffer(const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
                              int8_t* row_buffer,
                              const size_t key_width,
                              const size_t row_size,
                              const size_t first,
                              const size_t last) {
  switch (key_width) {
    case 4:
      thrust::for_each(exec,
                       thrust::make_counting_iterator(first),
                       thrust::make_counting_iterator(last),
                       KeyReseter<int32_t>(row_buffer, row_size, static_cast<int32_t>(EMPTY_KEY_32)));
      break;
    case 8:
      thrust::for_each(exec,
                       thrust::make_counting_iterator(first),
                       thrust::make_counting_iterator(last),
                       KeyReseter<int64_t>(row_buffer, row_size, static_cast<int64_t>(EMPTY_KEY_64)));
      break;
    default:
      CHECK(false);
  }
}

std::vector<int8_t> pop_n_rows_from_merged_heaps_gpu(Data_Namespace::DataMgr* data_mgr,
                                                     const int64_t* dev_heaps,
                                                     const size_t heaps_size,
                                                     const size_t n,
                                                     const PodOrderEntry& oe,
                                                     const GroupByBufferLayoutInfo& layout,
                                                     const size_t group_key_bytes,
                                                     const size_t thread_count,
                                                     const int device_id) {
  const auto row_size = layout.row_bytes;
  CHECK_EQ(heaps_size, streaming_top_n::get_heap_size(row_size, n, thread_count));
  const int8_t* rows_ptr =
      reinterpret_cast<const int8_t*>(dev_heaps) + streaming_top_n::get_rows_offset_of_heaps(n, thread_count);
  const auto total_entry_count = n * thread_count;
  ThrustAllocator thrust_allocator(data_mgr, device_id);
  auto d_indices = get_device_ptr<int32_t>(total_entry_count, thrust_allocator);
  thrust::sequence(d_indices, d_indices + total_entry_count);
  auto separator =
      (group_key_bytes == 4)
          ? thrust::partition(d_indices, d_indices + total_entry_count, is_taken_entry<int32_t>(rows_ptr, row_size))
          : thrust::partition(d_indices, d_indices + total_entry_count, is_taken_entry<int64_t>(rows_ptr, row_size));
  const size_t actual_entry_count = separator - d_indices;
  if (!actual_entry_count) {
    std::vector<int8_t> top_rows(row_size * n);
    reset_keys_in_row_buffer(thrust::host, &top_rows[0], layout.col_bytes, row_size, 0, n);
    return top_rows;
  }

  const auto& oe_type = layout.oe_target_info.sql_type;
  if (oe_type.get_notnull()) {
    do_radix_sort(d_indices, actual_entry_count, rows_ptr, oe, layout, thrust_allocator);
  } else {
    auto separator = partition_by_null(d_indices,
                                       d_indices + actual_entry_count,
                                       null_val_bit_pattern(oe_type, false),
                                       oe.nulls_first,
                                       rows_ptr,
                                       layout);
    if (oe.nulls_first) {
      const size_t null_count = separator - d_indices;
      if (null_count < actual_entry_count) {
        do_radix_sort(separator, actual_entry_count - null_count, rows_ptr, oe, layout, thrust_allocator);
      }
    } else {
      const size_t nonnull_count = separator - d_indices;
      if (nonnull_count > 0) {
        do_radix_sort(d_indices, nonnull_count, rows_ptr, oe, layout, thrust_allocator);
      }
    }
  }

  const auto final_entry_count = std::min(n, actual_entry_count);
  auto d_top_rows = get_device_ptr<int8_t>(row_size * n, thrust_allocator);
  thrust::for_each(thrust::make_counting_iterator(size_t(0)),
                   thrust::make_counting_iterator(final_entry_count),
                   RowFetcher<int32_t>(
                       thrust::raw_pointer_cast(d_top_rows), rows_ptr, thrust::raw_pointer_cast(d_indices), row_size));

  if (final_entry_count < n) {
    reset_keys_in_row_buffer(
        thrust::device, thrust::raw_pointer_cast(d_top_rows), layout.col_bytes, row_size, final_entry_count, n);
  }

  std::vector<int8_t> top_rows(row_size * n);
  thrust::copy(d_top_rows, d_top_rows + row_size * n, top_rows.begin());
  return top_rows;
}
