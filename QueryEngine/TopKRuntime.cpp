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

/*
 * @file    TopKRuntime.cpp
 * @author  Minggang Yu <miyu@mapd.com>
 * @brief   Structures and runtime functions of streaming top-k heap
 *
 * Copyright (c) 2017 MapD Technologies, Inc.  All rights reserved.
 */
#include "../Shared/funcannotations.h"

enum class HeapOrdering { MIN, MAX };

enum class NullsOrdering { FIRST, LAST };

template <typename KeyT = int64_t, typename IndexT = int32_t>
struct KeyAccessor {
  DEVICE KeyAccessor(const int8_t* key_buff, const size_t key_stride, const size_t key_idx)
      : buffer(key_buff), stride(key_stride), index(key_idx) {}
  ALWAYS_INLINE DEVICE KeyT get(const IndexT rowid) const {
    auto keys_ptr = reinterpret_cast<const KeyT*>(buffer + stride * rowid);
    return keys_ptr[index];
  }

  const int8_t* buffer;
  const size_t stride;
  const size_t index;
};

template <typename KeyT = int64_t>
struct KeyComparator {
  DEVICE KeyComparator(const HeapOrdering hp_order,
                       const bool nullable,
                       const KeyT null_val,
                       const NullsOrdering null_order)
      : heap_ordering(hp_order), has_nulls(nullable), null_key(null_val), nulls_ordering(null_order) {}
  ALWAYS_INLINE DEVICE bool operator()(const KeyT lhs, const KeyT rhs) const {
    if (has_nulls) {
      if (nulls_ordering == NullsOrdering::FIRST) {
        if (rhs == null_key) {
          return true;
        }
        if (lhs == null_key) {
          return false;
        }
      } else {
        if (lhs == null_key) {
          return true;
        }
        if (rhs == null_key) {
          return false;
        }
      }
    }
    return heap_ordering == HeapOrdering::MIN ? (lhs < rhs) : (lhs > rhs);
  }
  const HeapOrdering heap_ordering;
  const bool has_nulls;
  const KeyT null_key;
  const NullsOrdering nulls_ordering;
};

template <typename KeyT = int64_t, typename NodeT = int64_t>
ALWAYS_INLINE DEVICE void sift_down(NodeT* heap,
                                    const size_t heap_size,
                                    const NodeT curr_idx,
                                    const KeyComparator<KeyT>& compare,
                                    const KeyAccessor<KeyT, NodeT>& accessor) {
  for (NodeT i = curr_idx, last = static_cast<NodeT>(heap_size); i < last;) {
#ifdef __CUDACC__
    const auto left_child = min(2 * i + 1, last);
    const auto right_child = min(2 * i + 2, last);
#else
    const auto left_child = std::min(2 * i + 1, last);
    const auto right_child = std::min(2 * i + 2, last);
#endif
    auto candidate_idx = last;
    if (left_child < last) {
      if (right_child < last) {
        const auto left_key = accessor.get(heap[left_child]);
        const auto right_key = accessor.get(heap[right_child]);
        candidate_idx = compare(left_key, right_key) ? left_child : right_child;
      } else {
        candidate_idx = left_child;
      }
    } else {
      candidate_idx = right_child;
    }
    if (candidate_idx >= last) {
      break;
    }
    const auto curr_key = accessor.get(heap[i]);
    const auto candidate_key = accessor.get(heap[candidate_idx]);
    if (compare(curr_key, candidate_key)) {
      break;
    }
    auto temp_id = heap[i];
    heap[i] = heap[candidate_idx];
    heap[candidate_idx] = temp_id;
    i = candidate_idx;
  }
}

template <typename KeyT = int64_t, typename NodeT = int64_t>
ALWAYS_INLINE DEVICE void sift_up(NodeT* heap,
                                  const NodeT curr_idx,
                                  const KeyComparator<KeyT>& compare,
                                  const KeyAccessor<KeyT, NodeT>& accessor) {
  for (NodeT i = curr_idx; i > 0 && (i - 1) < i;) {
    const auto parent = (i - 1) / 2;
    const auto curr_key = accessor.get(heap[i]);
    const auto parent_key = accessor.get(heap[parent]);
    if (compare(parent_key, curr_key)) {
      break;
    }
    auto temp_id = heap[i];
    heap[i] = heap[parent];
    heap[parent] = temp_id;
    i = parent;
  }
}

template <typename KeyT = int64_t, typename NodeT = int64_t>
ALWAYS_INLINE DEVICE void push_heap(int64_t* heap_ptr,
                                    int64_t* rows_ptr,
                                    NodeT& node_count,
                                    const uint32_t row_size_quad,
                                    const uint32_t key_offset,
                                    const KeyComparator<KeyT>& comparator,
                                    const KeyAccessor<KeyT, NodeT>& accessor,
                                    const KeyT curr_key) {
  const NodeT bin_index = node_count++;
  heap_ptr[bin_index] = bin_index;
  int8_t* row_ptr = reinterpret_cast<int8_t*>(rows_ptr + bin_index * row_size_quad);
  auto key_ptr = reinterpret_cast<KeyT*>(row_ptr + key_offset);
  *key_ptr = curr_key;
  // sift up
  sift_up<KeyT, NodeT>(heap_ptr, bin_index, comparator, accessor);
}

template <typename KeyT = int64_t, typename NodeT = int64_t>
ALWAYS_INLINE DEVICE bool pop_and_push_heap(int64_t* heap_ptr,
                                            int64_t* rows_ptr,
                                            const NodeT node_count,
                                            const uint32_t row_size_quad,
                                            const uint32_t key_offset,
                                            const KeyComparator<KeyT>& compare,
                                            const KeyAccessor<KeyT, NodeT>& accessor,
                                            const KeyT curr_key) {
  const NodeT top_bin_idx = static_cast<NodeT>(heap_ptr[0]);
  int8_t* top_row_ptr = reinterpret_cast<int8_t*>(rows_ptr + top_bin_idx * row_size_quad);
  auto top_key = reinterpret_cast<KeyT*>(top_row_ptr + key_offset);
  if (compare(curr_key, *top_key)) {
    return false;
  }
  // kick out
  *top_key = curr_key;
  // sift down
  sift_down<KeyT, NodeT>(heap_ptr, node_count, 0, compare, accessor);
  return true;
}

// This function only works on rowwise layout.
template <typename KeyT = int64_t>
ALWAYS_INLINE DEVICE int64_t* get_bin_from_k_heap_impl(int64_t* heaps,
                                                       const uint32_t k,
                                                       const uint32_t row_size_quad,
                                                       const uint32_t key_offset,
                                                       const bool min_heap,
                                                       const bool has_null,
                                                       const bool nulls_first,
                                                       const KeyT null_key,
                                                       const KeyT curr_key) {
  const int32_t thread_global_index = pos_start_impl(nullptr);
  const int32_t thread_count = pos_step_impl();
  int64_t& node_count = heaps[thread_global_index];
  int64_t* heap_ptr = heaps + thread_count + thread_global_index * k;
  int64_t* rows_ptr = heaps + thread_count + thread_count * k + thread_global_index * row_size_quad * k;
  KeyComparator<KeyT> compare((min_heap ? HeapOrdering::MIN : HeapOrdering::MAX),
                              has_null,
                              null_key,
                              nulls_first ? NullsOrdering::FIRST : NullsOrdering::LAST);
  KeyAccessor<KeyT, int64_t> accessor(
      reinterpret_cast<int8_t*>(rows_ptr), row_size_quad * sizeof(int64_t), key_offset / sizeof(KeyT));
  if (node_count < static_cast<int64_t>(k)) {
    push_heap(heap_ptr, rows_ptr, node_count, row_size_quad, key_offset, compare, accessor, curr_key);
    const auto last_bin_index = node_count - 1;
    auto row_ptr = rows_ptr + last_bin_index * row_size_quad;
    row_ptr[0] = last_bin_index;
    return row_ptr + 1;
  } else {
    const int64_t top_bin_idx = heap_ptr[0];
    const bool rejected =
        !pop_and_push_heap(heap_ptr, rows_ptr, node_count, row_size_quad, key_offset, compare, accessor, curr_key);
    if (rejected) {
      return nullptr;
    }
    auto row_ptr = rows_ptr + top_bin_idx * row_size_quad;
    row_ptr[0] = top_bin_idx;
    return row_ptr + 1;
  }
}

#define DEF_GET_BIN_FROM_K_HEAP(key_type)                                                              \
  extern "C" NEVER_INLINE DEVICE int64_t* get_bin_from_k_heap_##key_type(int64_t* heaps,               \
                                                                         const uint32_t k,             \
                                                                         const uint32_t row_size_quad, \
                                                                         const uint32_t key_offset,    \
                                                                         const bool min_heap,          \
                                                                         const bool has_null,          \
                                                                         const bool nulls_first,       \
                                                                         const key_type null_key,      \
                                                                         const key_type curr_key) {    \
    return get_bin_from_k_heap_impl(                                                                   \
        heaps, k, row_size_quad, key_offset, min_heap, has_null, nulls_first, null_key, curr_key);     \
  }

DEF_GET_BIN_FROM_K_HEAP(int32_t)
DEF_GET_BIN_FROM_K_HEAP(int64_t)
DEF_GET_BIN_FROM_K_HEAP(float)
DEF_GET_BIN_FROM_K_HEAP(double)
