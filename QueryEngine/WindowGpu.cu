/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "WindowGpu.h"

#include "DataMgr/Allocators/ThrustAllocator.h"
#include "Logger/Logger.h"

#include <cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <cmath>
#include <cstdint>

#define checkCudaErrors(err) CHECK_EQ(err, CUDA_SUCCESS)

namespace {

constexpr int32_t kWarpSize = 32;
constexpr int32_t kWarpsPerBlock = 4;
constexpr int32_t kDirectRankMaxPartitionSize = kWarpSize;
constexpr int32_t kExtremaThreadsPerBlock = 256;

struct ScopedThrustBuffer {
  ScopedThrustBuffer(ThrustAllocator& allocator, const size_t bytes)
      : allocator_(allocator), bytes_(bytes), ptr_(allocator.allocate(bytes)) {}

  ~ScopedThrustBuffer() {
    if (ptr_) {
      allocator_.deallocate(ptr_, bytes_);
    }
  }

  ScopedThrustBuffer(const ScopedThrustBuffer&) = delete;
  ScopedThrustBuffer& operator=(const ScopedThrustBuffer&) = delete;

  template <typename T>
  T* as() const {
    return reinterpret_cast<T*>(ptr_);
  }

 private:
  ThrustAllocator& allocator_;
  const size_t bytes_;
  int8_t* ptr_;
};

__global__ void fill_window_sort_positions(int32_t* positions,
                                           int32_t* partition_ids,
                                           const int32_t* offsets,
                                           const int32_t* counts,
                                           const int32_t partition_count) {
  const auto partition_idx = static_cast<int32_t>(blockIdx.x);
  if (partition_idx >= partition_count) {
    return;
  }
  const auto offset = offsets[partition_idx];
  const auto count = counts[partition_idx];
  for (int32_t i = threadIdx.x; i < count; i += blockDim.x) {
    const auto pos = offset + i;
    positions[pos] = pos;
    partition_ids[pos] = partition_idx;
  }
}

template <typename T, typename NullPatternT>
struct WindowOrderTraits {
  static __host__ __device__ bool is_null(const T val,
                                          const bool nullable,
                                          const int64_t null_pattern) {
    if (!nullable) {
      return false;
    }
    return static_cast<NullPatternT>(val) == static_cast<NullPatternT>(null_pattern);
  }
};

template <typename T, typename NullPatternT>
struct WindowNullValue {
  static __host__ __device__ T get(const int64_t null_pattern) {
    return static_cast<T>(static_cast<NullPatternT>(null_pattern));
  }
};

template <typename T>
struct WindowExtremaTraits {
  static __device__ bool is_nan(const T /*value*/) { return false; }
};

template <>
struct WindowExtremaTraits<double> {
  static __device__ bool is_nan(const double value) { return isnan(value); }
};

template <>
struct WindowNullValue<double, int64_t> {
  static __host__ __device__ double get(const int64_t null_pattern) {
#ifdef __CUDA_ARCH__
    return __longlong_as_double(null_pattern);
#else
    return *reinterpret_cast<const double*>(&null_pattern);
#endif
  }
};

template <>
struct WindowOrderTraits<float, int32_t> {
  static __host__ __device__ bool is_null(const float val,
                                          const bool nullable,
                                          const int64_t null_pattern) {
    if (!nullable) {
      return false;
    }
    const auto bits = *reinterpret_cast<const int32_t*>(&val);
    return bits == static_cast<int32_t>(null_pattern);
  }
};

template <>
struct WindowOrderTraits<double, int64_t> {
  static __host__ __device__ bool is_null(const double val,
                                          const bool nullable,
                                          const int64_t null_pattern) {
    if (!nullable) {
      return false;
    }
    const auto bits = *reinterpret_cast<const int64_t*>(&val);
    return bits == null_pattern;
  }
};

template <typename T, typename NullPatternT>
struct WindowRankSortComparator {
  const int32_t* payload;
  const int32_t* partition_ids;
  const T* order_col;
  bool nullable;
  int64_t null_pattern;
  bool desc;
  bool nulls_first;

  __host__ __device__ bool operator()(const int32_t lhs_pos,
                                      const int32_t rhs_pos) const {
    const auto lhs_partition = partition_ids[lhs_pos];
    const auto rhs_partition = partition_ids[rhs_pos];
    if (lhs_partition != rhs_partition) {
      return lhs_partition < rhs_partition;
    }

    const auto lhs_val = order_col[payload[lhs_pos]];
    const auto rhs_val = order_col[payload[rhs_pos]];
    const auto lhs_null =
        WindowOrderTraits<T, NullPatternT>::is_null(lhs_val, nullable, null_pattern);
    const auto rhs_null =
        WindowOrderTraits<T, NullPatternT>::is_null(rhs_val, nullable, null_pattern);
    if (lhs_null != rhs_null) {
      return lhs_null ? nulls_first : !nulls_first;
    }
    if (lhs_null) {
      return lhs_pos < rhs_pos;
    }
    if (lhs_val < rhs_val) {
      return !desc;
    }
    if (lhs_val > rhs_val) {
      return desc;
    }
    return lhs_pos < rhs_pos;
  }
};

template <typename T, typename NullPatternT>
__device__ int compare_window_order_values(const int32_t lhs_row,
                                           const int32_t rhs_row,
                                           const T* order_col,
                                           const bool nullable,
                                           const int64_t null_pattern,
                                           const bool desc,
                                           const bool nulls_first) {
  const auto lhs_val = order_col[lhs_row];
  const auto rhs_val = order_col[rhs_row];
  const auto lhs_null =
      WindowOrderTraits<T, NullPatternT>::is_null(lhs_val, nullable, null_pattern);
  const auto rhs_null =
      WindowOrderTraits<T, NullPatternT>::is_null(rhs_val, nullable, null_pattern);
  if (lhs_null || rhs_null) {
    if (lhs_null == rhs_null) {
      return 0;
    }
    return lhs_null ? (nulls_first ? -1 : 1) : (nulls_first ? 1 : -1);
  }
  if (lhs_val < rhs_val) {
    return desc ? 1 : -1;
  }
  if (lhs_val > rhs_val) {
    return desc ? -1 : 1;
  }
  return 0;
}

template <typename T, typename NullPatternT>
__device__ bool same_window_order_value(const int32_t lhs_pos,
                                        const int32_t rhs_pos,
                                        const int32_t* payload,
                                        const T* order_col,
                                        const bool nullable,
                                        const int64_t null_pattern) {
  return compare_window_order_values<T, NullPatternT>(payload[lhs_pos],
                                                      payload[rhs_pos],
                                                      order_col,
                                                      nullable,
                                                      null_pattern,
                                                      false,
                                                      false) == 0;
}

template <typename T, typename NullPatternT>
__global__ void fill_rank_output_for_partitions(const int32_t* sorted_positions,
                                                const int32_t* payload,
                                                const int32_t* offsets,
                                                const int32_t* counts,
                                                const int32_t partition_count,
                                                const T* order_col,
                                                const bool nullable,
                                                const int64_t null_pattern,
                                                const GpuWindowRankKind rank_kind,
                                                int64_t* output) {
  const auto partition_idx = static_cast<int32_t>(blockIdx.x);
  if (partition_idx >= partition_count || threadIdx.x != 0) {
    return;
  }

  const auto offset = offsets[partition_idx];
  const auto count = counts[partition_idx];
  int64_t rank = 1;
  int64_t dense_rank = 1;
  for (int32_t i = 0; i < count; ++i) {
    const auto sorted_idx = offset + i;
    const auto current_pos = sorted_positions[sorted_idx];
    if (i > 0) {
      const auto previous_pos = sorted_positions[sorted_idx - 1];
      if (!same_window_order_value<T, NullPatternT>(
              previous_pos, current_pos, payload, order_col, nullable, null_pattern)) {
        rank = static_cast<int64_t>(i) + 1;
        ++dense_rank;
      }
    }
    output[payload[current_pos]] =
        rank_kind == GpuWindowRankKind::Rank ? rank : dense_rank;
  }
}

template <typename T, typename NullPatternT>
__global__ void fill_partition_extrema_output(const int32_t* payload,
                                              const int32_t* offsets,
                                              const int32_t* counts,
                                              const int32_t partition_count,
                                              const T* input_col,
                                              const bool nullable,
                                              const int64_t null_pattern,
                                              const GpuWindowExtremaKind extrema_kind,
                                              T* value_output,
                                              int64_t* row_position_output) {
  __shared__ T values[kExtremaThreadsPerBlock];
  __shared__ int32_t valid[kExtremaThreadsPerBlock];
  __shared__ int32_t value_positions[kExtremaThreadsPerBlock];
  __shared__ T first_values[kExtremaThreadsPerBlock];
  __shared__ int32_t first_positions[kExtremaThreadsPerBlock];

  const auto partition_idx = static_cast<int32_t>(blockIdx.x);
  if (partition_idx >= partition_count) {
    return;
  }

  const auto offset = offsets[partition_idx];
  const auto count = counts[partition_idx];
  bool thread_valid = false;
  T thread_value{};
  int32_t thread_value_position = count;
  T thread_first_value{};
  int32_t thread_first_position = count;
  for (int32_t i = threadIdx.x; i < count; i += blockDim.x) {
    const auto row_id = payload[offset + i];
    const auto value = input_col[row_id];
    if (WindowOrderTraits<T, NullPatternT>::is_null(value, nullable, null_pattern)) {
      continue;
    }
    if (thread_first_position == count) {
      thread_first_value = value;
      thread_first_position = i;
    }
    // CPU MIN/MAX retains a NaN only when it is the first non-null value. Preserve
    // that order-sensitive behavior while keeping NaNs out of the parallel extrema.
    if (WindowExtremaTraits<T>::is_nan(value)) {
      continue;
    }
    if (!thread_valid) {
      thread_value = value;
      thread_value_position = i;
      thread_valid = true;
    } else if (extrema_kind == GpuWindowExtremaKind::Min) {
      if (value < thread_value) {
        thread_value = value;
        thread_value_position = i;
      }
    } else {
      if (value > thread_value) {
        thread_value = value;
        thread_value_position = i;
      }
    }
  }

  values[threadIdx.x] = thread_value;
  valid[threadIdx.x] = thread_valid ? 1 : 0;
  value_positions[threadIdx.x] = thread_value_position;
  first_values[threadIdx.x] = thread_first_value;
  first_positions[threadIdx.x] = thread_first_position;
  __syncthreads();

  for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      const auto other_idx = threadIdx.x + stride;
      if (first_positions[other_idx] < first_positions[threadIdx.x]) {
        first_values[threadIdx.x] = first_values[other_idx];
        first_positions[threadIdx.x] = first_positions[other_idx];
      }
      if (valid[other_idx]) {
        if (!valid[threadIdx.x]) {
          values[threadIdx.x] = values[other_idx];
          value_positions[threadIdx.x] = value_positions[other_idx];
          valid[threadIdx.x] = 1;
        } else {
          const auto other_value = values[other_idx];
          const auto current_value = values[threadIdx.x];
          const bool other_is_better = extrema_kind == GpuWindowExtremaKind::Min
                                           ? other_value < current_value
                                           : other_value > current_value;
          const bool values_are_equal =
              !(other_value < current_value) && !(other_value > current_value);
          if (other_is_better || (values_are_equal && value_positions[other_idx] <
                                                          value_positions[threadIdx.x])) {
            values[threadIdx.x] = other_value;
            value_positions[threadIdx.x] = value_positions[other_idx];
          }
        }
      }
    }
    __syncthreads();
  }

  const auto has_non_null_value = first_positions[0] < count;
  const auto first_value_is_nan =
      has_non_null_value && WindowExtremaTraits<T>::is_nan(first_values[0]);
  const auto partition_value = !has_non_null_value
                                   ? WindowNullValue<T, NullPatternT>::get(null_pattern)
                                   : (first_value_is_nan ? first_values[0] : values[0]);
  for (int32_t i = threadIdx.x; i < count; i += blockDim.x) {
    const auto partition_pos = offset + i;
    const auto row_id = payload[partition_pos];
    value_output[row_id] = partition_value;
    row_position_output[partition_pos] = row_id;
  }
}

template <typename T, typename NullPatternT>
__global__ void fill_rank_output_direct_for_small_partitions(
    const int32_t* payload,
    const int32_t* offsets,
    const int32_t* counts,
    const int32_t partition_count,
    const T* order_col,
    const bool nullable,
    const int64_t null_pattern,
    const bool desc,
    const bool nulls_first,
    const GpuWindowRankKind rank_kind,
    int64_t* output) {
  const auto warp_id = threadIdx.x / kWarpSize;
  const auto lane = threadIdx.x % kWarpSize;
  const auto partition_idx = static_cast<int32_t>(blockIdx.x) * kWarpsPerBlock + warp_id;
  if (partition_idx >= partition_count) {
    return;
  }

  const auto offset = offsets[partition_idx];
  const auto count = counts[partition_idx];
  if (lane >= count) {
    return;
  }

  const auto current_row = payload[offset + lane];
  int64_t rank = 1;
  int64_t dense_rank = 1;
  for (int32_t peer_idx = 0; peer_idx < count; ++peer_idx) {
    const auto peer_row = payload[offset + peer_idx];
    const auto peer_cmp = compare_window_order_values<T, NullPatternT>(
        peer_row, current_row, order_col, nullable, null_pattern, desc, nulls_first);
    if (peer_cmp >= 0) {
      continue;
    }
    ++rank;

    if (rank_kind == GpuWindowRankKind::DenseRank) {
      bool peer_value_already_seen = false;
      for (int32_t prior_peer_idx = 0; prior_peer_idx < peer_idx; ++prior_peer_idx) {
        const auto prior_peer_row = payload[offset + prior_peer_idx];
        if (compare_window_order_values<T, NullPatternT>(prior_peer_row,
                                                         peer_row,
                                                         order_col,
                                                         nullable,
                                                         null_pattern,
                                                         desc,
                                                         nulls_first) == 0) {
          peer_value_already_seen = true;
          break;
        }
      }
      if (!peer_value_already_seen) {
        ++dense_rank;
      }
    }
  }
  output[current_row] = rank_kind == GpuWindowRankKind::Rank ? rank : dense_rank;
}

template <typename T, typename NullPatternT>
bool compute_window_rank_on_gpu_direct_typed(const int32_t* payload,
                                             const int32_t* offsets,
                                             const int32_t* counts,
                                             const int32_t partition_count,
                                             const int8_t* order_col,
                                             const bool order_col_nullable,
                                             const int64_t order_null_pattern,
                                             const bool desc,
                                             const bool nulls_first,
                                             const GpuWindowRankKind rank_kind,
                                             int64_t* output,
                                             CUstream cuda_stream) {
  if (partition_count == 0) {
    return true;
  }

  constexpr int threads_per_block = kWarpSize * kWarpsPerBlock;
  const auto block_count = (partition_count + kWarpsPerBlock - 1) / kWarpsPerBlock;
  fill_rank_output_direct_for_small_partitions<T, NullPatternT>
      <<<block_count, threads_per_block, 0, cuda_stream>>>(
          payload,
          offsets,
          counts,
          partition_count,
          reinterpret_cast<const T*>(order_col),
          order_col_nullable,
          order_null_pattern,
          desc,
          nulls_first,
          rank_kind,
          output);
  checkCudaErrors(cuStreamSynchronize(cuda_stream));
  return true;
}

template <typename T, typename NullPatternT>
bool compute_window_rank_on_gpu_typed(const int32_t* payload,
                                      const int32_t* offsets,
                                      const int32_t* counts,
                                      const int64_t elem_count,
                                      const int32_t partition_count,
                                      const int32_t max_partition_count,
                                      const int8_t* order_col,
                                      const bool order_col_nullable,
                                      const int64_t order_null_pattern,
                                      const bool desc,
                                      const bool nulls_first,
                                      const GpuWindowRankKind rank_kind,
                                      int64_t* output,
                                      ThrustAllocator& allocator,
                                      CUstream cuda_stream) {
  if (elem_count == 0 || partition_count == 0) {
    return true;
  }
  if (max_partition_count <= kDirectRankMaxPartitionSize) {
    return compute_window_rank_on_gpu_direct_typed<T, NullPatternT>(payload,
                                                                    offsets,
                                                                    counts,
                                                                    partition_count,
                                                                    order_col,
                                                                    order_col_nullable,
                                                                    order_null_pattern,
                                                                    desc,
                                                                    nulls_first,
                                                                    rank_kind,
                                                                    output,
                                                                    cuda_stream);
  }

  ScopedThrustBuffer positions_buffer(allocator, elem_count * sizeof(int32_t));
  ScopedThrustBuffer partition_ids_buffer(allocator, elem_count * sizeof(int32_t));
  auto positions = positions_buffer.as<int32_t>();
  auto partition_ids = partition_ids_buffer.as<int32_t>();

  constexpr int threads_per_block = 256;
  fill_window_sort_positions<<<partition_count, threads_per_block, 0, cuda_stream>>>(
      positions, partition_ids, offsets, counts, partition_count);
  checkCudaErrors(cuStreamSynchronize(cuda_stream));

  auto positions_begin = thrust::device_pointer_cast(positions);
  thrust::sort(
      thrust::cuda::par(allocator).on(cuda_stream),
      positions_begin,
      positions_begin + elem_count,
      WindowRankSortComparator<T, NullPatternT>{payload,
                                                partition_ids,
                                                reinterpret_cast<const T*>(order_col),
                                                order_col_nullable,
                                                order_null_pattern,
                                                desc,
                                                nulls_first});
  checkCudaErrors(cuStreamSynchronize(cuda_stream));

  fill_rank_output_for_partitions<T, NullPatternT>
      <<<partition_count, 1, 0, cuda_stream>>>(positions,
                                               payload,
                                               offsets,
                                               counts,
                                               partition_count,
                                               reinterpret_cast<const T*>(order_col),
                                               order_col_nullable,
                                               order_null_pattern,
                                               rank_kind,
                                               output);
  checkCudaErrors(cuStreamSynchronize(cuda_stream));
  return true;
}

}  // namespace

bool compute_window_rank_on_gpu(const int32_t* payload,
                                const int32_t* offsets,
                                const int32_t* counts,
                                const int64_t elem_count,
                                const int32_t partition_count,
                                const int32_t max_partition_count,
                                const int8_t* order_col,
                                const SQLTypes order_type,
                                const int order_type_size,
                                const bool order_col_nullable,
                                const int64_t order_null_pattern,
                                const bool desc,
                                const bool nulls_first,
                                const GpuWindowRankKind rank_kind,
                                int64_t* output,
                                ThrustAllocator& allocator,
                                CUstream cuda_stream) {
  switch (order_type) {
    case kBOOLEAN:
    case kTINYINT:
      return compute_window_rank_on_gpu_typed<int8_t, int8_t>(payload,
                                                              offsets,
                                                              counts,
                                                              elem_count,
                                                              partition_count,
                                                              max_partition_count,
                                                              order_col,
                                                              order_col_nullable,
                                                              order_null_pattern,
                                                              desc,
                                                              nulls_first,
                                                              rank_kind,
                                                              output,
                                                              allocator,
                                                              cuda_stream);
    case kSMALLINT:
      return compute_window_rank_on_gpu_typed<int16_t, int16_t>(payload,
                                                                offsets,
                                                                counts,
                                                                elem_count,
                                                                partition_count,
                                                                max_partition_count,
                                                                order_col,
                                                                order_col_nullable,
                                                                order_null_pattern,
                                                                desc,
                                                                nulls_first,
                                                                rank_kind,
                                                                output,
                                                                allocator,
                                                                cuda_stream);
    case kINT:
    case kDATE:
      return order_type_size == 2
                 ? compute_window_rank_on_gpu_typed<int16_t, int16_t>(payload,
                                                                      offsets,
                                                                      counts,
                                                                      elem_count,
                                                                      partition_count,
                                                                      max_partition_count,
                                                                      order_col,
                                                                      order_col_nullable,
                                                                      order_null_pattern,
                                                                      desc,
                                                                      nulls_first,
                                                                      rank_kind,
                                                                      output,
                                                                      allocator,
                                                                      cuda_stream)
                 : compute_window_rank_on_gpu_typed<int32_t, int32_t>(payload,
                                                                      offsets,
                                                                      counts,
                                                                      elem_count,
                                                                      partition_count,
                                                                      max_partition_count,
                                                                      order_col,
                                                                      order_col_nullable,
                                                                      order_null_pattern,
                                                                      desc,
                                                                      nulls_first,
                                                                      rank_kind,
                                                                      output,
                                                                      allocator,
                                                                      cuda_stream);
    case kBIGINT:
    case kTIME:
    case kTIMESTAMP:
    case kINTERVAL_DAY_TIME:
    case kINTERVAL_YEAR_MONTH:
    case kDECIMAL:
    case kNUMERIC:
      switch (order_type_size) {
        case 1:
          return compute_window_rank_on_gpu_typed<int8_t, int8_t>(payload,
                                                                  offsets,
                                                                  counts,
                                                                  elem_count,
                                                                  partition_count,
                                                                  max_partition_count,
                                                                  order_col,
                                                                  order_col_nullable,
                                                                  order_null_pattern,
                                                                  desc,
                                                                  nulls_first,
                                                                  rank_kind,
                                                                  output,
                                                                  allocator,
                                                                  cuda_stream);
        case 2:
          return compute_window_rank_on_gpu_typed<int16_t, int16_t>(payload,
                                                                    offsets,
                                                                    counts,
                                                                    elem_count,
                                                                    partition_count,
                                                                    max_partition_count,
                                                                    order_col,
                                                                    order_col_nullable,
                                                                    order_null_pattern,
                                                                    desc,
                                                                    nulls_first,
                                                                    rank_kind,
                                                                    output,
                                                                    allocator,
                                                                    cuda_stream);
        case 4:
          return compute_window_rank_on_gpu_typed<int32_t, int32_t>(payload,
                                                                    offsets,
                                                                    counts,
                                                                    elem_count,
                                                                    partition_count,
                                                                    max_partition_count,
                                                                    order_col,
                                                                    order_col_nullable,
                                                                    order_null_pattern,
                                                                    desc,
                                                                    nulls_first,
                                                                    rank_kind,
                                                                    output,
                                                                    allocator,
                                                                    cuda_stream);
        case 8:
          return compute_window_rank_on_gpu_typed<int64_t, int64_t>(payload,
                                                                    offsets,
                                                                    counts,
                                                                    elem_count,
                                                                    partition_count,
                                                                    max_partition_count,
                                                                    order_col,
                                                                    order_col_nullable,
                                                                    order_null_pattern,
                                                                    desc,
                                                                    nulls_first,
                                                                    rank_kind,
                                                                    output,
                                                                    allocator,
                                                                    cuda_stream);
        default:
          return false;
      }
    case kFLOAT:
      return compute_window_rank_on_gpu_typed<float, int32_t>(payload,
                                                              offsets,
                                                              counts,
                                                              elem_count,
                                                              partition_count,
                                                              max_partition_count,
                                                              order_col,
                                                              order_col_nullable,
                                                              order_null_pattern,
                                                              desc,
                                                              nulls_first,
                                                              rank_kind,
                                                              output,
                                                              allocator,
                                                              cuda_stream);
    case kDOUBLE:
      return compute_window_rank_on_gpu_typed<double, int64_t>(payload,
                                                               offsets,
                                                               counts,
                                                               elem_count,
                                                               partition_count,
                                                               max_partition_count,
                                                               order_col,
                                                               order_col_nullable,
                                                               order_null_pattern,
                                                               desc,
                                                               nulls_first,
                                                               rank_kind,
                                                               output,
                                                               allocator,
                                                               cuda_stream);
    default:
      return false;
  }
}

template <typename T, typename NullPatternT>
bool compute_window_partition_extrema_on_gpu_typed(
    const int32_t* payload,
    const int32_t* offsets,
    const int32_t* counts,
    const int32_t partition_count,
    const int8_t* input_col,
    const bool input_col_nullable,
    const int64_t input_null_pattern,
    const GpuWindowExtremaKind extrema_kind,
    int8_t* value_output,
    int64_t* row_position_output,
    CUstream cuda_stream) {
  if (partition_count == 0) {
    return true;
  }
  CHECK(value_output);
  CHECK(row_position_output);
  fill_partition_extrema_output<T, NullPatternT>
      <<<partition_count, kExtremaThreadsPerBlock, 0, cuda_stream>>>(
          payload,
          offsets,
          counts,
          partition_count,
          reinterpret_cast<const T*>(input_col),
          input_col_nullable,
          input_null_pattern,
          extrema_kind,
          reinterpret_cast<T*>(value_output),
          row_position_output);
  checkCudaErrors(cuStreamSynchronize(cuda_stream));
  return true;
}

bool compute_window_partition_extrema_on_gpu(const int32_t* payload,
                                             const int32_t* offsets,
                                             const int32_t* counts,
                                             const int64_t elem_count,
                                             const int32_t partition_count,
                                             const int8_t* input_col,
                                             const SQLTypes input_type,
                                             const int input_type_size,
                                             const bool input_col_nullable,
                                             const int64_t input_null_pattern,
                                             const GpuWindowExtremaKind extrema_kind,
                                             int8_t* value_output,
                                             int64_t* row_position_output,
                                             CUstream cuda_stream) {
  if (elem_count == 0 || partition_count == 0) {
    return true;
  }
  if (input_type == kDOUBLE && input_type_size == 8) {
    return compute_window_partition_extrema_on_gpu_typed<double, int64_t>(
        payload,
        offsets,
        counts,
        partition_count,
        input_col,
        input_col_nullable,
        input_null_pattern,
        extrema_kind,
        value_output,
        row_position_output,
        cuda_stream);
  }
  if (input_type_size == 8) {
    return compute_window_partition_extrema_on_gpu_typed<int64_t, int64_t>(
        payload,
        offsets,
        counts,
        partition_count,
        input_col,
        input_col_nullable,
        input_null_pattern,
        extrema_kind,
        value_output,
        row_position_output,
        cuda_stream);
  }
  return false;
}
