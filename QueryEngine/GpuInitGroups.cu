/*
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda.h>

#include "BufferCompaction.h"
#include "GpuInitGroups.h"
#include "GpuRtConstants.h"
#include "Logger/Logger.h"
#include "MurmurHash1Inl.h"
#include "Shared/sqldefs.h"

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/transform_reduce.h>
#include <algorithm>
#include <cfloat>
#include <limits>

#define checkCudaErrors(err) CHECK_EQ(err, cudaSuccess)

namespace {

constexpr size_t compact_block_size = 256;
constexpr size_t compact_max_grid_size = 65535;
constexpr int64_t kNoTranslatedGroupbyNull = std::numeric_limits<int64_t>::min();

size_t compact_grid_size(const size_t entry_count) {
  const auto block_count =
      entry_count / compact_block_size + (entry_count % compact_block_size != 0);
  return std::max<size_t>(1, std::min(compact_max_grid_size, block_count));
}

bool is_supported_int_width(const size_t width) {
  return width == sizeof(int8_t) || width == sizeof(int16_t) ||
         width == sizeof(int32_t) || width == sizeof(int64_t);
}

__device__ bool baseline_row_is_non_empty(const int8_t* row_ptr, const size_t key_width) {
  switch (key_width) {
    case 4:
      return *reinterpret_cast<const int32_t*>(row_ptr) != EMPTY_KEY_32;
    case 8:
      return *reinterpret_cast<const int64_t*>(row_ptr) != EMPTY_KEY_64;
    default:
      return false;
  }
}

__device__ int64_t read_device_int(const int8_t* ptr, const uint8_t width) {
  switch (width) {
    case 8:
      return *reinterpret_cast<const int64_t*>(ptr);
    case 4:
      return *reinterpret_cast<const int32_t*>(ptr);
    case 2:
      return *reinterpret_cast<const int16_t*>(ptr);
    case 1:
      return *reinterpret_cast<const int8_t*>(ptr);
    default:
      return 0;
  }
}

__device__ void write_device_int(int8_t* ptr, const size_t width, const int64_t value) {
  switch (width) {
    case sizeof(int8_t):
      *reinterpret_cast<int8_t*>(ptr) = static_cast<int8_t>(value);
      break;
    case sizeof(int16_t):
      *reinterpret_cast<int16_t*>(ptr) = static_cast<int16_t>(value);
      break;
    case sizeof(int32_t):
      *reinterpret_cast<int32_t*>(ptr) = static_cast<int32_t>(value);
      break;
    case sizeof(int64_t):
      *reinterpret_cast<int64_t*>(ptr) = value;
      break;
    default:
      break;
  }
}

__device__ bool keyless_row_is_non_empty(const int8_t* row_ptr,
                                         const size_t key_slot_offset,
                                         const size_t key_slot_width,
                                         const int64_t key_init_val) {
  return read_device_int(row_ptr + key_slot_offset, key_slot_width) != key_init_val;
}

template <typename T>
__device__ bool compare_device_scalar(const T lhs, const T rhs, const int32_t op) {
  switch (op) {
    case kEQ:
    case kBW_EQ:
      return lhs == rhs;
    case kNE:
      return lhs != rhs;
    case kLT:
      return lhs < rhs;
    case kGT:
      return lhs > rhs;
    case kLE:
      return lhs <= rhs;
    case kGE:
      return lhs >= rhs;
    default:
      return false;
  }
}

__device__ bool keyless_row_matches_filter(
    const int8_t* row_ptr,
    const DeviceResultSetEntryComparison* comparisons,
    const size_t comparison_count) {
  for (size_t comparison_idx = 0; comparison_idx < comparison_count; ++comparison_idx) {
    const auto& comparison = comparisons[comparison_idx];
    const auto target_bits =
        read_device_int(row_ptr + comparison.target_offset, comparison.target_width);
    if (comparison.nullable && target_bits == comparison.null_bits) {
      return false;
    }
    if (comparison.is_fp) {
      double target_value{0.0};
      if (comparison.is_float) {
        const auto float_bits = static_cast<int32_t>(target_bits);
        target_value = static_cast<double>(*reinterpret_cast<const float*>(&float_bits));
      } else {
        target_value = *reinterpret_cast<const double*>(&target_bits);
      }
      if (!compare_device_scalar(target_value, comparison.fp_literal, comparison.op)) {
        return false;
      }
    } else if (!compare_device_scalar(
                   target_bits, comparison.int_literal, comparison.op)) {
      return false;
    }
  }
  return true;
}

__device__ bool preserved_key_matches(const int64_t key,
                                      const int64_t* preserved_keys,
                                      const size_t preserved_key_count) {
  size_t begin = 0;
  size_t end = preserved_key_count;
  while (begin < end) {
    const auto mid = begin + (end - begin) / 2;
    const auto candidate = preserved_keys[mid];
    if (candidate == key) {
      return true;
    }
    if (candidate < key) {
      begin = mid + 1;
    } else {
      end = mid;
    }
  }
  return false;
}

__device__ int64_t baseline_row_key(const int8_t* row_ptr, const size_t key_width) {
  switch (key_width) {
    case 4:
      return static_cast<int64_t>(*reinterpret_cast<const int32_t*>(row_ptr));
    case 8:
      return *reinterpret_cast<const int64_t*>(row_ptr);
    default:
      return 0;
  }
}

__device__ bool baseline_row_matches_filter(
    const int8_t* row_ptr,
    const size_t key_width,
    const DeviceResultSetEntryComparison* comparisons,
    const size_t comparison_count,
    const int64_t* preserved_keys,
    const size_t preserved_key_count) {
  if (!baseline_row_is_non_empty(row_ptr, key_width)) {
    return false;
  }
  if (keyless_row_matches_filter(row_ptr, comparisons, comparison_count)) {
    return true;
  }
  return preserved_key_count > 0 &&
         preserved_key_matches(
             baseline_row_key(row_ptr, key_width), preserved_keys, preserved_key_count);
}

__device__ bool baseline_keys_match(const int8_t* lhs,
                                    const int8_t* rhs,
                                    const size_t key_width,
                                    const size_t key_count) {
  switch (key_width) {
    case 4: {
      const auto lhs_key = reinterpret_cast<const int32_t*>(lhs);
      const auto rhs_key = reinterpret_cast<const int32_t*>(rhs);
      for (size_t key_idx = 0; key_idx < key_count; ++key_idx) {
        if (lhs_key[key_idx] != rhs_key[key_idx]) {
          return false;
        }
      }
      return true;
    }
    case 8: {
      const auto lhs_key = reinterpret_cast<const int64_t*>(lhs);
      const auto rhs_key = reinterpret_cast<const int64_t*>(rhs);
      for (size_t key_idx = 0; key_idx < key_count; ++key_idx) {
        if (lhs_key[key_idx] != rhs_key[key_idx]) {
          return false;
        }
      }
      return true;
    }
    default:
      return false;
  }
}

template <typename T>
__device__ T atomic_cas_value(T* address, const T compare, const T val);

template <>
__device__ int32_t atomic_cas_value<int32_t>(int32_t* address,
                                             const int32_t compare,
                                             const int32_t val) {
  return atomicCAS(address, compare, val);
}

template <>
__device__ int64_t atomic_cas_value<int64_t>(int64_t* address,
                                             const int64_t compare,
                                             const int64_t val) {
  return static_cast<int64_t>(atomicCAS(reinterpret_cast<unsigned long long*>(address),
                                        static_cast<unsigned long long>(compare),
                                        static_cast<unsigned long long>(val)));
}

template <typename T>
__device__ T atomic_load_value(const T* address) {
  return atomic_cas_value(const_cast<T*>(address), T{0}, T{0});
}

template <typename T>
__device__ T atomic_exchange_value(T* address, const T value);

template <>
__device__ int32_t atomic_exchange_value<int32_t>(int32_t* address, const int32_t value) {
  return atomicExch(address, value);
}

template <>
__device__ int64_t atomic_exchange_value<int64_t>(int64_t* address, const int64_t value) {
  return static_cast<int64_t>(atomicExch(reinterpret_cast<unsigned long long*>(address),
                                         static_cast<unsigned long long>(value)));
}

template <typename T>
__device__ void publish_atomic_value(T* address, const T value) {
  __threadfence();
  atomic_exchange_value(address, value);
}

template <typename T>
__device__ void atomic_add_value(T* address, const T value);

template <>
__device__ void atomic_add_value<int32_t>(int32_t* address, const int32_t value) {
  atomicAdd(address, value);
}

template <>
__device__ void atomic_add_value<int64_t>(int64_t* address, const int64_t value) {
  atomicAdd(reinterpret_cast<unsigned long long*>(address),
            static_cast<unsigned long long>(value));
}

template <>
__device__ void atomic_add_value<float>(float* address, const float value) {
  atomicAdd(address, value);
}

template <>
__device__ void atomic_add_value<double>(double* address, const double value) {
  atomicAdd(address, value);
}

template <typename T, typename Update>
__device__ void atomic_update_value(T* dest, const T source_value, Update update) {
  auto old_value = atomic_load_value(dest);
  while (true) {
    const auto new_value = update(old_value, source_value);
    const auto observed = atomic_cas_value(dest, old_value, new_value);
    if (observed == old_value) {
      return;
    }
    old_value = observed;
  }
}

template <typename T, typename Update>
__device__ void atomic_update_skip_init(T* dest,
                                        const T source_value,
                                        const T init_value,
                                        Update update) {
  if (source_value == init_value) {
    return;
  }
  auto old_value = atomic_load_value(dest);
  while (true) {
    const auto new_value =
        old_value == init_value ? source_value : update(old_value, source_value);
    const auto observed = atomic_cas_value(dest, old_value, new_value);
    if (observed == old_value) {
      return;
    }
    old_value = observed;
  }
}

template <typename T>
__device__ void reduce_baseline_hash_slot(int8_t* destination_row,
                                          const int8_t* source_row,
                                          const DeviceBaselineHashReductionSlot& slot) {
  auto dest = reinterpret_cast<T*>(destination_row + slot.offset);
  const auto source_value = *reinterpret_cast<const T*>(source_row + slot.offset);
  const auto init_value = static_cast<T>(slot.init_val);
  switch (slot.op) {
    case DeviceBaselineHashReductionSlot::Sum:
      if (slot.skip_null_val) {
        atomic_update_skip_init<T>(
            dest, source_value, init_value, [] __device__(T lhs, T rhs) {
              return static_cast<T>(lhs + rhs);
            });
      } else {
        atomic_add_value(dest, source_value);
      }
      break;
    case DeviceBaselineHashReductionSlot::Min:
      if (slot.skip_null_val) {
        atomic_update_skip_init<T>(
            dest, source_value, init_value, [] __device__(T lhs, T rhs) {
              return lhs < rhs ? lhs : rhs;
            });
      } else {
        atomic_update_value<T>(dest, source_value, [] __device__(T lhs, T rhs) {
          return lhs < rhs ? lhs : rhs;
        });
      }
      break;
    case DeviceBaselineHashReductionSlot::Max:
      if (slot.skip_null_val) {
        atomic_update_skip_init<T>(
            dest, source_value, init_value, [] __device__(T lhs, T rhs) {
              return lhs > rhs ? lhs : rhs;
            });
      } else {
        atomic_update_value<T>(dest, source_value, [] __device__(T lhs, T rhs) {
          return lhs > rhs ? lhs : rhs;
        });
      }
      break;
    default:
      break;
  }
}

template <typename FpType, typename BitsType>
__device__ void reduce_baseline_hash_fp_sum_slot(
    int8_t* destination_row,
    const int8_t* source_row,
    const DeviceBaselineHashReductionSlot& slot) {
  auto dest = reinterpret_cast<FpType*>(destination_row + slot.offset);
  const auto source_value = *reinterpret_cast<const FpType*>(source_row + slot.offset);
  const auto source_bits = *reinterpret_cast<const BitsType*>(&source_value);
  const auto init_bits = static_cast<BitsType>(slot.init_val);
  if (!slot.skip_null_val) {
    atomic_add_value(dest, source_value);
    return;
  }
  if (source_bits == init_bits) {
    return;
  }
  auto dest_bits = reinterpret_cast<BitsType*>(dest);
  auto old_bits = atomic_load_value(dest_bits);
  while (true) {
    FpType new_value;
    if (old_bits == init_bits) {
      new_value = source_value;
    } else {
      const auto old_value = *reinterpret_cast<const FpType*>(&old_bits);
      new_value = old_value + source_value;
    }
    const auto new_bits = *reinterpret_cast<const BitsType*>(&new_value);
    const auto observed = atomic_cas_value(dest_bits, old_bits, new_bits);
    if (observed == old_bits) {
      return;
    }
    old_bits = observed;
  }
}

__device__ void reduce_baseline_hash_slots(int8_t* destination_row,
                                           const int8_t* source_row,
                                           const DeviceBaselineHashReductionSlot* slots,
                                           const size_t slot_count) {
  for (size_t slot_idx = 0; slot_idx < slot_count; ++slot_idx) {
    const auto& slot = slots[slot_idx];
    if (slot.is_fp) {
      if (slot.op != DeviceBaselineHashReductionSlot::Sum) {
        continue;
      }
      switch (slot.width) {
        case sizeof(float):
          reduce_baseline_hash_fp_sum_slot<float, int32_t>(
              destination_row, source_row, slot);
          break;
        case sizeof(double):
          reduce_baseline_hash_fp_sum_slot<double, int64_t>(
              destination_row, source_row, slot);
          break;
        default:
          break;
      }
      continue;
    }
    switch (slot.width) {
      case sizeof(int32_t):
        reduce_baseline_hash_slot<int32_t>(destination_row, source_row, slot);
        break;
      case sizeof(int64_t):
        reduce_baseline_hash_slot<int64_t>(destination_row, source_row, slot);
        break;
      default:
        break;
    }
  }
}

template <typename T>
__device__ void reduce_perfect_hash_slot(int8_t* destination_row,
                                         const int8_t* source_row,
                                         const DeviceBaselineHashReductionSlot& slot) {
  auto dest = reinterpret_cast<T*>(destination_row + slot.offset);
  const auto source_value = *reinterpret_cast<const T*>(source_row + slot.offset);
  const auto init_value = static_cast<T>(slot.init_val);
  if (slot.skip_null_val && source_value == init_value) {
    return;
  }
  const auto dest_value = *dest;
  switch (slot.op) {
    case DeviceBaselineHashReductionSlot::Sum:
      if (slot.skip_null_val && dest_value == init_value) {
        *dest = source_value;
      } else {
        *dest = static_cast<T>(dest_value + source_value);
      }
      break;
    case DeviceBaselineHashReductionSlot::Min:
      if (slot.skip_null_val && dest_value == init_value) {
        *dest = source_value;
      } else {
        *dest = dest_value < source_value ? dest_value : source_value;
      }
      break;
    case DeviceBaselineHashReductionSlot::Max:
      if (slot.skip_null_val && dest_value == init_value) {
        *dest = source_value;
      } else {
        *dest = dest_value > source_value ? dest_value : source_value;
      }
      break;
    default:
      break;
  }
}

template <typename FpType, typename BitsType>
__device__ void reduce_perfect_hash_fp_sum_slot(
    int8_t* destination_row,
    const int8_t* source_row,
    const DeviceBaselineHashReductionSlot& slot) {
  auto dest = reinterpret_cast<FpType*>(destination_row + slot.offset);
  const auto source_value = *reinterpret_cast<const FpType*>(source_row + slot.offset);
  const auto source_bits = *reinterpret_cast<const BitsType*>(&source_value);
  const auto init_bits = static_cast<BitsType>(slot.init_val);
  if (slot.skip_null_val && source_bits == init_bits) {
    return;
  }
  if (!slot.skip_null_val) {
    *dest += source_value;
    return;
  }
  auto dest_bits = reinterpret_cast<BitsType*>(dest);
  if (*dest_bits == init_bits) {
    *dest = source_value;
  } else {
    *dest += source_value;
  }
}

__device__ void reduce_perfect_hash_slots(int8_t* destination_row,
                                          const int8_t* source_row,
                                          const DeviceBaselineHashReductionSlot* slots,
                                          const size_t slot_count) {
  for (size_t slot_idx = 0; slot_idx < slot_count; ++slot_idx) {
    const auto& slot = slots[slot_idx];
    if (slot.is_fp) {
      if (slot.op != DeviceBaselineHashReductionSlot::Sum) {
        continue;
      }
      switch (slot.width) {
        case sizeof(float):
          reduce_perfect_hash_fp_sum_slot<float, int32_t>(
              destination_row, source_row, slot);
          break;
        case sizeof(double):
          reduce_perfect_hash_fp_sum_slot<double, int64_t>(
              destination_row, source_row, slot);
          break;
        default:
          break;
      }
      continue;
    }
    switch (slot.width) {
      case sizeof(int32_t):
        reduce_perfect_hash_slot<int32_t>(destination_row, source_row, slot);
        break;
      case sizeof(int64_t):
        reduce_perfect_hash_slot<int64_t>(destination_row, source_row, slot);
        break;
      default:
        break;
    }
  }
}

__device__ void copy_row(int8_t* destination_row,
                         const int8_t* source_row,
                         const size_t row_size) {
  const auto row_qw_count = row_size / sizeof(int64_t);
  auto destination_qw = reinterpret_cast<int64_t*>(destination_row);
  auto source_qw = reinterpret_cast<const int64_t*>(source_row);
  for (size_t qw_idx = 0; qw_idx < row_qw_count; ++qw_idx) {
    destination_qw[qw_idx] = source_qw[qw_idx];
  }
}

__device__ void copy_baseline_hash_row_after_pending_key(int8_t* destination_row,
                                                         const int8_t* source_row,
                                                         const size_t row_size,
                                                         const size_t key_width) {
  for (size_t byte_idx = key_width; byte_idx < row_size; ++byte_idx) {
    destination_row[byte_idx] = source_row[byte_idx];
  }
}

__device__ size_t baseline_hash_row_partition(const int8_t* row_ptr,
                                              const size_t key_width,
                                              const size_t key_count,
                                              const size_t partition_count) {
  const auto hash =
      MurmurHash64AImpl(row_ptr, static_cast<int>(key_count * key_width), 0);
  return hash % partition_count;
}

__global__ void count_baseline_hash_partition_rows_kernel(const int8_t* groups_buffer,
                                                          uint64_t* partition_counts,
                                                          const size_t entry_count,
                                                          const size_t row_size,
                                                          const size_t key_width,
                                                          const size_t key_count,
                                                          const size_t partition_count) {
  const size_t start = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t step = blockDim.x * gridDim.x;
  for (size_t entry_idx = start; entry_idx < entry_count; entry_idx += step) {
    const auto row_ptr = groups_buffer + entry_idx * row_size;
    if (!baseline_row_is_non_empty(row_ptr, key_width)) {
      continue;
    }
    const auto partition_idx =
        baseline_hash_row_partition(row_ptr, key_width, key_count, partition_count);
    atomicAdd(reinterpret_cast<unsigned long long*>(&partition_counts[partition_idx]),
              1ULL);
  }
}

__global__ void partition_baseline_hash_rows_kernel(const int8_t* groups_buffer,
                                                    int8_t* partitioned_buffer,
                                                    uint64_t* partition_write_counts,
                                                    const uint64_t* partition_offsets,
                                                    const size_t entry_count,
                                                    const size_t row_size,
                                                    const size_t key_width,
                                                    const size_t key_count,
                                                    const size_t partition_count) {
  const size_t start = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t step = blockDim.x * gridDim.x;
  for (size_t entry_idx = start; entry_idx < entry_count; entry_idx += step) {
    const auto source_row = groups_buffer + entry_idx * row_size;
    if (!baseline_row_is_non_empty(source_row, key_width)) {
      continue;
    }
    const auto partition_idx =
        baseline_hash_row_partition(source_row, key_width, key_count, partition_count);
    const auto partition_row_idx = atomicAdd(
        reinterpret_cast<unsigned long long*>(&partition_write_counts[partition_idx]),
        1ULL);
    const auto output_row_idx = partition_offsets[partition_idx] + partition_row_idx;
    auto destination_row = partitioned_buffer + output_row_idx * row_size;
    copy_row(destination_row, source_row, row_size);
  }
}

template <typename KeyType>
__device__ bool insert_or_reduce_baseline_hash_row(
    int8_t* destination_buffer,
    const size_t destination_entry_count,
    const int8_t* source_row,
    const size_t row_size,
    const size_t key_width,
    const size_t key_count,
    const DeviceBaselineHashReductionSlot* slots,
    const size_t slot_count) {
  const auto source_key = *reinterpret_cast<const KeyType*>(source_row);
  const auto empty_key =
      static_cast<KeyType>(key_width == sizeof(int32_t) ? EMPTY_KEY_32 : EMPTY_KEY_64);
  const auto write_pending = static_cast<KeyType>(empty_key - 1);
  // The reducer borrows one otherwise valid key value while publishing a new row.
  // Refuse that shape before probing so no thread can mistake a stored group key for
  // an in-progress writer. The caller will use the CPU reduction path instead.
  if (source_key == write_pending) {
    return false;
  }
  const auto hash =
      MurmurHash64AImpl(source_row, static_cast<int>(key_count * key_width), 0);
  const auto start_slot = hash % destination_entry_count;
  for (size_t probe_count = 0; probe_count < destination_entry_count; ++probe_count) {
    const auto entry_idx = (start_slot + probe_count) % destination_entry_count;
    auto destination_row = destination_buffer + entry_idx * row_size;
    auto destination_key = reinterpret_cast<KeyType*>(destination_row);
    auto old_key = atomic_cas_value(destination_key, empty_key, write_pending);
    if (old_key == empty_key) {
      copy_baseline_hash_row_after_pending_key(
          destination_row, source_row, row_size, key_width);
      publish_atomic_value(destination_key, source_key);
      return true;
    }
    while (old_key == write_pending) {
      old_key = atomic_load_value(destination_key);
    }
    if (old_key == source_key &&
        baseline_keys_match(destination_row, source_row, key_width, key_count)) {
      reduce_baseline_hash_slots(destination_row, source_row, slots, slot_count);
      return true;
    }
  }
  return false;
}

__global__ void reduce_baseline_hash_rows_kernel(
    int8_t* destination_buffer,
    const size_t destination_entry_count,
    const int8_t* source_buffer,
    const size_t source_entry_count,
    const size_t row_size,
    const size_t key_width,
    const size_t key_count,
    const DeviceBaselineHashReductionSlot* slots,
    const size_t slot_count,
    int* error_code) {
  const size_t start = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t step = blockDim.x * gridDim.x;
  for (size_t entry_idx = start; entry_idx < source_entry_count; entry_idx += step) {
    const auto source_row = source_buffer + entry_idx * row_size;
    if (!baseline_row_is_non_empty(source_row, key_width)) {
      continue;
    }
    bool success{false};
    switch (key_width) {
      case sizeof(int32_t):
        success = insert_or_reduce_baseline_hash_row<int32_t>(destination_buffer,
                                                              destination_entry_count,
                                                              source_row,
                                                              row_size,
                                                              key_width,
                                                              key_count,
                                                              slots,
                                                              slot_count);
        break;
      case sizeof(int64_t):
        success = insert_or_reduce_baseline_hash_row<int64_t>(destination_buffer,
                                                              destination_entry_count,
                                                              source_row,
                                                              row_size,
                                                              key_width,
                                                              key_count,
                                                              slots,
                                                              slot_count);
        break;
      default:
        break;
    }
    if (!success) {
      atomicCAS(error_code, 0, 1);
      return;
    }
  }
}

__global__ void reduce_baseline_hash_buffers_kernel(
    int8_t* destination_buffer,
    const size_t destination_entry_count,
    const int8_t* source_buffers,
    const size_t source_entry_count,
    const size_t source_buffer_stride,
    const size_t source_buffer_count,
    const size_t row_size,
    const size_t key_width,
    const size_t key_count,
    const DeviceBaselineHashReductionSlot* slots,
    const size_t slot_count,
    int* error_code) {
  const auto total_source_entries = source_entry_count * source_buffer_count;
  const size_t start = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t step = blockDim.x * gridDim.x;
  for (size_t global_entry_idx = start; global_entry_idx < total_source_entries;
       global_entry_idx += step) {
    const auto buffer_idx = global_entry_idx / source_entry_count;
    const auto entry_idx = global_entry_idx % source_entry_count;
    const auto source_row =
        source_buffers + buffer_idx * source_buffer_stride + entry_idx * row_size;
    if (!baseline_row_is_non_empty(source_row, key_width)) {
      continue;
    }
    bool success{false};
    switch (key_width) {
      case sizeof(int32_t):
        success = insert_or_reduce_baseline_hash_row<int32_t>(destination_buffer,
                                                              destination_entry_count,
                                                              source_row,
                                                              row_size,
                                                              key_width,
                                                              key_count,
                                                              slots,
                                                              slot_count);
        break;
      case sizeof(int64_t):
        success = insert_or_reduce_baseline_hash_row<int64_t>(destination_buffer,
                                                              destination_entry_count,
                                                              source_row,
                                                              row_size,
                                                              key_width,
                                                              key_count,
                                                              slots,
                                                              slot_count);
        break;
      default:
        break;
    }
    if (!success) {
      atomicCAS(error_code, 0, 1);
      return;
    }
  }
}

__global__ void reduce_perfect_hash_rows_kernel(
    int8_t* destination_buffer,
    const int8_t* source_buffer,
    const size_t entry_count,
    const size_t row_size,
    const size_t key_width,
    const size_t key_count,
    const bool keyless,
    const size_t key_slot_offset,
    const size_t key_slot_width,
    const int64_t key_init_val,
    const DeviceBaselineHashReductionSlot* slots,
    const size_t slot_count,
    int* error_code) {
  const size_t start = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t step = blockDim.x * gridDim.x;
  for (size_t entry_idx = start; entry_idx < entry_count; entry_idx += step) {
    const auto source_row = source_buffer + entry_idx * row_size;
    const auto source_non_empty =
        keyless ? keyless_row_is_non_empty(
                      source_row, key_slot_offset, key_slot_width, key_init_val)
                : baseline_row_is_non_empty(source_row, key_width);
    if (!source_non_empty) {
      continue;
    }
    auto destination_row = destination_buffer + entry_idx * row_size;
    const auto destination_non_empty =
        keyless ? keyless_row_is_non_empty(
                      destination_row, key_slot_offset, key_slot_width, key_init_val)
                : baseline_row_is_non_empty(destination_row, key_width);
    if (!destination_non_empty) {
      copy_row(destination_row, source_row, row_size);
      continue;
    }
    if (!keyless &&
        !baseline_keys_match(destination_row, source_row, key_width, key_count)) {
      atomicCAS(error_code, 0, 1);
      continue;
    }
    reduce_perfect_hash_slots(destination_row, source_row, slots, slot_count);
  }
}

__global__ void count_non_empty_baseline_hash_rows_kernel(const int8_t* groups_buffer,
                                                          uint64_t* row_count,
                                                          const size_t entry_count,
                                                          const size_t row_size,
                                                          const size_t key_width) {
  const size_t start = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t step = blockDim.x * gridDim.x;
  uint64_t local_count = 0;
  for (size_t entry_idx = start; entry_idx < entry_count; entry_idx += step) {
    if (baseline_row_is_non_empty(groups_buffer + entry_idx * row_size, key_width)) {
      ++local_count;
    }
  }
  if (local_count) {
    atomicAdd(reinterpret_cast<unsigned long long*>(row_count),
              static_cast<unsigned long long>(local_count));
  }
}

__global__ void count_non_empty_keyless_hash_rows_kernel(const int8_t* groups_buffer,
                                                         uint64_t* row_count,
                                                         const size_t entry_count,
                                                         const size_t row_size,
                                                         const size_t key_slot_offset,
                                                         const size_t key_slot_width,
                                                         const int64_t key_init_val) {
  const size_t start = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t step = blockDim.x * gridDim.x;
  uint64_t local_count = 0;
  for (size_t entry_idx = start; entry_idx < entry_count; entry_idx += step) {
    if (keyless_row_is_non_empty(groups_buffer + entry_idx * row_size,
                                 key_slot_offset,
                                 key_slot_width,
                                 key_init_val)) {
      ++local_count;
    }
  }
  if (local_count) {
    atomicAdd(reinterpret_cast<unsigned long long*>(row_count),
              static_cast<unsigned long long>(local_count));
  }
}

__global__ void count_matching_baseline_hash_rows_kernel(
    const int8_t* groups_buffer,
    uint64_t* row_count,
    const size_t entry_count,
    const size_t row_size,
    const size_t key_width,
    const DeviceResultSetEntryComparison* comparisons,
    const size_t comparison_count,
    const int64_t* preserved_keys,
    const size_t preserved_key_count) {
  const size_t start = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t step = blockDim.x * gridDim.x;
  uint64_t local_count = 0;
  for (size_t entry_idx = start; entry_idx < entry_count; entry_idx += step) {
    if (baseline_row_matches_filter(groups_buffer + entry_idx * row_size,
                                    key_width,
                                    comparisons,
                                    comparison_count,
                                    preserved_keys,
                                    preserved_key_count)) {
      ++local_count;
    }
  }
  if (local_count) {
    atomicAdd(reinterpret_cast<unsigned long long*>(row_count),
              static_cast<unsigned long long>(local_count));
  }
}

__global__ void count_baseline_hash_rows_excluding_keys_kernel(
    const int8_t* groups_buffer,
    uint64_t* row_count,
    const size_t entry_count,
    const size_t row_size,
    const size_t key_width,
    const int64_t* excluded_keys,
    const size_t excluded_key_count) {
  const size_t start = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t step = blockDim.x * gridDim.x;
  uint64_t local_count = 0;
  for (size_t entry_idx = start; entry_idx < entry_count; entry_idx += step) {
    const auto source_row = groups_buffer + entry_idx * row_size;
    if (baseline_row_is_non_empty(source_row, key_width) &&
        !preserved_key_matches(
            baseline_row_key(source_row, key_width), excluded_keys, excluded_key_count)) {
      ++local_count;
    }
  }
  if (local_count) {
    atomicAdd(reinterpret_cast<unsigned long long*>(row_count),
              static_cast<unsigned long long>(local_count));
  }
}

__global__ void compact_baseline_hash_rows_kernel(const int8_t* groups_buffer,
                                                  int8_t* compacted_buffer,
                                                  uint64_t* compacted_row_count,
                                                  const size_t entry_count,
                                                  const size_t row_size,
                                                  const size_t key_width) {
  const size_t start = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t step = blockDim.x * gridDim.x;
  const size_t row_qw_count = row_size / sizeof(int64_t);
  for (size_t entry_idx = start; entry_idx < entry_count; entry_idx += step) {
    const auto source_row = groups_buffer + entry_idx * row_size;
    if (!baseline_row_is_non_empty(source_row, key_width)) {
      continue;
    }
    const auto compacted_idx =
        atomicAdd(reinterpret_cast<unsigned long long*>(compacted_row_count),
                  static_cast<unsigned long long>(1));
    auto compacted_row = compacted_buffer + compacted_idx * row_size;
    auto source_qw = reinterpret_cast<const int64_t*>(source_row);
    auto compacted_qw = reinterpret_cast<int64_t*>(compacted_row);
    for (size_t qw_idx = 0; qw_idx < row_qw_count; ++qw_idx) {
      compacted_qw[qw_idx] = source_qw[qw_idx];
    }
  }
}

__global__ void compact_matching_baseline_hash_rows_kernel(
    const int8_t* groups_buffer,
    int8_t* compacted_buffer,
    uint64_t* compacted_row_count,
    const size_t entry_count,
    const size_t row_size,
    const size_t key_width,
    const DeviceResultSetEntryComparison* comparisons,
    const size_t comparison_count,
    const int64_t* preserved_keys,
    const size_t preserved_key_count) {
  const size_t start = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t step = blockDim.x * gridDim.x;
  const size_t row_qw_count = row_size / sizeof(int64_t);
  for (size_t entry_idx = start; entry_idx < entry_count; entry_idx += step) {
    const auto source_row = groups_buffer + entry_idx * row_size;
    if (!baseline_row_matches_filter(source_row,
                                     key_width,
                                     comparisons,
                                     comparison_count,
                                     preserved_keys,
                                     preserved_key_count)) {
      continue;
    }
    const auto compacted_idx =
        atomicAdd(reinterpret_cast<unsigned long long*>(compacted_row_count),
                  static_cast<unsigned long long>(1));
    auto compacted_row = compacted_buffer + compacted_idx * row_size;
    auto source_qw = reinterpret_cast<const int64_t*>(source_row);
    auto compacted_qw = reinterpret_cast<int64_t*>(compacted_row);
    for (size_t qw_idx = 0; qw_idx < row_qw_count; ++qw_idx) {
      compacted_qw[qw_idx] = source_qw[qw_idx];
    }
  }
}

__global__ void compact_keyless_hash_rows_kernel(const int8_t* groups_buffer,
                                                 int8_t* compacted_buffer,
                                                 uint64_t* compacted_row_count,
                                                 uint64_t* compacted_entry_indices,
                                                 const size_t entry_count,
                                                 const size_t row_size,
                                                 const size_t key_slot_offset,
                                                 const size_t key_slot_width,
                                                 const int64_t key_init_val) {
  const size_t start = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t step = blockDim.x * gridDim.x;
  const size_t row_qw_count = row_size / sizeof(int64_t);
  for (size_t entry_idx = start; entry_idx < entry_count; entry_idx += step) {
    const auto source_row = groups_buffer + entry_idx * row_size;
    if (!keyless_row_is_non_empty(
            source_row, key_slot_offset, key_slot_width, key_init_val)) {
      continue;
    }
    const auto compacted_idx =
        atomicAdd(reinterpret_cast<unsigned long long*>(compacted_row_count),
                  static_cast<unsigned long long>(1));
    if (compacted_entry_indices) {
      compacted_entry_indices[compacted_idx] = entry_idx;
    }
    auto compacted_row = compacted_buffer + compacted_idx * row_size;
    auto source_qw = reinterpret_cast<const int64_t*>(source_row);
    auto compacted_qw = reinterpret_cast<int64_t*>(compacted_row);
    for (size_t qw_idx = 0; qw_idx < row_qw_count; ++qw_idx) {
      compacted_qw[qw_idx] = source_qw[qw_idx];
    }
  }
}

__global__ void compact_matching_keyless_hash_rows_kernel(
    int8_t* groups_buffer,
    int8_t* compacted_buffer,
    uint64_t* compacted_row_count,
    uint64_t* compacted_entry_indices,
    const size_t entry_count,
    const size_t row_size,
    const size_t key_slot_offset,
    const size_t key_slot_width,
    const int64_t key_init_val,
    const DeviceResultSetEntryComparison* comparisons,
    const size_t comparison_count) {
  const size_t start = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t step = blockDim.x * gridDim.x;
  const size_t row_qw_count = row_size / sizeof(int64_t);
  for (size_t entry_idx = start; entry_idx < entry_count; entry_idx += step) {
    auto source_row = groups_buffer + entry_idx * row_size;
    if (!keyless_row_is_non_empty(
            source_row, key_slot_offset, key_slot_width, key_init_val)) {
      continue;
    }
    if (!keyless_row_matches_filter(source_row, comparisons, comparison_count)) {
      write_device_int(source_row + key_slot_offset, key_slot_width, key_init_val);
      continue;
    }
    const auto compacted_idx =
        atomicAdd(reinterpret_cast<unsigned long long*>(compacted_row_count),
                  static_cast<unsigned long long>(1));
    if (compacted_entry_indices) {
      compacted_entry_indices[compacted_idx] = entry_idx;
    }
    auto compacted_row = compacted_buffer + compacted_idx * row_size;
    auto source_qw = reinterpret_cast<const int64_t*>(source_row);
    auto compacted_qw = reinterpret_cast<int64_t*>(compacted_row);
    for (size_t qw_idx = 0; qw_idx < row_qw_count; ++qw_idx) {
      compacted_qw[qw_idx] = source_qw[qw_idx];
    }
  }
}

__global__ void synthesize_perfect_hash_group_key_column_kernel(
    const uint64_t* entry_indices,
    int8_t* columnar_buffer,
    const size_t entry_count,
    const size_t output_width,
    const int64_t min_val,
    const int64_t bucket,
    const int64_t source_null_val,
    const int64_t normalized_null_val) {
  const size_t start = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t step = blockDim.x * gridDim.x;
  for (size_t output_idx = start; output_idx < entry_count; output_idx += step) {
    const auto source_entry_idx = static_cast<int64_t>(entry_indices[output_idx]);
    const auto raw_value = min_val + source_entry_idx * bucket;
    const auto output_value =
        source_null_val != kNoTranslatedGroupbyNull && raw_value == source_null_val
            ? normalized_null_val
            : raw_value;
    write_device_int(
        columnar_buffer + output_idx * output_width, output_width, output_value);
  }
}

__global__ void compact_baseline_hash_rows_excluding_keys_kernel(
    const int8_t* groups_buffer,
    int8_t* compacted_buffer,
    uint64_t* compacted_row_count,
    const size_t entry_count,
    const size_t row_size,
    const size_t key_width,
    const int64_t* excluded_keys,
    const size_t excluded_key_count) {
  const size_t start = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t step = blockDim.x * gridDim.x;
  const size_t row_qw_count = row_size / sizeof(int64_t);
  for (size_t entry_idx = start; entry_idx < entry_count; entry_idx += step) {
    const auto source_row = groups_buffer + entry_idx * row_size;
    if (!baseline_row_is_non_empty(source_row, key_width) ||
        preserved_key_matches(
            baseline_row_key(source_row, key_width), excluded_keys, excluded_key_count)) {
      continue;
    }
    const auto compacted_idx =
        atomicAdd(reinterpret_cast<unsigned long long*>(compacted_row_count),
                  static_cast<unsigned long long>(1));
    auto compacted_row = compacted_buffer + compacted_idx * row_size;
    auto source_qw = reinterpret_cast<const int64_t*>(source_row);
    auto compacted_qw = reinterpret_cast<int64_t*>(compacted_row);
    for (size_t qw_idx = 0; qw_idx < row_qw_count; ++qw_idx) {
      compacted_qw[qw_idx] = source_qw[qw_idx];
    }
  }
}

__global__ void compact_baseline_hash_rows_matching_keys_kernel(
    int8_t* groups_buffer,
    int8_t* compacted_buffer,
    uint64_t* compacted_row_count,
    const size_t entry_count,
    const size_t row_size,
    const size_t key_width,
    const int64_t* matching_keys,
    const size_t matching_key_count,
    const bool clear_matching_keys) {
  const size_t start = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t step = blockDim.x * gridDim.x;
  const size_t row_qw_count = row_size / sizeof(int64_t);
  for (size_t entry_idx = start; entry_idx < entry_count; entry_idx += step) {
    auto source_row = groups_buffer + entry_idx * row_size;
    if (!baseline_row_is_non_empty(source_row, key_width) ||
        !preserved_key_matches(
            baseline_row_key(source_row, key_width), matching_keys, matching_key_count)) {
      continue;
    }
    const auto compacted_idx =
        atomicAdd(reinterpret_cast<unsigned long long*>(compacted_row_count),
                  static_cast<unsigned long long>(1));
    auto compacted_row = compacted_buffer + compacted_idx * row_size;
    auto source_qw = reinterpret_cast<const int64_t*>(source_row);
    auto compacted_qw = reinterpret_cast<int64_t*>(compacted_row);
    for (size_t qw_idx = 0; qw_idx < row_qw_count; ++qw_idx) {
      compacted_qw[qw_idx] = source_qw[qw_idx];
    }
    if (clear_matching_keys) {
      switch (key_width) {
        case 4:
          *reinterpret_cast<int32_t*>(source_row) = EMPTY_KEY_32;
          break;
        case 8:
          *reinterpret_cast<int64_t*>(source_row) = EMPTY_KEY_64;
          break;
        default:
          break;
      }
    }
  }
}

__global__ void extract_fixed_width_column_from_rows_kernel(
    const int8_t* rowwise_buffer,
    int8_t* columnar_buffer,
    const size_t entry_count,
    const size_t row_size,
    const size_t source_offset,
    const size_t source_width,
    const size_t output_width,
    const int64_t dict_entry_count,
    const int64_t source_null_val,
    const int64_t normalized_null_val) {
  const size_t start = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t step = blockDim.x * gridDim.x;
  const bool normalize_dict_string_null = dict_entry_count >= 0;
  const bool normalize_source_null = source_null_val != kNoTranslatedGroupbyNull;
  for (size_t entry_idx = start; entry_idx < entry_count; entry_idx += step) {
    const auto source = rowwise_buffer + entry_idx * row_size + source_offset;
    auto dest = columnar_buffer + entry_idx * output_width;
    if (!normalize_dict_string_null && !normalize_source_null &&
        source_width == output_width) {
      switch (output_width) {
        case sizeof(int8_t):
          *reinterpret_cast<int8_t*>(dest) = *reinterpret_cast<const int8_t*>(source);
          break;
        case sizeof(int16_t):
          *reinterpret_cast<int16_t*>(dest) = *reinterpret_cast<const int16_t*>(source);
          break;
        case sizeof(int32_t):
          *reinterpret_cast<int32_t*>(dest) = *reinterpret_cast<const int32_t*>(source);
          break;
        case sizeof(int64_t):
          *reinterpret_cast<int64_t*>(dest) = *reinterpret_cast<const int64_t*>(source);
          break;
        default:
          for (size_t byte_idx = 0; byte_idx < output_width; ++byte_idx) {
            dest[byte_idx] = source[byte_idx];
          }
          break;
      }
      continue;
    }

    int64_t value{0};
    switch (source_width) {
      case sizeof(int8_t):
        value = *reinterpret_cast<const int8_t*>(source);
        break;
      case sizeof(int16_t):
        value = *reinterpret_cast<const int16_t*>(source);
        break;
      case sizeof(int32_t):
        value = *reinterpret_cast<const int32_t*>(source);
        break;
      case sizeof(int64_t):
        value = *reinterpret_cast<const int64_t*>(source);
        break;
    }
    if (normalize_source_null && value == source_null_val) {
      value = normalized_null_val;
    } else if (normalize_dict_string_null && (value < 0 || value >= dict_entry_count)) {
      value = normalized_null_val;
    }
    switch (output_width) {
      case sizeof(int8_t):
        *reinterpret_cast<int8_t*>(dest) = static_cast<int8_t>(value);
        break;
      case sizeof(int16_t):
        *reinterpret_cast<int16_t*>(dest) = static_cast<int16_t>(value);
        break;
      case sizeof(int32_t):
        *reinterpret_cast<int32_t*>(dest) = static_cast<int32_t>(value);
        break;
      case sizeof(int64_t):
        *reinterpret_cast<int64_t*>(dest) = value;
        break;
    }
  }
}

template <typename T>
struct DeviceColumnStatsAccumulator {
  T min;
  T max;
  bool has_nulls;
  bool has_values;
};

template <typename T>
__host__ __device__ T device_stats_max();

template <>
__host__ __device__ int8_t device_stats_max<int8_t>() {
  return INT8_MAX;
}

template <>
__host__ __device__ int16_t device_stats_max<int16_t>() {
  return INT16_MAX;
}

template <>
__host__ __device__ int32_t device_stats_max<int32_t>() {
  return INT32_MAX;
}

template <>
__host__ __device__ int64_t device_stats_max<int64_t>() {
  return INT64_MAX;
}

template <>
__host__ __device__ float device_stats_max<float>() {
  return FLT_MAX;
}

template <>
__host__ __device__ double device_stats_max<double>() {
  return DBL_MAX;
}

template <typename T>
__host__ __device__ T device_stats_lowest();

template <>
__host__ __device__ int8_t device_stats_lowest<int8_t>() {
  return INT8_MIN;
}

template <>
__host__ __device__ int16_t device_stats_lowest<int16_t>() {
  return INT16_MIN;
}

template <>
__host__ __device__ int32_t device_stats_lowest<int32_t>() {
  return INT32_MIN;
}

template <>
__host__ __device__ int64_t device_stats_lowest<int64_t>() {
  return INT64_MIN;
}

template <>
__host__ __device__ float device_stats_lowest<float>() {
  return -FLT_MAX;
}

template <>
__host__ __device__ double device_stats_lowest<double>() {
  return -DBL_MAX;
}

template <typename T>
struct DeviceColumnStatsTransform {
  T null_val;

  __host__ __device__ DeviceColumnStatsAccumulator<T> operator()(const T val) const {
    if (val == null_val) {
      return DeviceColumnStatsAccumulator<T>{
          device_stats_max<T>(), device_stats_lowest<T>(), true, false};
    }
    return DeviceColumnStatsAccumulator<T>{val, val, false, true};
  }
};

template <typename T>
struct DeviceColumnStatsReduce {
  __host__ __device__ DeviceColumnStatsAccumulator<T> operator()(
      const DeviceColumnStatsAccumulator<T>& lhs,
      const DeviceColumnStatsAccumulator<T>& rhs) const {
    DeviceColumnStatsAccumulator<T> result{
        device_stats_max<T>(), device_stats_lowest<T>(), false, false};
    result.has_nulls = lhs.has_nulls || rhs.has_nulls;
    result.has_values = lhs.has_values || rhs.has_values;
    if (lhs.has_values && rhs.has_values) {
      result.min = lhs.min < rhs.min ? lhs.min : rhs.min;
      result.max = lhs.max > rhs.max ? lhs.max : rhs.max;
    } else if (lhs.has_values) {
      result.min = lhs.min;
      result.max = lhs.max;
    } else if (rhs.has_values) {
      result.min = rhs.min;
      result.max = rhs.max;
    }
    return result;
  }
};

template <typename T>
DeviceColumnStatsAccumulator<T> compute_columnar_fragment_stats_typed(
    const int8_t* column_buffer,
    const size_t entry_count,
    const T null_val,
    CUstream cuda_stream) {
  CHECK(column_buffer);
  if (!entry_count) {
    return DeviceColumnStatsAccumulator<T>{
        device_stats_max<T>(), device_stats_lowest<T>(), false, false};
  }
  auto begin = thrust::device_pointer_cast(reinterpret_cast<const T*>(column_buffer));
  auto end = begin + entry_count;
  const DeviceColumnStatsAccumulator<T> init{
      device_stats_max<T>(), device_stats_lowest<T>(), false, false};
  return thrust::transform_reduce(thrust::cuda::par.on(cuda_stream),
                                  begin,
                                  end,
                                  DeviceColumnStatsTransform<T>{null_val},
                                  init,
                                  DeviceColumnStatsReduce<T>{});
}

}  // namespace

template <typename T>
__device__ int8_t* init_columnar_buffer(T* buffer_ptr,
                                        const T init_val,
                                        const uint32_t entry_count,
                                        const int32_t start,
                                        const int32_t step) {
  for (int32_t i = start; i < entry_count; i += step) {
    buffer_ptr[i] = init_val;
  }
  return reinterpret_cast<int8_t*>(buffer_ptr + entry_count);
}

extern "C" __device__ void init_columnar_group_by_buffer_gpu_impl(
    int64_t* groups_buffer,
    const int64_t* init_vals,
    const uint32_t groups_buffer_entry_count,
    const uint32_t key_count,
    const uint32_t agg_col_count,
    const int8_t* col_sizes,
    const bool need_padding,
    const bool keyless,
    const int8_t key_size) {
  const int32_t start = blockIdx.x * blockDim.x + threadIdx.x;
  const int32_t step = blockDim.x * gridDim.x;

  int8_t* buffer_ptr = reinterpret_cast<int8_t*>(groups_buffer);
  if (!keyless) {
    for (uint32_t i = 0; i < key_count; ++i) {
      switch (key_size) {
        case 1:
          buffer_ptr = init_columnar_buffer<int8_t>(
              buffer_ptr, EMPTY_KEY_8, groups_buffer_entry_count, start, step);
          break;
        case 2:
          buffer_ptr =
              init_columnar_buffer<int16_t>(reinterpret_cast<int16_t*>(buffer_ptr),
                                            EMPTY_KEY_16,
                                            groups_buffer_entry_count,
                                            start,
                                            step);
          break;
        case 4:
          buffer_ptr =
              init_columnar_buffer<int32_t>(reinterpret_cast<int32_t*>(buffer_ptr),
                                            EMPTY_KEY_32,
                                            groups_buffer_entry_count,
                                            start,
                                            step);
          break;
        case 8:
          buffer_ptr =
              init_columnar_buffer<int64_t>(reinterpret_cast<int64_t*>(buffer_ptr),
                                            EMPTY_KEY_64,
                                            groups_buffer_entry_count,
                                            start,
                                            step);
          break;
        default:
          // FIXME(miyu): CUDA linker doesn't accept assertion on GPU yet right now.
          break;
      }
      buffer_ptr = align_to_int64(buffer_ptr);
    }
  }
  int32_t init_idx = 0;
  for (int32_t i = 0; i < agg_col_count; ++i) {
    if (need_padding) {
      buffer_ptr = align_to_int64(buffer_ptr);
    }
    switch (col_sizes[i]) {
      case 1:
        buffer_ptr = init_columnar_buffer<int8_t>(
            buffer_ptr, init_vals[init_idx++], groups_buffer_entry_count, start, step);
        break;
      case 2:
        buffer_ptr = init_columnar_buffer<int16_t>(reinterpret_cast<int16_t*>(buffer_ptr),
                                                   init_vals[init_idx++],
                                                   groups_buffer_entry_count,
                                                   start,
                                                   step);
        break;
      case 4:
        buffer_ptr = init_columnar_buffer<int32_t>(reinterpret_cast<int32_t*>(buffer_ptr),
                                                   init_vals[init_idx++],
                                                   groups_buffer_entry_count,
                                                   start,
                                                   step);
        break;
      case 8:
        buffer_ptr = init_columnar_buffer<int64_t>(reinterpret_cast<int64_t*>(buffer_ptr),
                                                   init_vals[init_idx++],
                                                   groups_buffer_entry_count,
                                                   start,
                                                   step);
        break;
      case 0:
        continue;
      default:
        // FIXME(miyu): CUDA linker doesn't accept assertion on GPU yet now.
        break;
    }
  }
  __syncthreads();
}

template <typename K>
inline __device__ void fill_empty_device_key(K* keys_ptr,
                                             const uint32_t key_count,
                                             const K empty_key) {
  for (uint32_t i = 0; i < key_count; ++i) {
    keys_ptr[i] = empty_key;
  }
}

__global__ void init_group_by_buffer_gpu(int64_t* groups_buffer,
                                         const int64_t* init_vals,
                                         const size_t groups_buffer_entry_count,
                                         const uint32_t key_count,
                                         const uint32_t key_width,
                                         const uint32_t row_size_quad,
                                         const bool keyless,
                                         const int8_t warp_size) {
  const size_t start = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t step = blockDim.x * gridDim.x;
  if (keyless) {
    for (size_t i = start;
         i < groups_buffer_entry_count * row_size_quad * static_cast<size_t>(warp_size);
         i += step) {
      groups_buffer[i] = init_vals[i % row_size_quad];
    }
    __syncthreads();
    return;
  }

  for (size_t i = start; i < groups_buffer_entry_count; i += step) {
    int64_t* keys_ptr = groups_buffer + i * row_size_quad;
    switch (key_width) {
      case 4:
        fill_empty_device_key(
            reinterpret_cast<int32_t*>(keys_ptr), key_count, EMPTY_KEY_32);
        break;
      case 8:
        fill_empty_device_key(
            reinterpret_cast<int64_t*>(keys_ptr), key_count, EMPTY_KEY_64);
        break;
      default:
        break;
    }
  }

  const size_t values_off_quad = align_to_int64(key_count * key_width) / sizeof(int64_t);
  for (size_t i = start; i < groups_buffer_entry_count; i += step) {
    int64_t* vals_ptr = groups_buffer + i * row_size_quad + values_off_quad;
    const size_t val_count =
        row_size_quad - values_off_quad;  // value slots are always 64-bit
    for (size_t j = 0; j < val_count; ++j) {
      vals_ptr[j] = init_vals[j];
    }
  }
  __syncthreads();
}

__global__ void init_columnar_group_by_buffer_gpu_wrapper(
    int64_t* groups_buffer,
    const int64_t* init_vals,
    const uint32_t groups_buffer_entry_count,
    const uint32_t key_count,
    const uint32_t agg_col_count,
    const int8_t* col_sizes,
    const bool need_padding,
    const bool keyless,
    const int8_t key_size) {
  init_columnar_group_by_buffer_gpu_impl(groups_buffer,
                                         init_vals,
                                         groups_buffer_entry_count,
                                         key_count,
                                         agg_col_count,
                                         col_sizes,
                                         need_padding,
                                         keyless,
                                         key_size);
}

void init_group_by_buffer_on_device(int64_t* groups_buffer,
                                    const int64_t* init_vals,
                                    const size_t groups_buffer_entry_count,
                                    const uint32_t key_count,
                                    const uint32_t key_width,
                                    const uint32_t row_size_quad,
                                    const bool keyless,
                                    const int8_t warp_size,
                                    const size_t block_size_x,
                                    const size_t grid_size_x,
                                    CUstream cuda_stream) {
  init_group_by_buffer_gpu<<<grid_size_x, block_size_x, 0, cuda_stream>>>(
      groups_buffer,
      init_vals,
      groups_buffer_entry_count,
      key_count,
      key_width,
      row_size_quad,
      keyless,
      warp_size);
  checkCudaErrors(cudaStreamSynchronize(cuda_stream));
}

size_t count_non_empty_baseline_hash_rows_on_device(const int8_t* groups_buffer,
                                                    const size_t entry_count,
                                                    const size_t row_size,
                                                    const size_t key_width,
                                                    uint64_t* row_count,
                                                    const int device_id,
                                                    CUstream cuda_stream) {
  CHECK(groups_buffer);
  CHECK(row_count);
  CHECK_GT(row_size, size_t(0));
  CHECK_EQ(size_t(0), row_size % sizeof(int64_t));
  CHECK(key_width == size_t(4) || key_width == size_t(8));
  checkCudaErrors(cudaMemsetAsync(row_count, 0, sizeof(uint64_t), cuda_stream));
  count_non_empty_baseline_hash_rows_kernel<<<compact_grid_size(entry_count),
                                              compact_block_size,
                                              0,
                                              cuda_stream>>>(
      groups_buffer, row_count, entry_count, row_size, key_width);
  checkCudaErrors(cudaGetLastError());
  uint64_t host_row_count{0};
  checkCudaErrors(cudaMemcpyAsync(
      &host_row_count, row_count, sizeof(uint64_t), cudaMemcpyDeviceToHost, cuda_stream));
  checkCudaErrors(cudaStreamSynchronize(cuda_stream));
  return static_cast<size_t>(host_row_count);
}

void compact_baseline_hash_rows_on_device(const int8_t* groups_buffer,
                                          int8_t* compacted_buffer,
                                          uint64_t* compacted_row_count,
                                          const size_t entry_count,
                                          const size_t row_size,
                                          const size_t key_width,
                                          const int device_id,
                                          CUstream cuda_stream) {
  CHECK(groups_buffer);
  CHECK(compacted_buffer);
  CHECK(compacted_row_count);
  CHECK_GT(row_size, size_t(0));
  CHECK_EQ(size_t(0), row_size % sizeof(int64_t));
  CHECK(key_width == size_t(4) || key_width == size_t(8));
  checkCudaErrors(cudaMemsetAsync(compacted_row_count, 0, sizeof(uint64_t), cuda_stream));
  compact_baseline_hash_rows_kernel<<<compact_grid_size(entry_count),
                                      compact_block_size,
                                      0,
                                      cuda_stream>>>(groups_buffer,
                                                     compacted_buffer,
                                                     compacted_row_count,
                                                     entry_count,
                                                     row_size,
                                                     key_width);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaStreamSynchronize(cuda_stream));
}

void count_baseline_hash_partition_rows_on_device(const int8_t* groups_buffer,
                                                  uint64_t* partition_counts,
                                                  const size_t entry_count,
                                                  const size_t row_size,
                                                  const size_t key_width,
                                                  const size_t key_count,
                                                  const size_t partition_count,
                                                  const int device_id,
                                                  CUstream cuda_stream) {
  CHECK(groups_buffer);
  CHECK(partition_counts);
  CHECK_GT(partition_count, size_t(0));
  CHECK_LE(partition_count, std::numeric_limits<size_t>::max() / sizeof(uint64_t));
  CHECK_GT(row_size, size_t(0));
  CHECK_EQ(size_t(0), row_size % sizeof(int64_t));
  CHECK(key_width == size_t(4) || key_width == size_t(8));
  CHECK_GT(key_count, size_t(0));
  checkCudaErrors(cudaMemsetAsync(
      partition_counts, 0, partition_count * sizeof(uint64_t), cuda_stream));
  count_baseline_hash_partition_rows_kernel<<<compact_grid_size(entry_count),
                                              compact_block_size,
                                              0,
                                              cuda_stream>>>(groups_buffer,
                                                             partition_counts,
                                                             entry_count,
                                                             row_size,
                                                             key_width,
                                                             key_count,
                                                             partition_count);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaStreamSynchronize(cuda_stream));
}

void partition_baseline_hash_rows_on_device(const int8_t* groups_buffer,
                                            int8_t* partitioned_buffer,
                                            uint64_t* partition_write_counts,
                                            const uint64_t* partition_offsets,
                                            const size_t entry_count,
                                            const size_t row_size,
                                            const size_t key_width,
                                            const size_t key_count,
                                            const size_t partition_count,
                                            const int device_id,
                                            CUstream cuda_stream) {
  CHECK(groups_buffer);
  CHECK(partitioned_buffer);
  CHECK(partition_write_counts);
  CHECK(partition_offsets);
  CHECK_GT(partition_count, size_t(0));
  CHECK_LE(partition_count, std::numeric_limits<size_t>::max() / sizeof(uint64_t));
  CHECK_GT(row_size, size_t(0));
  CHECK_EQ(size_t(0), row_size % sizeof(int64_t));
  CHECK(key_width == size_t(4) || key_width == size_t(8));
  CHECK_GT(key_count, size_t(0));
  checkCudaErrors(cudaMemsetAsync(
      partition_write_counts, 0, partition_count * sizeof(uint64_t), cuda_stream));
  partition_baseline_hash_rows_kernel<<<compact_grid_size(entry_count),
                                        compact_block_size,
                                        0,
                                        cuda_stream>>>(groups_buffer,
                                                       partitioned_buffer,
                                                       partition_write_counts,
                                                       partition_offsets,
                                                       entry_count,
                                                       row_size,
                                                       key_width,
                                                       key_count,
                                                       partition_count);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaStreamSynchronize(cuda_stream));
}

size_t count_non_empty_keyless_hash_rows_on_device(const int8_t* groups_buffer,
                                                   const size_t entry_count,
                                                   const size_t row_size,
                                                   const size_t key_slot_offset,
                                                   const size_t key_slot_width,
                                                   const int64_t key_init_val,
                                                   uint64_t* row_count,
                                                   const int device_id,
                                                   CUstream cuda_stream) {
  CHECK(groups_buffer);
  CHECK(row_count);
  CHECK_GT(row_size, size_t(0));
  CHECK_EQ(size_t(0), row_size % sizeof(int64_t));
  CHECK(key_slot_width == size_t(1) || key_slot_width == size_t(2) ||
        key_slot_width == size_t(4) || key_slot_width == size_t(8));
  CHECK_LE(key_slot_offset, row_size);
  CHECK_LE(key_slot_width, row_size - key_slot_offset);
  checkCudaErrors(cudaMemsetAsync(row_count, 0, sizeof(uint64_t), cuda_stream));
  count_non_empty_keyless_hash_rows_kernel<<<compact_grid_size(entry_count),
                                             compact_block_size,
                                             0,
                                             cuda_stream>>>(groups_buffer,
                                                            row_count,
                                                            entry_count,
                                                            row_size,
                                                            key_slot_offset,
                                                            key_slot_width,
                                                            key_init_val);
  checkCudaErrors(cudaGetLastError());
  uint64_t host_row_count{0};
  checkCudaErrors(cudaMemcpyAsync(
      &host_row_count, row_count, sizeof(uint64_t), cudaMemcpyDeviceToHost, cuda_stream));
  checkCudaErrors(cudaStreamSynchronize(cuda_stream));
  return static_cast<size_t>(host_row_count);
}

void compact_keyless_hash_rows_on_device(const int8_t* groups_buffer,
                                         int8_t* compacted_buffer,
                                         uint64_t* compacted_row_count,
                                         uint64_t* compacted_entry_indices,
                                         const size_t entry_count,
                                         const size_t row_size,
                                         const size_t key_slot_offset,
                                         const size_t key_slot_width,
                                         const int64_t key_init_val,
                                         const int device_id,
                                         CUstream cuda_stream) {
  CHECK(groups_buffer);
  CHECK(compacted_buffer);
  CHECK(compacted_row_count);
  CHECK_GT(row_size, size_t(0));
  CHECK_EQ(size_t(0), row_size % sizeof(int64_t));
  CHECK(key_slot_width == size_t(1) || key_slot_width == size_t(2) ||
        key_slot_width == size_t(4) || key_slot_width == size_t(8));
  CHECK_LE(key_slot_offset, row_size);
  CHECK_LE(key_slot_width, row_size - key_slot_offset);
  checkCudaErrors(cudaMemsetAsync(compacted_row_count, 0, sizeof(uint64_t), cuda_stream));
  compact_keyless_hash_rows_kernel<<<compact_grid_size(entry_count),
                                     compact_block_size,
                                     0,
                                     cuda_stream>>>(groups_buffer,
                                                    compacted_buffer,
                                                    compacted_row_count,
                                                    compacted_entry_indices,
                                                    entry_count,
                                                    row_size,
                                                    key_slot_offset,
                                                    key_slot_width,
                                                    key_init_val);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaStreamSynchronize(cuda_stream));
}

size_t count_matching_baseline_hash_rows_on_device(
    const int8_t* groups_buffer,
    const size_t entry_count,
    const size_t row_size,
    const size_t key_width,
    const DeviceResultSetEntryComparison* comparisons,
    const size_t comparison_count,
    const int64_t* preserved_keys,
    const size_t preserved_key_count,
    uint64_t* row_count,
    const int device_id,
    CUstream cuda_stream) {
  CHECK(groups_buffer);
  CHECK(comparison_count == 0 || comparisons);
  CHECK(preserved_key_count == 0 || preserved_keys);
  CHECK(row_count);
  CHECK_GT(row_size, size_t(0));
  CHECK_EQ(size_t(0), row_size % sizeof(int64_t));
  CHECK(key_width == size_t(4) || key_width == size_t(8));
  checkCudaErrors(cudaMemsetAsync(row_count, 0, sizeof(uint64_t), cuda_stream));
  count_matching_baseline_hash_rows_kernel<<<compact_grid_size(entry_count),
                                             compact_block_size,
                                             0,
                                             cuda_stream>>>(groups_buffer,
                                                            row_count,
                                                            entry_count,
                                                            row_size,
                                                            key_width,
                                                            comparisons,
                                                            comparison_count,
                                                            preserved_keys,
                                                            preserved_key_count);
  checkCudaErrors(cudaGetLastError());
  uint64_t host_row_count{0};
  checkCudaErrors(cudaMemcpyAsync(
      &host_row_count, row_count, sizeof(uint64_t), cudaMemcpyDeviceToHost, cuda_stream));
  checkCudaErrors(cudaStreamSynchronize(cuda_stream));
  return static_cast<size_t>(host_row_count);
}

void compact_matching_baseline_hash_rows_on_device(
    const int8_t* groups_buffer,
    int8_t* compacted_buffer,
    uint64_t* compacted_row_count,
    const size_t entry_count,
    const size_t row_size,
    const size_t key_width,
    const DeviceResultSetEntryComparison* comparisons,
    const size_t comparison_count,
    const int64_t* preserved_keys,
    const size_t preserved_key_count,
    const int device_id,
    CUstream cuda_stream) {
  CHECK(groups_buffer);
  CHECK(compacted_buffer);
  CHECK(compacted_row_count);
  CHECK(comparison_count == 0 || comparisons);
  CHECK(preserved_key_count == 0 || preserved_keys);
  CHECK_GT(row_size, size_t(0));
  CHECK_EQ(size_t(0), row_size % sizeof(int64_t));
  CHECK(key_width == size_t(4) || key_width == size_t(8));
  checkCudaErrors(cudaMemsetAsync(compacted_row_count, 0, sizeof(uint64_t), cuda_stream));
  compact_matching_baseline_hash_rows_kernel<<<compact_grid_size(entry_count),
                                               compact_block_size,
                                               0,
                                               cuda_stream>>>(groups_buffer,
                                                              compacted_buffer,
                                                              compacted_row_count,
                                                              entry_count,
                                                              row_size,
                                                              key_width,
                                                              comparisons,
                                                              comparison_count,
                                                              preserved_keys,
                                                              preserved_key_count);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaStreamSynchronize(cuda_stream));
}

void compact_matching_keyless_hash_rows_on_device(
    int8_t* groups_buffer,
    int8_t* compacted_buffer,
    uint64_t* compacted_row_count,
    uint64_t* compacted_entry_indices,
    const size_t entry_count,
    const size_t row_size,
    const size_t key_slot_offset,
    const size_t key_slot_width,
    const int64_t key_init_val,
    const DeviceResultSetEntryComparison* comparisons,
    const size_t comparison_count,
    const int device_id,
    CUstream cuda_stream) {
  CHECK(groups_buffer);
  CHECK(compacted_buffer);
  CHECK(compacted_row_count);
  CHECK(comparisons);
  CHECK_GT(comparison_count, size_t(0));
  CHECK_GT(row_size, size_t(0));
  CHECK_EQ(size_t(0), row_size % sizeof(int64_t));
  CHECK(key_slot_width == size_t(1) || key_slot_width == size_t(2) ||
        key_slot_width == size_t(4) || key_slot_width == size_t(8));
  CHECK_LE(key_slot_offset, row_size);
  CHECK_LE(key_slot_width, row_size - key_slot_offset);
  checkCudaErrors(cudaMemsetAsync(compacted_row_count, 0, sizeof(uint64_t), cuda_stream));
  compact_matching_keyless_hash_rows_kernel<<<compact_grid_size(entry_count),
                                              compact_block_size,
                                              0,
                                              cuda_stream>>>(groups_buffer,
                                                             compacted_buffer,
                                                             compacted_row_count,
                                                             compacted_entry_indices,
                                                             entry_count,
                                                             row_size,
                                                             key_slot_offset,
                                                             key_slot_width,
                                                             key_init_val,
                                                             comparisons,
                                                             comparison_count);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaStreamSynchronize(cuda_stream));
}

void synthesize_perfect_hash_group_key_column_on_device(const uint64_t* entry_indices,
                                                        int8_t* columnar_buffer,
                                                        const size_t entry_count,
                                                        const size_t output_width,
                                                        const int64_t min_val,
                                                        const int64_t bucket,
                                                        const int64_t source_null_val,
                                                        const int64_t normalized_null_val,
                                                        const int device_id,
                                                        CUstream cuda_stream) {
  CHECK(entry_indices);
  CHECK(columnar_buffer);
  CHECK_GT(entry_count, size_t(0));
  CHECK(is_supported_int_width(output_width));
  synthesize_perfect_hash_group_key_column_kernel<<<compact_grid_size(entry_count),
                                                    compact_block_size,
                                                    0,
                                                    cuda_stream>>>(entry_indices,
                                                                   columnar_buffer,
                                                                   entry_count,
                                                                   output_width,
                                                                   min_val,
                                                                   bucket ? bucket : 1,
                                                                   source_null_val,
                                                                   normalized_null_val);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaStreamSynchronize(cuda_stream));
}

size_t count_baseline_hash_rows_excluding_keys_on_device(const int8_t* groups_buffer,
                                                         const size_t entry_count,
                                                         const size_t row_size,
                                                         const size_t key_width,
                                                         const int64_t* excluded_keys,
                                                         const size_t excluded_key_count,
                                                         uint64_t* row_count,
                                                         const int device_id,
                                                         CUstream cuda_stream) {
  CHECK(groups_buffer);
  CHECK(row_count);
  CHECK(excluded_keys);
  CHECK_GT(excluded_key_count, size_t(0));
  CHECK_GT(row_size, size_t(0));
  CHECK_EQ(size_t(0), row_size % sizeof(int64_t));
  CHECK(key_width == size_t(4) || key_width == size_t(8));
  checkCudaErrors(cudaMemsetAsync(row_count, 0, sizeof(uint64_t), cuda_stream));
  count_baseline_hash_rows_excluding_keys_kernel<<<compact_grid_size(entry_count),
                                                   compact_block_size,
                                                   0,
                                                   cuda_stream>>>(groups_buffer,
                                                                  row_count,
                                                                  entry_count,
                                                                  row_size,
                                                                  key_width,
                                                                  excluded_keys,
                                                                  excluded_key_count);
  checkCudaErrors(cudaGetLastError());
  uint64_t host_row_count{0};
  checkCudaErrors(cudaMemcpyAsync(
      &host_row_count, row_count, sizeof(uint64_t), cudaMemcpyDeviceToHost, cuda_stream));
  checkCudaErrors(cudaStreamSynchronize(cuda_stream));
  return static_cast<size_t>(host_row_count);
}

void compact_baseline_hash_rows_excluding_keys_on_device(const int8_t* groups_buffer,
                                                         int8_t* compacted_buffer,
                                                         uint64_t* compacted_row_count,
                                                         const size_t entry_count,
                                                         const size_t row_size,
                                                         const size_t key_width,
                                                         const int64_t* excluded_keys,
                                                         const size_t excluded_key_count,
                                                         const int device_id,
                                                         CUstream cuda_stream) {
  CHECK(groups_buffer);
  CHECK(compacted_buffer);
  CHECK(compacted_row_count);
  CHECK(excluded_keys);
  CHECK_GT(excluded_key_count, size_t(0));
  CHECK_GT(row_size, size_t(0));
  CHECK_EQ(size_t(0), row_size % sizeof(int64_t));
  CHECK(key_width == size_t(4) || key_width == size_t(8));
  checkCudaErrors(cudaMemsetAsync(compacted_row_count, 0, sizeof(uint64_t), cuda_stream));
  compact_baseline_hash_rows_excluding_keys_kernel<<<compact_grid_size(entry_count),
                                                     compact_block_size,
                                                     0,
                                                     cuda_stream>>>(groups_buffer,
                                                                    compacted_buffer,
                                                                    compacted_row_count,
                                                                    entry_count,
                                                                    row_size,
                                                                    key_width,
                                                                    excluded_keys,
                                                                    excluded_key_count);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaStreamSynchronize(cuda_stream));
}

void compact_baseline_hash_rows_matching_keys_on_device(int8_t* groups_buffer,
                                                        int8_t* compacted_buffer,
                                                        uint64_t* compacted_row_count,
                                                        const size_t entry_count,
                                                        const size_t row_size,
                                                        const size_t key_width,
                                                        const int64_t* matching_keys,
                                                        const size_t matching_key_count,
                                                        const bool clear_matching_keys,
                                                        const int device_id,
                                                        CUstream cuda_stream) {
  CHECK(groups_buffer);
  CHECK(compacted_buffer);
  CHECK(compacted_row_count);
  CHECK(matching_keys);
  CHECK_GT(matching_key_count, size_t(0));
  CHECK_GT(row_size, size_t(0));
  CHECK_EQ(size_t(0), row_size % sizeof(int64_t));
  CHECK(key_width == size_t(4) || key_width == size_t(8));
  checkCudaErrors(cudaMemsetAsync(compacted_row_count, 0, sizeof(uint64_t), cuda_stream));
  compact_baseline_hash_rows_matching_keys_kernel<<<compact_grid_size(entry_count),
                                                    compact_block_size,
                                                    0,
                                                    cuda_stream>>>(groups_buffer,
                                                                   compacted_buffer,
                                                                   compacted_row_count,
                                                                   entry_count,
                                                                   row_size,
                                                                   key_width,
                                                                   matching_keys,
                                                                   matching_key_count,
                                                                   clear_matching_keys);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaStreamSynchronize(cuda_stream));
}

void extract_fixed_width_column_from_rows_on_device(const int8_t* rowwise_buffer,
                                                    int8_t* columnar_buffer,
                                                    const size_t entry_count,
                                                    const size_t row_size,
                                                    const size_t source_offset,
                                                    const size_t source_width,
                                                    const size_t output_width,
                                                    const int64_t dict_entry_count,
                                                    const int64_t source_null_val,
                                                    const int64_t normalized_null_val,
                                                    const int device_id,
                                                    CUstream cuda_stream) {
  CHECK(rowwise_buffer);
  CHECK(columnar_buffer);
  CHECK_GT(entry_count, size_t(0));
  CHECK_GT(row_size, size_t(0));
  CHECK_GT(output_width, size_t(0));
  CHECK_GT(source_width, size_t(0));
  if (source_width != output_width) {
    CHECK(is_supported_int_width(source_width));
    CHECK(is_supported_int_width(output_width));
  }
  if (dict_entry_count >= 0) {
    CHECK(is_supported_int_width(source_width));
    CHECK(is_supported_int_width(output_width));
  }
  if (source_null_val != kNoTranslatedGroupbyNull) {
    CHECK(is_supported_int_width(source_width));
    CHECK(is_supported_int_width(output_width));
  }
  CHECK_LE(source_offset, row_size);
  CHECK_LE(source_width, row_size - source_offset);
  extract_fixed_width_column_from_rows_kernel<<<compact_grid_size(entry_count),
                                                compact_block_size,
                                                0,
                                                cuda_stream>>>(rowwise_buffer,
                                                               columnar_buffer,
                                                               entry_count,
                                                               row_size,
                                                               source_offset,
                                                               source_width,
                                                               output_width,
                                                               dict_entry_count,
                                                               source_null_val,
                                                               normalized_null_val);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaStreamSynchronize(cuda_stream));
}

bool reduce_baseline_hash_rows_on_device(int8_t* destination_buffer,
                                         const size_t destination_entry_count,
                                         const int8_t* source_buffer,
                                         const size_t source_entry_count,
                                         const size_t row_size,
                                         const size_t key_width,
                                         const size_t key_count,
                                         const DeviceBaselineHashReductionSlot* slots,
                                         const size_t slot_count,
                                         int* error_code,
                                         const int device_id,
                                         CUstream cuda_stream) {
  CHECK(destination_buffer);
  CHECK(source_buffer);
  CHECK_GT(destination_entry_count, size_t(0));
  CHECK_GT(source_entry_count, size_t(0));
  CHECK_GT(row_size, size_t(0));
  CHECK_EQ(size_t(0), row_size % sizeof(int64_t));
  CHECK(key_width == size_t(4) || key_width == size_t(8));
  CHECK_GT(key_count, size_t(0));
  CHECK(slot_count == 0 || slots);
  CHECK(error_code);
  checkCudaErrors(cudaMemsetAsync(error_code, 0, sizeof(int), cuda_stream));
  reduce_baseline_hash_rows_kernel<<<compact_grid_size(source_entry_count),
                                     compact_block_size,
                                     0,
                                     cuda_stream>>>(destination_buffer,
                                                    destination_entry_count,
                                                    source_buffer,
                                                    source_entry_count,
                                                    row_size,
                                                    key_width,
                                                    key_count,
                                                    slots,
                                                    slot_count,
                                                    error_code);
  checkCudaErrors(cudaGetLastError());
  int host_error_code{0};
  checkCudaErrors(cudaMemcpyAsync(&host_error_code,
                                  error_code,
                                  sizeof(host_error_code),
                                  cudaMemcpyDeviceToHost,
                                  cuda_stream));
  checkCudaErrors(cudaStreamSynchronize(cuda_stream));
  return host_error_code == 0;
}

bool reduce_baseline_hash_buffers_on_device(int8_t* destination_buffer,
                                            const size_t destination_entry_count,
                                            const int8_t* source_buffers,
                                            const size_t source_entry_count,
                                            const size_t source_buffer_stride,
                                            const size_t source_buffer_count,
                                            const size_t row_size,
                                            const size_t key_width,
                                            const size_t key_count,
                                            const DeviceBaselineHashReductionSlot* slots,
                                            const size_t slot_count,
                                            int* error_code,
                                            const int device_id,
                                            CUstream cuda_stream) {
  CHECK(destination_buffer);
  CHECK(source_buffers);
  CHECK_GT(destination_entry_count, size_t(0));
  CHECK_GT(source_entry_count, size_t(0));
  CHECK_GT(source_buffer_count, size_t(0));
  CHECK_GT(row_size, size_t(0));
  CHECK_LE(source_entry_count, std::numeric_limits<size_t>::max() / row_size);
  CHECK_GE(source_buffer_stride, source_entry_count * row_size);
  CHECK_EQ(size_t(0), row_size % sizeof(int64_t));
  CHECK(key_width == size_t(4) || key_width == size_t(8));
  CHECK_GT(key_count, size_t(0));
  CHECK(slot_count == 0 || slots);
  CHECK(error_code);
  CHECK_LE(source_buffer_count, std::numeric_limits<size_t>::max() / source_entry_count);
  const auto total_source_entries = source_entry_count * source_buffer_count;
  checkCudaErrors(cudaMemsetAsync(error_code, 0, sizeof(int), cuda_stream));
  reduce_baseline_hash_buffers_kernel<<<compact_grid_size(total_source_entries),
                                        compact_block_size,
                                        0,
                                        cuda_stream>>>(destination_buffer,
                                                       destination_entry_count,
                                                       source_buffers,
                                                       source_entry_count,
                                                       source_buffer_stride,
                                                       source_buffer_count,
                                                       row_size,
                                                       key_width,
                                                       key_count,
                                                       slots,
                                                       slot_count,
                                                       error_code);
  checkCudaErrors(cudaGetLastError());
  int host_error_code{0};
  checkCudaErrors(cudaMemcpyAsync(&host_error_code,
                                  error_code,
                                  sizeof(host_error_code),
                                  cudaMemcpyDeviceToHost,
                                  cuda_stream));
  checkCudaErrors(cudaStreamSynchronize(cuda_stream));
  return host_error_code == 0;
}

bool reduce_perfect_hash_rows_on_device(int8_t* destination_buffer,
                                        const int8_t* source_buffer,
                                        const size_t entry_count,
                                        const size_t row_size,
                                        const size_t key_width,
                                        const size_t key_count,
                                        const bool keyless,
                                        const size_t key_slot_offset,
                                        const size_t key_slot_width,
                                        const int64_t key_init_val,
                                        const DeviceBaselineHashReductionSlot* slots,
                                        const size_t slot_count,
                                        int* error_code,
                                        const int device_id,
                                        CUstream cuda_stream) {
  CHECK(destination_buffer);
  CHECK(source_buffer);
  CHECK_GT(entry_count, size_t(0));
  CHECK_GT(row_size, size_t(0));
  CHECK_EQ(size_t(0), row_size % sizeof(int64_t));
  if (keyless) {
    CHECK(key_slot_width == size_t(1) || key_slot_width == size_t(2) ||
          key_slot_width == size_t(4) || key_slot_width == size_t(8));
    CHECK_LE(key_slot_offset, row_size);
    CHECK_LE(key_slot_width, row_size - key_slot_offset);
  } else {
    CHECK(key_width == size_t(4) || key_width == size_t(8));
    CHECK_GT(key_count, size_t(0));
  }
  CHECK(slot_count == 0 || slots);
  CHECK(error_code);
  checkCudaErrors(cudaMemsetAsync(error_code, 0, sizeof(int), cuda_stream));
  reduce_perfect_hash_rows_kernel<<<compact_grid_size(entry_count),
                                    compact_block_size,
                                    0,
                                    cuda_stream>>>(destination_buffer,
                                                   source_buffer,
                                                   entry_count,
                                                   row_size,
                                                   key_width,
                                                   key_count,
                                                   keyless,
                                                   key_slot_offset,
                                                   key_slot_width,
                                                   key_init_val,
                                                   slots,
                                                   slot_count,
                                                   error_code);
  checkCudaErrors(cudaGetLastError());
  int host_error_code{0};
  checkCudaErrors(cudaMemcpyAsync(&host_error_code,
                                  error_code,
                                  sizeof(host_error_code),
                                  cudaMemcpyDeviceToHost,
                                  cuda_stream));
  checkCudaErrors(cudaStreamSynchronize(cuda_stream));
  return host_error_code == 0;
}

bool compute_columnar_fragment_int_stats_on_device(const int8_t* column_buffer,
                                                   const size_t entry_count,
                                                   const size_t elem_size,
                                                   const int64_t null_val,
                                                   const int device_id,
                                                   DeviceColumnFragmentStats& stats,
                                                   CUstream cuda_stream) {
  CHECK(column_buffer);
  switch (elem_size) {
    case sizeof(int8_t): {
      const auto result = compute_columnar_fragment_stats_typed<int8_t>(
          column_buffer, entry_count, static_cast<int8_t>(null_val), cuda_stream);
      stats.int_min = result.min;
      stats.int_max = result.max;
      stats.has_nulls = result.has_nulls;
      stats.has_values = result.has_values;
      break;
    }
    case sizeof(int16_t): {
      const auto result = compute_columnar_fragment_stats_typed<int16_t>(
          column_buffer, entry_count, static_cast<int16_t>(null_val), cuda_stream);
      stats.int_min = result.min;
      stats.int_max = result.max;
      stats.has_nulls = result.has_nulls;
      stats.has_values = result.has_values;
      break;
    }
    case sizeof(int32_t): {
      const auto result = compute_columnar_fragment_stats_typed<int32_t>(
          column_buffer, entry_count, static_cast<int32_t>(null_val), cuda_stream);
      stats.int_min = result.min;
      stats.int_max = result.max;
      stats.has_nulls = result.has_nulls;
      stats.has_values = result.has_values;
      break;
    }
    case sizeof(int64_t): {
      const auto result = compute_columnar_fragment_stats_typed<int64_t>(
          column_buffer, entry_count, static_cast<int64_t>(null_val), cuda_stream);
      stats.int_min = result.min;
      stats.int_max = result.max;
      stats.has_nulls = result.has_nulls;
      stats.has_values = result.has_values;
      break;
    }
    default:
      return false;
  }
  return true;
}

bool compute_columnar_fragment_fp_stats_on_device(const int8_t* column_buffer,
                                                  const size_t entry_count,
                                                  const size_t elem_size,
                                                  const double null_val,
                                                  const int device_id,
                                                  DeviceColumnFragmentStats& stats,
                                                  CUstream cuda_stream) {
  CHECK(column_buffer);
  switch (elem_size) {
    case sizeof(float): {
      const auto result = compute_columnar_fragment_stats_typed<float>(
          column_buffer, entry_count, static_cast<float>(null_val), cuda_stream);
      stats.fp_min = result.min;
      stats.fp_max = result.max;
      stats.has_nulls = result.has_nulls;
      stats.has_values = result.has_values;
      break;
    }
    case sizeof(double): {
      const auto result = compute_columnar_fragment_stats_typed<double>(
          column_buffer, entry_count, static_cast<double>(null_val), cuda_stream);
      stats.fp_min = result.min;
      stats.fp_max = result.max;
      stats.has_nulls = result.has_nulls;
      stats.has_values = result.has_values;
      break;
    }
    default:
      return false;
  }
  return true;
}

void init_columnar_group_by_buffer_on_device(int64_t* groups_buffer,
                                             const int64_t* init_vals,
                                             const uint32_t groups_buffer_entry_count,
                                             const uint32_t key_count,
                                             const uint32_t agg_col_count,
                                             const int8_t* col_sizes,
                                             const bool need_padding,
                                             const bool keyless,
                                             const int8_t key_size,
                                             const size_t block_size_x,
                                             const size_t grid_size_x,
                                             CUstream cuda_stream) {
  init_columnar_group_by_buffer_gpu_wrapper<<<grid_size_x,
                                              block_size_x,
                                              0,
                                              cuda_stream>>>(groups_buffer,
                                                             init_vals,
                                                             groups_buffer_entry_count,
                                                             key_count,
                                                             agg_col_count,
                                                             col_sizes,
                                                             need_padding,
                                                             keyless,
                                                             key_size);
  checkCudaErrors(cudaStreamSynchronize(cuda_stream));
}
