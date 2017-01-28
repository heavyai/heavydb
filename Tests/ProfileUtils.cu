/**
* @file    ProfileUtils.cu
* @author  Minggang Yu <miyu@mapd.com>
* @brief   Unit tests for microbenchmark.
*
* Copyright (c) 2016 MapD Technologies, Inc.  All rights reserved.
*/
#include "ProfileTest.h"

#if defined(HAVE_CUDA) && CUDA_VERSION >= 8000
#include <stdio.h>
#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#else
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "../QueryEngine/cuda_mapd_rt.cu"
#ifdef __clang__
#pragma clang diagnostic pop
#else
#pragma GCC diagnostic pop
#endif

namespace {
// Number of threads to put in a thread block.
const unsigned c_block_size = 512;

// Number of blocks to put along each axis of the grid.
const unsigned c_grid_size = 16384;

dim3 compute_grid_dim(unsigned n) {
  dim3 grid((n + c_block_size - 1) / c_block_size);
  if (grid.x > c_grid_size) {
    grid.y = (grid.x + c_grid_size - 1) / c_grid_size;
    grid.x = c_grid_size;
  }
  return grid;
}

template <typename KeyT = int64_t>
__device__ inline bool is_empty_slot(const KeyT k) {
  static_assert(thrust::detail::is_same<KeyT, int64_t>::value,
                "Unsupported template parameter other than int64_t for now");
  return k == EMPTY_KEY_64;
}

template <typename KeyT = int64_t>
__device__ inline void reset_entry(KeyT* entry_ptr) {
  static_assert(thrust::detail::is_same<KeyT, int64_t>::value,
                "Unsupported template parameter other than int64_t for now");
  *entry_ptr = static_cast<KeyT>(EMPTY_KEY_64);
}

template <typename T = int64_t>
struct DeviceIntGenerator {
  static_assert(thrust::detail::is_integral<T>::value, "Template type is not integral");
  DeviceIntGenerator(int8_t* ptr, size_t gap, T min, T max, T seed = 0)
      : buff_ptr(ptr), stride(gap), engine(seed), uni_dist(min, max) {}

  __device__ void operator()(const int index) {
    engine.discard(index);
    *reinterpret_cast<T*>(buff_ptr + index * stride) = uni_dist(engine);
  }

  int8_t* buff_ptr;
  size_t stride;
  thrust::default_random_engine engine;
  thrust::uniform_int_distribution<T> uni_dist;
};

template <typename T = int64_t>
bool generate_numbers(int8_t* random_numbers,
                      const size_t num_random_numbers,
                      const T min_number,
                      const T max_number,
                      const DIST_KIND dist,
                      const size_t stride = 1) {
  if (dist != DIST_KIND::UNI) {
    return false;
  }
  static T seed = 0;
  thrust::for_each(thrust::make_counting_iterator(size_t(0)),
                   thrust::make_counting_iterator(num_random_numbers),
                   DeviceIntGenerator<T>(random_numbers, stride, min_number, max_number, seed++));
  return true;
}

}  // namespace

bool generate_columns_on_device(int8_t* buffers,
                                const size_t row_count,
                                const size_t col_count,
                                const std::vector<size_t>& col_widths,
                                const std::vector<std::pair<int64_t, int64_t>>& ranges,
                                const bool is_columnar,
                                const std::vector<DIST_KIND>& dists) {
  if (buffers == nullptr) {
    return false;
  }
  CHECK_EQ(col_widths.size(), col_count);
  CHECK_EQ(ranges.size(), col_count);
  size_t row_size = 0;
  for (auto& wid : col_widths) {
    row_size += wid;
  }
  for (size_t i = 0; i < col_count; buffers += (is_columnar ? row_count : 1) * col_widths[i++]) {
    if (dists[i] == DIST_KIND::INVALID) {
      continue;
    }
    CHECK(ranges[i].first <= ranges[i].second);
    switch (col_widths[i]) {
      case 4:
        if (!generate_numbers(buffers,
                              row_count,
                              static_cast<int32_t>(ranges[i].first),
                              static_cast<int32_t>(ranges[i].second),
                              dists[i],
                              (is_columnar ? 4 : row_size))) {
          return false;
        }
        break;
      case 8:
        if (!generate_numbers(
                buffers, row_count, ranges[i].first, ranges[i].second, dists[i], (is_columnar ? 8 : row_size))) {
          return false;
        }
        break;
      default:
        CHECK(false);
    }
  }
  return true;
}

namespace {

template <bool isColumnar = true>
__global__ void init_group(int8_t* groups,
                           const size_t group_count,
                           const size_t col_count,
                           const size_t* col_widths,
                           const size_t* init_vals) {
  const auto thread_index = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;
  if (thread_index >= group_count) {
    return;
  }
  for (size_t i = 0; i < col_count; groups += col_widths[i++] * group_count) {
    switch (col_widths[i]) {
      case 4:
        *reinterpret_cast<uint32_t*>(groups) = *reinterpret_cast<const uint32_t*>(init_vals + i);
        break;
      case 8:
        reinterpret_cast<size_t*>(groups)[thread_index] = init_vals[i];
        break;
      default:;
    }
  }
}

template <>
__global__ void init_group<false>(int8_t* groups,
                                  const size_t group_count,
                                  const size_t col_count,
                                  const size_t* col_widths,
                                  const size_t* init_vals) {
  const auto thread_index = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;
  if (thread_index >= group_count) {
    return;
  }
  size_t row_size = 0;
  for (size_t i = 0; i < col_count; ++i) {
    row_size += col_widths[i];
  }
  int8_t* group_base = groups + row_size * thread_index;
  for (size_t i = 0; i < col_count; group_base += col_widths[i++]) {
    switch (col_widths[i]) {
      case 4:
        *reinterpret_cast<uint32_t*>(group_base) = *reinterpret_cast<const uint32_t*>(init_vals + i);
        break;
      case 8:
        *reinterpret_cast<size_t*>(group_base) = init_vals[i];
        break;
      default:;
    }
  }
}

}  // namespace

void init_groups_on_device(int8_t* groups,
                           const size_t group_count,
                           const size_t col_count,
                           const std::vector<size_t>& col_widths,
                           const std::vector<size_t>& init_vals,
                           const bool is_columnar) {
  thrust::device_vector<size_t> dev_col_widths(col_widths);
  thrust::device_vector<size_t> dev_init_vals(init_vals);
  if (is_columnar) {
    init_group<true><<<compute_grid_dim(group_count), c_block_size>>>(groups,
                                                                      group_count,
                                                                      col_count,
                                                                      thrust::raw_pointer_cast(dev_col_widths.data()),
                                                                      thrust::raw_pointer_cast(dev_init_vals.data()));
  } else {
    init_group<false><<<compute_grid_dim(group_count), c_block_size>>>(groups,
                                                                       group_count,
                                                                       col_count,
                                                                       thrust::raw_pointer_cast(dev_col_widths.data()),
                                                                       thrust::raw_pointer_cast(dev_init_vals.data()));
  }
}

#ifdef TRY_COLUMNAR
namespace {
__global__ void columnarize_groups(int8_t* columnar_buffer,
                                   const int8_t* rowwise_buffer,
                                   const size_t row_count,
                                   const size_t col_count,
                                   const size_t* col_widths,
                                   const size_t row_size) {
  const auto thread_index = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;
  if (thread_index >= row_count) {
    return;
  }
  auto read_ptr = rowwise_buffer + thread_index * row_size;
  auto col_base = columnar_buffer;
  for (size_t i = 0; i < col_count; ++i) {
    switch (col_widths[i]) {
      case 8: {
        int64_t* write_ptr = reinterpret_cast<int64_t*>(col_base) + thread_index;
        *write_ptr = *reinterpret_cast<const int64_t*>(read_ptr);
      } break;
      case 4: {
        int32_t* write_ptr = reinterpret_cast<int32_t*>(col_base) + thread_index;
        *write_ptr = *reinterpret_cast<const int32_t*>(read_ptr);
      } break;
      default:;
    }
    col_base += col_widths[i] * row_count;
    read_ptr += col_widths[i];  // WARN(miyu): No padding!!
  }
}
}  // namespace

void columnarize_groups_on_device(int8_t* columnar_buffer,
                                  const int8_t* rowwise_buffer,
                                  const size_t row_count,
                                  const std::vector<size_t>& col_widths) {
  size_t row_size = 0;
  for (size_t i = 0; i < col_widths.size(); ++i) {
    row_size += col_widths[i];
  }
  thrust::device_vector<size_t> dev_col_widths(col_widths);
  columnarize_groups<<<compute_grid_dim(row_count), c_block_size>>>(columnar_buffer,
                                                                    rowwise_buffer,
                                                                    row_count,
                                                                    col_widths.size(),
                                                                    thrust::raw_pointer_cast(dev_col_widths.data()),
                                                                    row_size);
}
#endif

namespace {

__device__ inline void row_func(int8_t* write_base,
                                const size_t write_stride,
                                const int8_t* read_base,
                                const size_t read_stride,
                                const size_t val_count,
                                const size_t* val_widths,
                                const OP_KIND* agg_ops) {
  for (size_t i = 0; i < val_count; ++i) {
    switch (val_widths[i]) {
      case 4: {
        auto write_ptr = reinterpret_cast<int32_t*>(write_base);
        const auto value = *reinterpret_cast<const int32_t*>(read_base);
        switch (agg_ops[i]) {
          case OP_COUNT:
            agg_count_int32_shared(reinterpret_cast<uint32_t*>(write_ptr), value);
            break;
          case OP_SUM:
            agg_sum_int32_shared(write_ptr, value);
            break;
          case OP_MIN:
            agg_min_int32_shared(write_ptr, value);
            break;
          case OP_MAX:
            agg_max_int32_shared(write_ptr, value);
            break;
          default:;
        }
      } break;
      case 8: {
        auto write_ptr = reinterpret_cast<int64_t*>(write_base);
        const auto value = *reinterpret_cast<const int64_t*>(read_base);
        switch (agg_ops[i]) {
          case OP_COUNT:
            agg_count_shared(reinterpret_cast<uint64_t*>(write_ptr), value);
            break;
          case OP_SUM:
            agg_sum_shared(write_ptr, value);
            break;
          case OP_MIN:
            agg_min_shared(write_ptr, value);
            break;
          case OP_MAX:
            agg_max_shared(write_ptr, value);
            break;
          default:;
        }
      } break;
      default:;
    }
    write_base += val_widths[i] * write_stride;
    read_base += val_widths[i] * read_stride;
  }
}

__device__ inline uint64_t rotl(uint64_t x, int8_t r) {
  return (x << r) | (x >> (64 - r));
}

__device__ inline uint64_t fmix(uint64_t k) {
  k ^= k >> 33;
  k *= 0xff51afd7ed558ccdULL;
  k ^= k >> 33;
  k *= 0xc4ceb9fe1a85ec53ULL;
  k ^= k >> 33;

  return k;
}

__device__ inline uint64_t murmur_hash3(const int64_t* key, const size_t key_count, const size_t qw_stride = 1) {
  if (key_count == 1) {
    return key[0];
  }
  uint64_t h = 0;
  const uint64_t c1 = 0x87c37b91114253d5ULL;
  const uint64_t c2 = 0x4cf5ad432745937fULL;

  for (int i = 0; i < key_count; i++) {
    uint64_t k = static_cast<uint64_t>(key[i * qw_stride]);

    k *= c1;
    k = rotl(k, 31);
    k *= c2;
    h ^= k;
    h = rotl(h, 27);
    h = h * 5 + 0x52dce729;
  }

  h ^= key_count * sizeof(int64_t);
  h = fmix(h);

  return h;
}

__device__ inline uint64_t key_hash_strided(const int64_t* key, const size_t key_count, const size_t qw_stride = 1) {
  return murmur_hash3(key, key_count, qw_stride);
}

__device__ int64_t* get_matching_group_value_strided(int64_t* groups_buffer,
                                                     const size_t groups_count,
                                                     const uint64_t h,
                                                     const int64_t* key,
                                                     const size_t key_count,
                                                     const size_t key_qw_stride = 1) {
  const auto off = h;
  const auto gb_qw_stride = groups_count;
  {
    const uint64_t old = atomicCAS(reinterpret_cast<unsigned long long*>(groups_buffer + off), EMPTY_KEY_64, *key);
    if (EMPTY_KEY_64 == old) {
      for (size_t i = 1; i < key_count; ++i) {
        atomicExch(reinterpret_cast<unsigned long long*>(groups_buffer + i * gb_qw_stride + off),
                   key[i * key_qw_stride]);
      }
    }
  }
  if (key_count > 1) {
    while (atomicAdd(reinterpret_cast<unsigned long long*>(groups_buffer + (key_count - 1) * gb_qw_stride + off), 0) ==
           EMPTY_KEY_64) {
      // spin until the winning thread has finished writing the entire key and the init value
    }
  }
  bool match = true;
  for (uint32_t i = 0; i < key_count; ++i) {
    if (groups_buffer[off + i * gb_qw_stride] != key[i * key_qw_stride]) {
      match = false;
      break;
    }
  }
  return match ? groups_buffer + key_count * gb_qw_stride + off : NULL;
}

__device__ int64_t* get_group_value_columnar(int64_t* groups_buffer,
                                             const size_t groups_count,
                                             const int64_t* key,
                                             const size_t key_count,
                                             const size_t key_qw_stride) {
  const auto h = key_hash_strided(key, key_count, key_qw_stride) % groups_count;
  auto matching_group = get_matching_group_value_strided(groups_buffer, groups_count, h, key, key_count, key_qw_stride);
  if (matching_group) {
    return matching_group;
  }
  auto h_probe = (h + 1) % groups_count;
  while (h_probe != h) {
    matching_group =
        get_matching_group_value_strided(groups_buffer, groups_count, h_probe, key, key_count, key_qw_stride);
    if (matching_group) {
      return matching_group;
    }
    h_probe = (h_probe + 1) % groups_count;
  }
  return NULL;
}

template <typename KeyT = int64_t, typename ValT = int64_t>
__global__ void column_runner(int8_t* groups_buffer,
                              const size_t group_count,
                              const int8_t* row_buffer,
                              const size_t row_size,
                              const size_t row_count,
                              const size_t key_count,
                              const size_t val_count,
                              const size_t* val_widths,
                              const OP_KIND* agg_ops) {
  static_assert(thrust::detail::is_same<KeyT, int64_t>::value && thrust::detail::is_same<ValT, int64_t>::value,
                "Unsupported template parameter other than int64_t for now");
  const auto thread_index = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;
  if (thread_index >= row_count) {
    return;
  }
  auto keys_base = row_buffer + sizeof(KeyT) * thread_index;
  auto read_base = row_buffer + sizeof(KeyT) * row_count * key_count + sizeof(ValT) * thread_index;
  auto write_base = reinterpret_cast<int8_t*>(get_group_value_columnar(reinterpret_cast<int64_t*>(groups_buffer),
                                                                       group_count,
                                                                       reinterpret_cast<const int64_t*>(keys_base),
                                                                       key_count,
                                                                       row_count));
  if (write_base) {
    row_func(write_base, group_count, read_base, row_count, val_count, val_widths, agg_ops);
  }
}

template <typename KeyT = int64_t, typename ValT = int64_t>
__global__ void row_runner(int8_t* groups_buffer,
                           const size_t group_count,
                           const int8_t* row_buffer,
                           const size_t row_size,
                           const size_t row_count,
                           const size_t key_count,
                           const size_t val_count,
                           const size_t* val_widths,
                           const OP_KIND* agg_ops) {
  static_assert(thrust::detail::is_same<KeyT, int64_t>::value && thrust::detail::is_same<ValT, int64_t>::value,
                "Unsupported template parameter other than int64_t for now");
  const auto thread_index = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;
  if (thread_index >= row_count) {
    return;
  }
  auto keys_base = row_buffer + row_size * thread_index;
  auto read_base = keys_base + sizeof(KeyT) * key_count;
  auto write_base = reinterpret_cast<int8_t*>(get_group_value(reinterpret_cast<int64_t*>(groups_buffer),
                                                              static_cast<uint32_t>(group_count),
                                                              reinterpret_cast<const int64_t*>(keys_base),
                                                              static_cast<uint32_t>(key_count),
                                                              static_cast<uint32_t>(row_size / sizeof(int64_t)),
                                                              NULL));
  if (write_base) {
    row_func(write_base, 1, read_base, 1, val_count, val_widths, agg_ops);
  }
}

}  // namespace

void run_query_on_device(int8_t* groups_buffer,
                         const size_t group_count,
                         const int8_t* row_buffer,
                         const size_t row_count,
                         const size_t key_count,
                         const size_t val_count,
                         const std::vector<size_t>& col_widths,
                         const std::vector<OP_KIND>& agg_ops,
                         const bool is_columnar) {
  CHECK_EQ(val_count, agg_ops.size());
  CHECK_EQ(key_count + val_count, col_widths.size());
  size_t row_size = 0;
  for (size_t i = 0; i < col_widths.size(); ++i) {
    row_size += col_widths[i];
  }
  thrust::device_vector<size_t> dev_col_widths(col_widths);
  thrust::device_vector<OP_KIND> dev_agg_ops(agg_ops);
  if (is_columnar) {
    column_runner<<<compute_grid_dim(row_count), c_block_size>>>(
        groups_buffer,
        group_count,
        row_buffer,
        row_size,
        row_count,
        key_count,
        val_count,
        thrust::raw_pointer_cast(dev_col_widths.data() + key_count),
        thrust::raw_pointer_cast(dev_agg_ops.data()));
  } else {
    row_runner<<<compute_grid_dim(row_count), c_block_size>>>(
        groups_buffer,
        group_count,
        row_buffer,
        row_size,
        row_count,
        key_count,
        val_count,
        thrust::raw_pointer_cast(dev_col_widths.data() + key_count),
        thrust::raw_pointer_cast(dev_agg_ops.data()));
  }
}

#if defined(TRY_MASH) || defined(TRY_MASH_COLUMNAR)
namespace {

__device__ int64_t* mash_get_matching_group_value(int64_t* groups_buffer,
                                                  const uint32_t h,
                                                  const int64_t* key,
                                                  const uint32_t key_qw_count,
                                                  const uint32_t entry_size_quad,
                                                  const int64_t* init_vals) {
#ifdef SAVE_MASH_BUF
  const uint32_t keys_size_quad = 1;
#else
  const auto keys_size_quad = key_qw_count;
#endif
  const uint32_t off = h * entry_size_quad;
  const uint64_t value = key_qw_count == 1 ? key[0] : reinterpret_cast<uint64_t>(key);
  const uint64_t old = atomicCAS(reinterpret_cast<unsigned long long*>(groups_buffer + off), EMPTY_KEY_64, value);
  if (EMPTY_KEY_64 == old) {
    return groups_buffer + off + keys_size_quad;
  }
  if (key_qw_count == 1) {
    return groups_buffer[off] == static_cast<int64_t>(value) ? groups_buffer + off + keys_size_quad : NULL;
  }
  bool match = true;
  const auto curr_key = reinterpret_cast<int64_t*>(groups_buffer[off]);
  for (uint32_t i = 0; i < key_qw_count; ++i) {
    if (curr_key[i] != key[i]) {
      match = false;
      break;
    }
  }
  return match ? groups_buffer + off + keys_size_quad : NULL;
}

__device__ int64_t* mash_get_group_value(int64_t* groups_buffer,
                                         const uint32_t groups_buffer_entry_count,
                                         const int64_t* key,
                                         const uint32_t key_qw_count,
                                         const uint32_t entry_size_quad,
                                         const int64_t* init_vals) {
  const uint32_t h = key_hash(key, key_qw_count) % groups_buffer_entry_count;
  int64_t* matching_group =
      mash_get_matching_group_value(groups_buffer, h, key, key_qw_count, entry_size_quad, init_vals);
  if (matching_group) {
    return matching_group;
  }
  uint32_t h_probe = (h + 1) % groups_buffer_entry_count;
  while (h_probe != h) {
    matching_group =
        mash_get_matching_group_value(groups_buffer, h_probe, key, key_qw_count, entry_size_quad, init_vals);
    if (matching_group) {
      return matching_group;
    }
    h_probe = (h_probe + 1) % groups_buffer_entry_count;
  }
  return NULL;
}

__device__ int64_t* mash_get_matching_group_value_strided(int64_t* groups_buffer,
                                                          const size_t groups_count,
                                                          const uint64_t h,
                                                          const int64_t* key,
                                                          const size_t key_count,
                                                          const size_t key_qw_stride = 1) {
#ifdef SAVE_MASH_BUF
  const uint32_t actual_key_count = 1;
#else
  const auto actual_key_count = key_count;
#endif
  const auto off = h;
  const auto gb_qw_stride = groups_count;
  const uint64_t value = key_count == 1 ? key[0] : reinterpret_cast<uint64_t>(key);
  const uint64_t old = atomicCAS(reinterpret_cast<unsigned long long*>(groups_buffer + off), EMPTY_KEY_64, value);
  if (EMPTY_KEY_64 == old) {
    return groups_buffer + actual_key_count * gb_qw_stride + off;
  }
  if (key_count == 1) {
    return groups_buffer[off] == static_cast<int64_t>(value) ? groups_buffer + actual_key_count * gb_qw_stride + off
                                                             : NULL;
  }
  bool match = true;
  const auto curr_key = reinterpret_cast<int64_t*>(groups_buffer[off]);
  for (uint32_t i = 0; i < key_count; ++i) {
    if (curr_key[i * key_qw_stride] != key[i * key_qw_stride]) {
      match = false;
      break;
    }
  }
  return match ? groups_buffer + actual_key_count * gb_qw_stride + off : NULL;
}

__device__ int64_t* mash_get_group_value_columnar(int64_t* groups_buffer,
                                                  const size_t groups_count,
                                                  const int64_t* key,
                                                  const size_t key_count,
                                                  const size_t key_qw_stride) {
  const auto h = key_hash_strided(key, key_count, key_qw_stride) % groups_count;
  int64_t* matching_group =
      mash_get_matching_group_value_strided(groups_buffer, groups_count, h, key, key_count, key_qw_stride);
  if (matching_group) {
    return matching_group;
  }
  auto h_probe = (h + 1) % groups_count;
  while (h_probe != h) {
    matching_group =
        mash_get_matching_group_value_strided(groups_buffer, groups_count, h_probe, key, key_count, key_qw_stride);
    if (matching_group) {
      return matching_group;
    }
    h_probe = (h_probe + 1) % groups_count;
  }
  return NULL;
}

template <typename KeyT = int64_t, typename ValT = int64_t>
__global__ void mash_column_runner(int8_t* groups_buffer,
                                   const size_t group_count,
                                   const int8_t* row_buffer,
                                   const size_t row_size,
                                   const size_t row_count,
                                   const size_t key_count,
                                   const size_t val_count,
                                   const size_t* val_widths,
                                   const OP_KIND* agg_ops) {
  static_assert(thrust::detail::is_same<KeyT, int64_t>::value && thrust::detail::is_same<ValT, int64_t>::value,
                "Unsupported template parameter other than int64_t for now");
  const auto thread_index = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;
  if (thread_index >= row_count) {
    return;
  }
  auto keys_base = row_buffer + sizeof(KeyT) * thread_index;
  auto read_base = row_buffer + sizeof(KeyT) * row_count * key_count + sizeof(ValT) * thread_index;
  auto write_base = reinterpret_cast<int8_t*>(mash_get_group_value_columnar(reinterpret_cast<int64_t*>(groups_buffer),
                                                                            group_count,
                                                                            reinterpret_cast<const int64_t*>(keys_base),
                                                                            key_count,
                                                                            row_count));
  if (write_base) {
    row_func(write_base, group_count, read_base, row_count, val_count, val_widths, agg_ops);
  }
}

template <typename KeyT = int64_t, typename ValT = int64_t>
__global__ void mash_row_runner(int8_t* groups_buffer,
                                const size_t group_count,
                                const size_t entry_size,
                                const int8_t* row_buffer,
                                const size_t row_size,
                                const size_t row_count,
                                const size_t key_count,
                                const size_t val_count,
                                const size_t* val_widths,
                                const OP_KIND* agg_ops) {
  static_assert(thrust::detail::is_same<KeyT, int64_t>::value && thrust::detail::is_same<ValT, int64_t>::value,
                "Unsupported template parameter other than int64_t for now");
  const auto thread_index = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;
  if (thread_index >= row_count) {
    return;
  }
  auto keys_base = row_buffer + row_size * thread_index;
  auto write_base = reinterpret_cast<int8_t*>(mash_get_group_value(reinterpret_cast<int64_t*>(groups_buffer),
                                                                   static_cast<uint32_t>(group_count),
                                                                   reinterpret_cast<const int64_t*>(keys_base),
                                                                   static_cast<uint32_t>(key_count),
                                                                   static_cast<uint32_t>(entry_size / sizeof(int64_t)),
                                                                   NULL));
  if (write_base) {
    auto read_base = keys_base + sizeof(KeyT) * key_count;
    row_func(write_base, 1, read_base, 1, val_count, val_widths, agg_ops);
  }
}

template <typename T = int64_t, bool isColumnar = false>
struct PtrRestorer {
  static_assert(thrust::detail::is_same<T, int64_t>::value,
                "Unsupported template parameter other than int64_t for now");
  PtrRestorer(int8_t* buff, const size_t key_num, const size_t entry_num, const size_t row_size, const size_t row_num)
      : buff_ptr(buff), key_count(key_num), entry_count(entry_num), entry_size(row_size), row_count(row_num) {}
  __device__ void operator()(const int index) {
    const auto value = *reinterpret_cast<T*>(buff_ptr + (isColumnar ? sizeof(T) : entry_size) * index);
    if (is_empty_slot(value)) {
      return;
    }
    auto dst = reinterpret_cast<T*>(buff_ptr + (isColumnar ? sizeof(T) : entry_size) * index);
    const size_t key_stride = isColumnar ? row_count : 1;
    const size_t dst_stride = isColumnar ? entry_count : 1;

    auto keys_ptr = reinterpret_cast<T*>(value);
    for (size_t i = 0; i < key_count; ++i) {
      dst[i * dst_stride] = keys_ptr[i * key_stride];
    }
  }
  int8_t* buff_ptr;
  const size_t key_count;
  const size_t entry_count;
  const size_t entry_size;
  const size_t row_count;
};

#ifdef SAVE_MASH_BUF
template <typename T = int64_t, bool isColumnar = false>
struct IdxRestorer {
  static_assert(thrust::detail::is_same<T, int64_t>::value,
                "Unsupported template parameter other than int64_t for now");
  IdxRestorer(int8_t* output, const size_t entry_sz, const int8_t* input, const size_t row_sz, const size_t row_num)
      : output_ptr(output), entry_size(entry_sz), input_ptr(input), row_size(row_sz), row_count(row_num) {}
  __device__ void operator()(const int index) {
    const auto value = *reinterpret_cast<T*>(output_ptr + (isColumnar ? sizeof(T) : entry_size) * index);
    if (is_empty_slot(value)) {
      return;
    }
    auto dst_ptr = reinterpret_cast<T*>(output_ptr + (isColumnar ? sizeof(T) : entry_size) * index);
    auto row_ptr = reinterpret_cast<int8_t*>(value);
    *dst_ptr = (row_ptr - input_ptr) / (isColumnar ? sizeof(T) : row_size);
  }
  int8_t* output_ptr;
  const size_t entry_size;
  const int8_t* input_ptr;
  const size_t row_size;
  const size_t row_count;
};
#endif

template <bool isColumnar = false>
void mash_restore_keys(int8_t* groups_buffer,
                       const size_t group_count,
                       const size_t entry_size,
                       const int8_t* input_buffer,
                       const size_t row_count,
                       const size_t key_count,
                       const size_t row_size) {
#ifdef SAVE_MASH_BUF
  thrust::for_each(thrust::make_counting_iterator(size_t(0)),
                   thrust::make_counting_iterator(group_count),
                   IdxRestorer<int64_t, isColumnar>(groups_buffer, entry_size, input_buffer, row_size, row_count));
#else
  thrust::for_each(thrust::make_counting_iterator(size_t(0)),
                   thrust::make_counting_iterator(group_count),
                   PtrRestorer<int64_t, isColumnar>(groups_buffer, key_count, group_count, row_size, row_count));
#endif
}

}  // namespace

void mash_run_query_on_device(int8_t* groups_buffer,
                              const size_t group_count,
                              const int8_t* row_buffer,
                              const size_t row_count,
                              const size_t key_count,
                              const size_t val_count,
                              const std::vector<size_t>& col_widths,
                              const std::vector<OP_KIND>& agg_ops,
                              const bool is_columnar) {
  CHECK_EQ(val_count, agg_ops.size());
  CHECK_EQ(key_count + val_count, col_widths.size());
  size_t row_size = 0;
  for (size_t i = 0; i < col_widths.size(); ++i) {
    row_size += col_widths[i];
  }
#ifdef SAVE_MASH_BUF
  size_t entry_size = sizeof(int64_t);
  for (size_t i = key_count; i < col_widths.size(); ++i) {
    entry_size += col_widths[i];
  }
#else
  const auto entry_size = row_size;
#endif
  thrust::device_vector<size_t> dev_col_widths(col_widths);
  thrust::device_vector<OP_KIND> dev_agg_ops(agg_ops);
  if (is_columnar) {
    mash_column_runner<<<compute_grid_dim(row_count), c_block_size>>>(
        groups_buffer,
        group_count,
        row_buffer,
        row_size,
        row_count,
        key_count,
        val_count,
        thrust::raw_pointer_cast(dev_col_widths.data() + key_count),
        thrust::raw_pointer_cast(dev_agg_ops.data()));
    if (key_count > 1) {
      mash_restore_keys<true>(groups_buffer, group_count, entry_size, row_buffer, row_count, key_count, row_size);
    }
  } else {
    mash_row_runner<<<compute_grid_dim(row_count), c_block_size>>>(
        groups_buffer,
        group_count,
        entry_size,
        row_buffer,
        row_size,
        row_count,
        key_count,
        val_count,
        thrust::raw_pointer_cast(dev_col_widths.data() + key_count),
        thrust::raw_pointer_cast(dev_agg_ops.data()));
    if (key_count > 1) {
      mash_restore_keys<false>(groups_buffer, group_count, entry_size, row_buffer, row_count, key_count, row_size);
    }
  }
}
#endif  // TRY_MASH || TRY_MASH_COLUMNAR

namespace {

__device__ int32_t* get_matching_bucket(int32_t* hash_table,
                                        const uint32_t h,
                                        const int64_t* key,
                                        const int8_t* row_buffer,
                                        const size_t key_count,
                                        const size_t row_size,
                                        bool* is_owner) {
  const auto value = static_cast<int>(reinterpret_cast<const int8_t*>(key) - row_buffer) / row_size;
  const auto old = atomicCAS(reinterpret_cast<int*>(hash_table + h), int(EMPTY_KEY_32), value);
  if (EMPTY_KEY_32 == old) {
    *is_owner = true;
    return hash_table + h;
  }
  bool match = true;
  const auto curr_key = reinterpret_cast<const int64_t*>(row_buffer + hash_table[h] * row_size);
  for (uint32_t i = 0; i < key_count; ++i) {
    if (curr_key[i] != key[i]) {
      match = false;
      break;
    }
  }
  if (match) {
    *is_owner = false;
  }
  return match ? hash_table + h : NULL;
}

__device__ bool acquire_bucket(int32_t* hash_table,
                               const uint32_t bucket_count,
                               const int64_t* key,
                               const int8_t* row_buffer,
                               const size_t key_count,
                               const size_t row_size) {
  const auto h = key_hash(key, key_count) % bucket_count;
  bool is_owner = false;
  auto matching_bucket = get_matching_bucket(hash_table, h, key, row_buffer, key_count, row_size, &is_owner);
  if (matching_bucket) {
    return is_owner;
  }
  uint32_t h_probe = (h + 1) % bucket_count;
  while (h_probe != h) {
    matching_bucket = get_matching_bucket(hash_table, h_probe, key, row_buffer, key_count, row_size, &is_owner);
    if (matching_bucket) {
      return is_owner;
    }
    h_probe = (h_probe + 1) % bucket_count;
  }
  return false;
}

template <typename KeyT = int64_t>
__global__ void column_deduplicater(int8_t* row_buffer,
                                    const size_t row_count,
                                    const size_t row_size,
                                    const size_t key_count,
                                    int32_t* hash_table,
                                    const size_t bucket_count) {
  static_assert(thrust::detail::is_same<KeyT, int64_t>::value,
                "Unsupported template parameter other than int64_t for now");
}

template <typename KeyT = int64_t>
__global__ void row_deduplicater(int8_t* row_buffer,
                                 const size_t row_count,
                                 const size_t row_size,
                                 const size_t key_count,
                                 int32_t* hash_table,
                                 const size_t bucket_count) {
  static_assert(thrust::detail::is_same<KeyT, int64_t>::value,
                "Unsupported template parameter other than int64_t for now");
  const auto thread_index = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;

  if (thread_index >= row_count) {
    return;
  }

  auto keys_base = row_buffer + row_size * thread_index;
  auto keys_i64 = reinterpret_cast<KeyT*>(keys_base);
  bool is_owner =
      acquire_bucket(hash_table, static_cast<uint32_t>(bucket_count), keys_i64, row_buffer, key_count, row_size);
  if (!is_owner) {
    reset_entry(keys_i64);
  }
}

template <bool isColumnar = false, typename T = int64_t>
struct RowCounter {
  static_assert(thrust::detail::is_same<T, int64_t>::value,
                "Unsupported template parameter other than int64_t for now");
  RowCounter(uint32_t* row_num, const int8_t* buff, const size_t entry_sz)
      : row_count(row_num), buff_ptr(buff), entry_size(entry_sz) {}
  __device__ void operator()(const int index) {
    const auto value = *reinterpret_cast<const T*>(buff_ptr + (isColumnar ? sizeof(T) : entry_size) * index);
    if (is_empty_slot(value)) {
      return;
    }
    atomicAdd(row_count, 1UL);
  }

  uint32_t* row_count;
  const int8_t* buff_ptr;
  const size_t entry_size;
};

template <bool isColumnar = false>
inline size_t count_rows(const int8_t* input_buffer, const size_t entry_count, const size_t row_size) {
  thrust::device_vector<uint32_t> row_count(1, 0);
  thrust::for_each(thrust::make_counting_iterator(size_t(0)),
                   thrust::make_counting_iterator(entry_count),
                   RowCounter<isColumnar>(thrust::raw_pointer_cast(row_count.data()), input_buffer, row_size));
  return static_cast<size_t>(row_count[0]);
}

template <bool isColumnar = false, typename T = int64_t>
struct InplaceCompactor {
  static_assert(thrust::detail::is_same<T, int64_t>::value,
                "Unsupported template parameter other than int64_t for now");
  InplaceCompactor(uint32_t* w,
                   const uint32_t e,
                   int8_t* b,
                   const size_t e_cnt,
                   const size_t c_cnt,
                   const size_t* c_wids,
                   const size_t e_sz)
      : walker(w), end(e), buff_ptr(b), entry_count(e_cnt), col_count(c_cnt), col_widths(c_wids), entry_size(e_sz) {}
  __device__ void operator()(const int index) {
    const auto stride = isColumnar ? sizeof(T) : entry_size;
    const auto value = *reinterpret_cast<T*>(buff_ptr + stride * index);
    if (!is_empty_slot(static_cast<T>(value))) {
      return;
    }
    bool found = false;
    T* key_ptr = nullptr;
    for (auto curr_row = atomicSub(walker, 1UL); curr_row > end; curr_row = atomicSub(walker, 1UL)) {
      key_ptr = reinterpret_cast<T*>(buff_ptr + stride * curr_row);
      if (!is_empty_slot(*key_ptr)) {
        found = true;
        break;
      }
    }
    if (found) {
      auto dst_ptr = buff_ptr + stride * index;
      if (isColumnar) {
        auto src_ptr = reinterpret_cast<int8_t*>(key_ptr);
        for (size_t i = 0; i < col_count; ++i) {
          switch (col_widths[i]) {
            case 4:
              *reinterpret_cast<int32_t*>(dst_ptr) = *reinterpret_cast<int32_t*>(src_ptr);
              break;
            case 8:
              *reinterpret_cast<int64_t*>(dst_ptr) = *reinterpret_cast<int64_t*>(src_ptr);
              break;
            default:;
          }
          dst_ptr += col_widths[i] * entry_count;
          src_ptr += col_widths[i] * entry_count;
        }
      } else {
        memcpy(dst_ptr, key_ptr, entry_size);
      }
      reset_entry(key_ptr);
    }
  }

  uint32_t* walker;
  uint32_t end;
  int8_t* buff_ptr;
  const size_t entry_count;
  const size_t col_count;
  const size_t* col_widths;
  const size_t entry_size;
};

template <bool isColumnar = false>
inline size_t compact_buffer(int8_t* input_buffer, const size_t entry_count, const std::vector<size_t>& col_widths) {
  const auto col_count = col_widths.size();
  size_t entry_size = 0;
  for (size_t i = 0; i < col_count; ++i) {
    entry_size += col_widths[i];
  }
  const auto actual_row_count = count_rows<isColumnar>(input_buffer, entry_count, entry_size);
  if (actual_row_count > static_cast<size_t>(entry_count * 0.4f)) {
    return entry_count;
  }
  thrust::device_vector<size_t> dev_col_widths(col_widths);
  thrust::device_vector<uint32_t> walker(1, entry_count - 1);
  thrust::for_each(thrust::make_counting_iterator(size_t(0)),
                   thrust::make_counting_iterator(actual_row_count),
                   InplaceCompactor<isColumnar>(thrust::raw_pointer_cast(walker.data()),
                                                static_cast<uint32_t>(actual_row_count - 1),
                                                input_buffer,
                                                entry_count,
                                                col_count,
                                                thrust::raw_pointer_cast(dev_col_widths.data()),
                                                entry_size));
  return actual_row_count;
}

template <bool isColumnar = false, typename T = int64_t>
struct Checker {
  static_assert(thrust::detail::is_same<T, int64_t>::value,
                "Unsupported template parameter other than int64_t for now");
  Checker(uint32_t* cnt, const int8_t* buff, const size_t entry_sz)
      : count(cnt), buff_ptr(buff), entry_size(entry_sz) {}
  __device__ void operator()(const int index) {
    const auto value = *reinterpret_cast<const T*>(buff_ptr + (isColumnar ? sizeof(T) : entry_size) * index);
    const auto next_value = *reinterpret_cast<const T*>(buff_ptr + (isColumnar ? sizeof(T) : entry_size) * (index + 1));
    if ((!is_empty_slot(value) && is_empty_slot(next_value)) || (is_empty_slot(value) && !is_empty_slot(next_value))) {
      atomicAdd(count, 1UL);
    }
  }

  uint32_t* count;
  const int8_t* buff_ptr;
  const size_t entry_size;
};

template <bool isColumnar = false>
inline bool is_compacted(const int8_t* input_buffer, const size_t entry_count, const size_t row_size) {
  thrust::device_vector<uint32_t> count(1, 0);
  thrust::for_each(thrust::make_counting_iterator(size_t(0)),
                   thrust::make_counting_iterator(entry_count - 1),
                   Checker<isColumnar>(thrust::raw_pointer_cast(count.data()), input_buffer, row_size));
  return count[0] == 1;
}

}  // namespace

size_t deduplicate_rows_on_device(int8_t* row_buffer,
                                  const size_t row_count,
                                  const size_t key_count,
                                  const std::vector<size_t>& col_widths,
                                  const bool is_columnar) {
  CHECK_GT(col_widths.size(), key_count);
  size_t row_size = 0;
  for (auto wid : col_widths) {
    row_size += wid;
  }
  const auto bucket_count = static_cast<size_t>(row_count * 1.3f);
  thrust::device_vector<int32_t> hash_table(bucket_count, int32_t(EMPTY_KEY_32));
  if (is_columnar) {
    column_deduplicater<<<compute_grid_dim(row_count), c_block_size>>>(
        row_buffer, row_count, row_size, key_count, thrust::raw_pointer_cast(hash_table.data()), bucket_count);
  } else {
    row_deduplicater<<<compute_grid_dim(row_count), c_block_size>>>(
        row_buffer, row_count, row_size, key_count, thrust::raw_pointer_cast(hash_table.data()), bucket_count);
  }

  return (is_columnar ? count_rows<true>(row_buffer, row_count, row_size)
                      : count_rows<false>(row_buffer, row_count, row_size));
}

namespace {
template <bool isColumnar = false, typename T = int64_t>
struct Dropper {
  static_assert(thrust::detail::is_same<T, int64_t>::value,
                "Unsupported template parameter other than int64_t for now");
  Dropper(int8_t* buff, uint32_t* ub, const size_t row_cnt, const size_t entry_sz)
      : buff_ptr(buff), upper_bound(ub), row_count(row_cnt), entry_size(entry_sz) {}
  __device__ void operator()(const int index) {
    auto key_ptr = reinterpret_cast<T*>(buff_ptr + (isColumnar ? sizeof(T) : entry_size) * index);
    if (is_empty_slot(*key_ptr)) {
      return;
    }
    if (atomicAdd(upper_bound, 1UL) <= row_count) {
      reset_entry(key_ptr);
    }
  }

  int8_t* buff_ptr;
  uint32_t* upper_bound;
  const uint32_t row_count;
  const size_t entry_size;
};

}  // namespace

size_t drop_rows(int8_t* row_buffer,
                 const size_t entry_count,
                 const size_t entry_size,
                 const size_t row_count,
                 const float fill_rate,
                 const bool is_columnar) {
  auto limit = static_cast<size_t>(entry_count * fill_rate);
  if (row_count < limit) {
    return row_count;
  }
  thrust::device_vector<uint32_t> upper_bound(1, static_cast<uint32_t>(limit));

  if (is_columnar) {
    thrust::for_each(thrust::make_counting_iterator(size_t(0)),
                     thrust::make_counting_iterator(entry_count),
                     Dropper<true>(row_buffer, thrust::raw_pointer_cast(upper_bound.data()), row_count, entry_size));
  } else {
    thrust::for_each(thrust::make_counting_iterator(size_t(0)),
                     thrust::make_counting_iterator(entry_count),
                     Dropper<false>(row_buffer, thrust::raw_pointer_cast(upper_bound.data()), row_count, entry_size));
  }
  return limit;
}

namespace {

__device__ inline void reduce_func(int8_t* write_base,
                                   const size_t write_stride,
                                   const int8_t* read_base,
                                   const size_t read_stride,
                                   const size_t val_count,
                                   const size_t* val_widths,
                                   const OP_KIND* agg_ops) {
  for (size_t i = 0; i < val_count; ++i) {
    switch (val_widths[i]) {
      case 4: {
        auto write_ptr = reinterpret_cast<int32_t*>(write_base);
        const auto value = *reinterpret_cast<const int32_t*>(read_base);
        switch (agg_ops[i]) {
          case OP_COUNT:
          case OP_SUM:
            agg_sum_int32_shared(write_ptr, value);
            break;
          case OP_MIN:
            agg_min_int32_shared(write_ptr, value);
            break;
          case OP_MAX:
            agg_max_int32_shared(write_ptr, value);
            break;
          default:;
        }
      } break;
      case 8: {
        auto write_ptr = reinterpret_cast<int64_t*>(write_base);
        const auto value = *reinterpret_cast<const int64_t*>(read_base);
        switch (agg_ops[i]) {
          case OP_COUNT:
          case OP_SUM:
            agg_sum_shared(write_ptr, value);
            break;
          case OP_MIN:
            agg_min_shared(write_ptr, value);
            break;
          case OP_MAX:
            agg_max_shared(write_ptr, value);
            break;
          default:;
        }
      } break;
      default:;
    }
    write_base += val_widths[i] * write_stride;
    read_base += val_widths[i] * read_stride;
  }
}

template <typename KeyT = int64_t, typename ValT = int64_t>
__global__ void column_reducer(int8_t* this_buffer,
                               const size_t this_entry_count,
                               int8_t* that_buffer,
                               const size_t that_entry_count,
                               const size_t entry_size,
                               const size_t key_count,
                               const size_t val_count,
                               const size_t* val_widths,
                               const OP_KIND* agg_ops) {
  static_assert(thrust::detail::is_same<KeyT, int64_t>::value && thrust::detail::is_same<ValT, int64_t>::value,
                "Unsupported template parameter other than int64_t for now");
}

template <typename KeyT = int64_t, typename ValT = int64_t>
__global__ void row_reducer(int8_t* this_buffer,
                            const size_t this_entry_count,
                            int8_t* that_buffer,
                            const size_t that_entry_count,
                            const size_t entry_size,
                            const size_t key_count,
                            const size_t val_count,
                            const size_t* val_widths,
                            const OP_KIND* agg_ops) {
  static_assert(thrust::detail::is_same<KeyT, int64_t>::value && thrust::detail::is_same<ValT, int64_t>::value,
                "Unsupported template parameter other than int64_t for now");
  const auto thread_index = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;
  auto keys_base = that_buffer + entry_size * thread_index;
  auto keys_i64 = reinterpret_cast<KeyT*>(keys_base);

  if (thread_index >= that_entry_count || is_empty_slot(*keys_i64)) {
    return;
  }

  auto write_base = reinterpret_cast<int8_t*>(get_group_value(reinterpret_cast<int64_t*>(this_buffer),
                                                              static_cast<uint32_t>(this_entry_count),
                                                              keys_i64,
                                                              static_cast<uint32_t>(key_count),
                                                              static_cast<uint32_t>(entry_size / sizeof(int64_t)),
                                                              NULL));
  if (write_base) {
    auto read_base = keys_base + sizeof(KeyT) * key_count;
    reduce_func(write_base, 1, read_base, 1, val_count, val_widths, agg_ops);
  }
}

}  // namespace

int8_t* get_hashed_copy(int8_t* dev_buffer,
                        const size_t entry_count,
                        const size_t new_entry_count,
                        const std::vector<size_t>& col_widths,
                        const std::vector<OP_KIND>& agg_ops,
                        const std::vector<size_t>& init_vals,
                        const bool is_columnar) {
  const size_t val_count = agg_ops.size();
  const size_t key_count = col_widths.size() - val_count;
  size_t entry_size = 0;
  for (size_t i = 0; i < col_widths.size(); ++i) {
    entry_size += col_widths[i];
  }
  thrust::device_vector<size_t> dev_col_widths(col_widths);
  thrust::device_vector<OP_KIND> dev_agg_ops(agg_ops);
  thrust::device_vector<size_t> dev_init_vals(init_vals);

  int8_t* dev_copy = nullptr;
  cudaMalloc(&dev_copy, entry_size * new_entry_count);
  if (is_columnar) {
    init_group<true><<<compute_grid_dim(new_entry_count), c_block_size>>>(
        dev_copy,
        new_entry_count,
        dev_col_widths.size(),
        thrust::raw_pointer_cast(dev_col_widths.data()),
        thrust::raw_pointer_cast(dev_init_vals.data()));
    column_reducer<<<compute_grid_dim(entry_count), c_block_size>>>(
        dev_copy,
        new_entry_count,
        dev_buffer,
        entry_count,
        entry_size,
        key_count,
        val_count,
        thrust::raw_pointer_cast(dev_col_widths.data() + key_count),
        thrust::raw_pointer_cast(dev_agg_ops.data()));
  } else {
    init_group<false><<<compute_grid_dim(new_entry_count), c_block_size>>>(
        dev_copy,
        new_entry_count,
        col_widths.size(),
        thrust::raw_pointer_cast(dev_col_widths.data()),
        thrust::raw_pointer_cast(dev_init_vals.data()));
    row_reducer<<<compute_grid_dim(entry_count), c_block_size>>>(
        dev_copy,
        new_entry_count,
        dev_buffer,
        entry_count,
        entry_size,
        key_count,
        val_count,
        thrust::raw_pointer_cast(dev_col_widths.data() + key_count),
        thrust::raw_pointer_cast(dev_agg_ops.data()));
  }
  return dev_copy;
}

void reduce_on_device(int8_t*& this_dev_buffer,
                      const size_t this_dev_id,
                      size_t& this_entry_count,
                      int8_t* that_dev_buffer,
                      const size_t that_dev_id,
                      const size_t that_entry_count,
                      const size_t that_actual_row_count,
                      const std::vector<size_t>& col_widths,
                      const std::vector<OP_KIND>& agg_ops,
                      const std::vector<size_t>& init_vals,
                      const bool is_columnar) {
  CHECK_EQ(col_widths.size(), init_vals.size());
  const size_t val_count = agg_ops.size();
  const size_t key_count = col_widths.size() - val_count;
  size_t entry_size = 0;
  for (size_t i = 0; i < col_widths.size(); ++i) {
    entry_size += col_widths[i];
  }

  thrust::device_vector<size_t> dev_col_widths(col_widths);
  thrust::device_vector<OP_KIND> dev_agg_ops(agg_ops);
  auto total_row_count = (is_columnar ? count_rows<true>(this_dev_buffer, this_entry_count, entry_size)
                                      : count_rows<false>(this_dev_buffer, this_entry_count, entry_size)) +
                         that_actual_row_count;
  const auto threshold = static_cast<size_t>(total_row_count * 1.3f);
  if (threshold > this_entry_count) {
    total_row_count = std::min(threshold, this_entry_count + that_entry_count);
    thrust::device_vector<size_t> dev_init_vals(init_vals);
    auto this_dev_copy = get_hashed_copy(
        this_dev_buffer, this_entry_count, total_row_count, col_widths, agg_ops, init_vals, is_columnar);

    cudaFree(this_dev_buffer);
    this_dev_buffer = this_dev_copy;
    this_entry_count = total_row_count;
  }
  int8_t* that_dev_copy = nullptr;
  if (that_dev_id != this_dev_id) {
    cudaMalloc(&that_dev_copy, that_entry_count * entry_size);
    int canAccessPeer;
    cudaDeviceCanAccessPeer(&canAccessPeer, this_dev_id, that_dev_id);
    if (canAccessPeer) {
      cudaDeviceEnablePeerAccess(that_dev_id, 0);
    }
    cudaMemcpyPeer(that_dev_copy, this_dev_id, that_dev_buffer, that_dev_id, that_entry_count * entry_size);
  } else {
    that_dev_copy = that_dev_buffer;
  }

  if (is_columnar) {
    column_reducer<<<compute_grid_dim(that_entry_count), c_block_size>>>(
        this_dev_buffer,
        this_entry_count,
        that_dev_copy,
        that_entry_count,
        entry_size,
        key_count,
        val_count,
        thrust::raw_pointer_cast(dev_col_widths.data() + key_count),
        thrust::raw_pointer_cast(dev_agg_ops.data()));
    this_entry_count = compact_buffer<true>(this_dev_buffer, this_entry_count, col_widths);
  } else {
    row_reducer<<<compute_grid_dim(that_entry_count), c_block_size>>>(
        this_dev_buffer,
        this_entry_count,
        that_dev_copy,
        that_entry_count,
        entry_size,
        key_count,
        val_count,
        thrust::raw_pointer_cast(dev_col_widths.data() + key_count),
        thrust::raw_pointer_cast(dev_agg_ops.data()));
    this_entry_count = compact_buffer<false>(this_dev_buffer, this_entry_count, col_widths);
  }

  cudaFree(that_dev_copy);
}

namespace {

template <typename KeyT = int64_t>
__device__ size_t get_perfect_hash_index(const KeyT* key_base,
                                         const size_t key_count,
                                         const KeyT* min_keys,
                                         const KeyT* max_keys,
                                         const size_t stride) {
  size_t hash_index = 0;
  for (size_t k = 0; k < key_count; ++k) {
    if (k > 0) {
      hash_index *= static_cast<size_t>(max_keys[k] - min_keys[k] + 1);
    }
    hash_index += static_cast<size_t>(key_base[k * stride] - min_keys[k]);
  }
  return hash_index;
}

template <typename KeyT = int64_t, typename ValT = int64_t>
__global__ void col_perfect_placer(int8_t* new_buffer,
                                   const size_t write_stride,
                                   const int8_t* old_buffer,
                                   const size_t entry_count,
                                   const size_t read_stride,
                                   const size_t key_count,
                                   const KeyT* min_keys,
                                   const KeyT* max_keys,
                                   const size_t val_count,
                                   const size_t* val_widths) {
  static_assert(thrust::detail::is_same<KeyT, int64_t>::value && thrust::detail::is_same<ValT, int64_t>::value,
                "Unsupported template parameter other than int64_t for now");
  const auto thread_index = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;

  auto keys_base = old_buffer + sizeof(KeyT) * thread_index;
  auto keys_i64 = reinterpret_cast<const KeyT*>(keys_base);

  if (thread_index >= entry_count || is_empty_slot(*keys_i64)) {
    return;
  }

  auto read_base = keys_base;
  auto write_base =
      new_buffer + sizeof(KeyT) * get_perfect_hash_index(keys_i64, key_count, min_keys, max_keys, read_stride);
  const auto old = atomicCAS(
      reinterpret_cast<unsigned long long*>(write_base), EMPTY_KEY_64, static_cast<unsigned long long>(*keys_i64));

  if (is_empty_slot(static_cast<KeyT>(old))) {
    for (size_t i = 0; i < key_count;
         ++i, write_base += write_stride * sizeof(KeyT), read_base += read_stride * sizeof(KeyT)) {
      *reinterpret_cast<KeyT*>(write_base) = *reinterpret_cast<const KeyT*>(read_base);
    }
    for (size_t i = 0; i < val_count;
         ++i, write_base += write_stride * sizeof(ValT), read_base += read_stride * sizeof(ValT)) {
      *reinterpret_cast<ValT*>(write_base) = *reinterpret_cast<const ValT*>(read_base);
    }
  }
}

template <typename KeyT = int64_t, typename ValT = int64_t>
__global__ void row_perfect_placer(int8_t* new_buffer,
                                   const int8_t* old_buffer,
                                   const size_t entry_count,
                                   const size_t entry_size,
                                   const size_t key_count,
                                   const KeyT* min_keys,
                                   const KeyT* max_keys,
                                   const size_t val_count,
                                   const size_t* val_widths) {
  static_assert(thrust::detail::is_same<KeyT, int64_t>::value && thrust::detail::is_same<ValT, int64_t>::value,
                "Unsupported template parameter other than int64_t for now");
  const auto thread_index = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;

  auto keys_base = old_buffer + entry_size * thread_index;
  auto keys_i64 = reinterpret_cast<const KeyT*>(keys_base);

  if (thread_index >= entry_count || is_empty_slot(*keys_i64)) {
    return;
  }

  auto write_base = new_buffer + entry_size * get_perfect_hash_index(keys_i64, key_count, min_keys, max_keys, 1);
  const auto old = atomicCAS(
      reinterpret_cast<unsigned long long*>(write_base), EMPTY_KEY_64, static_cast<unsigned long long>(*keys_i64));

  if (is_empty_slot(static_cast<KeyT>(old))) {
    memcpy(write_base, keys_base, entry_size);
  }
}

}  // namespace

std::pair<int8_t*, size_t> get_perfect_hashed_copy(int8_t* dev_buffer,
                                                   const size_t entry_count,
                                                   const std::vector<size_t>& col_widths,
                                                   const std::vector<std::pair<int64_t, int64_t>>& ranges,
                                                   const std::vector<OP_KIND>& agg_ops,
                                                   const std::vector<size_t>& init_vals,
                                                   const bool is_columnar) {
  const size_t val_count = agg_ops.size();
  const size_t key_count = col_widths.size() - val_count;
  size_t entry_size = 0;
  for (size_t i = 0; i < col_widths.size(); ++i) {
    entry_size += col_widths[i];
  }
  thrust::device_vector<size_t> dev_col_widths(col_widths);
  thrust::device_vector<OP_KIND> dev_agg_ops(agg_ops);
  thrust::device_vector<size_t> dev_init_vals(init_vals);
  std::vector<int64_t> min_keys(key_count, 0);
  std::vector<int64_t> max_keys(key_count, 0);
  for (size_t k = 0; k < key_count; ++k) {
    min_keys[k] = ranges[k].first;
    max_keys[k] = ranges[k].second;
  }
  thrust::device_vector<int64_t> dev_min_keys(min_keys);
  thrust::device_vector<int64_t> dev_max_keys(max_keys);
  int8_t* dev_copy = nullptr;
  cudaMalloc(&dev_copy, entry_size * entry_count);
  if (is_columnar) {
    init_group<true><<<compute_grid_dim(entry_count), c_block_size>>>(dev_copy,
                                                                      entry_count,
                                                                      col_widths.size(),
                                                                      thrust::raw_pointer_cast(dev_col_widths.data()),
                                                                      thrust::raw_pointer_cast(dev_init_vals.data()));
    col_perfect_placer<<<compute_grid_dim(entry_count), c_block_size>>>(
        dev_copy,
        entry_count,
        dev_buffer,
        entry_count,
        entry_count,
        key_count,
        thrust::raw_pointer_cast(dev_min_keys.data()),
        thrust::raw_pointer_cast(dev_max_keys.data()),
        val_count,
        thrust::raw_pointer_cast(dev_col_widths.data() + key_count));
  } else {
    init_group<false><<<compute_grid_dim(entry_count), c_block_size>>>(dev_copy,
                                                                       entry_count,
                                                                       col_widths.size(),
                                                                       thrust::raw_pointer_cast(dev_col_widths.data()),
                                                                       thrust::raw_pointer_cast(dev_init_vals.data()));

    row_perfect_placer<<<compute_grid_dim(entry_count), c_block_size>>>(
        dev_copy,
        dev_buffer,
        entry_count,
        entry_size,
        key_count,
        thrust::raw_pointer_cast(dev_min_keys.data()),
        thrust::raw_pointer_cast(dev_max_keys.data()),
        val_count,
        thrust::raw_pointer_cast(dev_col_widths.data() + key_count));
  }

  const auto actual_entry_count = (is_columnar ? count_rows<true>(dev_copy, entry_count, entry_size)
                                               : count_rows<false>(dev_copy, entry_count, entry_size));
  return {dev_copy, actual_entry_count};
}

int8_t* fetch_segs_from_others(std::vector<int8_t*>& dev_reduced_buffers,
                               const size_t entry_count,
                               const size_t dev_id,
                               const size_t dev_count,
                               const std::vector<size_t>& col_widths,
                               const bool is_columnar,
                               const size_t start,
                               const size_t end) {
  const size_t col_count = col_widths.size();
  size_t entry_size = 0;
  for (size_t i = 0; i < col_widths.size(); ++i) {
    entry_size += col_widths[i];
  }
  const size_t seg_entry_count = end - start;
  const size_t read_stride = entry_count;
  const size_t write_stride = seg_entry_count * (dev_count - 1);
  int8_t* dev_segs_copy = nullptr;
  cudaMalloc(&dev_segs_copy, write_stride * entry_size);
  for (size_t i = (dev_id + 1) % dev_count, offset = 0; i != dev_id; i = (i + 1) % dev_count) {
    int canAccessPeer;
    cudaDeviceCanAccessPeer(&canAccessPeer, dev_id, i);
    if (canAccessPeer) {
      cudaDeviceEnablePeerAccess(i, 0);
    }
    if (is_columnar) {
      auto read_ptr = dev_reduced_buffers[i] + start * col_widths[0];
      auto write_ptr = dev_segs_copy + offset * col_widths[0];
      for (size_t j = 0; j < col_count; ++j) {
        cudaMemcpyPeer(write_ptr, dev_id, read_ptr, i, seg_entry_count * col_widths[j]);
        read_ptr += read_stride * col_widths[j];
        write_ptr += write_stride * col_widths[j];
      }
      offset += seg_entry_count;
    } else {
      cudaMemcpyPeer(
          dev_segs_copy + offset, dev_id, dev_reduced_buffers[i] + start * entry_size, i, seg_entry_count * entry_size);
      offset += seg_entry_count * entry_size;
    }
  }
  return dev_segs_copy;
}

namespace {

template <typename KeyT = int64_t, typename ValT = int64_t>
__global__ void col_perfect_reducer(int8_t* this_seg,
                                    const size_t entry_count,
                                    const size_t write_stride,
                                    const int8_t* other_segs,
                                    const size_t seg_count,
                                    const size_t read_stride,
                                    const size_t key_count,
                                    const size_t val_count,
                                    const size_t* val_widths,
                                    const OP_KIND* agg_ops) {
  static_assert(thrust::detail::is_same<KeyT, int64_t>::value && thrust::detail::is_same<ValT, int64_t>::value,
                "Unsupported template parameter other than int64_t for now");
  const auto thread_index = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;
  const auto thread_count = seg_count == size_t(1) ? entry_count : entry_count * (seg_count - 1);

  auto keys_base = other_segs + sizeof(KeyT) * thread_index;
  auto keys_i64 = reinterpret_cast<const KeyT*>(keys_base);

  if (thread_index >= thread_count || is_empty_slot(*keys_i64)) {
    return;
  }

  auto read_base = keys_base;
  auto write_base = this_seg + sizeof(KeyT) * (thread_index % entry_count);
  const auto old = atomicCAS(
      reinterpret_cast<unsigned long long*>(write_base), EMPTY_KEY_64, static_cast<unsigned long long>(*keys_i64));

  if (is_empty_slot(static_cast<KeyT>(old))) {
    for (size_t i = 0; i < key_count;
         ++i, write_base += write_stride * sizeof(KeyT), read_base += read_stride * sizeof(KeyT)) {
      *reinterpret_cast<KeyT*>(write_base) = *reinterpret_cast<const KeyT*>(read_base);
    }
  }

  write_base = this_seg + sizeof(KeyT) * (thread_index % entry_count) + sizeof(KeyT) * write_stride * key_count;
  read_base = keys_base + sizeof(KeyT) * read_stride * key_count;
  reduce_func(write_base, write_stride, read_base, read_stride, val_count, val_widths, agg_ops);
}

template <typename KeyT = int64_t, typename ValT = int64_t>
__global__ void row_perfect_reducer(int8_t* this_seg,
                                    const size_t entry_count,
                                    const int8_t* other_segs,
                                    const size_t seg_count,
                                    const size_t entry_size,
                                    const size_t key_count,
                                    const size_t val_count,
                                    const size_t* val_widths,
                                    const OP_KIND* agg_ops) {
  static_assert(thrust::detail::is_same<KeyT, int64_t>::value && thrust::detail::is_same<ValT, int64_t>::value,
                "Unsupported template parameter other than int64_t for now");
  const auto thread_index = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;
  const auto thread_count = seg_count == size_t(1) ? entry_count : entry_count * (seg_count - 1);

  auto keys_base = other_segs + entry_size * thread_index;
  auto keys_i64 = reinterpret_cast<const KeyT*>(keys_base);

  if (thread_index >= thread_count || is_empty_slot(*keys_i64)) {
    return;
  }

  auto write_base = this_seg + entry_size * (thread_index % entry_count);
  const auto old = atomicCAS(
      reinterpret_cast<unsigned long long*>(write_base), EMPTY_KEY_64, static_cast<unsigned long long>(*keys_i64));

  if (is_empty_slot(static_cast<KeyT>(old))) {
    memcpy(write_base, keys_base, sizeof(KeyT) * key_count);
  }

  write_base += sizeof(KeyT) * key_count;
  auto read_base = keys_base + sizeof(KeyT) * key_count;
  reduce_func(write_base, 1, read_base, 1, val_count, val_widths, agg_ops);
}

}  // namespace

void reduce_segment_on_device(int8_t* dev_seg_buf,
                              const int8_t* dev_other_segs,
                              const size_t entry_count,
                              const size_t seg_count,
                              const std::vector<size_t>& col_widths,
                              const std::vector<OP_KIND>& agg_ops,
                              const bool is_columnar,
                              const size_t start,
                              const size_t end) {
  const size_t val_count = agg_ops.size();
  const size_t key_count = col_widths.size() - val_count;
  size_t entry_size = 0;
  for (size_t i = 0; i < col_widths.size(); ++i) {
    entry_size += col_widths[i];
  }

  thrust::device_vector<size_t> dev_col_widths(col_widths);
  thrust::device_vector<OP_KIND> dev_agg_ops(agg_ops);
  const auto thread_count = seg_count == size_t(1) ? entry_count : entry_count * (seg_count - 1);
  if (is_columnar) {
    const size_t write_stride = entry_count;
    const size_t read_stride = (end - start) * (seg_count == size_t(1) ? 1 : (seg_count - 1));
    col_perfect_reducer<<<compute_grid_dim(thread_count), c_block_size>>>(
        dev_seg_buf + start * sizeof(int64_t),
        end - start,
        write_stride,
        dev_other_segs,
        seg_count,
        read_stride,
        key_count,
        val_count,
        thrust::raw_pointer_cast(dev_col_widths.data() + key_count),
        thrust::raw_pointer_cast(dev_agg_ops.data()));
  } else {
    row_perfect_reducer<<<compute_grid_dim(thread_count), c_block_size>>>(
        dev_seg_buf + start * entry_size,
        end - start,
        dev_other_segs,
        seg_count,
        entry_size,
        key_count,
        val_count,
        thrust::raw_pointer_cast(dev_col_widths.data() + key_count),
        thrust::raw_pointer_cast(dev_agg_ops.data()));
  }
}

#endif  // HAVE_CUDA
