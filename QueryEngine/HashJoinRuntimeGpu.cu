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
#include "HashJoinRuntime.cpp"

#include <thrust/device_ptr.h>
#include <thrust/scan.h>

__global__ void fill_hash_join_buff_wrapper(int32_t* buff,
                                            const int32_t invalid_slot_val,
                                            const JoinColumn join_column,
                                            const JoinColumnTypeInfo type_info,
                                            int* err) {
  int partial_err = SUFFIX(fill_hash_join_buff)(
      buff, invalid_slot_val, join_column, type_info, NULL, NULL, -1, -1);
  atomicCAS(err, 0, partial_err);
}

void fill_hash_join_buff_on_device(int32_t* buff,
                                   const int32_t invalid_slot_val,
                                   int* dev_err_buff,
                                   const JoinColumn join_column,
                                   const JoinColumnTypeInfo type_info,
                                   const size_t block_size_x,
                                   const size_t grid_size_x) {
  fill_hash_join_buff_wrapper<<<grid_size_x, block_size_x>>>(
      buff, invalid_slot_val, join_column, type_info, dev_err_buff);
}

__global__ void fill_hash_join_buff_wrapper_sharded(int32_t* buff,
                                                    const int32_t invalid_slot_val,
                                                    const JoinColumn join_column,
                                                    const JoinColumnTypeInfo type_info,
                                                    const ShardInfo shard_info,
                                                    int* err) {
  int partial_err = SUFFIX(fill_hash_join_buff_sharded)(
      buff, invalid_slot_val, join_column, type_info, shard_info, NULL, NULL, -1, -1);
  atomicCAS(err, 0, partial_err);
}

void fill_hash_join_buff_on_device_sharded(int32_t* buff,
                                           const int32_t invalid_slot_val,
                                           int* dev_err_buff,
                                           const JoinColumn join_column,
                                           const JoinColumnTypeInfo type_info,
                                           const ShardInfo shard_info,
                                           const size_t block_size_x,
                                           const size_t grid_size_x) {
  fill_hash_join_buff_wrapper_sharded<<<grid_size_x, block_size_x>>>(
      buff, invalid_slot_val, join_column, type_info, shard_info, dev_err_buff);
}

__global__ void init_hash_join_buff_wrapper(int32_t* buff,
                                            const int32_t hash_entry_count,
                                            const int32_t invalid_slot_val) {
  SUFFIX(init_hash_join_buff)(buff, hash_entry_count, invalid_slot_val, -1, -1);
}

void init_hash_join_buff_on_device(int32_t* buff,
                                   const int32_t hash_entry_count,
                                   const int32_t invalid_slot_val,
                                   const size_t block_size_x,
                                   const size_t grid_size_x) {
  init_hash_join_buff_wrapper<<<grid_size_x, block_size_x>>>(
      buff, hash_entry_count, invalid_slot_val);
}

#define VALID_POS_FLAG 0

__global__ void set_valid_pos_flag(int32_t* pos_buff,
                                   const int32_t* count_buff,
                                   const int32_t entry_count) {
  const int32_t start = threadIdx.x + blockDim.x * blockIdx.x;
  const int32_t step = blockDim.x * gridDim.x;
  for (int32_t i = start; i < entry_count; i += step) {
    if (count_buff[i]) {
      pos_buff[i] = VALID_POS_FLAG;
    }
  }
}

__global__ void set_valid_pos(int32_t* pos_buff,
                              int32_t* count_buff,
                              const int32_t entry_count) {
  const int32_t start = threadIdx.x + blockDim.x * blockIdx.x;
  const int32_t step = blockDim.x * gridDim.x;
  for (int32_t i = start; i < entry_count; i += step) {
    if (VALID_POS_FLAG == pos_buff[i]) {
      pos_buff[i] = !i ? 0 : count_buff[i - 1];
    }
  }
}

void fill_one_to_many_hash_table_on_device(int32_t* buff,
                                           const int32_t hash_entry_count,
                                           const int32_t invalid_slot_val,
                                           const JoinColumn& join_column,
                                           const JoinColumnTypeInfo& type_info,
                                           const size_t block_size_x,
                                           const size_t grid_size_x) {
  int32_t* pos_buff = buff;
  int32_t* count_buff = buff + hash_entry_count;
  cudaMemset(count_buff, 0, hash_entry_count * sizeof(int32_t));
  SUFFIX(count_matches)<<<grid_size_x, block_size_x>>>(
      count_buff, invalid_slot_val, join_column, type_info);

  set_valid_pos_flag<<<grid_size_x, block_size_x>>>(
      pos_buff, count_buff, hash_entry_count);

  auto count_buff_dev_ptr = thrust::device_pointer_cast(count_buff);
  thrust::inclusive_scan(
      count_buff_dev_ptr, count_buff_dev_ptr + hash_entry_count, count_buff_dev_ptr);
  set_valid_pos<<<grid_size_x, block_size_x>>>(pos_buff, count_buff, hash_entry_count);
  cudaMemset(count_buff, 0, hash_entry_count * sizeof(int32_t));
  SUFFIX(fill_row_ids)<<<grid_size_x, block_size_x>>>(
      buff, hash_entry_count, invalid_slot_val, join_column, type_info);
}

void fill_one_to_many_hash_table_on_device_sharded(int32_t* buff,
                                                   const int32_t hash_entry_count,
                                                   const int32_t invalid_slot_val,
                                                   const JoinColumn& join_column,
                                                   const JoinColumnTypeInfo& type_info,
                                                   const ShardInfo& shard_info,
                                                   const size_t block_size_x,
                                                   const size_t grid_size_x) {
  int32_t* pos_buff = buff;
  int32_t* count_buff = buff + hash_entry_count;
  cudaMemset(count_buff, 0, hash_entry_count * sizeof(int32_t));
  SUFFIX(count_matches_sharded)<<<grid_size_x, block_size_x>>>(
      count_buff, invalid_slot_val, join_column, type_info, shard_info);

  set_valid_pos_flag<<<grid_size_x, block_size_x>>>(
      pos_buff, count_buff, hash_entry_count);

  auto count_buff_dev_ptr = thrust::device_pointer_cast(count_buff);
  thrust::inclusive_scan(
      count_buff_dev_ptr, count_buff_dev_ptr + hash_entry_count, count_buff_dev_ptr);
  set_valid_pos<<<grid_size_x, block_size_x>>>(pos_buff, count_buff, hash_entry_count);
  cudaMemset(count_buff, 0, hash_entry_count * sizeof(int32_t));
  SUFFIX(fill_row_ids_sharded)<<<grid_size_x, block_size_x>>>(
      buff, hash_entry_count, invalid_slot_val, join_column, type_info, shard_info);
}

template <typename T, typename KEY_HANDLER>
void fill_one_to_many_baseline_hash_table_on_device(int32_t* buff,
                                                    const T* composite_key_dict,
                                                    const size_t hash_entry_count,
                                                    const int32_t invalid_slot_val,
                                                    const KEY_HANDLER* key_handler,
                                                    const size_t num_elems,
                                                    const size_t block_size_x,
                                                    const size_t grid_size_x) {
  auto pos_buff = buff;
  auto count_buff = buff + hash_entry_count;
  cudaMemset(count_buff, 0, hash_entry_count * sizeof(int32_t));
  count_matches_baseline_gpu<<<grid_size_x, block_size_x>>>(
      count_buff, composite_key_dict, hash_entry_count, key_handler, num_elems);

  set_valid_pos_flag<<<grid_size_x, block_size_x>>>(
      pos_buff, count_buff, hash_entry_count);

  auto count_buff_dev_ptr = thrust::device_pointer_cast(count_buff);
  thrust::inclusive_scan(
      count_buff_dev_ptr, count_buff_dev_ptr + hash_entry_count, count_buff_dev_ptr);
  set_valid_pos<<<grid_size_x, block_size_x>>>(pos_buff, count_buff, hash_entry_count);
  cudaMemset(count_buff, 0, hash_entry_count * sizeof(int32_t));
  fill_row_ids_baseline_gpu<<<grid_size_x, block_size_x>>>(buff,
                                                           composite_key_dict,
                                                           hash_entry_count,
                                                           invalid_slot_val,
                                                           key_handler,
                                                           num_elems);
}

template <typename T>
__global__ void init_baseline_hash_join_buff_wrapper(int8_t* hash_join_buff,
                                                     const size_t entry_count,
                                                     const size_t key_component_count,
                                                     const bool with_val_slot,
                                                     const int32_t invalid_slot_val) {
  SUFFIX(init_baseline_hash_join_buff)<T>(hash_join_buff,
                                          entry_count,
                                          key_component_count,
                                          with_val_slot,
                                          invalid_slot_val,
                                          -1,
                                          -1);
}

void init_baseline_hash_join_buff_on_device_32(int8_t* hash_join_buff,
                                               const int32_t entry_count,
                                               const size_t key_component_count,
                                               const bool with_val_slot,
                                               const int32_t invalid_slot_val,
                                               const size_t block_size_x,
                                               const size_t grid_size_x) {
  init_baseline_hash_join_buff_wrapper<int32_t><<<grid_size_x, block_size_x>>>(
      hash_join_buff, entry_count, key_component_count, with_val_slot, invalid_slot_val);
}

void init_baseline_hash_join_buff_on_device_64(int8_t* hash_join_buff,
                                               const int32_t entry_count,
                                               const size_t key_component_count,
                                               const bool with_val_slot,
                                               const int32_t invalid_slot_val,
                                               const size_t block_size_x,
                                               const size_t grid_size_x) {
  init_baseline_hash_join_buff_wrapper<int64_t><<<grid_size_x, block_size_x>>>(
      hash_join_buff, entry_count, key_component_count, with_val_slot, invalid_slot_val);
}

template <typename T, typename KEY_HANDLER>
__global__ void fill_baseline_hash_join_buff_wrapper(int8_t* hash_buff,
                                                     const size_t entry_count,
                                                     const int32_t invalid_slot_val,
                                                     const size_t key_component_count,
                                                     const bool with_val_slot,
                                                     int* err,
                                                     const KEY_HANDLER* key_handler,
                                                     const size_t num_elems) {
  int partial_err = SUFFIX(fill_baseline_hash_join_buff)<T>(hash_buff,
                                                            entry_count,
                                                            invalid_slot_val,
                                                            key_component_count,
                                                            with_val_slot,
                                                            key_handler,
                                                            num_elems,
                                                            -1,
                                                            -1);
  atomicCAS(err, 0, partial_err);
}

void fill_baseline_hash_join_buff_on_device_32(int8_t* hash_buff,
                                               const size_t entry_count,
                                               const int32_t invalid_slot_val,
                                               const size_t key_component_count,
                                               const bool with_val_slot,
                                               int* dev_err_buff,
                                               const GenericKeyHandler* key_handler,
                                               const size_t num_elems,
                                               const size_t block_size_x,
                                               const size_t grid_size_x) {
  fill_baseline_hash_join_buff_wrapper<int32_t>
      <<<grid_size_x, block_size_x>>>(hash_buff,
                                      entry_count,
                                      invalid_slot_val,
                                      key_component_count,
                                      with_val_slot,
                                      dev_err_buff,
                                      key_handler,
                                      num_elems);
}

void fill_baseline_hash_join_buff_on_device_64(int8_t* hash_buff,
                                               const size_t entry_count,
                                               const int32_t invalid_slot_val,
                                               const size_t key_component_count,
                                               const bool with_val_slot,
                                               int* dev_err_buff,
                                               const GenericKeyHandler* key_handler,
                                               const size_t num_elems,
                                               const size_t block_size_x,
                                               const size_t grid_size_x) {
  fill_baseline_hash_join_buff_wrapper<unsigned long long>
      <<<grid_size_x, block_size_x>>>(hash_buff,
                                      entry_count,
                                      invalid_slot_val,
                                      key_component_count,
                                      with_val_slot,
                                      dev_err_buff,
                                      key_handler,
                                      num_elems);
}

void overlaps_fill_baseline_hash_join_buff_on_device_64(
    int8_t* hash_buff,
    const size_t entry_count,
    const int32_t invalid_slot_val,
    const size_t key_component_count,
    const bool with_val_slot,
    int* dev_err_buff,
    const OverlapsKeyHandler* key_handler,
    const size_t num_elems,
    const size_t block_size_x,
    const size_t grid_size_x) {
  fill_baseline_hash_join_buff_wrapper<unsigned long long>
      <<<grid_size_x, block_size_x>>>(hash_buff,
                                      entry_count,
                                      invalid_slot_val,
                                      key_component_count,
                                      with_val_slot,
                                      dev_err_buff,
                                      key_handler,
                                      num_elems);
}

void fill_one_to_many_baseline_hash_table_on_device_32(
    int32_t* buff,
    const int32_t* composite_key_dict,
    const size_t hash_entry_count,
    const int32_t invalid_slot_val,
    const size_t key_component_count,
    const GenericKeyHandler* key_handler,
    const size_t num_elems,
    const size_t block_size_x,
    const size_t grid_size_x) {
  fill_one_to_many_baseline_hash_table_on_device<int32_t>(buff,
                                                          composite_key_dict,
                                                          hash_entry_count,
                                                          invalid_slot_val,
                                                          key_handler,
                                                          num_elems,
                                                          block_size_x,
                                                          grid_size_x);
}

void fill_one_to_many_baseline_hash_table_on_device_64(
    int32_t* buff,
    const int64_t* composite_key_dict,
    const size_t hash_entry_count,
    const int32_t invalid_slot_val,
    const GenericKeyHandler* key_handler,
    const size_t num_elems,
    const size_t block_size_x,
    const size_t grid_size_x) {
  fill_one_to_many_baseline_hash_table_on_device<int64_t>(buff,
                                                          composite_key_dict,
                                                          hash_entry_count,
                                                          invalid_slot_val,
                                                          key_handler,
                                                          num_elems,
                                                          block_size_x,
                                                          grid_size_x);
}

void overlaps_fill_one_to_many_baseline_hash_table_on_device_64(
    int32_t* buff,
    const int64_t* composite_key_dict,
    const size_t hash_entry_count,
    const int32_t invalid_slot_val,
    const OverlapsKeyHandler* key_handler,
    const size_t num_elems,
    const size_t block_size_x,
    const size_t grid_size_x) {
  fill_one_to_many_baseline_hash_table_on_device<int64_t>(buff,
                                                          composite_key_dict,
                                                          hash_entry_count,
                                                          invalid_slot_val,
                                                          key_handler,
                                                          num_elems,
                                                          block_size_x,
                                                          grid_size_x);
}

void approximate_distinct_tuples_on_device_overlaps(uint8_t* hll_buffer,
                                                    const uint32_t b,
                                                    int32_t* row_counts_buffer,
                                                    const OverlapsKeyHandler* key_handler,
                                                    const size_t num_elems,
                                                    const size_t block_size_x,
                                                    const size_t grid_size_x) {
  approximate_distinct_tuples_impl_gpu<<<grid_size_x, block_size_x>>>(
      hll_buffer, row_counts_buffer, b, num_elems, key_handler);

  auto row_counts_buffer_ptr = thrust::device_pointer_cast(row_counts_buffer);
  thrust::inclusive_scan(
      row_counts_buffer_ptr, row_counts_buffer_ptr + num_elems, row_counts_buffer_ptr);
}

void approximate_distinct_tuples_on_device(uint8_t* hll_buffer,
                                           const uint32_t b,
                                           const GenericKeyHandler* key_handler,
                                           const size_t num_elems,
                                           const size_t block_size_x,
                                           const size_t grid_size_x) {
  approximate_distinct_tuples_impl_gpu<<<grid_size_x, block_size_x>>>(
      hll_buffer, nullptr, b, num_elems, key_handler);
}

void compute_bucket_sizes_on_device(double* bucket_sizes_buffer,
                                    const JoinColumn* join_column,
                                    const double bucket_sz_threshold,
                                    const size_t block_size_x,
                                    const size_t grid_size_x) {
  compute_bucket_sizes_impl_gpu<2><<<grid_size_x, block_size_x>>>(
      bucket_sizes_buffer, join_column, bucket_sz_threshold, block_size_x, grid_size_x);
}