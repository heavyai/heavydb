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

#define checkCudaErrors(err) CHECK_EQ(err, cudaSuccess)

template <typename F, typename... ARGS>
void cuda_kernel_launch_wrapper(F func, ARGS&&... args) {
  int grid_size = -1;
  int block_size = -1;
  checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size, func));
  func<<<grid_size, block_size>>>(std::forward<ARGS>(args)...);
  checkCudaErrors(cudaGetLastError());
}

__global__ void fill_hash_join_buff_wrapper(int32_t* buff,
                                            const int32_t invalid_slot_val,
                                            const bool for_semi_join,
                                            const JoinColumn join_column,
                                            const JoinColumnTypeInfo type_info,
                                            int* err) {
  int partial_err = SUFFIX(fill_hash_join_buff)(
      buff, invalid_slot_val, for_semi_join, join_column, type_info, NULL, NULL, -1, -1);
  atomicCAS(err, 0, partial_err);
}

__global__ void fill_hash_join_buff_bucketized_wrapper(
    int32_t* buff,
    const int32_t invalid_slot_val,
    const bool for_semi_join,
    const JoinColumn join_column,
    const JoinColumnTypeInfo type_info,
    int* err,
    const int64_t bucket_normalization) {
  int partial_err = SUFFIX(fill_hash_join_buff_bucketized)(buff,
                                                           invalid_slot_val,
                                                           for_semi_join,
                                                           join_column,
                                                           type_info,
                                                           NULL,
                                                           NULL,
                                                           -1,
                                                           -1,
                                                           bucket_normalization);
  atomicCAS(err, 0, partial_err);
}

void fill_hash_join_buff_on_device_bucketized(int32_t* buff,
                                              const int32_t invalid_slot_val,
                                              const bool for_semi_join,
                                              int* dev_err_buff,
                                              const JoinColumn join_column,
                                              const JoinColumnTypeInfo type_info,
                                              const int64_t bucket_normalization) {
  cuda_kernel_launch_wrapper(fill_hash_join_buff_bucketized_wrapper,
                             buff,
                             invalid_slot_val,
                             for_semi_join,
                             join_column,
                             type_info,
                             dev_err_buff,
                             bucket_normalization);
}

void fill_hash_join_buff_on_device(int32_t* buff,
                                   const int32_t invalid_slot_val,
                                   const bool for_semi_join,
                                   int* dev_err_buff,
                                   const JoinColumn join_column,
                                   const JoinColumnTypeInfo type_info) {
  cuda_kernel_launch_wrapper(fill_hash_join_buff_wrapper,
                             buff,
                             invalid_slot_val,
                             for_semi_join,
                             join_column,
                             type_info,
                             dev_err_buff);
}

__global__ void init_hash_join_buff_wrapper(int32_t* buff,
                                            const int64_t hash_entry_count,
                                            const int32_t invalid_slot_val) {
  SUFFIX(init_hash_join_buff)(buff, hash_entry_count, invalid_slot_val, -1, -1);
}

void init_hash_join_buff_on_device(int32_t* buff,
                                   const int64_t hash_entry_count,
                                   const int32_t invalid_slot_val) {
  cuda_kernel_launch_wrapper(
      init_hash_join_buff_wrapper, buff, hash_entry_count, invalid_slot_val);
}

#define VALID_POS_FLAG 0

__global__ void set_valid_pos_flag(int32_t* pos_buff,
                                   const int32_t* count_buff,
                                   const int64_t entry_count) {
  const int32_t start = threadIdx.x + blockDim.x * blockIdx.x;
  const int32_t step = blockDim.x * gridDim.x;
  for (int64_t i = start; i < entry_count; i += step) {
    if (count_buff[i]) {
      pos_buff[i] = VALID_POS_FLAG;
    }
  }
}

__global__ void set_valid_pos(int32_t* pos_buff,
                              int32_t* count_buff,
                              const int64_t entry_count) {
  const int32_t start = threadIdx.x + blockDim.x * blockIdx.x;
  const int32_t step = blockDim.x * gridDim.x;
  for (int64_t i = start; i < entry_count; i += step) {
    if (VALID_POS_FLAG == pos_buff[i]) {
      pos_buff[i] = !i ? 0 : count_buff[i - 1];
    }
  }
}

template <typename COUNT_MATCHES_FUNCTOR, typename FILL_ROW_IDS_FUNCTOR>
void fill_one_to_many_hash_table_on_device_impl(int32_t* buff,
                                                const int64_t hash_entry_count,
                                                const int32_t invalid_slot_val,
                                                const JoinColumn& join_column,
                                                const JoinColumnTypeInfo& type_info,
                                                COUNT_MATCHES_FUNCTOR count_matches_func,
                                                FILL_ROW_IDS_FUNCTOR fill_row_ids_func) {
  int32_t* pos_buff = buff;
  int32_t* count_buff = buff + hash_entry_count;
  cudaMemset(count_buff, 0, hash_entry_count * sizeof(int32_t));
  count_matches_func();

  cuda_kernel_launch_wrapper(set_valid_pos_flag, pos_buff, count_buff, hash_entry_count);

  auto count_buff_dev_ptr = thrust::device_pointer_cast(count_buff);
  thrust::inclusive_scan(
      count_buff_dev_ptr, count_buff_dev_ptr + hash_entry_count, count_buff_dev_ptr);

  cuda_kernel_launch_wrapper(set_valid_pos, pos_buff, count_buff, hash_entry_count);
  cudaMemset(count_buff, 0, hash_entry_count * sizeof(int32_t));
  fill_row_ids_func();
}

void fill_one_to_many_hash_table_on_device(int32_t* buff,
                                           const HashEntryInfo hash_entry_info,
                                           const int32_t invalid_slot_val,
                                           const JoinColumn& join_column,
                                           const JoinColumnTypeInfo& type_info) {
  auto hash_entry_count = hash_entry_info.hash_entry_count;
  auto count_matches_func = [hash_entry_count,
                             count_buff = buff + hash_entry_count,
                             invalid_slot_val,
                             join_column,
                             type_info] {
    cuda_kernel_launch_wrapper(
        SUFFIX(count_matches), count_buff, invalid_slot_val, join_column, type_info);
  };

  auto fill_row_ids_func =
      [buff, hash_entry_count, invalid_slot_val, join_column, type_info] {
        cuda_kernel_launch_wrapper(SUFFIX(fill_row_ids),
                                   buff,
                                   hash_entry_count,
                                   invalid_slot_val,
                                   join_column,
                                   type_info);
      };

  fill_one_to_many_hash_table_on_device_impl(buff,
                                             hash_entry_count,
                                             invalid_slot_val,
                                             join_column,
                                             type_info,
                                             count_matches_func,
                                             fill_row_ids_func);
}

void fill_one_to_many_hash_table_on_device_bucketized(
    int32_t* buff,
    const HashEntryInfo hash_entry_info,
    const int32_t invalid_slot_val,
    const JoinColumn& join_column,
    const JoinColumnTypeInfo& type_info) {
  auto hash_entry_count = hash_entry_info.getNormalizedHashEntryCount();
  auto count_matches_func = [count_buff = buff + hash_entry_count,
                             invalid_slot_val,
                             join_column,
                             type_info,
                             bucket_normalization =
                                 hash_entry_info.bucket_normalization] {
    cuda_kernel_launch_wrapper(SUFFIX(count_matches_bucketized),
                               count_buff,
                               invalid_slot_val,
                               join_column,
                               type_info,
                               bucket_normalization);
  };

  auto fill_row_ids_func = [buff,
                            hash_entry_count =
                                hash_entry_info.getNormalizedHashEntryCount(),
                            invalid_slot_val,
                            join_column,
                            type_info,
                            bucket_normalization = hash_entry_info.bucket_normalization] {
    cuda_kernel_launch_wrapper(SUFFIX(fill_row_ids_bucketized),
                               buff,
                               hash_entry_count,
                               invalid_slot_val,
                               join_column,
                               type_info,
                               bucket_normalization);
  };

  fill_one_to_many_hash_table_on_device_impl(buff,
                                             hash_entry_count,
                                             invalid_slot_val,
                                             join_column,
                                             type_info,
                                             count_matches_func,
                                             fill_row_ids_func);
}

template <typename T, typename KEY_HANDLER>
void fill_one_to_many_baseline_hash_table_on_device(int32_t* buff,
                                                    const T* composite_key_dict,
                                                    const int64_t hash_entry_count,
                                                    const int32_t invalid_slot_val,
                                                    const KEY_HANDLER* key_handler,
                                                    const size_t num_elems) {
  auto pos_buff = buff;
  auto count_buff = buff + hash_entry_count;
  cudaMemset(count_buff, 0, hash_entry_count * sizeof(int32_t));
  cuda_kernel_launch_wrapper(count_matches_baseline_gpu<T, KEY_HANDLER>,
                             count_buff,
                             composite_key_dict,
                             hash_entry_count,
                             key_handler,
                             num_elems);

  cuda_kernel_launch_wrapper(set_valid_pos_flag, pos_buff, count_buff, hash_entry_count);

  auto count_buff_dev_ptr = thrust::device_pointer_cast(count_buff);
  thrust::inclusive_scan(
      count_buff_dev_ptr, count_buff_dev_ptr + hash_entry_count, count_buff_dev_ptr);
  cuda_kernel_launch_wrapper(set_valid_pos, pos_buff, count_buff, hash_entry_count);
  cudaMemset(count_buff, 0, hash_entry_count * sizeof(int32_t));

  cuda_kernel_launch_wrapper(fill_row_ids_baseline_gpu<T, KEY_HANDLER>,
                             buff,
                             composite_key_dict,
                             hash_entry_count,
                             invalid_slot_val,
                             key_handler,
                             num_elems);
}

template <typename T>
__global__ void init_baseline_hash_join_buff_wrapper(int8_t* hash_join_buff,
                                                     const int64_t entry_count,
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
                                               const int64_t entry_count,
                                               const size_t key_component_count,
                                               const bool with_val_slot,
                                               const int32_t invalid_slot_val) {
  cuda_kernel_launch_wrapper(init_baseline_hash_join_buff_wrapper<int32_t>,
                             hash_join_buff,
                             entry_count,
                             key_component_count,
                             with_val_slot,
                             invalid_slot_val);
}

void init_baseline_hash_join_buff_on_device_64(int8_t* hash_join_buff,
                                               const int64_t entry_count,
                                               const size_t key_component_count,
                                               const bool with_val_slot,
                                               const int32_t invalid_slot_val) {
  cuda_kernel_launch_wrapper(init_baseline_hash_join_buff_wrapper<int64_t>,
                             hash_join_buff,
                             entry_count,
                             key_component_count,
                             with_val_slot,
                             invalid_slot_val);
}

template <typename T, typename KEY_HANDLER>
__global__ void fill_baseline_hash_join_buff_wrapper(int8_t* hash_buff,
                                                     const int64_t entry_count,
                                                     const int32_t invalid_slot_val,
                                                     const bool for_semi_join,
                                                     const size_t key_component_count,
                                                     const bool with_val_slot,
                                                     int* err,
                                                     const KEY_HANDLER* key_handler,
                                                     const int64_t num_elems) {
  int partial_err = SUFFIX(fill_baseline_hash_join_buff)<T>(hash_buff,
                                                            entry_count,
                                                            invalid_slot_val,
                                                            for_semi_join,
                                                            key_component_count,
                                                            with_val_slot,
                                                            key_handler,
                                                            num_elems,
                                                            -1,
                                                            -1);
  atomicCAS(err, 0, partial_err);
}

void fill_baseline_hash_join_buff_on_device_32(int8_t* hash_buff,
                                               const int64_t entry_count,
                                               const int32_t invalid_slot_val,
                                               const bool for_semi_join,
                                               const size_t key_component_count,
                                               const bool with_val_slot,
                                               int* dev_err_buff,
                                               const GenericKeyHandler* key_handler,
                                               const int64_t num_elems) {
  cuda_kernel_launch_wrapper(
      fill_baseline_hash_join_buff_wrapper<int32_t, GenericKeyHandler>,
      hash_buff,
      entry_count,
      invalid_slot_val,
      for_semi_join,
      key_component_count,
      with_val_slot,
      dev_err_buff,
      key_handler,
      num_elems);
}

void fill_baseline_hash_join_buff_on_device_64(int8_t* hash_buff,
                                               const int64_t entry_count,
                                               const int32_t invalid_slot_val,
                                               const bool for_semi_join,
                                               const size_t key_component_count,
                                               const bool with_val_slot,
                                               int* dev_err_buff,
                                               const GenericKeyHandler* key_handler,
                                               const int64_t num_elems) {
  cuda_kernel_launch_wrapper(
      fill_baseline_hash_join_buff_wrapper<unsigned long long, GenericKeyHandler>,
      hash_buff,
      entry_count,
      invalid_slot_val,
      for_semi_join,
      key_component_count,
      with_val_slot,
      dev_err_buff,
      key_handler,
      num_elems);
}

void fill_one_to_many_baseline_hash_table_on_device_32(
    int32_t* buff,
    const int32_t* composite_key_dict,
    const int64_t hash_entry_count,
    const int32_t invalid_slot_val,
    const size_t key_component_count,
    const GenericKeyHandler* key_handler,
    const int64_t num_elems) {
  fill_one_to_many_baseline_hash_table_on_device<int32_t>(buff,
                                                          composite_key_dict,
                                                          hash_entry_count,
                                                          invalid_slot_val,
                                                          key_handler,
                                                          num_elems);
}

void fill_one_to_many_baseline_hash_table_on_device_64(
    int32_t* buff,
    const int64_t* composite_key_dict,
    const int64_t hash_entry_count,
    const int32_t invalid_slot_val,
    const GenericKeyHandler* key_handler,
    const int64_t num_elems) {
  fill_one_to_many_baseline_hash_table_on_device<int64_t>(buff,
                                                          composite_key_dict,
                                                          hash_entry_count,
                                                          invalid_slot_val,
                                                          key_handler,
                                                          num_elems);
}

void approximate_distinct_tuples_on_device(uint8_t* hll_buffer,
                                           const uint32_t b,
                                           const GenericKeyHandler* key_handler,
                                           const int64_t num_elems) {
  cuda_kernel_launch_wrapper(approximate_distinct_tuples_impl_gpu<GenericKeyHandler>,
                             hll_buffer,
                             nullptr,
                             b,
                             num_elems,
                             key_handler);
}

void compute_bucket_sizes_on_device(double* bucket_sizes_buffer,
                                    const JoinColumn* join_column,
                                    const JoinColumnTypeInfo* type_info,
                                    const double* bucket_sz_threshold) {
  cuda_kernel_launch_wrapper(compute_bucket_sizes_impl_gpu<2>,
                             bucket_sizes_buffer,
                             join_column,
                             type_info,
                             bucket_sz_threshold);
}
