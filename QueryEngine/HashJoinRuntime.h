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
 * @file    HashJoinRuntime.h
 * @author  Alex Suhan <alex@mapd.com>
 *
 * Copyright (c) 2015 MapD Technologies, Inc.  All rights reserved.
 */

#ifndef QUERYENGINE_HASHJOINRUNTIME_H
#define QUERYENGINE_HASHJOINRUNTIME_H

#include <stddef.h>
#include <stdint.h>
#include <vector>
#include "../Shared/sqltypes.h"
#include "SqlTypesLayout.h"

struct GenericKeyHandler;
struct OverlapsKeyHandler;

const size_t g_maximum_conditions_to_coalesce{8};

void init_hash_join_buff(int32_t* buff,
                         const int32_t entry_count,
                         const int32_t invalid_slot_val,
                         const int32_t cpu_thread_idx,
                         const int32_t cpu_thread_count);

void init_hash_join_buff_on_device(int32_t* buff,
                                   const int32_t entry_count,
                                   const int32_t invalid_slot_val,
                                   const size_t block_size_x,
                                   const size_t grid_size_x);

void init_baseline_hash_join_buff_32(int8_t* hash_join_buff,
                                     const int32_t entry_count,
                                     const size_t key_component_count,
                                     const bool with_val_slot,
                                     const int32_t invalid_slot_val,
                                     const int32_t cpu_thread_idx,
                                     const int32_t cpu_thread_count);

void init_baseline_hash_join_buff_64(int8_t* hash_join_buff,
                                     const int32_t entry_count,
                                     const size_t key_component_count,
                                     const bool with_val_slot,
                                     const int32_t invalid_slot_val,
                                     const int32_t cpu_thread_idx,
                                     const int32_t cpu_thread_count);

void init_baseline_hash_join_buff_on_device_32(int8_t* hash_join_buff,
                                               const int32_t entry_count,
                                               const size_t key_component_count,
                                               const bool with_val_slot,
                                               const int32_t invalid_slot_val,
                                               const size_t block_size_x,
                                               const size_t grid_size_x);

void init_baseline_hash_join_buff_on_device_64(int8_t* hash_join_buff,
                                               const int32_t entry_count,
                                               const size_t key_component_count,
                                               const bool with_val_slot,
                                               const int32_t invalid_slot_val,
                                               const size_t block_size_x,
                                               const size_t grid_size_x);

enum ColumnType { SmallDate = 0, Signed = 1, Unsigned = 2 };

struct JoinColumn {
  const int8_t* col_buff;
  const size_t num_elems;
};

struct JoinColumnTypeInfo {
  const size_t elem_sz;
  const int64_t min_val;
  const int64_t null_val;
  const bool uses_bw_eq;
  const int64_t translated_null_val;
  const ColumnType column_type;
};

inline ColumnType get_join_column_type_kind(const SQLTypeInfo& ti) {
  if (ti.is_date_in_days()) {
    return SmallDate;
  } else {
    return is_unsigned_type(ti) ? Unsigned : Signed;
  }
}

struct JoinBucketInfo {
  std::vector<double> bucket_sizes_for_dimension;
  bool is_double;  // TODO(adb): assume float otherwise (?)
};

int fill_hash_join_buff(int32_t* buff,
                        const int32_t invalid_slot_val,
                        const JoinColumn join_column,
                        const JoinColumnTypeInfo type_info,
                        const void* sd_inner,
                        const void* sd_outer,
                        const int32_t cpu_thread_idx,
                        const int32_t cpu_thread_count);

void fill_hash_join_buff_on_device(int32_t* buff,
                                   const int32_t invalid_slot_val,
                                   int* dev_err_buff,
                                   const JoinColumn join_column,
                                   const JoinColumnTypeInfo type_info,
                                   const size_t block_size_x,
                                   const size_t grid_size_x);

struct ShardInfo {
  const size_t shard;
  const size_t entry_count_per_shard;
  const size_t num_shards;
  const int device_count;
};

void fill_hash_join_buff_on_device_sharded(int32_t* buff,
                                           const int32_t invalid_slot_val,
                                           int* dev_err_buff,
                                           const JoinColumn join_column,
                                           const JoinColumnTypeInfo type_info,
                                           const ShardInfo shard_info,
                                           const size_t block_size_x,
                                           const size_t grid_size_x);

void fill_one_to_many_hash_table(int32_t* buff,
                                 const int32_t hash_entry_count,
                                 const int32_t invalid_slot_val,
                                 const JoinColumn& join_column,
                                 const JoinColumnTypeInfo& type_info,
                                 const void* sd_inner_proxy,
                                 const void* sd_outer_proxy,
                                 const int32_t cpu_thread_count);

void fill_one_to_many_hash_table_sharded(int32_t* buff,
                                         const int32_t hash_entry_count,
                                         const int32_t invalid_slot_val,
                                         const JoinColumn& join_column,
                                         const JoinColumnTypeInfo& type_info,
                                         const ShardInfo& shard_info,
                                         const void* sd_inner_proxy,
                                         const void* sd_outer_proxy,
                                         const int32_t cpu_thread_count);

void fill_one_to_many_hash_table_on_device(int32_t* buff,
                                           const int32_t hash_entry_count,
                                           const int32_t invalid_slot_val,
                                           const JoinColumn& join_column,
                                           const JoinColumnTypeInfo& type_info,
                                           const size_t block_size_x,
                                           const size_t grid_size_x);

void fill_one_to_many_hash_table_on_device_sharded(int32_t* buff,
                                                   const int32_t hash_entry_count,
                                                   const int32_t invalid_slot_val,
                                                   const JoinColumn& join_column,
                                                   const JoinColumnTypeInfo& type_info,
                                                   const ShardInfo& shard_info,
                                                   const size_t block_size_x,
                                                   const size_t grid_size_x);

int fill_baseline_hash_join_buff_32(int8_t* hash_buff,
                                    const size_t entry_count,
                                    const int32_t invalid_slot_val,
                                    const size_t key_component_count,
                                    const bool with_val_slot,
                                    const GenericKeyHandler* key_handler,
                                    const size_t num_elems,
                                    const int32_t cpu_thread_idx,
                                    const int32_t cpu_thread_count);

int overlaps_fill_baseline_hash_join_buff_32(int8_t* hash_buff,
                                             const size_t entry_count,
                                             const int32_t invalid_slot_val,
                                             const size_t key_component_count,
                                             const bool with_val_slot,
                                             const OverlapsKeyHandler* key_handler,
                                             const size_t num_elems,
                                             const int32_t cpu_thread_idx,
                                             const int32_t cpu_thread_count);

int fill_baseline_hash_join_buff_64(int8_t* hash_buff,
                                    const size_t entry_count,
                                    const int32_t invalid_slot_val,
                                    const size_t key_component_count,
                                    const bool with_val_slot,
                                    const GenericKeyHandler* key_handler,
                                    const size_t num_elems,
                                    const int32_t cpu_thread_idx,
                                    const int32_t cpu_thread_count);

int overlaps_fill_baseline_hash_join_buff_64(int8_t* hash_buff,
                                             const size_t entry_count,
                                             const int32_t invalid_slot_val,
                                             const size_t key_component_count,
                                             const bool with_val_slot,
                                             const OverlapsKeyHandler* key_handler,
                                             const size_t num_elems,
                                             const int32_t cpu_thread_idx,
                                             const int32_t cpu_thread_count);

void fill_baseline_hash_join_buff_on_device_32(int8_t* hash_buff,
                                               const size_t entry_count,
                                               const int32_t invalid_slot_val,
                                               const size_t key_component_count,
                                               const bool with_val_slot,
                                               int* dev_err_buff,
                                               const GenericKeyHandler* key_handler,
                                               const size_t num_elems,
                                               const size_t block_size_x,
                                               const size_t grid_size_x);

void fill_baseline_hash_join_buff_on_device_64(int8_t* hash_buff,
                                               const size_t entry_count,
                                               const int32_t invalid_slot_val,
                                               const size_t key_component_count,
                                               const bool with_val_slot,
                                               int* dev_err_buff,
                                               const GenericKeyHandler* key_handler,
                                               const size_t num_elems,
                                               const size_t block_size_x,
                                               const size_t grid_size_x);

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
    const size_t grid_size_x);

void fill_one_to_many_baseline_hash_table_32(
    int32_t* buff,
    const int32_t* composite_key_dict,
    const size_t hash_entry_count,
    const int32_t invalid_slot_val,
    const size_t key_component_count,
    const std::vector<JoinColumn>& join_column_per_key,
    const std::vector<JoinColumnTypeInfo>& type_info_per_key,
    const std::vector<JoinBucketInfo>& join_bucket_info,
    const std::vector<const void*>& sd_inner_proxy_per_key,
    const std::vector<const void*>& sd_outer_proxy_per_key,
    const int32_t cpu_thread_count);

void fill_one_to_many_baseline_hash_table_64(
    int32_t* buff,
    const int64_t* composite_key_dict,
    const size_t hash_entry_count,
    const int32_t invalid_slot_val,
    const size_t key_component_count,
    const std::vector<JoinColumn>& join_column_per_key,
    const std::vector<JoinColumnTypeInfo>& type_info_per_key,
    const std::vector<JoinBucketInfo>& join_bucket_info,
    const std::vector<const void*>& sd_inner_proxy_per_key,
    const std::vector<const void*>& sd_outer_proxy_per_key,
    const int32_t cpu_thread_count);

void fill_one_to_many_baseline_hash_table_on_device_32(
    int32_t* buff,
    const int32_t* composite_key_dict,
    const size_t hash_entry_count,
    const int32_t invalid_slot_val,
    const size_t key_component_count,
    const GenericKeyHandler* key_handler,
    const size_t num_elems,
    const size_t block_size_x,
    const size_t grid_size_x);

void fill_one_to_many_baseline_hash_table_on_device_64(
    int32_t* buff,
    const int64_t* composite_key_dict,
    const size_t hash_entry_count,
    const int32_t invalid_slot_val,
    const GenericKeyHandler* key_handler,
    const size_t num_elems,
    const size_t block_size_x,
    const size_t grid_size_x);

void overlaps_fill_one_to_many_baseline_hash_table_on_device_64(
    int32_t* buff,
    const int64_t* composite_key_dict,
    const size_t hash_entry_count,
    const int32_t invalid_slot_val,
    const OverlapsKeyHandler* key_handler,
    const size_t num_elems,
    const size_t block_size_x,
    const size_t grid_size_x);

void approximate_distinct_tuples(uint8_t* hll_buffer_all_cpus,
                                 const uint32_t b,
                                 const size_t padded_size_bytes,
                                 const std::vector<JoinColumn>& join_column_per_key,
                                 const std::vector<JoinColumnTypeInfo>& type_info_per_key,
                                 const int thread_count);

void approximate_distinct_tuples_overlaps(
    uint8_t* hll_buffer_all_cpus,
    std::vector<int32_t>& row_counts,
    const uint32_t b,
    const size_t padded_size_bytes,
    const std::vector<JoinColumn>& join_column_per_key,
    const std::vector<JoinColumnTypeInfo>& type_info_per_key,
    const std::vector<JoinBucketInfo>& join_buckets_per_key,
    const int thread_count);

void approximate_distinct_tuples_on_device(uint8_t* hll_buffer,
                                           const uint32_t b,
                                           const GenericKeyHandler* key_handler,
                                           const size_t num_elems,
                                           const size_t block_size_x,
                                           const size_t grid_size_x);

void approximate_distinct_tuples_on_device_overlaps(uint8_t* hll_buffer,
                                                    const uint32_t b,
                                                    int32_t* row_counts_buffer,
                                                    const OverlapsKeyHandler* key_handler,
                                                    const size_t num_elems,
                                                    const size_t block_size_x,
                                                    const size_t grid_size_x);

void compute_bucket_sizes(std::vector<double>& bucket_sizes_for_dimension,
                          const JoinColumn& join_column,
                          const double bucket_size_threshold,
                          const int thread_count);

void compute_bucket_sizes_on_device(double* bucket_sizes_buffer,
                                    const JoinColumn* join_column_for_key,
                                    const double bucket_sz_threshold,
                                    const size_t block_size_x,
                                    const size_t grid_size_x);

#endif  // QUERYENGINE_HASHJOINRUNTIME_H
