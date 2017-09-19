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

struct JoinColumn {
  const int8_t* col_buff;
  size_t num_elems;
};

struct JoinColumnTypeInfo {
  size_t elem_sz;
  int64_t min_val;
  int64_t null_val;
  bool uses_bw_eq;
  int64_t translated_null_val;
  bool is_unsigned;
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
                                    const std::vector<JoinColumn>& join_column_per_key,
                                    const std::vector<JoinColumnTypeInfo>& type_info_per_key,
                                    const std::vector<const void*>& sd_inner_proxy_per_key,
                                    const std::vector<const void*>& sd_outer_proxy_per_key,
                                    const int32_t cpu_thread_idx,
                                    const int32_t cpu_thread_count);

int fill_baseline_hash_join_buff_64(int8_t* hash_buff,
                                    const size_t entry_count,
                                    const int32_t invalid_slot_val,
                                    const size_t key_component_count,
                                    const bool with_val_slot,
                                    const std::vector<JoinColumn>& join_column_per_key,
                                    const std::vector<JoinColumnTypeInfo>& type_info_per_key,
                                    const std::vector<const void*>& sd_inner_proxy_per_key,
                                    const std::vector<const void*>& sd_outer_proxy_per_key,
                                    const int32_t cpu_thread_idx,
                                    const int32_t cpu_thread_count);

void fill_baseline_hash_join_buff_on_device_32(int8_t* hash_buff,
                                               const size_t entry_count,
                                               const int32_t invalid_slot_val,
                                               const size_t key_component_count,
                                               const bool with_val_slot,
                                               int* dev_err_buff,
                                               const JoinColumn* join_column_per_key,
                                               const JoinColumnTypeInfo* type_info_per_key,
                                               const size_t block_size_x,
                                               const size_t grid_size_x);

void fill_baseline_hash_join_buff_on_device_64(int8_t* hash_buff,
                                               const size_t entry_count,
                                               const int32_t invalid_slot_val,
                                               const size_t key_component_count,
                                               const bool with_val_slot,
                                               int* dev_err_buff,
                                               const JoinColumn* join_column_per_key,
                                               const JoinColumnTypeInfo* type_info_per_key,
                                               const size_t block_size_x,
                                               const size_t grid_size_x);

void fill_one_to_many_baseline_hash_table_32(int32_t* buff,
                                             const int32_t* composite_key_dict,
                                             const size_t hash_entry_count,
                                             const int32_t invalid_slot_val,
                                             const size_t key_component_count,
                                             const std::vector<JoinColumn>& join_column_per_key,
                                             const std::vector<JoinColumnTypeInfo>& type_info_per_key,
                                             const std::vector<const void*>& sd_inner_proxy_per_key,
                                             const std::vector<const void*>& sd_outer_proxy_per_key,
                                             const int32_t cpu_thread_count);

void fill_one_to_many_baseline_hash_table_64(int32_t* buff,
                                             const int64_t* composite_key_dict,
                                             const size_t hash_entry_count,
                                             const int32_t invalid_slot_val,
                                             const size_t key_component_count,
                                             const std::vector<JoinColumn>& join_column_per_key,
                                             const std::vector<JoinColumnTypeInfo>& type_info_per_key,
                                             const std::vector<const void*>& sd_inner_proxy_per_key,
                                             const std::vector<const void*>& sd_outer_proxy_per_key,
                                             const int32_t cpu_thread_count);

void fill_one_to_many_baseline_hash_table_on_device_32(int32_t* buff,
                                                       const int32_t* composite_key_dict,
                                                       const size_t hash_entry_count,
                                                       const int32_t invalid_slot_val,
                                                       const size_t key_component_count,
                                                       const JoinColumn* join_column_per_key,
                                                       const JoinColumnTypeInfo* type_info_per_key,
                                                       const size_t block_size_x,
                                                       const size_t grid_size_x);

void fill_one_to_many_baseline_hash_table_on_device_64(int32_t* buff,
                                                       const int64_t* composite_key_dict,
                                                       const size_t hash_entry_count,
                                                       const int32_t invalid_slot_val,
                                                       const size_t key_component_count,
                                                       const JoinColumn* join_column_per_key,
                                                       const JoinColumnTypeInfo* type_info_per_key,
                                                       const size_t block_size_x,
                                                       const size_t grid_size_x);

void approximate_distinct_tuples(uint8_t* hll_buffer_all_cpus,
                                 const uint32_t b,
                                 const size_t padded_size_bytes,
                                 const std::vector<JoinColumn>& join_column_per_key,
                                 const std::vector<JoinColumnTypeInfo>& type_info_per_key,
                                 const int thread_count);

void approximate_distinct_tuples_on_device(uint8_t* hll_buffer,
                                           const uint32_t b,
                                           const size_t padded_size_bytes,
                                           const JoinColumn* join_column_per_key,
                                           const JoinColumnTypeInfo* type_info_per_key,
                                           const size_t block_size_x,
                                           const size_t grid_size_x);

#endif  // QUERYENGINE_HASHJOINRUNTIME_H
