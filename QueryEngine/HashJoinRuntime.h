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

struct JoinColumn {
  const int8_t* col_buff;
  const size_t num_elems;
};

struct JoinColumnTypeInfo {
  const size_t elem_sz;
  const int64_t min_val;
  const int64_t null_val;
  const int64_t translated_null_val;
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

#endif  // QUERYENGINE_HASHJOINRUNTIME_H
