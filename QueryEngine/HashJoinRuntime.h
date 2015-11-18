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

int fill_hash_join_buff(int32_t* buff,
                        const int32_t invalid_slot_val,
                        const int8_t* col_buff,
                        const size_t num_elems,
                        const size_t elem_sz,
                        const int64_t min_val,
                        const int64_t null_val,
                        const int64_t translated_null_val,
                        const void* sd_inner,
                        const void* sd_outer,
                        const int32_t cpu_thread_idx,
                        const int32_t cpu_thread_count);

void fill_hash_join_buff_on_device(int32_t* buff,
                                   const int32_t invalid_slot_val,
                                   int* dev_err_buff,
                                   const int8_t* col_buff,
                                   const size_t num_elems,
                                   const size_t elem_sz,
                                   const int64_t min_val,
                                   const int64_t null_val,
                                   const int64_t translated_null_val,
                                   const size_t block_size_x,
                                   const size_t grid_size_x);

#endif  // QUERYENGINE_HASHJOINRUNTIME_H
