/*
 * @file    HashJoinRuntime.h
 * @author  Alex Suhan <alex@mapd.com>
 *
 * Copyright (c) 2015 MapD Technologies, Inc.  All rights reserved.
 */

#ifndef QUERYENGINE_HASHJOINRUNTIME_H
#define QUERYENGINE_HASHJOINRUNTIME_H

#include <cstddef>
#include <cstdint>

void init_hash_join_buff(int64_t* buff,
                         const int32_t groups_buffer_entry_count,
                         const int8_t* col_buff,
                         const size_t num_elems,
                         const size_t elem_sz,
                         const int64_t min_val);

#endif  // QUERYENGINE_HASHJOINRUNTIME_H
