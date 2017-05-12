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
 * @file    JoinHashImpl.h
 * @author  Alex Suhan <alex@mapd.com>
 *
 * Copyright (c) 2015 MapD Technologies, Inc.  All rights reserved.
 */

#ifndef QUERYENGINE_GROUPBYFASTIMPL_H
#define QUERYENGINE_GROUPBYFASTIMPL_H

#include "../Shared/funcannotations.h"
#include <stdint.h>

extern "C" ALWAYS_INLINE DEVICE int32_t* SUFFIX(get_hash_slot)(int32_t* buff,
                                                               const int64_t key,
                                                               const int64_t min_key) {
  return buff + (key - min_key);
}

extern "C" ALWAYS_INLINE DEVICE int32_t* SUFFIX(get_hash_slot_sharded)(int32_t* buff,
                                                                       const int64_t key,
                                                                       const int64_t min_key,
                                                                       const uint32_t entry_count_per_shard,
                                                                       const uint32_t num_shards,
                                                                       const uint32_t device_count) {
  const uint32_t shard = key % num_shards;
  const uint32_t shard_buffer_index = shard / device_count;  // shard sub-buffer index within `buff`
  int32_t* shard_buffer = buff + shard_buffer_index * entry_count_per_shard;
  return shard_buffer + (key - min_key) / num_shards;
}

#endif  // QUERYENGINE_GROUPBYFASTIMPL_H
