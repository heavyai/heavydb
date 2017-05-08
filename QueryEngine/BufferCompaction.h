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
 * @file    BufferCompaction.h
 * @author  Minggang Yu <miyu@mapd.com>
 * @brief   Macros and functions for groupby buffer compaction
 *
 * Copyright (c) 2016 MapD Technologies, Inc.  All rights reserved.
 */

#ifndef BUFFER_COMPACTION_H
#define BUFFER_COMPACTION_H

#include "../Shared/funcannotations.h"
#include <stdint.h>

#ifndef __CUDACC__
#include <algorithm>
#endif

#define MAX_BYTE_WIDTH_SUPPORTED 8

#ifndef __CUDACC__
inline unsigned compact_byte_width(unsigned qw, unsigned low_bound) {
  return std::max(qw, low_bound);
}
#endif

template <typename T>
FORCE_INLINE HOST DEVICE T align_to_int64(T addr) {
  addr += sizeof(int64_t) - 1;
  return (T)(((uint64_t)addr >> 3) << 3);
}

#endif /* BUFFER_COMPACTION_H */
