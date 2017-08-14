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

#ifndef QUERYENGINE_HYPERLOGLOGRT_H
#define QUERYENGINE_HYPERLOGLOGRT_H

#include "../Shared/funcannotations.h"

#ifdef __CUDACC__
inline __device__ int32_t get_rank(uint64_t x, uint32_t b) {
  return min(b, static_cast<uint32_t>(x ? __clzll(x) : 64)) + 1;
}
#else
FORCE_INLINE uint8_t get_rank(uint64_t x, uint32_t b) {
  return std::min(b, static_cast<uint32_t>(x ? __builtin_clzl(x) : 64)) + 1;
}
#endif

#endif  // QUERYENGINE_HYPERLOGLOGRT_H
