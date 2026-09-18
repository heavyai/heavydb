/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
