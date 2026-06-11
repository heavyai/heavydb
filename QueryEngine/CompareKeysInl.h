/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#ifdef __CUDACC__
#include "GpuRtConstants.h"
#else
#include "RuntimeFunctions.h"
#endif  // __CUDACC__
#include <cstdlib>
#include "../Shared/funcannotations.h"

template <typename T = int64_t>
inline DEVICE T SUFFIX(get_invalid_key)() {
  return EMPTY_KEY_64;
}

template <>
inline DEVICE int32_t SUFFIX(get_invalid_key)() {
  return EMPTY_KEY_32;
}

#ifdef __CUDACC__
template <typename T>
inline __device__ bool keys_are_equal(const T* key1,
                                      const T* key2,
                                      const size_t key_component_count) {
  for (size_t i = 0; i < key_component_count; ++i) {
    if (key1[i] != key2[i]) {
      return false;
    }
  }
  return true;
}
#else
#include <cstring>

template <typename T>
inline bool keys_are_equal(const T* key1,
                           const T* key2,
                           const size_t key_component_count) {
  return memcmp(key1, key2, key_component_count * sizeof(T)) == 0;
}
#endif  // __CUDACC__
