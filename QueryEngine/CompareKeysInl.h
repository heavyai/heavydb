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

#pragma once

#ifdef __CUDACC__
#include "GpuRtConstants.h"
#else
#include "RuntimeFunctions.h"
#endif  // __CUDACC__
#include "../Shared/funcannotations.h"
#include <cstdlib>

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
inline __device__ bool keys_are_equal(const T* key1, const T* key2, const size_t key_component_count) {
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
inline bool keys_are_equal(const T* key1, const T* key2, const size_t key_component_count) {
  return memcmp(key1, key2, key_component_count * sizeof(T)) == 0;
}
#endif  // __CUDACC__
