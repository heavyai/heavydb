/*
 * Copyright 2022 HEAVY.AI, Inc.
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

#ifndef QUERYENGINE_SORTUTILS_CUH
#define QUERYENGINE_SORTUTILS_CUH

#include <thrust/device_vector.h>
#include "DataMgr/Allocators/ThrustAllocator.h"

template <typename T>
inline thrust::device_ptr<T> get_device_ptr(const size_t host_vec_size,
                                            ThrustAllocator& thrust_allocator) {
  CHECK_GT(host_vec_size, size_t(0));
  const auto host_vec_bytes = host_vec_size * sizeof(T);
  T* dev_ptr = reinterpret_cast<T*>(
      thrust_allocator.allocateScopedBuffer(align_to_int64(host_vec_bytes)));
  return thrust::device_ptr<T>(dev_ptr);
}

#endif  // QUERYENGINE_SORTUTILS_CUH
