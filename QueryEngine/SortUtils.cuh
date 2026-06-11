/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
