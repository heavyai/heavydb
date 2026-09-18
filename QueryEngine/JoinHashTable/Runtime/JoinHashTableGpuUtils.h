/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef QUERYENGINE_JOINHASHTABLE_GPUUTILS_H
#define QUERYENGINE_JOINHASHTABLE_GPUUTILS_H

#include "DataMgr/Allocators/DeviceAllocator.h"
#include "QueryEngine/GpuMemUtils.h"

template <class T>
T* transfer_vector_of_flat_objects_to_gpu(const std::vector<T>& vec,
                                          DeviceAllocator& allocator,
                                          std::string_view tag) {
  static_assert(std::is_trivially_copyable<T>::value && std::is_standard_layout<T>::value,
                "Transferring a vector to GPU only works for flat object elements");
  const auto vec_bytes = vec.size() * sizeof(T);
  auto gpu_vec = allocator.alloc(vec_bytes);
  allocator.copyToDevice(
      gpu_vec, reinterpret_cast<const int8_t*>(vec.data()), vec_bytes, tag);
  return reinterpret_cast<T*>(gpu_vec);
}

template <class T>
T* transfer_flat_object_to_gpu(const T& object,
                               DeviceAllocator& allocator,
                               std::string_view tag) {
  static_assert(std::is_standard_layout<T>::value,
                "Transferring an object to GPU only works for standard layout elements");
  const auto bytes = sizeof(T);
  auto gpu_ptr = allocator.alloc(bytes);
  allocator.copyToDevice(gpu_ptr, reinterpret_cast<const int8_t*>(&object), bytes, tag);
  return reinterpret_cast<T*>(gpu_ptr);
}

#endif  // QUERYENGINE_JOINHASHTABLE_GPUUTILS_H
