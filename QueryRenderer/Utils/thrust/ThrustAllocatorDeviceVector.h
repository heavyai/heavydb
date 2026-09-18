/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <thrust/device_vector.h>

#include "DataMgr/Allocators/thrust/TypedThrustAllocator.h"
#include "QueryRenderer/Utils/thrust/ThrustExecutionContext.h"

namespace QueryRenderer {

/**
 * @brief a utility function for creating thrust device vectors using a ThrustAllocator as
 * a custom allocator. This handles some of the boilerplate otherwise required.
 */
template <typename T, typename... Targs>
auto make_device_vector_from_context(DataMgrThrustContext& context, Targs&&... Fargs) {
  using DeviceAllocator = Data_Namespace::TypedThrustAllocator<T>;
  return ::thrust::device_vector<T, DeviceAllocator>(
      std::forward<Targs>(Fargs)..., DeviceAllocator(context.getAllocator()));
}

}  // namespace QueryRenderer
