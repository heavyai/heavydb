/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif
#include <thrust/execution_policy.h>

#include "DataMgr/Allocators/ThrustAllocator.h"

namespace Data_Namespace {
/**
 * @brief This is a custom device execution policy to allow a custom allocator backed by a
 * DataMgr instance for any temp allocations done in thrust calls. This execution policy
 * is required when using thrust functions that do host <-> device transfers internally,
 * such as thrust::equal_range(). Without the use of this custom policy,
 * thrust::equal_range() will crash. For more clarity, if you were to use the
 * thrust::device(allocator) as the execution policy for thrust::equal_range, where
 * allocator is a ThrustAllocator instance, thrust will crash because that policy by
 * default uses a cuda default stream wrapper class (execute_on_stream_base):
 *
 * https://github.com/thrust/thrust/blob/cuda-10.2/thrust/system/cuda/detail/par.h#L74
 *
 * The get_stream friend function in that wrapper class attempts to get a stream from a
 * wrapper that is not defined, and therefore segfaults.
 *
 * So this custom device policy is a way around that problem. It ensures that
 * execute_on_stream_base is not used as a default stream policy and instead uses the
 * default stream.
 *
 * If streams are ever intended to be used in the future for any of this thrust work,
 * this will need to be adjusted, and likely care required in getting
 * thrust::equal_range to work.
 *
 * Note the above crash scenario was noticed while using cuda 10.2 on 06/03/2020.
 */
using DataMgrAllocationPolicy = typename thrust::detail::allocator_aware_execution_policy<
    thrust::device_execution_policy>::execute_with_allocator_type<ThrustAllocator>::type;
}  // namespace Data_Namespace
