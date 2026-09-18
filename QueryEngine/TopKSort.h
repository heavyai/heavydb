/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef QUERYENGINE_TOPKSORT_H
#define QUERYENGINE_TOPKSORT_H

#ifdef HAVE_CUDA
#include <cuda.h>
#include "../Shared/sqltypes.h"
#include "ResultSetSortImpl.h"

#include <vector>

namespace Data_Namespace {

class DataMgr;

}  // namespace Data_Namespace

class ThrustAllocator;

std::vector<int8_t> pop_n_rows_from_merged_heaps_gpu(
    Data_Namespace::DataMgr* data_mgr,
    CudaAllocator* cuda_allocator,
    const int64_t* dev_heaps,
    const size_t heaps_size,
    const size_t n,
    const PodOrderEntry& oe,
    const GroupByBufferLayoutInfo& layout,
    const size_t group_key_bytes,
    const size_t thread_count,
    const int device_id,
    CUstream cuda_stream);

#endif  // HAVE_CUDA

#endif  // QUERYENGINE_TOPKSORT_H
