/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file    InPlaceSort.h
 * @brief
 *
 */

#ifndef INPLACESORT_H
#define INPLACESORT_H

#include "GpuMemUtils.h"

#include <cstdint>
#include <list>

namespace Analyzer {
struct OrderEntry;
}
class QueryMemoryDescriptor;
namespace Data_Namespace {
class DataMgr;
}

void inplace_sort_gpu(const std::list<Analyzer::OrderEntry>&,
                      const QueryMemoryDescriptor&,
                      const GpuGroupByBuffers&,
                      CudaAllocator* cuda_allocator,
                      CUstream cuda_stream);

void sort_groups_cpu(int64_t* val_buff,
                     int32_t* key_buff,
                     const uint64_t entry_count,
                     const bool desc,
                     const uint32_t chosen_bytes);

void apply_permutation_cpu(int64_t* val_buff,
                           int32_t* idx_buff,
                           const uint64_t entry_count,
                           int64_t* tmp_buff,
                           const uint32_t chosen_bytes);

#endif  // INPLACESORT_H
