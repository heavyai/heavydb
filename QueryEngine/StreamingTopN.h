/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file    StreamingTopN.h
 * @brief   Streaming Top N algorithm.
 *
 */

#ifndef QUERYENGINE_STREAMINGTOPN_H
#define QUERYENGINE_STREAMINGTOPN_H

#include <cstddef>
#include <cstdint>
#include <vector>

namespace streaming_top_n {

size_t get_heap_size(const size_t row_size, const size_t n, const size_t thread_count);

size_t get_rows_offset_of_heaps(const size_t n, const size_t thread_count);

std::vector<int8_t> get_rows_copy_from_heaps(const int64_t* heaps,
                                             const size_t heaps_size,
                                             const size_t n,
                                             const size_t thread_count);

}  // namespace streaming_top_n

struct RelAlgExecutionUnit;

namespace Analyzer {
class Expr;
}  // namespace Analyzer

// Compute the slot index where the target given by target_idx is stored, where
// target_exprs is the list all projected expressions.
size_t get_heap_key_slot_index(const std::vector<Analyzer::Expr*>& target_exprs,
                               const size_t target_idx);

#ifdef HAVE_CUDA
namespace Data_Namespace {

class DataMgr;

}  // namespace Data_Namespace

class QueryMemoryDescriptor;
class CudaAllocator;
std::vector<int8_t> pick_top_n_rows_from_dev_heaps(
    Data_Namespace::DataMgr* data_mgr,
    CudaAllocator* cuda_allocator,
    const int64_t* dev_heaps,
    const RelAlgExecutionUnit& ra_exe_unit,
    const QueryMemoryDescriptor& query_mem_desc,
    const size_t thread_count,
    const int device_id,
    CUstream cuda_stream);
#endif  // HAVE_CUDA
#endif  // QUERYENGINE_STREAMINGTOPN_H
