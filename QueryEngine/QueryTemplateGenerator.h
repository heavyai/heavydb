/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef QUERYENGINE_QUERYTEMPLATEGENERATOR_H
#define QUERYENGINE_QUERYTEMPLATEGENERATOR_H

#include "GroupByAndAggregate.h"

#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>

#include <string>

std::tuple<llvm::Function*, llvm::CallInst*> query_template(
    llvm::Module*,
    const size_t aggr_col_count,
    const bool hoist_literals,
    const bool is_estimate_query,
    const GpuSharedMemoryContext& gpu_smem_context);
std::tuple<llvm::Function*, llvm::CallInst*> query_group_by_template(
    llvm::Module*,
    const bool hoist_literals,
    const QueryMemoryDescriptor& query_mem_desc,
    const ExecutorDeviceType,
    const bool check_scan_limit,
    const GpuSharedMemoryContext& gpu_smem_context);

#endif  // QUERYENGINE_QUERYTEMPLATEGENERATOR_H
