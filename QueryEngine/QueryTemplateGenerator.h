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

#ifndef QUERYENGINE_QUERYTEMPLATEGENERATOR_H
#define QUERYENGINE_QUERYTEMPLATEGENERATOR_H

#include <llvm/IR/Module.h>

std::tuple<llvm::Function*, llvm::CallInst*> query_template(
    llvm::Module* mod,
    const size_t aggr_col_count,
    const bool is_estimate_query,
    const bool hoist_literals,
    const QueryMemoryDescriptor& query_mem_desc,
    const ExecutorDeviceType device_type,
    const bool check_scan_limit,
    const GpuSharedMemoryContext& gpu_smem_context,
    const compiler::CodegenTraits& traits);

#endif  // QUERYENGINE_QUERYTEMPLATEGENERATOR_H
