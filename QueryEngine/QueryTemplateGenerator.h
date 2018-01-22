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

#include "GroupByAndAggregate.h"

#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>

#include <string>

std::string unique_name(const char* base_name, const bool is_nested);
llvm::Function* query_template(llvm::Module*,
                               const size_t aggr_col_count,
                               const bool is_nested,
                               const bool hoist_literals,
                               const bool is_estimate_query);
llvm::Function* query_group_by_template(llvm::Module*,
                                        const bool is_nested,
                                        const bool hoist_literals,
                                        const QueryMemoryDescriptor& query_mem_desc,
                                        const ExecutorDeviceType,
                                        const bool check_scan_limit);

#endif  // QUERYENGINE_QUERYTEMPLATEGENERATOR_H
