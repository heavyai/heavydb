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

/**
 * @file    OutputBufferInitialization.h
 * @author  Alex Suhan <alex@mapd.com>
 */

#ifndef QUERYENGINE_OUTPUTBUFFERINITIALIZATION_H
#define QUERYENGINE_OUTPUTBUFFERINITIALIZATION_H

#include "SqlTypesLayout.h"

#include <list>
#include <memory>
#include <utility>
#include <vector>

namespace Analyzer {
class Expr;
}  // namespace Analyzer

class QueryMemoryDescriptor;

std::pair<int64_t, int64_t> inline_int_max_min(const size_t byte_width);

std::pair<uint64_t, uint64_t> inline_uint_max_min(const size_t byte_width);

int64_t get_agg_initial_val(const SQLAgg agg,
                            const SQLTypeInfo& ti,
                            const bool enable_compaction,
                            const unsigned min_byte_width_to_compact);

std::vector<int64_t> init_agg_val_vec(
    const std::vector<Analyzer::Expr*>& targets,
    const std::list<std::shared_ptr<Analyzer::Expr>>& quals,
    const QueryMemoryDescriptor& query_mem_desc);

const Analyzer::Expr* agg_arg(const Analyzer::Expr* expr);

bool constrained_not_null(const Analyzer::Expr* expr,
                          const std::list<std::shared_ptr<Analyzer::Expr>>& quals);

void set_notnull(TargetInfo& target, const bool not_null);

#endif  // QUERYENGINE_OUTPUTBUFFERINITIALIZATION_H
