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

#ifndef QUERYENGINE_RELALGOPTIMIZER_H
#define QUERYENGINE_RELALGOPTIMIZER_H

#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

class RelAlgNode;

std::unordered_map<const RelAlgNode*, std::unordered_set<const RelAlgNode*>> build_du_web(
    const std::vector<std::shared_ptr<RelAlgNode>>& nodes) noexcept;
void eliminate_identical_copy(std::vector<std::shared_ptr<RelAlgNode>>& nodes) noexcept;
void eliminate_dead_columns(std::vector<std::shared_ptr<RelAlgNode>>& nodes) noexcept;
void fold_filters(std::vector<std::shared_ptr<RelAlgNode>>& nodes) noexcept;
void hoist_filter_cond_to_cross_join(std::vector<std::shared_ptr<RelAlgNode>>& nodes) noexcept;
void simplify_sort(std::vector<std::shared_ptr<RelAlgNode>>& nodes) noexcept;
void sink_projected_boolean_expr_to_join(std::vector<std::shared_ptr<RelAlgNode>>& nodes) noexcept;

#endif  // QUERYENGINE_RELALGOPTIMIZER_H
