/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef QUERYENGINE_RELALGOPTIMIZER_H
#define QUERYENGINE_RELALGOPTIMIZER_H

#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

class RelAlgNode;
class RexSubQuery;

std::unordered_map<const RelAlgNode*, std::unordered_set<const RelAlgNode*>> build_du_web(
    const std::vector<std::shared_ptr<RelAlgNode>>& nodes) noexcept;
void eliminate_identical_copy(std::vector<std::shared_ptr<RelAlgNode>>& nodes) noexcept;
void eliminate_dead_columns(std::vector<std::shared_ptr<RelAlgNode>>& nodes) noexcept;
void eliminate_dead_subqueries(std::vector<std::shared_ptr<RexSubQuery>>& subqueries,
                               RelAlgNode const* root);
void fold_filters(std::vector<std::shared_ptr<RelAlgNode>>& nodes) noexcept;
void hoist_filter_cond_to_cross_join(
    std::vector<std::shared_ptr<RelAlgNode>>& nodes) noexcept;
void simplify_sort(std::vector<std::shared_ptr<RelAlgNode>>& nodes) noexcept;
void sink_projected_boolean_expr_to_join(
    std::vector<std::shared_ptr<RelAlgNode>>& nodes) noexcept;

#endif  // QUERYENGINE_RELALGOPTIMIZER_H
