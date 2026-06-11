/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef QUERYENGINE_EQUIJOINCONDITION_H
#define QUERYENGINE_EQUIJOINCONDITION_H

#include <list>
#include <memory>

namespace Analyzer {
class BinOper;
class Expr;
}  // namespace Analyzer

// Go through the qualifiers and group consecutive equality operators to create
// a list of composite join conditions.
std::list<std::shared_ptr<Analyzer::Expr>> combine_equi_join_conditions(
    const std::list<std::shared_ptr<Analyzer::Expr>>& join_quals);

std::list<std::shared_ptr<Analyzer::Expr>> coalesce_singleton_equi_join(
    const std::shared_ptr<Analyzer::BinOper>& join_qual);

#endif  // QUERYENGINE_EQUIJOINCONDITION_H
