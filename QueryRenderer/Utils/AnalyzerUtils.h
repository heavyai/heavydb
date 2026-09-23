/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <utility>

namespace Analyzer {
class TargetEntry;
class Expr;
}  // namespace Analyzer

namespace QueryRenderer {

std::pair<int, int> get_table_id_col_id_from_target_expr(const Analyzer::Expr* expr);

}  // namespace QueryRenderer
