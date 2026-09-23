/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../Analyzer/Analyzer.h"

// Sum window function come from Calcite with a redundant case expression. We don't
// support complex expressions involving window functions, rewrite to just a sum.
std::shared_ptr<Analyzer::WindowFunction> rewrite_sum_window(const Analyzer::Expr* expr);

// Same as above, but for average. Additionally, replace the sum divided by count
// expression with an explicit average.
std::shared_ptr<Analyzer::WindowFunction> rewrite_avg_window(const Analyzer::Expr* expr);
