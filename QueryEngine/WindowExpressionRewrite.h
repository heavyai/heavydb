/*
 * Copyright 2019 OmniSci, Inc.
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

#pragma once

#include "../Analyzer/Analyzer.h"

// Sum window function come from Calcite with a redundant case expression. We don't
// support complex expressions involving window functions, rewrite to just a sum.
std::shared_ptr<Analyzer::WindowFunction> rewrite_sum_window(const Analyzer::Expr* expr);

// Same as above, but for average. Additionally, replace the sum divided by count
// expression with an explicit average.
std::shared_ptr<Analyzer::WindowFunction> rewrite_avg_window(const Analyzer::Expr* expr);
