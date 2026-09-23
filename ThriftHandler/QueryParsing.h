/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>

class Calcite;
class TPlanResult;
class TQueryParsingOption;
class TOptimizationOption;

namespace query_state {
class QueryStateProxy;
};

namespace query_parsing {

TPlanResult process_and_check_access_privileges(
    Calcite* calcite,
    query_state::QueryStateProxy query_state_proxy,
    std::string sql_string,
    const TQueryParsingOption& query_parsing_option,
    const TOptimizationOption& optimization_option,
    const bool check_privileges = true,
    const std::string& calcite_session_id = "");

}
