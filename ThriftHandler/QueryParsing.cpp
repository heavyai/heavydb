/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryParsing.h"

#include "Calcite/Calcite.h"
#include "ThriftHandler/QueryAuth.h"
#include "ThriftHandler/QueryState.h"

namespace query_parsing {

TPlanResult process_and_check_access_privileges(
    Calcite* calcite,
    query_state::QueryStateProxy query_state_proxy,
    std::string sql_string,
    const TQueryParsingOption& query_parsing_option,
    const TOptimizationOption& optimization_option,
    const bool check_privileges,
    const std::string& calcite_session_id) {
  auto plan_result = calcite->process(query_state_proxy,
                                      sql_string,
                                      query_parsing_option,
                                      optimization_option,
                                      calcite_session_id);
  if (check_privileges && !query_parsing_option.is_explain) {
    query_auth::check_access_privileges(query_state_proxy, plan_result);
  }
  return plan_result;
}

};  // namespace query_parsing
