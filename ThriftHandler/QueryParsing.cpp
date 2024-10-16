/*
 * Copyright 2024 HEAVY.AI, Inc.
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
