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
