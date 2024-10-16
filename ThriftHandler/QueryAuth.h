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
#include <vector>

class TPlanResult;
namespace query_state {
class QueryStateProxy;
};

namespace query_auth {

struct CapturedColumns {
  std::string db_name;
  std::string table_name;
  std::vector<std::string> column_names;
};

std::vector<CapturedColumns> capture_columns(const std::string& query_ra);

void check_access_privileges(query_state::QueryStateProxy query_state_proxy,
                             const TPlanResult& plan);

};  // namespace query_auth
