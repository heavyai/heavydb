/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
