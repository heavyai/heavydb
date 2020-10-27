/*
 * Copyright 2020 OmniSci, Inc.
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

/*
 * File:   MapDDistributedHandler.h
 * Author: Chris Root
 *
 * Created on Nov 6, 2017, 10:00 AM
 */

#pragma once

#include "../DBHandler.h"

class MapDAggHandler {
 public:
  ~MapDAggHandler() {}

 private:
  MapDAggHandler(DBHandler* mapd_handler) { CHECK(false); }

  void cluster_execute(TQueryResult& _return,
                       QueryStateProxy,
                       const std::string& query_str,
                       const bool column_format,
                       const std::string& nonce,
                       const int32_t first_n,
                       const int32_t at_most_n,
                       const SystemParameters& system_parameters) {
    CHECK(false);
  }
  friend class DBHandler;
};

class MapDLeafHandler {
 public:
  ~MapDLeafHandler() {}

 private:
  MapDLeafHandler(DBHandler* mapd_handler) { CHECK(false); }

  int64_t query_get_outer_fragment_count(const TSessionId& session,
                                         const std::string& select_query) {
    CHECK(false);
    return -1;
  };

  void check_table_consistency(TTableMeta& _return,
                               const TSessionId& session,
                               const int32_t table_id) {
    CHECK(false);
  };

  void start_query(TPendingQuery& _return,
                   const TSessionId& leaf_session,
                   const TSessionId& parent_session,
                   const std::string& query_ra,
                   const bool just_explain,
                   const std::vector<int64_t>& outer_fragment_indices) {
    CHECK(false);
  }

  void execute_query_step(TStepResult& _return,
                          const TPendingQuery& pending_query,
                          const TSubqueryId subquery_id) {
    CHECK(false);
  }

  void broadcast_serialized_rows(const TSerializedRows& serialized_rows,
                                 const TRowDescriptor& row_desc,
                                 const TQueryId query_id,
                                 const TSubqueryId subquery_id) {
    CHECK(false);
  }

  void flush_queue() { CHECK(false); }

  friend class DBHandler;
};
