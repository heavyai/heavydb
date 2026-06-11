/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * File:   MapDDistributedHandler.h
 * Author: Chris Root
 *
 * Created on Nov 6, 2017, 10:00 AM
 */

#pragma once

#include "../DBHandler.h"

class HeavyDBAggHandler {
 public:
  ~HeavyDBAggHandler() {}

 private:
  HeavyDBAggHandler(DBHandler* db_handler) { CHECK(false); }

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

class HeavyDBLeafHandler {
 public:
  ~HeavyDBLeafHandler() {}

 private:
  HeavyDBLeafHandler(DBHandler* db_handler) { CHECK(false); }

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
                   const std::string& start_time_str,
                   const bool just_explain,
                   const std::vector<int64_t>& outer_fragment_indices) {
    CHECK(false);
  }

  void execute_query_step(TStepResult& _return,
                          const TPendingQuery& pending_query,
                          const TSubqueryId subquery_id,
                          const std::string& start_time_str) {
    CHECK(false);
  }

  void broadcast_serialized_rows(const TSerializedRows& serialized_rows,
                                 const TRowDescriptor& row_desc,
                                 const TQueryId query_id,
                                 const TSubqueryId subquery_id,
                                 const bool is_final_subquery_result) {
    CHECK(false);
  }

  void flush_queue() { CHECK(false); }

  friend class DBHandler;
};
