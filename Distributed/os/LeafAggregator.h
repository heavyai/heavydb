/*
 * Copyright 2017 MapD Technologies, Inc.
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

#ifndef LEAFAGGREGATOR_H
#define LEAFAGGREGATOR_H

#include "../AggregatedResult.h"
#include "DataMgr/MemoryLevel.h"
#include "LeafHostInfo.h"
#include "QueryEngine/CompilationOptions.h"
#include "QueryEngine/TargetMetaInfo.h"
#include "gen-cpp/OmniSci.h"

#include "Logger/Logger.h"

class SessionInfo;

class ResultSet;

class LeafAggregator {
 public:
  LeafAggregator(const std::vector<LeafHostInfo>& leaves) { CHECK(leaves.empty()); }

  AggregatedResult execute(const SessionInfo& parent_session_info,
                           const std::string& query_ra,
                           const ExecutionOptions& eo,
                           const bool is_update_delete) {
    CHECK(false);
    return {nullptr, {}};
  }

  void leafCatalogConsistencyCheck(const SessionInfo& parent_session_info) {
    CHECK(false);
  }

  std::map<size_t, TQueryResult> forwardQueryToLeaves(
      const SessionInfo& parent_session_info,
      const std::string& query_str) {
    CHECK(false);
    return {};
  }

  TQueryResult forwardQueryToLeaf(const SessionInfo& parent_session_info,
                                  const std::string& query_str,
                                  const size_t leaf_idx) {
    CHECK(false);
    return {};
  }

  void insertDataToLeaf(const SessionInfo& parent_session_info,
                        const size_t leaf_idx,
                        const TInsertData& thrift_insert_data) {
    CHECK(false);
  }

  void checkpointLeafShardsWithAutoRollback(const SessionInfo& parent_session_info,
                                            const int32_t db_id,
                                            const int32_t table_id) {
    CHECK(false);
  }

  int32_t get_table_epochLeaf(const SessionInfo& parent_session_info,
                              const int32_t db_id,
                              const int32_t table_id) {
    CHECK(false);
    return 0;
  }

  void set_table_epochLeaf(const SessionInfo& parent_session_info,
                           const int32_t db_id,
                           const int32_t table_id,
                           const int32_t new_epoch) {
    CHECK(false);
  }

  void connect(const SessionInfo& parent_session_info,
               const std::string& user,
               const std::string& passwd,
               const std::string& dbname) {
    CHECK(false);
  }

  void disconnect(const TSessionId session) { CHECK(false); }

  void switch_database(const TSessionId session, const std::string& dbname) {
    CHECK(false);
  }

  void clone_session(const TSessionId session1, const TSessionId session2) {
    CHECK(false);
  };

  void interrupt(const TSessionId query_session, const TSessionId interrupt_session) {
    CHECK(false);
  }

  void set_execution_mode(const TSessionId session, const TExecuteMode::type mode) {
    CHECK(false);
  }

  size_t leafCount() const { return 0; }

  std::vector<TServerStatus> getLeafStatus(TSessionId session) {
    CHECK(false);
    return {};
  }

  std::vector<TNodeMemoryInfo> getLeafMemoryInfo(
      TSessionId session,
      Data_Namespace::MemoryLevel memory_level) {
    CHECK(false);
    return {};
  }

  TClusterHardwareInfo getHardwareInfo(TSessionId session) {
    CHECK(false);
    return {};
  }

  void clear_leaf_cpu_memory(const TSessionId session) { CHECK(false); }

  void clear_leaf_gpu_memory(const TSessionId session) { CHECK(false); }

  void set_cur_session(const TSessionId parent_session,
                       const std::string& start_time_str,
                       const std::string& label) {
    CHECK(false);
  }

  void invalidate_cur_session(const TSessionId parent_session,
                              const std::string& start_time_str,
                              const std::string& label) {
    CHECK(false);
  }

  std::vector<size_t> query_get_outer_fragment_counts(
      const SessionInfo& parent_session_info,
      std::string& sql_query) {
    CHECK(false);
    return {};
  }
};

#endif  // LEAFAGGREGATOR_H
