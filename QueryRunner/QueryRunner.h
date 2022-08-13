/*
 * Copyright 2022 HEAVY.AI, Inc.
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

#ifndef QUERY_RUNNER_H
#define QUERY_RUNNER_H

#include <fstream>
#include <memory>
#include <optional>
#include <string>

#include "Catalog/SessionInfo.h"
#include "Catalog/SysCatalog.h"
#include "Catalog/TableDescriptor.h"
#include "LeafAggregator.h"
#include "QueryEngine/CompilationOptions.h"
#include "QueryEngine/JoinHashTable/BaselineJoinHashTable.h"
#include "QueryEngine/JoinHashTable/HashJoin.h"
#include "QueryEngine/JoinHashTable/OverlapsJoinHashTable.h"
#include "QueryEngine/QueryDispatchQueue.h"
#include "QueryEngine/QueryEngine.h"
#include "QueryEngine/QueryHint.h"
#include "QueryEngine/QueryPlanDagExtractor.h"
#include "QueryEngine/RelAlgDag.h"
#include "QueryEngine/RelAlgExecutionUnit.h"
#include "QueryEngine/RelAlgTranslator.h"
#include "Shared/SysDefinitions.h"
#include "ThriftHandler/QueryState.h"

namespace Catalog_Namespace {
class Catalog;
struct UserMetadata;
}  // namespace Catalog_Namespace

class ResultSet;
class ExecutionResult;

namespace Parser {
class Stmt;
class CopyTableStmt;
}  // namespace Parser

using query_state::QueryStateProxy;

namespace import_export {
class Loader;
}

class Calcite;

namespace QueryRunner {

struct QueryPlanDagInfo {
  std::shared_ptr<const RelAlgNode> root_node;
  std::vector<unsigned> left_deep_trees_id;
  std::unordered_map<unsigned, JoinQualsPerNestingLevel> left_deep_trees_info;
  std::shared_ptr<RelAlgTranslator> rel_alg_translator;
};

// Keep original values of data recycler related flags
// and restore them when QR instance is destructed
// Our test environment checks various bare-metal components of the system
// including computing various relational operations and expressions,
// building hash table, and so on
// Thus, unless we explicitly test those cases, we must disable all of them
// in the test framework by default
// Since we enable data recycler and hash table recycler by default,
// we keep them as is, but disable resultset recycler and its relevant
// stuffs to keep our testing environment as is
class DataRecyclerFlagsDisabler {
 public:
  DataRecyclerFlagsDisabler()
      : orig_chunk_metadata_cache_(g_use_chunk_metadata_cache)
      , orig_resultset_cache_(g_use_query_resultset_cache)
      , orig_allow_query_step_skipping_(g_allow_query_step_skipping)
      , orig_allow_auto_resultset_caching_(g_allow_auto_resultset_caching) {
    g_use_chunk_metadata_cache = false;
    g_use_query_resultset_cache = false;
    g_allow_query_step_skipping = false;
    g_allow_auto_resultset_caching = false;
  }

  ~DataRecyclerFlagsDisabler() {
    // restore the flag values
    g_use_chunk_metadata_cache = orig_chunk_metadata_cache_;
    g_use_query_resultset_cache = orig_resultset_cache_;
    g_allow_query_step_skipping = orig_allow_query_step_skipping_;
    g_allow_auto_resultset_caching = orig_allow_auto_resultset_caching_;
  }

 private:
  bool orig_chunk_metadata_cache_;
  bool orig_resultset_cache_;
  bool orig_allow_query_step_skipping_;
  bool orig_allow_auto_resultset_caching_;
};

enum CacheItemStatus {
  CLEAN_ONLY,
  DIRTY_ONLY,
  ALL  // CLEAN + DIRTY
};

struct BufferPoolStats {
  size_t num_buffers;
  size_t num_bytes;
  size_t num_tables;
  size_t num_columns;
  size_t num_fragments;
  size_t num_chunks;

  void print() const {
    std::cout << std::endl
              << std::endl
              << "------------ Buffer Pool Stats  ------------" << std::endl;
    std::cout << "Num buffers: " << num_buffers << std::endl;
    std::cout << "Num bytes: " << num_bytes << std::endl;
    std::cout << "Num tables: " << num_tables << std::endl;
    std::cout << "Num columns: " << num_columns << std::endl;
    std::cout << "Num fragments: " << num_fragments << std::endl;
    std::cout << "Num chunks: " << num_chunks << std::endl;
    std::cout << "--------------------------------------------" << std::endl << std::endl;
  }
};

class QueryRunner {
 public:
  static QueryRunner* init(const char* db_path,
                           const std::string& udf_filename = "",
                           const size_t max_gpu_mem = 0,  // use all available mem
                           const int reserved_gpu_mem = 256 << 20);

  static QueryRunner* init(const File_Namespace::DiskCacheConfig* disk_cache_config,
                           const char* db_path,
                           const std::vector<LeafHostInfo>& string_servers = {},
                           const std::vector<LeafHostInfo>& leaf_servers = {});

  static QueryRunner* init(const char* db_path,
                           const std::vector<LeafHostInfo>& string_servers,
                           const std::vector<LeafHostInfo>& leaf_servers) {
    return QueryRunner::init(db_path,
                             shared::kRootUsername,
                             "HyperInteractive",
                             shared::kDefaultDbName,
                             string_servers,
                             leaf_servers);
  }

  static QueryRunner* init(const char* db_path,
                           const std::string& user,
                           const std::string& pass,
                           const std::string& db_name,
                           const std::vector<LeafHostInfo>& string_servers,
                           const std::vector<LeafHostInfo>& leaf_servers,
                           const std::string& udf_filename = "",
                           bool uses_gpus = true,
                           const size_t max_gpu_mem = 0,  // use all available mem
                           const int reserved_gpu_mem = 256 << 20,
                           const bool create_user = false,
                           const bool create_db = false,
                           const File_Namespace::DiskCacheConfig* config = nullptr);

  static QueryRunner* init(std::unique_ptr<Catalog_Namespace::SessionInfo>& session) {
    qr_instance_.reset(new QueryRunner(std::move(session)));
    return qr_instance_.get();
  }

  static QueryRunner* get() {
    if (!qr_instance_) {
      throw std::runtime_error("QueryRunner must be initialized before calling get().");
    }
    return qr_instance_.get();
  }

  static void reset();

  std::shared_ptr<Catalog_Namespace::SessionInfo> getSession() const {
    return session_info_;
  }

  void addSessionId(const std::string& session_id,
                    ExecutorDeviceType device_type = ExecutorDeviceType::GPU) {
    auto user_info = session_info_->get_currentUser();
    session_info_ = std::make_unique<Catalog_Namespace::SessionInfo>(
        session_info_->get_catalog_ptr(), user_info, device_type, session_id);
  }

  void clearSessionId() { session_info_ = nullptr; }

  std::shared_ptr<Catalog_Namespace::Catalog> getCatalog() const;
  std::shared_ptr<Calcite> getCalcite() const;
  std::shared_ptr<Executor> getExecutor() const;
  Catalog_Namespace::UserMetadata& getUserMetadata() const;

  bool gpusPresent() const;
  virtual void clearGpuMemory() const;
  virtual void clearCpuMemory() const;
  std::vector<MemoryInfo> getMemoryInfo(
      const Data_Namespace::MemoryLevel memory_level) const;
  BufferPoolStats getBufferPoolStats(const Data_Namespace::MemoryLevel memory_level,
                                     const bool current_db_only) const;

  virtual std::unique_ptr<Parser::Stmt> createStatement(const std::string&);
  virtual void runDDLStatement(const std::string&);
  virtual void validateDDLStatement(const std::string&);

  virtual std::shared_ptr<ResultSet> runSQL(const std::string& query_str,
                                            CompilationOptions co,
                                            ExecutionOptions eo);
  virtual std::shared_ptr<ExecutionResult> runSelectQuery(const std::string& query_str,
                                                          CompilationOptions co,
                                                          ExecutionOptions eo);
  static ExecutionOptions defaultExecutionOptionsForRunSQL(bool allow_loop_joins = true,
                                                           bool just_explain = false);

  // TODO: Refactor away functions such as runSQL() and runSelectQuery() with arbitrary
  // parameters that grow over time. Instead, pass CompilationOptions and
  // ExecutionOptions which can be extended without changing the function signatures.
  // Why?
  //  * Functions with a large number of parameters are hard to maintain and error-prone.
  //  * "Default arguments are banned on virtual functions"
  //    https://google.github.io/styleguide/cppguide.html#Default_Arguments
  virtual std::shared_ptr<ResultSet> runSQL(const std::string& query_str,
                                            const ExecutorDeviceType device_type,
                                            const bool hoist_literals = true,
                                            const bool allow_loop_joins = true);
  virtual std::shared_ptr<ExecutionResult> runSelectQuery(
      const std::string& query_str,
      const ExecutorDeviceType device_type,
      const bool hoist_literals,
      const bool allow_loop_joins,
      const bool just_explain = false);
  virtual std::shared_ptr<ResultSet> runSQLWithAllowingInterrupt(
      const std::string& query_str,
      const std::string& session_id,
      const ExecutorDeviceType device_type,
      const double running_query_check_freq = 0.9,
      const unsigned pending_query_check_freq = 1000);

  virtual std::vector<std::shared_ptr<ResultSet>> runMultipleStatements(
      const std::string&,
      const ExecutorDeviceType);
  virtual void runImport(Parser::CopyTableStmt* import_stmt);
  virtual std::unique_ptr<import_export::Loader> getLoader(
      const TableDescriptor* td) const;

  RegisteredQueryHint getParsedQueryHint(const std::string&);
  std::optional<
      std::unordered_map<size_t, std::unordered_map<unsigned, RegisteredQueryHint>>>
  getParsedQueryHints(const std::string& query_str);
  std::shared_ptr<const RelAlgNode> getRootNodeFromParsedQuery(
      const std::string& query_str);
  std::optional<RegisteredQueryHint> getParsedGlobalQueryHints(
      const std::string& query_str);
  RaExecutionSequence getRaExecutionSequence(const std::string& query_str);
  virtual std::shared_ptr<ResultSet> getCalcitePlan(const std::string& query_str,
                                                    bool enable_watchdog,
                                                    bool as_json_str) const;

  std::tuple<QueryPlanHash,
             std::shared_ptr<HashTable>,
             std::optional<HashtableCacheMetaInfo>>
  getCachedHashtableWithoutCacheKey(std::set<size_t>& visited,
                                    CacheItemType hash_table_type,
                                    DeviceIdentifier device_identifier);
  std::shared_ptr<CacheItemMetric> getCacheItemMetric(QueryPlanHash cache_key,
                                                      CacheItemType hash_table_type,
                                                      DeviceIdentifier device_identifier);
  size_t getNumberOfCachedItem(CacheItemStatus item_status,
                               CacheItemType hash_table_type,
                               bool with_overlaps_tuning_param = false) const;

  void resizeDispatchQueue(const size_t num_executors);

  QueryPlanDagInfo getQueryInfoForDataRecyclerTest(const std::string&);

  std::shared_ptr<RelAlgTranslator> getRelAlgTranslator(const std::string&, Executor*);

  ExtractedQueryPlanDag extractQueryPlanDag(const std::string&);

  std::unique_ptr<RelAlgDag> getRelAlgDag(const std::string&);

  QueryRunner(std::unique_ptr<Catalog_Namespace::SessionInfo> session);

  virtual ~QueryRunner() = default;

  static query_state::QueryStates query_states_;

  template <typename... Ts>
  static std::shared_ptr<query_state::QueryState> create_query_state(Ts&&... args) {
    return query_states_.create(std::forward<Ts>(args)...);
  }

  void setExplainType(const ExecutorExplainType explain_type) {
    explain_type_ = explain_type;
  }

 protected:
  QueryRunner(const char* db_path,
              const std::string& user,
              const std::string& pass,
              const std::string& db_name,
              const std::vector<LeafHostInfo>& string_servers,
              const std::vector<LeafHostInfo>& leaf_servers,
              const std::string& udf_filename,
              bool uses_gpus,
              const size_t max_gpu_mem,
              const int reserved_gpu_mem,
              const bool create_user,
              const bool create_db,
              const File_Namespace::DiskCacheConfig* disk_cache_config = nullptr);
  static std::unique_ptr<QueryRunner> qr_instance_;

  ExecutorExplainType explain_type_ = ExecutorExplainType::Default;

  Catalog_Namespace::DBMetadata db_metadata_;
  std::shared_ptr<Catalog_Namespace::SessionInfo> session_info_;
  std::unique_ptr<QueryDispatchQueue> dispatch_queue_;
  std::shared_ptr<Data_Namespace::DataMgr> data_mgr_;
  std::shared_ptr<QueryEngine> query_engine_;
};

class ImportDriver : public QueryRunner {
 public:
  ImportDriver(std::shared_ptr<Catalog_Namespace::Catalog> cat,
               const Catalog_Namespace::UserMetadata& user,
               const ExecutorDeviceType dt = ExecutorDeviceType::GPU,
               const std::string session_id = "");

  void importGeoTable(const std::string& file_path,
                      const std::string& table_name,
                      const bool compression,
                      const bool create_table,
                      const bool explode_collections);
};

}  // namespace QueryRunner

#endif  // QUERY_RUNNER_H
