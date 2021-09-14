/*
 * Copyright 2019 OmniSci, Inc.
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
#include "QueryEngine/QueryHint.h"
#include "QueryEngine/RelAlgDagBuilder.h"
#include "QueryEngine/RelAlgExecutionUnit.h"
#include "QueryEngine/RelAlgTranslator.h"
#include "ThriftHandler/QueryState.h"

namespace Catalog_Namespace {
class Catalog;
struct UserMetadata;
}  // namespace Catalog_Namespace

class ResultSet;
class ExecutionResult;

namespace Parser {
class DDLStmt;
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
                             std::string{OMNISCI_ROOT_USER},
                             "HyperInteractive",
                             std::string{OMNISCI_DEFAULT_DB},
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
    session_info_ =
        std::make_unique<Catalog_Namespace::SessionInfo>(session_info_->get_catalog_ptr(),
                                                         session_info_->get_currentUser(),
                                                         device_type,
                                                         session_id);
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
  BufferPoolStats getBufferPoolStats(
      const Data_Namespace::MemoryLevel memory_level) const;

  virtual std::unique_ptr<Parser::DDLStmt> createDDLStatement(const std::string&);
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
  virtual RegisteredQueryHint getParsedQueryHint(const std::string&);

  virtual void runImport(Parser::CopyTableStmt* import_stmt);
  virtual std::unique_ptr<import_export::Loader> getLoader(
      const TableDescriptor* td) const;

  const int32_t* getCachedJoinHashTable(size_t idx);
  const int8_t* getCachedBaselineHashTable(size_t idx);
  size_t getEntryCntCachedBaselineHashTable(size_t idx);
  size_t getNumberOfCachedJoinHashTables();
  size_t getNumberOfCachedBaselineJoinHashTables();
  size_t getNumberOfCachedOverlapsHashTables();

  void resizeDispatchQueue(const size_t num_executors);

  QueryPlanDagInfo getQueryInfoForDataRecyclerTest(const std::string&);

  std::shared_ptr<RelAlgTranslator> getRelAlgTranslator(const std::string&, Executor*);

  void printQueryPlanDagCache() const;

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

  std::shared_ptr<Catalog_Namespace::SessionInfo> session_info_;
  std::unique_ptr<QueryDispatchQueue> dispatch_queue_;
  std::shared_ptr<Data_Namespace::DataMgr> data_mgr_;
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
