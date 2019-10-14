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

#include "../QueryEngine/CompilationOptions.h"
#include "LeafAggregator.h"
#include "ThriftHandler/QueryState.h"

#include <Catalog/SessionInfo.h>
#include <Catalog/SysCatalog.h>
#include <Catalog/TableDescriptor.h>

#include <fstream>
#include <memory>
#include <string>

namespace Catalog_Namespace {
class SessionInfo;
class Catalog;
struct UserMetadata;
}  // namespace Catalog_Namespace

class ResultSet;
class ExecutionResult;

namespace Planner {
class RootPlan;
}

namespace Parser {
class CopyTableStmt;
}

using query_state::QueryStateProxy;

namespace Importer_NS {
class Loader;
}

class Calcite;

namespace QueryRunner {

struct IRFileWriter {
  IRFileWriter(const std::string& filename) : filename(filename) {
    ofs.open(filename, std::ios::trunc);
  }
  ~IRFileWriter() { ofs.close(); }
  std::string filename;
  std::ofstream ofs;

  void operator()(const std::string& query_str, const std::string& ir_str) {
    ofs << query_str << "\n\n" << ir_str << "\n\n";
  }
};

class QueryRunner {
 public:
  static QueryRunner* init(const char* db_path,
                           const std::string& udf_filename = "",
                           const size_t max_gpu_mem = 0,  // use all available mem
                           const int reserved_gpu_mem = 256 << 20) {
    return QueryRunner::init(db_path,
                             std::string{OMNISCI_ROOT_USER},
                             "HyperInteractive",
                             std::string{OMNISCI_DEFAULT_DB},
                             {},
                             {},
                             udf_filename,
                             true,
                             max_gpu_mem,
                             reserved_gpu_mem);
  }

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
                           const bool create_db = false);

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
  std::shared_ptr<Catalog_Namespace::Catalog> getCatalog() const;
  std::shared_ptr<Calcite> getCalcite() const;

  bool gpusPresent() const;
  virtual void clearGpuMemory() const;
  virtual void clearCpuMemory() const;

  virtual void runDDLStatement(const std::string&);
  virtual std::shared_ptr<ResultSet> runSQL(const std::string& query_str,
                                            const ExecutorDeviceType device_type,
                                            const bool hoist_literals = true,
                                            const bool allow_loop_joins = true);
  virtual ExecutionResult runSelectQuery(const std::string& query_str,
                                         const ExecutorDeviceType device_type,
                                         const bool hoist_literals,
                                         const bool allow_loop_joins,
                                         const bool just_explain = false);
  virtual std::vector<std::shared_ptr<ResultSet>> runMultipleStatements(
      const std::string&,
      const ExecutorDeviceType);

  virtual void runImport(Parser::CopyTableStmt* import_stmt);
  virtual std::unique_ptr<Importer_NS::Loader> getLoader(const TableDescriptor* td) const;

  virtual void setIRFilename(const std::string& filename) {
    ir_file_writer_ = std::make_unique<IRFileWriter>(filename);
  }

  virtual ~QueryRunner() {}

  QueryRunner(std::unique_ptr<Catalog_Namespace::SessionInfo> session);

  static query_state::QueryStates query_states_;

  template <typename... Ts>
  static std::shared_ptr<query_state::QueryState> create_query_state(Ts&&... args) {
    return query_states_.create(std::forward<Ts>(args)...);
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
              const bool create_db);

  Planner::RootPlan* parsePlanLegacy(const std::string& query_str);
  Planner::RootPlan* parsePlanCalcite(QueryStateProxy);
  Planner::RootPlan* parsePlan(QueryStateProxy);

  static std::unique_ptr<QueryRunner> qr_instance_;

  std::shared_ptr<Catalog_Namespace::SessionInfo> session_info_;

 private:
  std::unique_ptr<IRFileWriter> ir_file_writer_;
};

}  // namespace QueryRunner

#endif  // QUERY_RUNNER_H
