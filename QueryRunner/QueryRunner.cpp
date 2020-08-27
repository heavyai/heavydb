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

#include "QueryRunner.h"

#include "Calcite/Calcite.h"
#include "Catalog/Catalog.h"
#include "DistributedLoader.h"
#include "ImportExport/CopyParams.h"
#include "Parser/ParserWrapper.h"
#include "Parser/parser.h"
#include "QueryEngine/CalciteAdapter.h"
#include "QueryEngine/ExtensionFunctionsWhitelist.h"
#include "QueryEngine/QueryDispatchQueue.h"
#include "QueryEngine/RelAlgExecutor.h"
#include "QueryEngine/TableFunctions/TableFunctionsFactory.h"
#include "Shared/Logger.h"
#include "Shared/StringTransform.h"
#include "Shared/SystemParameters.h"
#include "Shared/geosupport.h"
#include "Shared/import_helpers.h"
#include "bcrypt.h"
#include "gen-cpp/CalciteServer.h"

#include <boost/filesystem/operations.hpp>
#include <csignal>
#include <random>

#define CALCITEPORT 3279

extern size_t g_leaf_count;
extern bool g_enable_filter_push_down;

double g_gpu_mem_limit_percent{0.9};

extern bool g_serialize_temp_tables;
bool g_enable_calcite_view_optimize{true};
std::mutex calcite_lock;

using namespace Catalog_Namespace;
namespace {

std::shared_ptr<Calcite> g_calcite = nullptr;

void calcite_shutdown_handler() noexcept {
  if (g_calcite) {
    g_calcite->close_calcite_server();
    g_calcite.reset();
  }
}

void mapd_signal_handler(int signal_number) {
  LOG(ERROR) << "Interrupt signal (" << signal_number << ") received.";
  calcite_shutdown_handler();
  // shut down logging force a flush
  logger::shutdown();
  // terminate program
  if (signal_number == SIGTERM) {
    std::exit(EXIT_SUCCESS);
  } else {
    std::exit(signal_number);
  }
}

void register_signal_handler() {
  std::signal(SIGTERM, mapd_signal_handler);
  std::signal(SIGSEGV, mapd_signal_handler);
  std::signal(SIGABRT, mapd_signal_handler);
}

}  // namespace

namespace QueryRunner {

std::unique_ptr<QueryRunner> QueryRunner::qr_instance_ = nullptr;

query_state::QueryStates QueryRunner::query_states_;

QueryRunner* QueryRunner::init(const char* db_path,
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
                               const bool create_db) {
  LOG_IF(FATAL, !leaf_servers.empty()) << "Distributed test runner not supported.";
  CHECK(leaf_servers.empty());
  qr_instance_.reset(new QueryRunner(db_path,
                                     user,
                                     pass,
                                     db_name,
                                     string_servers,
                                     leaf_servers,
                                     udf_filename,
                                     uses_gpus,
                                     max_gpu_mem,
                                     reserved_gpu_mem,
                                     create_user,
                                     create_db));
  return qr_instance_.get();
}

QueryRunner::QueryRunner(const char* db_path,
                         const std::string& user_name,
                         const std::string& passwd,
                         const std::string& db_name,
                         const std::vector<LeafHostInfo>& string_servers,
                         const std::vector<LeafHostInfo>& leaf_servers,
                         const std::string& udf_filename,
                         bool uses_gpus,
                         const size_t max_gpu_mem,
                         const int reserved_gpu_mem,
                         const bool create_user,
                         const bool create_db)
    : dispatch_queue_(std::make_unique<QueryDispatchQueue>(1)) {
  g_serialize_temp_tables = true;

  boost::filesystem::path base_path{db_path};
  CHECK(boost::filesystem::exists(base_path));
  auto system_db_file = base_path / "mapd_catalogs" / OMNISCI_DEFAULT_DB;
  CHECK(boost::filesystem::exists(system_db_file));
  auto data_dir = base_path / "mapd_data";
  DiskCacheConfig disk_cache_config{(base_path / "omnisci_disk_cache").string()};
  Catalog_Namespace::UserMetadata user;
  Catalog_Namespace::DBMetadata db;

  register_signal_handler();
  logger::set_once_fatal_func(&calcite_shutdown_handler);
  g_calcite =
      std::make_shared<Calcite>(-1, CALCITEPORT, db_path, 1024, 5000, true, udf_filename);
  ExtensionFunctionsWhitelist::add(g_calcite->getExtensionFunctionWhitelist());
  if (!udf_filename.empty()) {
    ExtensionFunctionsWhitelist::addUdfs(g_calcite->getUserDefinedFunctionWhitelist());
  }

  table_functions::TableFunctionsFactory::init();

#ifndef HAVE_CUDA
  uses_gpus = false;
#endif
  SystemParameters mapd_params;
  mapd_params.gpu_buffer_mem_bytes = max_gpu_mem;
  mapd_params.aggregator = !leaf_servers.empty();

  auto data_mgr = std::make_shared<Data_Namespace::DataMgr>(data_dir.string(),
                                                            mapd_params,
                                                            uses_gpus,
                                                            -1,
                                                            0,
                                                            reserved_gpu_mem,
                                                            0,
                                                            disk_cache_config);

  auto& sys_cat = Catalog_Namespace::SysCatalog::instance();

  sys_cat.init(base_path.string(),
               data_mgr,
               {},
               g_calcite,
               false,
               mapd_params.aggregator,
               string_servers);

  if (create_user) {
    if (!sys_cat.getMetadataForUser(user_name, user)) {
      sys_cat.createUser(user_name, passwd, false, "", true);
    }
  }
  CHECK(sys_cat.getMetadataForUser(user_name, user));
  CHECK(bcrypt_checkpw(passwd.c_str(), user.passwd_hash.c_str()) == 0);

  if (create_db) {
    if (!sys_cat.getMetadataForDB(db_name, db)) {
      sys_cat.createDatabase(db_name, user.userId);
    }
  }
  CHECK(sys_cat.getMetadataForDB(db_name, db));
  CHECK(user.isSuper || (user.userId == db.dbOwner));
  auto cat = std::make_shared<Catalog_Namespace::Catalog>(
      base_path.string(), db, data_mgr, string_servers, g_calcite, create_db);
  Catalog_Namespace::Catalog::set(cat->getCurrentDB().dbName, cat);
  session_info_ = std::make_unique<Catalog_Namespace::SessionInfo>(
      cat, user, ExecutorDeviceType::GPU, "");
}

void QueryRunner::resizeDispatchQueue(const size_t num_executors) {
  dispatch_queue_ = std::make_unique<QueryDispatchQueue>(num_executors);
}

QueryRunner::QueryRunner(std::unique_ptr<Catalog_Namespace::SessionInfo> session)
    : session_info_(std::move(session))
    , dispatch_queue_(std::make_unique<QueryDispatchQueue>(1)) {}

std::shared_ptr<Catalog_Namespace::Catalog> QueryRunner::getCatalog() const {
  CHECK(session_info_);
  return session_info_->get_catalog_ptr();
}

std::shared_ptr<Calcite> QueryRunner::getCalcite() const {
  // TODO: Embed Calcite shared_ptr ownership in QueryRunner
  return g_calcite;
}

bool QueryRunner::gpusPresent() const {
  CHECK(session_info_);
  return session_info_->getCatalog().getDataMgr().gpusPresent();
}

void QueryRunner::clearGpuMemory() const {
  CHECK(!Catalog_Namespace::SysCatalog::instance().isAggregator());
  Executor::clearMemory(Data_Namespace::MemoryLevel::GPU_LEVEL);
}

void QueryRunner::clearCpuMemory() const {
  CHECK(!Catalog_Namespace::SysCatalog::instance().isAggregator());
  Executor::clearMemory(Data_Namespace::MemoryLevel::CPU_LEVEL);
}

std::string apply_copy_to_shim(const std::string& query_str) {
  auto result = query_str;
  {
    boost::regex copy_to{R"(COPY\s*\(([^#])(.+)\)\s+TO\s)",
                         boost::regex::extended | boost::regex::icase};
    apply_shim(result, copy_to, [](std::string& result, const boost::smatch& what) {
      result.replace(
          what.position(), what.length(), "COPY (#~#" + what[1] + what[2] + "#~#) TO  ");
    });
  }
  return result;
}

QueryHint QueryRunner::getParsedQueryHintofQuery(const std::string& query_str) {
  CHECK(session_info_);
  CHECK(!Catalog_Namespace::SysCatalog::instance().isAggregator());
  auto query_state = create_query_state(session_info_, query_str);
  const auto& cat = session_info_->getCatalog();
  auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID);
  auto calcite_mgr = cat.getCalciteMgr();
  const auto query_ra = calcite_mgr
                            ->process(query_state->createQueryStateProxy(),
                                      pg_shim(query_str),
                                      {},
                                      true,
                                      false,
                                      false,
                                      true)
                            .plan_result;
  auto ra_executor = RelAlgExecutor(executor.get(), cat, query_ra);
  const auto& query_hints = ra_executor.getParsedQueryHints();
  return query_hints;
}

void QueryRunner::runDDLStatement(const std::string& stmt_str_in) {
  CHECK(session_info_);
  CHECK(!Catalog_Namespace::SysCatalog::instance().isAggregator());

  std::string stmt_str = stmt_str_in;
  // First remove special chars
  boost::algorithm::trim_left_if(stmt_str, boost::algorithm::is_any_of("\n"));
  // Then remove spaces
  boost::algorithm::trim_left(stmt_str);

  ParserWrapper pw{stmt_str};
  if (pw.is_copy_to) {
    stmt_str = apply_copy_to_shim(stmt_str_in);
  }

  auto query_state = create_query_state(session_info_, stmt_str);
  auto stdlog = STDLOG(query_state);

  SQLParser parser;
  std::list<std::unique_ptr<Parser::Stmt>> parse_trees;
  std::string last_parsed;
  CHECK_EQ(parser.parse(stmt_str, parse_trees, last_parsed), 0) << stmt_str_in;
  CHECK_EQ(parse_trees.size(), size_t(1));
  auto stmt = parse_trees.front().get();
  auto ddl = dynamic_cast<Parser::DDLStmt*>(stmt);
  CHECK(ddl);
  ddl->execute(*session_info_);
}

std::shared_ptr<ResultSet> QueryRunner::runSQL(const std::string& query_str,
                                               const ExecutorDeviceType device_type,
                                               const bool hoist_literals,
                                               const bool allow_loop_joins) {
  CHECK(session_info_);
  CHECK(!Catalog_Namespace::SysCatalog::instance().isAggregator());

  ParserWrapper pw{query_str};
  if (pw.isCalcitePathPermissable()) {
    const auto execution_result =
        runSelectQuery(query_str, device_type, hoist_literals, allow_loop_joins);
    VLOG(1) << session_info_->getCatalog().getDataMgr().getSystemMemoryUsage();
    return execution_result->getRows();
  }

  auto query_state = create_query_state(session_info_, query_str);
  auto stdlog = STDLOG(query_state);

  SQLParser parser;
  std::list<std::unique_ptr<Parser::Stmt>> parse_trees;
  std::string last_parsed;
  CHECK_EQ(parser.parse(query_str, parse_trees, last_parsed), 0) << query_str;
  CHECK_EQ(parse_trees.size(), size_t(1));
  auto stmt = parse_trees.front().get();
  auto insert_values_stmt = dynamic_cast<InsertValuesStmt*>(stmt);
  CHECK(insert_values_stmt);
  insert_values_stmt->execute(*session_info_);
  return nullptr;
}

std::shared_ptr<Executor> QueryRunner::getExecutor() const {
  CHECK(session_info_);
  CHECK(!Catalog_Namespace::SysCatalog::instance().isAggregator());
  auto query_state = create_query_state(session_info_, "");
  auto stdlog = STDLOG(query_state);
  auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID);
  return executor;
}

std::shared_ptr<ResultSet> QueryRunner::runSQLWithAllowingInterrupt(
    const std::string& query_str,
    std::shared_ptr<Executor> executor,
    const std::string& session_id,
    const ExecutorDeviceType device_type,
    const unsigned interrupt_check_freq) {
  CHECK(session_info_);
  CHECK(!Catalog_Namespace::SysCatalog::instance().isAggregator());
  auto session_info =
      std::make_shared<Catalog_Namespace::SessionInfo>(session_info_->get_catalog_ptr(),
                                                       session_info_->get_currentUser(),
                                                       ExecutorDeviceType::GPU,
                                                       session_id);
  auto query_state = create_query_state(session_info, query_str);
  auto stdlog = STDLOG(query_state);
  const auto& cat = query_state->getConstSessionInfo()->getCatalog();
  CompilationOptions co = CompilationOptions::defaults(device_type);

  ExecutionOptions eo = {g_enable_columnar_output,
                         true,
                         false,
                         true,
                         false,
                         false,
                         false,
                         false,
                         10000,
                         false,
                         false,
                         g_gpu_mem_limit_percent,
                         true,
                         interrupt_check_freq};
  std::string query_ra{""};
  {
    // async query initiation for interrupt test
    // incurs data race warning in TSAN since
    // calcite_mgr is shared across multiple query threads
    // so here we lock the manager during query parsing
    std::lock_guard<std::mutex> calcite_lock_guard(calcite_lock);
    auto calcite_mgr = cat.getCalciteMgr();
    query_ra = calcite_mgr
                   ->process(query_state->createQueryStateProxy(),
                             pg_shim(query_state->getQueryStr()),
                             {},
                             true,
                             false,
                             false,
                             true)
                   .plan_result;
  }
  auto result = RelAlgExecutor(executor.get(), cat, query_ra, query_state)
                    .executeRelAlgQuery(co, eo, false, nullptr);
  return result.getRows();
}

std::vector<std::shared_ptr<ResultSet>> QueryRunner::runMultipleStatements(
    const std::string& sql,
    const ExecutorDeviceType dt) {
  std::vector<std::shared_ptr<ResultSet>> results;
  // TODO: Need to properly handle escaped semicolons instead of doing a naive split().
  auto fields = split(sql, ";");
  for (const auto& field : fields) {
    auto text = strip(field) + ";";
    if (text == ";") {
      continue;
    }
    // TODO: Maybe remove this redundant parsing after enhancing Parser::Stmt?
    SQLParser parser;
    std::list<std::unique_ptr<Parser::Stmt>> parse_trees;
    std::string last_parsed;
    CHECK_EQ(parser.parse(text, parse_trees, last_parsed), 0);
    CHECK_EQ(parse_trees.size(), size_t(1));
    auto stmt = parse_trees.front().get();
    Parser::DDLStmt* ddl = dynamic_cast<Parser::DDLStmt*>(stmt);
    Parser::DMLStmt* dml = dynamic_cast<Parser::DMLStmt*>(stmt);
    if (ddl != nullptr && dml == nullptr) {
      runDDLStatement(text);
      results.push_back(nullptr);
    } else if (ddl == nullptr && dml != nullptr) {
      results.push_back(runSQL(text, dt, true, true));
    } else {
      throw std::runtime_error("Unexpected SQL statement type: " + text);
    }
  }
  return results;
}

void QueryRunner::runImport(Parser::CopyTableStmt* import_stmt) {
  CHECK(import_stmt);
  import_stmt->execute(*session_info_);
}

std::unique_ptr<import_export::Loader> QueryRunner::getLoader(
    const TableDescriptor* td) const {
  auto cat = getCatalog();
  return std::make_unique<import_export::Loader>(*cat, td);
}

namespace {

std::shared_ptr<ExecutionResult> run_select_query_with_filter_push_down(
    QueryStateProxy query_state_proxy,
    const ExecutorDeviceType device_type,
    const bool hoist_literals,
    const bool allow_loop_joins,
    const bool just_explain,
    const bool with_filter_push_down) {
  auto const& query_state = query_state_proxy.getQueryState();
  const auto& cat = query_state.getConstSessionInfo()->getCatalog();
  auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID);
  CompilationOptions co = CompilationOptions::defaults(device_type);
  co.opt_level = ExecutorOptLevel::LoopStrengthReduction;

  ExecutionOptions eo = {g_enable_columnar_output,
                         true,
                         just_explain,
                         allow_loop_joins,
                         false,
                         false,
                         false,
                         false,
                         10000,
                         with_filter_push_down,
                         false,
                         g_gpu_mem_limit_percent,
                         false,
                         1000};
  auto calcite_mgr = cat.getCalciteMgr();
  const auto query_ra = calcite_mgr
                            ->process(query_state_proxy,
                                      pg_shim(query_state.getQueryStr()),
                                      {},
                                      true,
                                      false,
                                      false,
                                      true)
                            .plan_result;
  auto ra_executor = RelAlgExecutor(executor.get(), cat, query_ra);
  const auto& query_hints = ra_executor.getParsedQueryHints();
  if (query_hints.cpu_mode) {
    co.device_type = ExecutorDeviceType::CPU;
  }
  auto result = std::make_shared<ExecutionResult>(
      ra_executor.executeRelAlgQuery(co, eo, false, nullptr));
  const auto& filter_push_down_requests = result->getPushedDownFilterInfo();
  if (!filter_push_down_requests.empty()) {
    std::vector<TFilterPushDownInfo> filter_push_down_info;
    for (const auto& req : filter_push_down_requests) {
      TFilterPushDownInfo filter_push_down_info_for_request;
      filter_push_down_info_for_request.input_prev = req.input_prev;
      filter_push_down_info_for_request.input_start = req.input_start;
      filter_push_down_info_for_request.input_next = req.input_next;
      filter_push_down_info.push_back(filter_push_down_info_for_request);
    }
    const auto new_query_ra = calcite_mgr
                                  ->process(query_state_proxy,
                                            pg_shim(query_state.getQueryStr()),
                                            filter_push_down_info,
                                            true,
                                            false,
                                            false,
                                            true)
                                  .plan_result;
    const ExecutionOptions eo_modified{eo.output_columnar_hint,
                                       eo.allow_multifrag,
                                       eo.just_explain,
                                       eo.allow_loop_joins,
                                       eo.with_watchdog,
                                       eo.jit_debug,
                                       eo.just_validate,
                                       eo.with_dynamic_watchdog,
                                       eo.dynamic_watchdog_time_limit,
                                       /*find_push_down_candidates=*/false,
                                       /*just_calcite_explain=*/false,
                                       eo.gpu_input_mem_limit_percent,
                                       eo.allow_runtime_query_interrupt};
    auto new_ra_executor = RelAlgExecutor(executor.get(), cat, new_query_ra);
    return std::make_shared<ExecutionResult>(
        new_ra_executor.executeRelAlgQuery(co, eo_modified, false, nullptr));
  } else {
    return result;
  }
}

}  // namespace

std::shared_ptr<ExecutionResult> QueryRunner::runSelectQuery(
    const std::string& query_str,
    const ExecutorDeviceType device_type,
    const bool hoist_literals,
    const bool allow_loop_joins,
    const bool just_explain) {
  CHECK(session_info_);
  CHECK(!Catalog_Namespace::SysCatalog::instance().isAggregator());
  auto query_state = create_query_state(session_info_, query_str);
  auto stdlog = STDLOG(query_state);
  if (g_enable_filter_push_down) {
    return run_select_query_with_filter_push_down(query_state->createQueryStateProxy(),
                                                  device_type,
                                                  hoist_literals,
                                                  allow_loop_joins,
                                                  just_explain,
                                                  g_enable_filter_push_down);
  }

  const auto& cat = session_info_->getCatalog();

  std::shared_ptr<ExecutionResult> result;
  auto query_launch_task =
      std::make_shared<QueryDispatchQueue::Task>([&cat,
                                                  &query_str,
                                                  &device_type,
                                                  &allow_loop_joins,
                                                  &just_explain,
                                                  &query_state,
                                                  &result](const size_t worker_id) {
        auto executor = Executor::getExecutor(worker_id);
        CompilationOptions co = CompilationOptions::defaults(device_type);
        co.opt_level = ExecutorOptLevel::LoopStrengthReduction;

        ExecutionOptions eo = {g_enable_columnar_output,
                               true,
                               just_explain,
                               allow_loop_joins,
                               false,
                               false,
                               false,
                               false,
                               10000,
                               false,
                               false,
                               g_gpu_mem_limit_percent,
                               false,
                               1000};
        auto calcite_mgr = cat.getCalciteMgr();
        const auto query_ra = calcite_mgr
                                  ->process(query_state->createQueryStateProxy(),
                                            pg_shim(query_str),
                                            {},
                                            true,
                                            false,
                                            g_enable_calcite_view_optimize,
                                            true)
                                  .plan_result;
        auto ra_executor = RelAlgExecutor(executor.get(), cat, query_ra);
        const auto& query_hints = ra_executor.getParsedQueryHints();
        if (query_hints.cpu_mode) {
          co.device_type = ExecutorDeviceType::CPU;
        }
        result = std::make_shared<ExecutionResult>(
            ra_executor.executeRelAlgQuery(co, eo, false, nullptr));
      });
  CHECK(dispatch_queue_);
  dispatch_queue_->submit(query_launch_task);
  auto result_future = query_launch_task->get_future();
  result_future.get();
  CHECK(result);
  return result;
}

const std::shared_ptr<std::vector<int32_t>>& QueryRunner::getCachedJoinHashTable(
    size_t idx) {
  return JoinHashTable::getCachedHashTable(idx);
};

const std::shared_ptr<std::vector<int8_t>>& QueryRunner::getCachedBaselineHashTable(
    size_t idx) {
  return BaselineJoinHashTable::getCachedHashTable(idx);
};

size_t QueryRunner::getEntryCntCachedBaselineHashTable(size_t idx) {
  return BaselineJoinHashTable::getEntryCntCachedHashTable(idx);
}

uint64_t QueryRunner::getNumberOfCachedJoinHashTables() {
  return JoinHashTable::getNumberOfCachedHashTables();
};

uint64_t QueryRunner::getNumberOfCachedBaselineJoinHashTables() {
  return BaselineJoinHashTable::getNumberOfCachedHashTables();
};

void QueryRunner::reset() {
  qr_instance_.reset(nullptr);
  calcite_shutdown_handler();
}

ImportDriver::ImportDriver(std::shared_ptr<Catalog_Namespace::Catalog> cat,
                           const Catalog_Namespace::UserMetadata& user,
                           const ExecutorDeviceType dt)
    : QueryRunner(std::make_unique<Catalog_Namespace::SessionInfo>(cat, user, dt, "")) {}

void ImportDriver::importGeoTable(const std::string& file_path,
                                  const std::string& table_name,
                                  const bool compression,
                                  const bool create_table,
                                  const bool explode_collections) {
  using namespace import_export;

  CHECK(session_info_);
  const std::string geo_column_name(OMNISCI_GEO_PREFIX);

  CopyParams copy_params;
  if (compression) {
    copy_params.geo_coords_encoding = EncodingType::kENCODING_GEOINT;
    copy_params.geo_coords_comp_param = 32;
  } else {
    copy_params.geo_coords_encoding = EncodingType::kENCODING_NONE;
    copy_params.geo_coords_comp_param = 0;
  }
  copy_params.geo_assign_render_groups = true;
  copy_params.geo_explode_collections = explode_collections;

  auto cds = Importer::gdalToColumnDescriptors(file_path, geo_column_name, copy_params);
  std::map<std::string, std::string> colname_to_src;
  for (auto& cd : cds) {
    const auto col_name_sanitized = ImportHelpers::sanitize_name(cd.columnName);
    const auto ret =
        colname_to_src.insert(std::make_pair(col_name_sanitized, cd.columnName));
    CHECK(ret.second);
    cd.columnName = col_name_sanitized;
  }

  auto& cat = session_info_->getCatalog();

  if (create_table) {
    const auto td = cat.getMetadataForTable(table_name);
    if (td != nullptr) {
      throw std::runtime_error("Error: Table " + table_name +
                               " already exists. Possible failure to correctly re-create "
                               "mapd_data directory.");
    }
    if (table_name != ImportHelpers::sanitize_name(table_name)) {
      throw std::runtime_error("Invalid characters in table name: " + table_name);
    }

    std::string stmt{"CREATE TABLE " + table_name};
    std::vector<std::string> col_stmts;

    for (auto& cd : cds) {
      if (cd.columnType.get_type() == SQLTypes::kINTERVAL_DAY_TIME ||
          cd.columnType.get_type() == SQLTypes::kINTERVAL_YEAR_MONTH) {
        throw std::runtime_error(
            "Unsupported type: INTERVAL_DAY_TIME or INTERVAL_YEAR_MONTH for col " +
            cd.columnName + " (table: " + table_name + ")");
      }

      if (cd.columnType.get_type() == SQLTypes::kDECIMAL) {
        if (cd.columnType.get_precision() == 0 && cd.columnType.get_scale() == 0) {
          cd.columnType.set_precision(14);
          cd.columnType.set_scale(7);
        }
      }

      std::string col_stmt;
      col_stmt.append(cd.columnName + " " + cd.columnType.get_type_name() + " ");

      if (cd.columnType.get_compression() != EncodingType::kENCODING_NONE) {
        col_stmt.append("ENCODING " + cd.columnType.get_compression_name() + " ");
      } else {
        if (cd.columnType.is_string()) {
          col_stmt.append("ENCODING NONE");
        } else if (cd.columnType.is_geometry()) {
          if (cd.columnType.get_output_srid() == 4326) {
            col_stmt.append("ENCODING NONE");
          }
        }
      }
      col_stmts.push_back(col_stmt);
    }

    stmt.append(" (" + boost::algorithm::join(col_stmts, ",") + ");");
    runDDLStatement(stmt);

    LOG(INFO) << "Created table: " << table_name;
  } else {
    LOG(INFO) << "Not creating table: " << table_name;
  }

  const auto td = cat.getMetadataForTable(table_name);
  if (td == nullptr) {
    throw std::runtime_error("Error: Failed to create table " + table_name);
  }

  import_export::Importer importer(cat, td, file_path, copy_params);
  auto ms = measure<>::execution([&]() { importer.importGDAL(colname_to_src); });
  LOG(INFO) << "Import Time for " << table_name << ": " << (double)ms / 1000.0 << " s";
}

}  // namespace QueryRunner
