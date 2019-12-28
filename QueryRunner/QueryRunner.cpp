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
#include "Parser/ParserWrapper.h"
#include "Parser/parser.h"
#include "QueryEngine/CalciteAdapter.h"
#include "QueryEngine/ExtensionFunctionsWhitelist.h"
#include "QueryEngine/RelAlgExecutor.h"
#include "QueryEngine/TableFunctions/TableFunctionsFactory.h"
#include "Shared/ConfigResolve.h"
#include "Shared/Logger.h"
#include "Shared/MapDParameters.h"
#include "Shared/StringTransform.h"
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

using namespace Catalog_Namespace;
namespace {

std::shared_ptr<Calcite> g_calcite = nullptr;

void calcite_shutdown_handler() {
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
                         const bool create_db) {
  g_serialize_temp_tables = true;

  boost::filesystem::path base_path{db_path};
  CHECK(boost::filesystem::exists(base_path));
  auto system_db_file = base_path / "mapd_catalogs" / OMNISCI_DEFAULT_DB;
  CHECK(boost::filesystem::exists(system_db_file));
  auto data_dir = base_path / "mapd_data";
  Catalog_Namespace::UserMetadata user;
  Catalog_Namespace::DBMetadata db;

  register_signal_handler();
  logger::set_once_fatal_func(&calcite_shutdown_handler);
  g_calcite = std::make_shared<Calcite>(-1, CALCITEPORT, db_path, 1024, udf_filename);
  ExtensionFunctionsWhitelist::add(g_calcite->getExtensionFunctionWhitelist());
  if (!udf_filename.empty()) {
    ExtensionFunctionsWhitelist::addUdfs(g_calcite->getUserDefinedFunctionWhitelist());
  }

  table_functions::TableFunctionsFactory::init();

  if (std::is_same<CudaBuildSelector, PreprocessorFalse>::value) {
    uses_gpus = false;
  }
  MapDParameters mapd_params;
  mapd_params.gpu_buffer_mem_bytes = max_gpu_mem;
  mapd_params.aggregator = !leaf_servers.empty();

  auto data_mgr = std::make_shared<Data_Namespace::DataMgr>(
      data_dir.string(), mapd_params, uses_gpus, -1, 0, reserved_gpu_mem);

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

QueryRunner::QueryRunner(std::unique_ptr<Catalog_Namespace::SessionInfo> session)
    : session_info_(std::move(session)) {}

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

void QueryRunner::runDDLStatement(const std::string& stmt_str_in) {
  CHECK(session_info_);
  CHECK(!Catalog_Namespace::SysCatalog::instance().isAggregator());

  auto stmt_str = stmt_str_in;
  // First remove special chars
  boost::algorithm::trim_left_if(stmt_str, boost::algorithm::is_any_of("\n"));
  // Then remove spaces
  boost::algorithm::trim_left(stmt_str);

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
    if (ir_file_writer_ && (pw.getDMLType() == ParserWrapper::DMLType::NotDML)) {
      try {
        const auto result = runSelectQuery(
            query_str, device_type, hoist_literals, allow_loop_joins, true);
        const auto crt_row = result.getRows()->getNextRow(true, true);
        CHECK_EQ(size_t(1), crt_row.size());
        const auto scalar_ir = boost::get<ScalarTargetValue>(&crt_row[0]);
        CHECK(scalar_ir);
        const auto ir_ns = boost::get<NullableString>(scalar_ir);
        CHECK(ir_ns);
        const auto ir_str = boost::get<std::string>(ir_ns);
        CHECK(ir_str);
        (*ir_file_writer_)(query_str, *ir_str);
      } catch (const std::exception& e) {
        LOG(WARNING) << "Failed to run EXPLAIN on SELECT query: " << query_str << " ("
                     << e.what() << "). Proceeding with query execution.";
      }
    }
    const auto execution_result =
        runSelectQuery(query_str, device_type, hoist_literals, allow_loop_joins);

    return execution_result.getRows();
  }

  auto query_state = create_query_state(session_info_, query_str);
  auto stdlog = STDLOG(query_state);

  const auto& cat = session_info_->getCatalog();
  auto executor = Executor::getExecutor(cat.getCurrentDB().dbId);

  auto plan =
      std::unique_ptr<Planner::RootPlan>(parsePlan(query_state->createQueryStateProxy()));

#ifdef HAVE_CUDA
  return executor->execute(plan.get(),
                           *session_info_,
                           hoist_literals,
                           device_type,
                           ExecutorOptLevel::LoopStrengthReduction,
                           true,
                           allow_loop_joins);
#else
  return executor->execute(plan.get(),
                           *session_info_,
                           hoist_literals,
                           device_type,
                           ExecutorOptLevel::LoopStrengthReduction,
                           false,
                           allow_loop_joins);
#endif
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

std::unique_ptr<Importer_NS::Loader> QueryRunner::getLoader(
    const TableDescriptor* td) const {
  auto cat = getCatalog();
  return std::make_unique<Importer_NS::Loader>(*cat, td);
}

namespace {

ExecutionResult run_select_query_with_filter_push_down(
    QueryStateProxy query_state_proxy,
    const ExecutorDeviceType device_type,
    const bool hoist_literals,
    const bool allow_loop_joins,
    const bool just_explain,
    const bool with_filter_push_down) {
  auto const& query_state = query_state_proxy.getQueryState();
  const auto& cat = query_state.getConstSessionInfo()->getCatalog();
  auto executor = Executor::getExecutor(cat.getCurrentDB().dbId);
  CompilationOptions co = {
      device_type, true, ExecutorOptLevel::LoopStrengthReduction, false};
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
                         g_gpu_mem_limit_percent};
  auto calcite_mgr = cat.getCalciteMgr();
  const auto query_ra = calcite_mgr
                            ->process(query_state_proxy,
                                      pg_shim(query_state.getQueryStr()),
                                      {},
                                      true,
                                      false,
                                      false)
                            .plan_result;
  RelAlgExecutor ra_executor(executor.get(), cat);

  auto result = ra_executor.executeRelAlgQuery(query_ra, co, eo, nullptr);
  const auto& filter_push_down_requests = result.getPushedDownFilterInfo();
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
                                            false)
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
                                       eo.gpu_input_mem_limit_percent};
    return ra_executor.executeRelAlgQuery(new_query_ra, co, eo_modified, nullptr);
  } else {
    return result;
  }
}

}  // namespace

ExecutionResult QueryRunner::runSelectQuery(const std::string& query_str,
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
  auto executor = Executor::getExecutor(cat.getCurrentDB().dbId);
  CompilationOptions co = {
      device_type, true, ExecutorOptLevel::LoopStrengthReduction, false};
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
                         g_gpu_mem_limit_percent};
  auto calcite_mgr = cat.getCalciteMgr();
  const auto query_ra = calcite_mgr
                            ->process(query_state->createQueryStateProxy(),
                                      pg_shim(query_str),
                                      {},
                                      true,
                                      false,
                                      false)
                            .plan_result;
  RelAlgExecutor ra_executor(executor.get(), cat);
  return ra_executor.executeRelAlgQuery(query_ra, co, eo, nullptr);
}

void QueryRunner::reset() {
  qr_instance_.reset(nullptr);
  calcite_shutdown_handler();
}

}  // namespace QueryRunner
