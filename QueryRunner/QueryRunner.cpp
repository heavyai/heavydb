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
#include "Parser/ParserWrapper.h"
#include "Parser/parser.h"
#include "QueryEngine/CalciteAdapter.h"
#include "Shared/ConfigResolve.h"
#include "Shared/MapDParameters.h"
#include "Shared/StringTransform.h"
#include "bcrypt.h"
#include "gen-cpp/CalciteServer.h"

#include "QueryEngine/ExtensionFunctionsWhitelist.h"
#include "QueryEngine/RelAlgExecutor.h"

#include "DistributedLoader.h"

#include <glog/logging.h>
#include <boost/filesystem/operations.hpp>
#include <csignal>
#include <random>

#define CALCITEPORT 36279

extern size_t g_leaf_count;
extern bool g_enable_filter_push_down;

double g_gpu_mem_limit_percent{0.9};
namespace {

Planner::RootPlan* parse_plan_legacy(
    const std::string& query_str,
    const std::unique_ptr<Catalog_Namespace::SessionInfo>& session) {
  const auto& cat = session->getCatalog();
  SQLParser parser;
  std::list<std::unique_ptr<Parser::Stmt>> parse_trees;
  std::string last_parsed;
  if (parser.parse(query_str, parse_trees, last_parsed)) {
    throw std::runtime_error("Failed to parse query");
  }
  CHECK_EQ(parse_trees.size(), size_t(1));
  const auto& stmt = parse_trees.front();
  Parser::DDLStmt* ddl = dynamic_cast<Parser::DDLStmt*>(stmt.get());
  CHECK(!ddl);
  Parser::DMLStmt* dml = dynamic_cast<Parser::DMLStmt*>(stmt.get());
  Analyzer::Query query;
  dml->analyze(cat, query);
  Planner::Optimizer optimizer(query, cat);
  return optimizer.optimize();
}

Planner::RootPlan* parse_plan_calcite(
    const std::string& query_str,
    const std::unique_ptr<Catalog_Namespace::SessionInfo>& session) {
  ParserWrapper pw{query_str};
  if (pw.isOtherExplain() || pw.is_ddl || pw.is_update_dml) {
    return parse_plan_legacy(query_str, session);
  }

  const auto& cat = session->getCatalog();
  auto calcite_mgr = cat.getCalciteMgr();
  const Catalog_Namespace::SessionInfo* sess = session.get();
  const auto query_ra =
      calcite_mgr->process(*sess,
                           pg_shim(query_str),
                           {},
                           true,
                           false,
                           false)
          .plan_result;  //  if we want to be able to check plans we may want to calc this
  return translate_query(query_ra, cat);
}

Planner::RootPlan* parse_plan(
    const std::string& query_str,
    const std::unique_ptr<Catalog_Namespace::SessionInfo>& session) {
  Planner::RootPlan* plan = parse_plan_calcite(query_str, session);
  return plan;
}

}  // namespace

namespace QueryRunner {

LeafAggregator* leaf_aggregator = nullptr;

LeafAggregator* get_leaf_aggregator() {
  return leaf_aggregator;
}

std::shared_ptr<Calcite> g_calcite = nullptr;

void calcite_shutdown_handler() {
  if (g_calcite) {
    g_calcite->close_calcite_server();
  }
}

void mapd_signal_handler(int signal_number) {
  LOG(ERROR) << "Interrupt signal (" << signal_number << ") received.";
  calcite_shutdown_handler();
  // shut down logging force a flush
  google::ShutdownGoogleLogging();
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

Catalog_Namespace::SessionInfo* get_session(
    const char* db_path,
    const std::string& user_name,
    const std::string& passwd,
    const std::string& db_name,
    const std::vector<LeafHostInfo>& string_servers,
    const std::vector<LeafHostInfo>& leaf_servers,
    bool uses_gpus,
    const bool create_user,
    const bool create_db) {
  boost::filesystem::path base_path{db_path};
  CHECK(boost::filesystem::exists(base_path));
  auto system_db_file = base_path / "mapd_catalogs" / OMNISCI_DEFAULT_DB;
  CHECK(boost::filesystem::exists(system_db_file));
  auto data_dir = base_path / "mapd_data";
  Catalog_Namespace::UserMetadata user;
  Catalog_Namespace::DBMetadata db;

  g_leaf_count = leaf_servers.size();
  register_signal_handler();
  google::InstallFailureFunction(&calcite_shutdown_handler);

  g_calcite = std::make_shared<Calcite>(-1, CALCITEPORT, db_path, 1024);

  ExtensionFunctionsWhitelist::add(g_calcite->getExtensionFunctionWhitelist());

  if (std::is_same<CudaBuildSelector, PreprocessorFalse>::value) {
    uses_gpus = false;
  }
  MapDParameters mapd_parms;
  mapd_parms.aggregator = !leaf_servers.empty();

  const auto reserved_gpu_mem = 256 << 20;
  auto dataMgr = std::make_shared<Data_Namespace::DataMgr>(
      data_dir.string(), mapd_parms, uses_gpus, -1, 0, reserved_gpu_mem);

  auto& sys_cat = Catalog_Namespace::SysCatalog::instance();

  sys_cat.init(base_path.string(),
               dataMgr,
               {},
               g_calcite,
               false,
               mapd_parms.aggregator,
               string_servers);

  if (create_user) {
    if (!sys_cat.getMetadataForUser(user_name, user)) {
      sys_cat.createUser(user_name, passwd, false, "");
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
      base_path.string(), db, dataMgr, string_servers, g_calcite, create_db);
  Catalog_Namespace::Catalog::set(cat->getCurrentDB().dbName, cat);
  Catalog_Namespace::SessionInfo* session =
      new Catalog_Namespace::SessionInfo(cat, user, ExecutorDeviceType::GPU, "");

  return session;
}

Catalog_Namespace::SessionInfo* get_session(
    const char* db_path,
    const std::vector<LeafHostInfo>& string_servers,
    const std::vector<LeafHostInfo>& leaf_servers) {
  std::string db_name{OMNISCI_DEFAULT_DB};
  std::string user_name{OMNISCI_ROOT_USER};
  std::string passwd{"HyperInteractive"};

  return get_session(db_path, user_name, passwd, db_name, string_servers, leaf_servers);
}

Catalog_Namespace::SessionInfo* get_session(const char* db_path) {
  return get_session(db_path, std::vector<LeafHostInfo>{}, std::vector<LeafHostInfo>{});
}

Catalog_Namespace::UserMetadata get_user_metadata(
    const Catalog_Namespace::SessionInfo* session) {
  return session->get_currentUser();
}

std::shared_ptr<Catalog_Namespace::Catalog> get_catalog(
    const Catalog_Namespace::SessionInfo* session) {
  return session->get_catalog_ptr();
}

ExecutionResult run_select_query(
    const std::string& query_str,
    const std::unique_ptr<Catalog_Namespace::SessionInfo>& session,
    const ExecutorDeviceType device_type,
    const bool hoist_literals,
    const bool allow_loop_joins,
    const bool just_explain) {
  CHECK(!Catalog_Namespace::SysCatalog::instance().isAggregator());
  if (g_enable_filter_push_down) {
    return run_select_query_with_filter_push_down(query_str,
                                                  session,
                                                  device_type,
                                                  hoist_literals,
                                                  allow_loop_joins,
                                                  just_explain,
                                                  g_enable_filter_push_down);
  }

  const auto& cat = session->getCatalog();
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
  const auto query_ra =
      calcite_mgr->process(*session, pg_shim(query_str), {}, true, false, false)
          .plan_result;
  RelAlgExecutor ra_executor(executor.get(), cat);
  return ra_executor.executeRelAlgQuery(query_ra, co, eo, nullptr);
}

ExecutionResult run_select_query_with_filter_push_down(
    const std::string& query_str,
    const std::unique_ptr<Catalog_Namespace::SessionInfo>& session,
    const ExecutorDeviceType device_type,
    const bool hoist_literals,
    const bool allow_loop_joins,
    const bool just_explain,
    const bool with_filter_push_down) {
  const auto& cat = session->getCatalog();
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
  const auto query_ra =
      calcite_mgr->process(*session, pg_shim(query_str), {}, true, false, false)
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
    const auto new_query_ra =
        calcite_mgr
            ->process(
                *session, pg_shim(query_str), filter_push_down_info, true, false, false)
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

TExecuteMode::type to_execute_mode(ExecutorDeviceType device_type) {
  switch (device_type) {
    case ExecutorDeviceType::GPU:
      return TExecuteMode::type::GPU;
    case ExecutorDeviceType::CPU:
      return TExecuteMode::type::CPU;
  }

  CHECK(false);
  return TExecuteMode::type::GPU;
}

void run_copy_distributed(
    Parser::CopyTableStmt* import_stmt,
    const std::unique_ptr<Catalog_Namespace::SessionInfo>& session) {
  auto leaf_aggregator = get_leaf_aggregator();
  auto importer = [&session, &leaf_aggregator](
                      const Catalog_Namespace::Catalog& cat,
                      const TableDescriptor* td,
                      const std::string& file_path,
                      const Importer_NS::CopyParams& copy_params) {
    return std::make_unique<Importer_NS::Importer>(
        new DistributedLoader(*session.get(), td, leaf_aggregator),
        file_path,
        copy_params);
  };
  import_stmt->execute(*session, importer);
  return;
}

std::shared_ptr<ResultSet> run_sql_distributed(
    const std::string& query_str,
    const std::unique_ptr<Catalog_Namespace::SessionInfo>& session,
    const ExecutorDeviceType device_type,
    bool allow_loop_joins) {
  return nullptr;
}

std::shared_ptr<ResultSet> run_multiple_agg(
    const std::string& query_str,
    const std::unique_ptr<Catalog_Namespace::SessionInfo>& session,
    const ExecutorDeviceType device_type,
    const bool hoist_literals,
    const bool allow_loop_joins,
    const std::unique_ptr<IROutputFile>& ir_output_file) {
  if (Catalog_Namespace::SysCatalog::instance().isAggregator()) {
    return run_sql_distributed(query_str, session, device_type, allow_loop_joins);
  }

  ParserWrapper pw{query_str};
  if (pw.isCalcitePathPermissable()) {
    if (ir_output_file && (pw.getDMLType() == ParserWrapper::DMLType::NotDML)) {
      try {
        const auto result = run_select_query(
            query_str, session, device_type, hoist_literals, allow_loop_joins, true);
        const auto crt_row = result.getRows()->getNextRow(true, true);
        CHECK_EQ(size_t(1), crt_row.size());
        const auto scalar_ir = boost::get<ScalarTargetValue>(&crt_row[0]);
        CHECK(scalar_ir);
        const auto ir_ns = boost::get<NullableString>(scalar_ir);
        CHECK(ir_ns);
        const auto ir_str = boost::get<std::string>(ir_ns);
        CHECK(ir_str);
        (*ir_output_file)(query_str, *ir_str);
      } catch (const std::exception& e) {
        LOG(INFO) << "Failed to run EXPLAIN on SELECT query: " << query_str << " ("
                  << e.what() << ")";
      }
    }
    const auto execution_result = run_select_query(
        query_str, session, device_type, hoist_literals, allow_loop_joins);

    return execution_result.getRows();
  }

  const auto& cat = session->getCatalog();
  auto executor = Executor::getExecutor(cat.getCurrentDB().dbId);

  auto plan = std::unique_ptr<Planner::RootPlan>(parse_plan(query_str, session));

#ifdef HAVE_CUDA
  return executor->execute(plan.get(),
                           *session,
                           hoist_literals,
                           device_type,
                           ExecutorOptLevel::LoopStrengthReduction,
                           true,
                           allow_loop_joins);
#else
  return executor->execute(plan.get(),
                           *session,
                           hoist_literals,
                           device_type,
                           ExecutorOptLevel::LoopStrengthReduction,
                           false,
                           allow_loop_joins);
#endif
}

void run_ddl_statement(const std::string& stmt_str_in,
                       const std::unique_ptr<Catalog_Namespace::SessionInfo>& session) {
  auto stmt_str = stmt_str_in;
  // First remove special chars
  boost::algorithm::trim_left_if(stmt_str, boost::algorithm::is_any_of("\n"));
  // Then remove spaces
  boost::algorithm::trim_left(stmt_str);
  if (Catalog_Namespace::SysCatalog::instance().isAggregator()) {
    run_sql_distributed(stmt_str, session, ExecutorDeviceType::CPU, false);
    return;
  }

  SQLParser parser;
  std::list<std::unique_ptr<Parser::Stmt>> parse_trees;
  std::string last_parsed;
  CHECK_EQ(parser.parse(stmt_str, parse_trees, last_parsed), 0);
  CHECK_EQ(parse_trees.size(), size_t(1));
  auto stmt = parse_trees.front().get();
  Parser::DDLStmt* ddl = dynamic_cast<Parser::DDLStmt*>(stmt);
  CHECK(ddl);
  if (ddl != nullptr) {
    ddl->execute(*session);
  }
}

namespace {

void clear_cpu_memory_distributed(
    const std::unique_ptr<Catalog_Namespace::SessionInfo>& session) {}

void clear_gpu_memory_distributed(
    const std::unique_ptr<Catalog_Namespace::SessionInfo>& session) {}

}  // namespace

void clear_gpu_memory(const std::unique_ptr<Catalog_Namespace::SessionInfo>& session) {
  if (Catalog_Namespace::SysCatalog::instance().isAggregator()) {
    clear_gpu_memory_distributed(session);
    return;
  }
  Catalog_Namespace::SysCatalog::clearGpuMemory();
}

void clear_cpu_memory(const std::unique_ptr<Catalog_Namespace::SessionInfo>& session) {
  if (Catalog_Namespace::SysCatalog::instance().isAggregator()) {
    clear_cpu_memory_distributed(session);
    return;
  }
  Catalog_Namespace::SysCatalog::clearCpuMemory();
}

void run_sql_statements(const std::string& sql,
                        const std::unique_ptr<Catalog_Namespace::SessionInfo>& session,
                        const ExecutorDeviceType device_type) {
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
      run_ddl_statement(text, session);
      // NOTE: The device_type is not needed in this branch because DDL
      // statements only affect the catalog which runs on CPU only.
    } else if (ddl == nullptr && dml != nullptr) {
      run_multiple_agg(text, session, device_type, true, true, nullptr);
    } else {
      throw std::runtime_error(std::string("unexpected SQL statement type: ") + text);
    }
  }
  // TODO: Maybe should return the result set from zero or more selects here?
}

}  // namespace QueryRunner
