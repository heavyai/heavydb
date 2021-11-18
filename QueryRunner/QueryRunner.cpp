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
#include "Catalog/DdlCommandExecutor.h"
#include "DataMgr/DataMgrBufferProvider.h"
#include "DataMgr/DataMgrDataProvider.h"
#include "DistributedLoader.h"
#include "ImportExport/CopyParams.h"
#include "Logger/Logger.h"
#include "Parser/ParserWrapper.h"
#include "Parser/parser.h"
#include "QueryEngine/CalciteAdapter.h"
#include "QueryEngine/DataRecycler/HashtableRecycler.h"
#include "QueryEngine/ExtensionFunctionsWhitelist.h"
#include "QueryEngine/JoinHashTable/PerfectJoinHashTable.h"
#include "QueryEngine/QueryDispatchQueue.h"
#include "QueryEngine/QueryPlanDagExtractor.h"
#include "QueryEngine/RelAlgExecutor.h"
#include "QueryEngine/TableFunctions/TableFunctionsFactory.h"
#include "QueryEngine/ThriftSerializers.h"
#include "Shared/Globals.h"
#include "Shared/StringTransform.h"
#include "Shared/SystemParameters.h"
#include "Shared/import_helpers.h"
#include "TestProcessSignalHandler.h"
#include "gen-cpp/CalciteServer.h"
#include "include/bcrypt.h"

#include <boost/filesystem/operations.hpp>
#include <csignal>
#include <random>

#define CALCITEPORT 3279

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

void setup_signal_handler() {
  TestProcessSignalHandler::registerSignalHandler();
  TestProcessSignalHandler::addShutdownCallback(calcite_shutdown_handler);
}

}  // namespace

namespace QueryRunner {

std::unique_ptr<QueryRunner> QueryRunner::qr_instance_ = nullptr;

query_state::QueryStates QueryRunner::query_states_;

QueryRunner* QueryRunner::init(const char* db_path,
                               const std::string& udf_filename,
                               const size_t max_gpu_mem,
                               const int reserved_gpu_mem) {
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

QueryRunner* QueryRunner::init(const File_Namespace::DiskCacheConfig* disk_cache_config,
                               const char* db_path,
                               const std::vector<LeafHostInfo>& string_servers,
                               const std::vector<LeafHostInfo>& leaf_servers) {
  return QueryRunner::init(db_path,
                           std::string{OMNISCI_ROOT_USER},
                           "HyperInteractive",
                           std::string{OMNISCI_DEFAULT_DB},
                           string_servers,
                           leaf_servers,
                           "",
                           true,
                           0,
                           256 << 20,
                           false,
                           false,
                           disk_cache_config);
}

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
                               const bool create_db,
                               const File_Namespace::DiskCacheConfig* disk_cache_config) {
  // Whitelist root path for tests by default
  ddl_utils::FilePathWhitelist::clear();
  ddl_utils::FilePathWhitelist::initialize(db_path, "[\"/\"]", "[\"/\"]");
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
                                     create_db,
                                     disk_cache_config));
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
                         const bool create_db,
                         const File_Namespace::DiskCacheConfig* cache_config)
    : dispatch_queue_(std::make_unique<QueryDispatchQueue>(1)) {
  g_serialize_temp_tables = true;
  boost::filesystem::path base_path{db_path};
  CHECK(boost::filesystem::exists(base_path));
  auto system_db_file = base_path / "mapd_catalogs" / OMNISCI_DEFAULT_DB;
  CHECK(boost::filesystem::exists(system_db_file));
  auto data_dir = base_path / "mapd_data";
  File_Namespace::DiskCacheConfig disk_cache_config{
      (base_path / "omnisci_disk_cache").string(), File_Namespace::DiskCacheLevel::fsi};
  if (cache_config) {
    disk_cache_config = *cache_config;
  }
  Catalog_Namespace::UserMetadata user;
  Catalog_Namespace::DBMetadata db;

  setup_signal_handler();
  logger::set_once_fatal_func(&calcite_shutdown_handler);
  g_calcite =
      std::make_shared<Calcite>(-1, CALCITEPORT, db_path, 1024, 5000, true, udf_filename);
  ExtensionFunctionsWhitelist::add(g_calcite->getExtensionFunctionWhitelist());
  if (!udf_filename.empty()) {
    ExtensionFunctionsWhitelist::addUdfs(g_calcite->getUserDefinedFunctionWhitelist());
  }

  table_functions::TableFunctionsFactory::init();
  auto udtfs = ThriftSerializers::to_thrift(
      table_functions::TableFunctionsFactory::get_table_funcs(/*is_runtime=*/false));
  std::vector<TUserDefinedFunction> udfs = {};
  g_calcite->setRuntimeExtensionFunctions(udfs, udtfs, /*is_runtime=*/false);

  std::unique_ptr<CudaMgr_Namespace::CudaMgr> cuda_mgr;
#ifdef HAVE_CUDA
  if (uses_gpus) {
    cuda_mgr = std::make_unique<CudaMgr_Namespace::CudaMgr>(-1, 0);
  }
#else
  uses_gpus = false;
#endif
  SystemParameters mapd_params;
  mapd_params.gpu_buffer_mem_bytes = max_gpu_mem;
  mapd_params.aggregator = !leaf_servers.empty();

  data_mgr_.reset(new Data_Namespace::DataMgr(data_dir.string(),
                                              mapd_params,
                                              std::move(cuda_mgr),
                                              uses_gpus,
                                              reserved_gpu_mem,
                                              0,
                                              disk_cache_config));

  auto& sys_cat = Catalog_Namespace::SysCatalog::instance();

  g_base_path = base_path.string();
  sys_cat.init(g_base_path,
               data_mgr_,
               {},
               g_calcite,
               false,
               mapd_params.aggregator,
               string_servers);

  if (create_user) {
    if (!sys_cat.getMetadataForUser(user_name, user)) {
      sys_cat.createUser(user_name, passwd, false, "", true, g_read_only);
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
  auto cat = sys_cat.getCatalog(db, create_db);
  CHECK(cat);
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

std::shared_ptr<Data_Namespace::DataMgr> QueryRunner::getDataMgr() const {
  return data_mgr_;
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
  Executor::clearMemory(Data_Namespace::MemoryLevel::GPU_LEVEL,
                        &Catalog_Namespace::SysCatalog::instance().getDataMgr());
}

void QueryRunner::clearCpuMemory() const {
  CHECK(!Catalog_Namespace::SysCatalog::instance().isAggregator());
  Executor::clearMemory(Data_Namespace::MemoryLevel::CPU_LEVEL,
                        &Catalog_Namespace::SysCatalog::instance().getDataMgr());
}

std::vector<MemoryInfo> QueryRunner::getMemoryInfo(
    const Data_Namespace::MemoryLevel memory_level) const {
  CHECK(!Catalog_Namespace::SysCatalog::instance().isAggregator());
  return session_info_->getCatalog().getDataMgr().getMemoryInfo(memory_level);
}

RegisteredQueryHint QueryRunner::getParsedQueryHint(const std::string& query_str) {
  CHECK(session_info_);
  CHECK(!Catalog_Namespace::SysCatalog::instance().isAggregator());
  auto query_state = create_query_state(session_info_, query_str);
  const auto& cat = session_info_->getCatalog();
  auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID,
                                        &cat.getDataMgr(),
                                        cat.getDataMgr().getBufferProvider());

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
  auto ra_executor = RelAlgExecutor(
      executor.get(), &cat, cat.getDataMgr().getDataProvider(), query_ra, query_state);
  auto query_hints =
      ra_executor.getParsedQueryHint(ra_executor.getRootRelAlgNodeShPtr().get());
  return query_hints ? *query_hints : RegisteredQueryHint::defaults();
}

std::optional<std::unordered_map<size_t, RegisteredQueryHint>>
QueryRunner::getParsedQueryHints(const std::string& query_str) {
  CHECK(session_info_);
  CHECK(!Catalog_Namespace::SysCatalog::instance().isAggregator());
  auto query_state = create_query_state(session_info_, query_str);
  const auto& cat = session_info_->getCatalog();
  auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID,
                                        &cat.getDataMgr(),
                                        cat.getDataMgr().getBufferProvider());

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
  auto ra_executor = RelAlgExecutor(
      executor.get(), &cat, cat.getDataMgr().getDataProvider(), query_ra, query_state);
  auto query_hints = ra_executor.getParsedQueryHints();
  return query_hints ? query_hints : std::nullopt;
}

// used to validate calcite ddl statements
void QueryRunner::validateDDLStatement(const std::string& stmt_str_in) {
  CHECK(session_info_);

  std::string stmt_str = stmt_str_in;
  // First remove special chars
  boost::algorithm::trim_left_if(stmt_str, boost::algorithm::is_any_of("\n"));
  // Then remove spaces
  boost::algorithm::trim_left(stmt_str);

  auto query_state = create_query_state(session_info_, stmt_str);
  auto stdlog = STDLOG(query_state);

  const auto& cat = session_info_->getCatalog();
  auto calcite_mgr = cat.getCalciteMgr();
  calcite_mgr->process(query_state->createQueryStateProxy(),
                       pg_shim(stmt_str),
                       {},
                       true,
                       false,
                       false,
                       true);
}

std::shared_ptr<RelAlgTranslator> QueryRunner::getRelAlgTranslator(
    const std::string& query_str,
    Executor* executor) {
  CHECK(session_info_);
  CHECK(!Catalog_Namespace::SysCatalog::instance().isAggregator());
  auto query_state = create_query_state(session_info_, query_str);
  const auto& cat = session_info_->getCatalog();
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
  executor->setSchemaProvider(
      std::make_shared<Catalog_Namespace::CatalogSchemaProvider>(&cat));
  executor->setDatabaseId(cat.getDatabaseId());
  auto ra_executor =
      RelAlgExecutor(executor, &cat, cat.getDataMgr().getDataProvider(), query_ra);
  auto root_node_shared_ptr = ra_executor.getRootRelAlgNodeShPtr();
  return ra_executor.getRelAlgTranslator(root_node_shared_ptr.get());
}

QueryPlanDagInfo QueryRunner::getQueryInfoForDataRecyclerTest(
    const std::string& query_str) {
  CHECK(session_info_);
  CHECK(!Catalog_Namespace::SysCatalog::instance().isAggregator());
  auto query_state = create_query_state(session_info_, query_str);
  const auto& cat = session_info_->getCatalog();
  auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID,
                                        &cat.getDataMgr(),
                                        cat.getDataMgr().getBufferProvider());

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
  executor->setSchemaProvider(
      std::make_shared<Catalog_Namespace::CatalogSchemaProvider>(&cat));
  executor->setDatabaseId(cat.getDatabaseId());
  auto ra_executor =
      RelAlgExecutor(executor.get(), &cat, cat.getDataMgr().getDataProvider(), query_ra);
  // note that we assume the test for data recycler that needs to have join_info
  // does not contain any ORDER BY clause; this is necessary to create work_unit
  // without actually performing the query
  auto root_node_shared_ptr = ra_executor.getRootRelAlgNodeShPtr();
  auto join_info = ra_executor.getJoinInfo(root_node_shared_ptr.get());
  auto relAlgTranslator = ra_executor.getRelAlgTranslator(root_node_shared_ptr.get());
  return {root_node_shared_ptr, join_info.first, join_info.second, relAlgTranslator};
}

std::unique_ptr<Parser::DDLStmt> QueryRunner::createDDLStatement(
    const std::string& stmt_str_in) {
  CHECK(session_info_);
  CHECK(!Catalog_Namespace::SysCatalog::instance().isAggregator());

  std::string stmt_str = stmt_str_in;
  // First remove special chars
  boost::algorithm::trim_left_if(stmt_str, boost::algorithm::is_any_of("\n"));
  // Then remove spaces
  boost::algorithm::trim_left(stmt_str);

  ParserWrapper pw{stmt_str};

  auto query_state = create_query_state(session_info_, stmt_str);
  auto stdlog = STDLOG(query_state);

  if (pw.isCalciteDdl()) {
    const auto& cat = session_info_->getCatalog();
    auto calcite_mgr = cat.getCalciteMgr();
    const auto query_json = calcite_mgr
                                ->process(query_state->createQueryStateProxy(),
                                          pg_shim(stmt_str),
                                          {},
                                          true,
                                          false,
                                          false,
                                          true)
                                .plan_result;
    std::unique_ptr<Parser::DDLStmt> ptr = create_ddl_from_calcite(query_json);
    return ptr;
  }

  // simply fail here as non-Calcite parsing is about to be removed
  UNREACHABLE();
  return nullptr;
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

  auto query_state = create_query_state(session_info_, stmt_str);
  auto stdlog = STDLOG(query_state);

  if (pw.isCalciteDdl()) {
    const auto& cat = session_info_->getCatalog();
    auto calcite_mgr = cat.getCalciteMgr();
    const auto query_ra = calcite_mgr
                              ->process(query_state->createQueryStateProxy(),
                                        pg_shim(stmt_str),
                                        {},
                                        true,
                                        false,
                                        false,
                                        true)
                              .plan_result;
    DdlCommandExecutor executor = DdlCommandExecutor(query_ra, session_info_);
    executor.execute();
    return;
  }

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
                                               CompilationOptions co,
                                               ExecutionOptions eo) {
  CHECK(session_info_);
  CHECK(!Catalog_Namespace::SysCatalog::instance().isAggregator());

  ParserWrapper pw{query_str};
  if (pw.isCalcitePathPermissable()) {
    const auto execution_result = runSelectQuery(query_str, std::move(co), std::move(eo));
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
  auto insert_values_stmt = dynamic_cast<Parser::InsertValuesStmt*>(stmt);
  if (insert_values_stmt) {
    insert_values_stmt->execute(*session_info_);
    return nullptr;
  }
  auto ctas_stmt = dynamic_cast<Parser::CreateTableAsSelectStmt*>(stmt);
  if (ctas_stmt) {
    ctas_stmt->execute(*session_info_);
    return nullptr;
  }
  auto itas_stmt = dynamic_cast<Parser::InsertIntoTableAsSelectStmt*>(stmt);
  if (itas_stmt) {
    itas_stmt->execute(*session_info_);
    return nullptr;
  }
  UNREACHABLE();
  return nullptr;
}

std::shared_ptr<ResultSet> QueryRunner::runSQL(const std::string& query_str,
                                               const ExecutorDeviceType device_type,
                                               const bool hoist_literals,
                                               const bool allow_loop_joins) {
  auto co = CompilationOptions::defaults(device_type);
  co.hoist_literals = hoist_literals;
  co.use_groupby_buffer_desc = g_use_groupby_buffer_desc;

  return runSQL(
      query_str, std::move(co), defaultExecutionOptionsForRunSQL(allow_loop_joins));
}

ExecutionOptions QueryRunner::defaultExecutionOptionsForRunSQL(bool allow_loop_joins,
                                                               bool just_explain) {
  return {g_enable_columnar_output,
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
}

std::shared_ptr<Executor> QueryRunner::getExecutor() const {
  CHECK(session_info_);
  CHECK(!Catalog_Namespace::SysCatalog::instance().isAggregator());
  auto query_state = create_query_state(session_info_, "");
  auto stdlog = STDLOG(query_state);
  auto& data_mgr = Catalog_Namespace::SysCatalog::instance().getDataMgr();
  auto executor = Executor::getExecutor(
      Executor::UNITARY_EXECUTOR_ID, &data_mgr, data_mgr.getBufferProvider());
  return executor;
}

std::shared_ptr<ResultSet> QueryRunner::runSQLWithAllowingInterrupt(
    const std::string& query_str,
    const std::string& session_id,
    const ExecutorDeviceType device_type,
    const double running_query_check_freq,
    const unsigned pending_query_check_freq) {
  CHECK(session_info_);
  CHECK(!Catalog_Namespace::SysCatalog::instance().isAggregator());
  auto current_user = session_info_->get_currentUser();
  auto session_info = std::make_shared<Catalog_Namespace::SessionInfo>(
      session_info_->get_catalog_ptr(), current_user, device_type, session_id);
  auto query_state = create_query_state(session_info, query_str);
  auto stdlog = STDLOG(query_state);
  const auto& cat = query_state->getConstSessionInfo()->getCatalog();
  std::string query_ra{""};

  std::shared_ptr<ExecutionResult> result;
  auto query_launch_task = std::make_shared<QueryDispatchQueue::Task>(
      [&cat,
       &query_ra,
       &device_type,
       &query_state,
       &result,
       &running_query_check_freq,
       &pending_query_check_freq](const size_t worker_id) {
        auto executor = Executor::getExecutor(
            worker_id, &cat.getDataMgr(), cat.getDataMgr().getBufferProvider());

        CompilationOptions co = CompilationOptions::defaults(device_type);
        co.opt_level = ExecutorOptLevel::LoopStrengthReduction;
        co.use_groupby_buffer_desc = g_use_groupby_buffer_desc;

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
                               running_query_check_freq,
                               pending_query_check_freq};
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
        auto ra_executor = RelAlgExecutor(executor.get(),
                                          &cat,
                                          cat.getDataMgr().getDataProvider(),
                                          query_ra,
                                          query_state);
        result = std::make_shared<ExecutionResult>(
            ra_executor.executeRelAlgQuery(co, eo, false, nullptr));
      });
  auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID,
                                        &cat.getDataMgr(),
                                        cat.getDataMgr().getBufferProvider());
  executor->enrollQuerySession(session_id,
                               query_str,
                               query_state->getQuerySubmittedTime(),
                               Executor::UNITARY_EXECUTOR_ID,
                               QuerySessionStatus::QueryStatus::PENDING_QUEUE);
  CHECK(dispatch_queue_);
  dispatch_queue_->submit(query_launch_task, /*is_update_delete=*/false);
  auto result_future = query_launch_task->get_future();
  result_future.get();
  CHECK(result);
  return result->getRows();
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

    ParserWrapper pw{text};
    if (pw.isCalciteDdl()) {
      runDDLStatement(text);
      results.push_back(nullptr);
    } else {
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
    const ExecutorExplainType explain_type,
    const bool with_filter_push_down) {
  auto const& query_state = query_state_proxy.getQueryState();
  const auto& cat = query_state.getConstSessionInfo()->getCatalog();
  auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID,
                                        &cat.getDataMgr(),
                                        cat.getDataMgr().getBufferProvider());

  CompilationOptions co = CompilationOptions::defaults(device_type);
  co.opt_level = ExecutorOptLevel::LoopStrengthReduction;
  co.explain_type = explain_type;
  co.use_groupby_buffer_desc = g_use_groupby_buffer_desc;

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
                         false};
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
  auto ra_executor =
      RelAlgExecutor(executor.get(), &cat, cat.getDataMgr().getDataProvider(), query_ra);
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
                                       eo.allow_runtime_query_interrupt,
                                       eo.running_query_interrupt_freq,
                                       eo.pending_query_interrupt_freq};
    auto new_ra_executor = RelAlgExecutor(
        executor.get(), &cat, cat.getDataMgr().getDataProvider(), new_query_ra);
    return std::make_shared<ExecutionResult>(
        new_ra_executor.executeRelAlgQuery(co, eo_modified, false, nullptr));
  } else {
    return result;
  }
}

}  // namespace

std::shared_ptr<ExecutionResult> QueryRunner::runSelectQuery(const std::string& query_str,
                                                             CompilationOptions co,
                                                             ExecutionOptions eo) {
  CHECK(session_info_);
  CHECK(!Catalog_Namespace::SysCatalog::instance().isAggregator());
  auto query_state = create_query_state(session_info_, query_str);
  auto stdlog = STDLOG(query_state);
  if (g_enable_filter_push_down) {
    return run_select_query_with_filter_push_down(query_state->createQueryStateProxy(),
                                                  co.device_type,
                                                  co.hoist_literals,
                                                  eo.allow_loop_joins,
                                                  eo.just_explain,
                                                  explain_type_,
                                                  g_enable_filter_push_down);
  }

  const auto& cat = session_info_->getCatalog();

  std::shared_ptr<ExecutionResult> result;
  auto query_launch_task =
      std::make_shared<QueryDispatchQueue::Task>([&cat,
                                                  &query_str,
                                                  &co,
                                                  explain_type = this->explain_type_,
                                                  &eo,
                                                  &query_state,
                                                  &result](const size_t worker_id) {
        auto executor = Executor::getExecutor(
            worker_id, &cat.getDataMgr(), cat.getDataMgr().getBufferProvider());

        // TODO The next line should be deleted since it overwrites co, but then
        // NycTaxiTest.RunSelectsEncodingDictWhereGreater fails due to co not getting
        // reset to its default values.
        co = CompilationOptions::defaults(co.device_type);
        co.explain_type = explain_type;
        co.opt_level = ExecutorOptLevel::LoopStrengthReduction;
        co.use_groupby_buffer_desc = g_use_groupby_buffer_desc;

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
        auto ra_executor = RelAlgExecutor(
            executor.get(), &cat, cat.getDataMgr().getDataProvider(), query_ra);
        result = std::make_shared<ExecutionResult>(
            ra_executor.executeRelAlgQuery(co, eo, false, nullptr));
      });
  CHECK(dispatch_queue_);
  dispatch_queue_->submit(query_launch_task, /*is_update_delete=*/false);
  auto result_future = query_launch_task->get_future();
  result_future.get();
  CHECK(result);
  return result;
}

std::shared_ptr<ExecutionResult> QueryRunner::runSelectQuery(
    const std::string& query_str,
    const ExecutorDeviceType device_type,
    const bool hoist_literals,
    const bool allow_loop_joins,
    const bool just_explain) {
  auto co = CompilationOptions::defaults(device_type);
  co.hoist_literals = hoist_literals;
  co.use_groupby_buffer_desc = g_use_groupby_buffer_desc;

  return runSelectQuery(query_str,
                        std::move(co),
                        defaultExecutionOptionsForRunSQL(allow_loop_joins, just_explain));
}

ExtractedPlanDag QueryRunner::extractQueryPlanDag(const std::string& query_str) {
  auto query_dag_info = getQueryInfoForDataRecyclerTest(query_str);
  auto& data_mgr = Catalog_Namespace::SysCatalog::instance().getDataMgr();
  auto executor =
      Executor::getExecutor(
          Executor::UNITARY_EXECUTOR_ID, &data_mgr, data_mgr.getBufferProvider())
          .get();
  auto extracted_dag_info =
      QueryPlanDagExtractor::extractQueryPlanDag(query_dag_info.root_node.get(),
                                                 executor->getSchemaProvider(),
                                                 std::nullopt,
                                                 query_dag_info.left_deep_trees_info,
                                                 *executor->getTemporaryTables(),
                                                 executor,
                                                 *query_dag_info.rel_alg_translator);
  return extracted_dag_info;
}

const int32_t* QueryRunner::getCachedPerfectHashTable(QueryPlan plan_dag) {
  auto hash_table_cache = PerfectJoinHashTable::getHashTableCache();
  CHECK(hash_table_cache);
  auto cache_key = boost::hash_value(plan_dag);
  auto hash_table = hash_table_cache->getItemFromCache(
      cache_key, CacheItemType::PERFECT_HT, DataRecyclerUtil::CPU_DEVICE_IDENTIFIER);
  CHECK(hash_table);
  return reinterpret_cast<int32_t*>(hash_table->getCpuBuffer());
}

const int8_t* QueryRunner::getCachedBaselineHashTable(QueryPlan plan_dag) {
  auto hash_table_cache = BaselineJoinHashTable::getHashTableCache();
  CHECK(hash_table_cache);
  auto cache_key = boost::hash_value(plan_dag);
  auto hash_table = hash_table_cache->getItemFromCache(
      cache_key, CacheItemType::BASELINE_HT, DataRecyclerUtil::CPU_DEVICE_IDENTIFIER);
  CHECK(hash_table);
  return hash_table->getCpuBuffer();
}

size_t QueryRunner::getEntryCntCachedBaselineHashTable(QueryPlan plan_dag) {
  auto hash_table_cache = BaselineJoinHashTable::getHashTableCache();
  CHECK(hash_table_cache);
  auto cache_key = boost::hash_value(plan_dag);
  auto hash_table = hash_table_cache->getItemFromCache(
      cache_key, CacheItemType::BASELINE_HT, DataRecyclerUtil::CPU_DEVICE_IDENTIFIER);
  CHECK(hash_table);
  return hash_table->getEntryCount();
}

// this function exists to test data recycler
// specifically, it is tricky to get a hashtable cache key when we only know
// a target query sql in test code
// so this function utilizes an incorrect way to manipulate our hashtable recycler
// but provides the cached hashtable for performing the test
// a set "visited" contains cached hashtable keys that we have retrieved so far
// based on that, this function iterates hashtable cache and return a cached one
// when its hashtable cache key has not been visited yet
// for instance, if we call this funtion with an empty "visited" key, we return
// the first hashtable that its iterator visits
std::tuple<QueryPlanHash,
           std::shared_ptr<HashTable>,
           std::optional<HashtableCacheMetaInfo>>
QueryRunner::getCachedHashtableWithoutCacheKey(std::set<size_t>& visited,
                                               CacheItemType hash_table_type,
                                               DeviceIdentifier device_identifier) {
  HashtableRecycler* hash_table_cache{nullptr};
  switch (hash_table_type) {
    case CacheItemType::PERFECT_HT: {
      hash_table_cache = PerfectJoinHashTable::getHashTableCache();
      break;
    }
    case CacheItemType::BASELINE_HT: {
      hash_table_cache = BaselineJoinHashTable::getHashTableCache();
      break;
    }
    case CacheItemType::OVERLAPS_HT: {
      hash_table_cache = OverlapsJoinHashTable::getHashTableCache();
      break;
    }
    default: {
      UNREACHABLE();
      break;
    }
  }
  CHECK(hash_table_cache);
  return hash_table_cache->getCachedHashtableWithoutCacheKey(
      visited, hash_table_type, device_identifier);
}

std::shared_ptr<CacheItemMetric> QueryRunner::getCacheItemMetric(
    QueryPlanHash cache_key,
    CacheItemType hash_table_type,
    DeviceIdentifier device_identifier) {
  HashtableRecycler* hash_table_cache{nullptr};
  switch (hash_table_type) {
    case CacheItemType::PERFECT_HT: {
      hash_table_cache = PerfectJoinHashTable::getHashTableCache();
      break;
    }
    case CacheItemType::BASELINE_HT: {
      hash_table_cache = BaselineJoinHashTable::getHashTableCache();
      break;
    }
    case CacheItemType::OVERLAPS_HT: {
      hash_table_cache = OverlapsJoinHashTable::getHashTableCache();
      break;
    }
    default: {
      UNREACHABLE();
      break;
    }
  }
  CHECK(hash_table_cache);
  return hash_table_cache->getCachedItemMetric(
      hash_table_type, device_identifier, cache_key);
}

size_t QueryRunner::getNumberOfCachedPerfectHashTables() {
  auto hash_table_cache = PerfectJoinHashTable::getHashTableCache();
  CHECK(hash_table_cache);
  return hash_table_cache->getCurrentNumCachedItems(
      CacheItemType::PERFECT_HT, DataRecyclerUtil::CPU_DEVICE_IDENTIFIER);
};

size_t QueryRunner::getNumberOfCachedBaselineJoinHashTables() {
  auto hash_table_cache = BaselineJoinHashTable::getHashTableCache();
  CHECK(hash_table_cache);
  return hash_table_cache->getCurrentNumCachedItems(
      CacheItemType::BASELINE_HT, DataRecyclerUtil::CPU_DEVICE_IDENTIFIER);
}

size_t QueryRunner::getNumberOfCachedOverlapsHashTables() {
  auto hash_table_cache = OverlapsJoinHashTable::getHashTableCache();
  CHECK(hash_table_cache);
  return hash_table_cache->getCurrentNumCachedItems(
      CacheItemType::OVERLAPS_HT, DataRecyclerUtil::CPU_DEVICE_IDENTIFIER);
}

size_t QueryRunner::getNumberOfCachedOverlapsHashTableTuringParams() {
  auto hash_table_cache = OverlapsJoinHashTable::getOverlapsTuningParamCache();
  CHECK(hash_table_cache);
  return hash_table_cache->getCurrentNumCachedItems(
      CacheItemType::OVERLAPS_AUTO_TUNER_PARAM, DataRecyclerUtil::CPU_DEVICE_IDENTIFIER);
}

size_t QueryRunner::getNumberOfCachedOverlapsHashTablesAndTuningParams() {
  return getNumberOfCachedOverlapsHashTables() +
         getNumberOfCachedOverlapsHashTableTuringParams();
}

void QueryRunner::reset() {
  qr_instance_.reset(nullptr);
  calcite_shutdown_handler();
}

ImportDriver::ImportDriver(std::shared_ptr<Catalog_Namespace::Catalog> cat,
                           const Catalog_Namespace::UserMetadata& user,
                           const ExecutorDeviceType dt,
                           const std::string session_id)
    : QueryRunner(
          std::make_unique<Catalog_Namespace::SessionInfo>(cat, user, dt, session_id)) {}

}  // namespace QueryRunner
