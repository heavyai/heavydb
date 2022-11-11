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

#include "QueryRunner.h"

#include "Calcite/Calcite.h"
#include "Catalog/Catalog.h"
#include "Catalog/DdlCommandExecutor.h"
#include "DistributedLoader.h"
#include "Geospatial/ColumnNames.h"
#include "ImportExport/CopyParams.h"
#include "Logger/Logger.h"
#include "Parser/ParserNode.h"
#include "Parser/ParserWrapper.h"
#include "QueryEngine/CalciteAdapter.h"
#include "QueryEngine/DataRecycler/HashtableRecycler.h"
#include "QueryEngine/ExtensionFunctionsWhitelist.h"
#include "QueryEngine/QueryDispatchQueue.h"
#include "QueryEngine/QueryPlanDagExtractor.h"
#include "QueryEngine/RelAlgExecutor.h"
#include "QueryEngine/TableFunctions/TableFunctionsFactory.h"
#include "QueryEngine/ThriftSerializers.h"
#include "Shared/StringTransform.h"
#include "Shared/SysDefinitions.h"
#include "Shared/SystemParameters.h"
#include "Shared/import_helpers.h"
#include "TestProcessSignalHandler.h"
#include "gen-cpp/CalciteServer.h"
#include "include/bcrypt.h"

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
                           shared::kRootUsername,
                           "HyperInteractive",
                           shared::kDefaultDbName,
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
                           shared::kRootUsername,
                           "HyperInteractive",
                           shared::kDefaultDbName,
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
  auto system_db_file =
      base_path / shared::kCatalogDirectoryName / shared::kDefaultDbName;
  CHECK(boost::filesystem::exists(system_db_file));
  auto data_dir = base_path / shared::kDataDirectoryName;
  File_Namespace::DiskCacheConfig disk_cache_config{
      (base_path / shared::kDefaultDiskCacheDirName).string(),
      File_Namespace::DiskCacheLevel::fsi};
  if (cache_config) {
    disk_cache_config = *cache_config;
  }
  Catalog_Namespace::UserMetadata user;

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

  query_engine_ = QueryEngine::createInstance(data_mgr_->getCudaMgr(), !uses_gpus);

  if (create_user) {
    if (!sys_cat.getMetadataForUser(user_name, user)) {
      sys_cat.createUser(
          user_name,
          UserAlterations{
              passwd, /*is_super=*/false, /*default_db=*/"", /*can_login=*/true},
          g_read_only);
    }
  }
  CHECK(sys_cat.getMetadataForUser(user_name, user));
  CHECK(bcrypt_checkpw(passwd.c_str(), user.passwd_hash.c_str()) == 0);

  if (create_db) {
    if (!sys_cat.getMetadataForDB(db_name, db_metadata_)) {
      sys_cat.createDatabase(db_name, user.userId);
    }
  }
  CHECK(sys_cat.getMetadataForDB(db_name, db_metadata_));
  CHECK(user.isSuper || (user.userId == db_metadata_.dbOwner));
  auto cat = sys_cat.getCatalog(db_metadata_, create_db);
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

std::vector<MemoryInfo> QueryRunner::getMemoryInfo(
    const Data_Namespace::MemoryLevel memory_level) const {
  CHECK(!Catalog_Namespace::SysCatalog::instance().isAggregator());
  return session_info_->getCatalog().getDataMgr().getMemoryInfo(memory_level);
}

BufferPoolStats QueryRunner::getBufferPoolStats(
    const Data_Namespace::MemoryLevel memory_level,
    const bool current_db_only) const {
  // Only works single-node for now
  CHECK(!Catalog_Namespace::SysCatalog::instance().isAggregator());
  const std::vector<MemoryInfo> memory_infos =
      session_info_->getCatalog().getDataMgr().getMemoryInfo(memory_level);
  if (memory_level == Data_Namespace::MemoryLevel::CPU_LEVEL) {
    CHECK_EQ(memory_infos.size(), static_cast<size_t>(1));
  }
  std::set<std::vector<int32_t>> chunk_keys;
  std::set<std::vector<int32_t>> table_keys;
  std::set<std::vector<int32_t>> column_keys;
  std::set<std::vector<int32_t>> fragment_keys;
  size_t total_num_buffers{
      0};  // can be greater than chunk keys set size due to table replication
  size_t total_num_bytes{0};
  for (auto& pool_memory_info : memory_infos) {
    const std::vector<MemoryData>& memory_data = pool_memory_info.nodeMemoryData;
    for (auto& memory_datum : memory_data) {
      total_num_buffers++;
      const auto& chunk_key = memory_datum.chunk_key;
      if (memory_datum.memStatus == Buffer_Namespace::MemStatus::FREE ||
          chunk_key.size() < 4) {
        continue;
      }
      if (current_db_only) {
        if (chunk_key[0] != db_metadata_.dbId) {
          continue;
        }
      }
      total_num_bytes += (memory_datum.numPages * pool_memory_info.pageSize);
      table_keys.insert({chunk_key[0], chunk_key[1]});
      column_keys.insert({chunk_key[0], chunk_key[1], chunk_key[2]});
      fragment_keys.insert({chunk_key[0], chunk_key[1], chunk_key[3]});
      chunk_keys.insert(chunk_key);
    }
  }
  return {total_num_buffers,
          total_num_bytes,
          table_keys.size(),
          column_keys.size(),
          fragment_keys.size(),
          chunk_keys.size()};
}

RegisteredQueryHint QueryRunner::getParsedQueryHint(const std::string& query_str) {
  CHECK(session_info_);
  CHECK(!Catalog_Namespace::SysCatalog::instance().isAggregator());
  auto query_state = create_query_state(session_info_, query_str);
  auto& cat = session_info_->getCatalog();
  auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID);

  auto calcite_mgr = cat.getCalciteMgr();
  const auto calciteQueryParsingOption =
      calcite_mgr->getCalciteQueryParsingOption(true, false, true);
  const auto calciteOptimizationOption =
      calcite_mgr->getCalciteOptimizationOption(false, g_enable_watchdog, {}, false);
  const auto query_ra = calcite_mgr
                            ->process(query_state->createQueryStateProxy(),
                                      pg_shim(query_str),
                                      calciteQueryParsingOption,
                                      calciteOptimizationOption)
                            .plan_result;
  auto ra_executor = RelAlgExecutor(executor.get(), cat, query_ra, query_state);
  auto query_hints =
      ra_executor.getParsedQueryHint(ra_executor.getRootRelAlgNodeShPtr().get());
  return query_hints ? *query_hints : RegisteredQueryHint::defaults();
}

std::shared_ptr<const RelAlgNode> QueryRunner::getRootNodeFromParsedQuery(
    const std::string& query_str) {
  CHECK(session_info_);
  CHECK(!Catalog_Namespace::SysCatalog::instance().isAggregator());
  auto query_state = create_query_state(session_info_, query_str);
  auto& cat = session_info_->getCatalog();
  auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID);

  auto calcite_mgr = cat.getCalciteMgr();
  const auto calciteQueryParsingOption =
      calcite_mgr->getCalciteQueryParsingOption(true, false, true);
  const auto calciteOptimizationOption =
      calcite_mgr->getCalciteOptimizationOption(false, g_enable_watchdog, {}, false);
  const auto query_ra = calcite_mgr
                            ->process(query_state->createQueryStateProxy(),
                                      pg_shim(query_str),
                                      calciteQueryParsingOption,
                                      calciteOptimizationOption)
                            .plan_result;
  auto ra_executor = RelAlgExecutor(executor.get(), cat, query_ra, query_state);
  return ra_executor.getRootRelAlgNodeShPtr();
}

std::optional<
    std::unordered_map<size_t, std::unordered_map<unsigned, RegisteredQueryHint>>>
QueryRunner::getParsedQueryHints(const std::string& query_str) {
  CHECK(session_info_);
  CHECK(!Catalog_Namespace::SysCatalog::instance().isAggregator());
  auto query_state = create_query_state(session_info_, query_str);
  auto& cat = session_info_->getCatalog();
  auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID);
  auto calcite_mgr = cat.getCalciteMgr();
  const auto calciteQueryParsingOption =
      calcite_mgr->getCalciteQueryParsingOption(true, false, true);
  const auto calciteOptimizationOption =
      calcite_mgr->getCalciteOptimizationOption(false, g_enable_watchdog, {}, false);
  const auto query_ra = calcite_mgr
                            ->process(query_state->createQueryStateProxy(),
                                      pg_shim(query_str),
                                      calciteQueryParsingOption,
                                      calciteOptimizationOption)
                            .plan_result;
  auto ra_executor = RelAlgExecutor(executor.get(), cat, query_ra, query_state);
  return ra_executor.getParsedQueryHints();
}

std::optional<RegisteredQueryHint> QueryRunner::getParsedGlobalQueryHints(
    const std::string& query_str) {
  CHECK(session_info_);
  CHECK(!Catalog_Namespace::SysCatalog::instance().isAggregator());
  auto query_state = create_query_state(session_info_, query_str);
  auto& cat = session_info_->getCatalog();
  auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID);
  auto calcite_mgr = cat.getCalciteMgr();
  const auto calciteQueryParsingOption =
      calcite_mgr->getCalciteQueryParsingOption(true, false, true);
  const auto calciteOptimizationOption =
      calcite_mgr->getCalciteOptimizationOption(false, g_enable_watchdog, {}, false);
  const auto query_ra = calcite_mgr
                            ->process(query_state->createQueryStateProxy(),
                                      pg_shim(query_str),
                                      calciteQueryParsingOption,
                                      calciteOptimizationOption)
                            .plan_result;
  auto ra_executor = RelAlgExecutor(executor.get(), cat, query_ra, query_state);
  return ra_executor.getGlobalQueryHint();
}

RaExecutionSequence QueryRunner::getRaExecutionSequence(const std::string& query_str) {
  CHECK(session_info_);
  CHECK(!Catalog_Namespace::SysCatalog::instance().isAggregator());
  auto query_state = create_query_state(session_info_, query_str);
  auto& cat = session_info_->getCatalog();
  auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID);
  auto calcite_mgr = cat.getCalciteMgr();
  const auto calciteQueryParsingOption =
      calcite_mgr->getCalciteQueryParsingOption(true, false, true);
  const auto calciteOptimizationOption =
      calcite_mgr->getCalciteOptimizationOption(false, g_enable_watchdog, {}, false);
  const auto query_ra = calcite_mgr
                            ->process(query_state->createQueryStateProxy(),
                                      pg_shim(query_str),
                                      calciteQueryParsingOption,
                                      calciteOptimizationOption)
                            .plan_result;
  auto ra_executor = RelAlgExecutor(executor.get(), cat, query_ra, query_state);
  return ra_executor.getRaExecutionSequence(ra_executor.getRootRelAlgNodeShPtr().get(),
                                            executor.get());
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

  auto& cat = session_info_->getCatalog();
  auto calcite_mgr = cat.getCalciteMgr();
  const auto calciteQueryParsingOption =
      calcite_mgr->getCalciteQueryParsingOption(true, false, true);
  const auto calciteOptimizationOption =
      calcite_mgr->getCalciteOptimizationOption(false, g_enable_watchdog, {}, false);
  calcite_mgr->process(query_state->createQueryStateProxy(),
                       pg_shim(stmt_str),
                       calciteQueryParsingOption,
                       calciteOptimizationOption);
}

std::shared_ptr<RelAlgTranslator> QueryRunner::getRelAlgTranslator(
    const std::string& query_str,
    Executor* executor) {
  CHECK(session_info_);
  CHECK(!Catalog_Namespace::SysCatalog::instance().isAggregator());
  auto query_state = create_query_state(session_info_, query_str);
  auto& cat = session_info_->getCatalog();
  auto calcite_mgr = cat.getCalciteMgr();
  const auto calciteQueryParsingOption =
      calcite_mgr->getCalciteQueryParsingOption(true, false, true);
  const auto calciteOptimizationOption =
      calcite_mgr->getCalciteOptimizationOption(false, g_enable_watchdog, {}, false);
  const auto query_ra = calcite_mgr
                            ->process(query_state->createQueryStateProxy(),
                                      pg_shim(query_str),
                                      calciteQueryParsingOption,
                                      calciteOptimizationOption)
                            .plan_result;
  executor->setCatalog(&cat);
  auto ra_executor = RelAlgExecutor(executor, cat, query_ra);
  auto root_node_shared_ptr = ra_executor.getRootRelAlgNodeShPtr();
  return ra_executor.getRelAlgTranslator(root_node_shared_ptr.get());
}

QueryPlanDagInfo QueryRunner::getQueryInfoForDataRecyclerTest(
    const std::string& query_str) {
  CHECK(session_info_);
  CHECK(!Catalog_Namespace::SysCatalog::instance().isAggregator());
  auto query_state = create_query_state(session_info_, query_str);
  auto& cat = session_info_->getCatalog();
  auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID);
  auto calcite_mgr = cat.getCalciteMgr();
  const auto calciteQueryParsingOption =
      calcite_mgr->getCalciteQueryParsingOption(true, false, true);
  const auto calciteOptimizationOption =
      calcite_mgr->getCalciteOptimizationOption(false, g_enable_watchdog, {}, false);
  const auto query_ra = calcite_mgr
                            ->process(query_state->createQueryStateProxy(),
                                      pg_shim(query_str),
                                      calciteQueryParsingOption,
                                      calciteOptimizationOption)
                            .plan_result;
  executor->setCatalog(&cat);
  auto ra_executor = RelAlgExecutor(executor.get(), cat, query_ra);
  // note that we assume the test for data recycler that needs to have join_info
  // does not contain any ORDER BY clause; this is necessary to create work_unit
  // without actually performing the query
  auto root_node_shared_ptr = ra_executor.getRootRelAlgNodeShPtr();
  auto join_info = ra_executor.getJoinInfo(root_node_shared_ptr.get());
  auto relAlgTranslator = ra_executor.getRelAlgTranslator(root_node_shared_ptr.get());
  return {root_node_shared_ptr, join_info.first, join_info.second, relAlgTranslator};
}

std::unique_ptr<Parser::Stmt> QueryRunner::createStatement(
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

  if (pw.is_ddl) {
    const auto& cat = session_info_->getCatalog();
    auto calcite_mgr = cat.getCalciteMgr();
    const auto calciteQueryParsingOption =
        calcite_mgr->getCalciteQueryParsingOption(true, false, true);
    const auto calciteOptimizationOption =
        calcite_mgr->getCalciteOptimizationOption(false, g_enable_watchdog, {}, false);
    const auto query_json = calcite_mgr
                                ->process(query_state->createQueryStateProxy(),
                                          pg_shim(stmt_str),
                                          calciteQueryParsingOption,
                                          calciteOptimizationOption)
                                .plan_result;
    return Parser::create_stmt_for_json(query_json);
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

  if (pw.is_ddl || pw.getDMLType() == ParserWrapper::DMLType::Insert) {
    auto& cat = session_info_->getCatalog();
    auto calcite_mgr = cat.getCalciteMgr();
    const auto calciteQueryParsingOption =
        calcite_mgr->getCalciteQueryParsingOption(true, false, true);
    const auto calciteOptimizationOption =
        calcite_mgr->getCalciteOptimizationOption(false, g_enable_watchdog, {}, false);
    const auto query_ra = calcite_mgr
                              ->process(query_state->createQueryStateProxy(),
                                        pg_shim(stmt_str),
                                        calciteQueryParsingOption,
                                        calciteOptimizationOption)
                              .plan_result;
    if (pw.getDMLType() == ParserWrapper::DMLType::Insert) {
      rapidjson::Document ddl_query;
      ddl_query.Parse(query_ra);
      CHECK(ddl_query.HasMember("payload"));
      CHECK(ddl_query["payload"].IsObject());
      auto stmt = Parser::InsertValuesStmt(ddl_query["payload"].GetObject());
      stmt.execute(*session_info_, false /* read only */);
      return;
    }
    DdlCommandExecutor executor = DdlCommandExecutor(query_ra, session_info_);
    executor.execute(false /* read only */);
    return;
  }
}

std::shared_ptr<ResultSet> QueryRunner::runSQL(const std::string& query_str,
                                               CompilationOptions co,
                                               ExecutionOptions eo) {
  CHECK(session_info_);
  CHECK(!Catalog_Namespace::SysCatalog::instance().isAggregator());

  ParserWrapper pw{query_str};
  if (pw.getDMLType() == ParserWrapper::DMLType::Insert) {
    runDDLStatement(query_str);
    return nullptr;
  }
  const auto execution_result = runSelectQuery(query_str, std::move(co), std::move(eo));
  VLOG(1) << session_info_->getCatalog().getDataMgr().getSystemMemoryUsage();
  return execution_result->getRows();
}

std::shared_ptr<ResultSet> QueryRunner::runSQL(const std::string& query_str,
                                               const ExecutorDeviceType device_type,
                                               const bool hoist_literals,
                                               const bool allow_loop_joins) {
  auto co = CompilationOptions::defaults(device_type);
  co.hoist_literals = hoist_literals;
  return runSQL(
      query_str, std::move(co), defaultExecutionOptionsForRunSQL(allow_loop_joins));
}

ExecutionOptions QueryRunner::defaultExecutionOptionsForRunSQL(bool allow_loop_joins,
                                                               bool just_explain) {
  return {g_enable_columnar_output,
          false,
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
          1000,
          false};
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
  auto& cat = query_state->getConstSessionInfo()->getCatalog();
  std::string query_ra{""};

  std::shared_ptr<ExecutionResult> result;
  auto query_launch_task = std::make_shared<QueryDispatchQueue::Task>(
      [&cat,
       &query_ra,
       &device_type,
       &query_state,
       &result,
       &running_query_check_freq,
       &pending_query_check_freq,
       parent_thread_local_ids = logger::thread_local_ids()](const size_t worker_id) {
        logger::LocalIdsScopeGuard lisg = parent_thread_local_ids.setNewThreadId();
        auto executor = Executor::getExecutor(worker_id);
        CompilationOptions co = CompilationOptions::defaults(device_type);

        ExecutionOptions eo = {g_enable_columnar_output,
                               false,
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
                               pending_query_check_freq,
                               false};
        {
          // async query initiation for interrupt test
          // incurs data race warning in TSAN since
          // calcite_mgr is shared across multiple query threads
          // so here we lock the manager during query parsing
          std::lock_guard<std::mutex> calcite_lock_guard(calcite_lock);
          auto calcite_mgr = cat.getCalciteMgr();
          const auto calciteQueryParsingOption =
              calcite_mgr->getCalciteQueryParsingOption(true, false, true);
          const auto calciteOptimizationOption =
              calcite_mgr->getCalciteOptimizationOption(
                  false, g_enable_watchdog, {}, false);
          query_ra = calcite_mgr
                         ->process(query_state->createQueryStateProxy(),
                                   pg_shim(query_state->getQueryStr()),
                                   calciteQueryParsingOption,
                                   calciteOptimizationOption)
                         .plan_result;
        }
        auto ra_executor = RelAlgExecutor(executor.get(), cat, query_ra, query_state);
        result = std::make_shared<ExecutionResult>(
            ra_executor.executeRelAlgQuery(co, eo, false, nullptr));
      });
  auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID);
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
    if (pw.is_ddl || pw.getDMLType() == ParserWrapper::DMLType::Insert) {
      runDDLStatement(text);
      results.push_back(nullptr);
    } else {
      // is not DDL, then assume it's DML and try to execute
      results.push_back(runSQL(text, dt, true, true));
    }
  }
  return results;
}

void QueryRunner::runImport(Parser::CopyTableStmt* import_stmt) {
  CHECK(import_stmt);
  import_stmt->execute(*session_info_, false /* read only */);
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
  auto& cat = query_state_proxy->getConstSessionInfo()->getCatalog();
  auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID);
  CompilationOptions co = CompilationOptions::defaults(device_type);
  co.explain_type = explain_type;

  ExecutionOptions eo = ExecutionOptions::defaults();
  eo.output_columnar_hint = g_enable_columnar_output;
  eo.just_explain = just_explain;
  eo.allow_loop_joins = allow_loop_joins;
  eo.find_push_down_candidates = with_filter_push_down;
  eo.gpu_input_mem_limit_percent = g_gpu_mem_limit_percent;

  auto calcite_mgr = cat.getCalciteMgr();
  const auto calciteQueryParsingOption =
      calcite_mgr->getCalciteQueryParsingOption(true, false, true);
  auto calciteOptimizationOption =
      calcite_mgr->getCalciteOptimizationOption(false, g_enable_watchdog, {}, false);
  const auto query_ra = calcite_mgr
                            ->process(query_state_proxy,
                                      pg_shim(query_state_proxy->getQueryStr()),
                                      calciteQueryParsingOption,
                                      calciteOptimizationOption)
                            .plan_result;
  auto ra_executor = RelAlgExecutor(executor.get(), cat, query_ra);
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
    calciteOptimizationOption.filter_push_down_info = filter_push_down_info;
    const auto new_query_ra = calcite_mgr
                                  ->process(query_state_proxy,
                                            pg_shim(query_state_proxy->getQueryStr()),
                                            calciteQueryParsingOption,
                                            calciteOptimizationOption)
                                  .plan_result;
    auto eo_modified = eo;
    eo_modified.find_push_down_candidates = false;
    eo_modified.just_calcite_explain = false;
    auto new_ra_executor = RelAlgExecutor(executor.get(), cat, new_query_ra);
    return std::make_shared<ExecutionResult>(
        new_ra_executor.executeRelAlgQuery(co, eo_modified, false, nullptr));
  } else {
    return result;
  }
}

}  // namespace

std::shared_ptr<ResultSet> QueryRunner::getCalcitePlan(const std::string& query_str,
                                                       bool enable_watchdog,
                                                       bool as_json_str) const {
  CHECK(!Catalog_Namespace::SysCatalog::instance().isAggregator());
  CHECK(session_info_);
  const auto& cat = session_info_->getCatalog();
  auto query_state = create_query_state(session_info_, query_str);
  auto stdlog = STDLOG(query_state);

  std::shared_ptr<ResultSet> result;
  auto query_launch_task = std::make_shared<QueryDispatchQueue::Task>(
      [&cat,
       &query_str,
       &enable_watchdog,
       &as_json_str,
       &query_state,
       &result,
       parent_thread_local_ids = logger::thread_local_ids()](const size_t worker_id) {
        logger::LocalIdsScopeGuard lisg = parent_thread_local_ids.setNewThreadId();
        auto executor = Executor::getExecutor(worker_id);
        auto calcite_mgr = cat.getCalciteMgr();
        const auto calciteQueryParsingOption =
            calcite_mgr->getCalciteQueryParsingOption(true, as_json_str, false);
        const auto calciteOptimizationOption = calcite_mgr->getCalciteOptimizationOption(
            g_enable_calcite_view_optimize, enable_watchdog, {}, false);
        const auto query_ra = calcite_mgr
                                  ->process(query_state->createQueryStateProxy(),
                                            pg_shim(query_str),
                                            calciteQueryParsingOption,
                                            calciteOptimizationOption)
                                  .plan_result;
        result = std::make_shared<ResultSet>(query_ra);
        return result;
      });
  CHECK(dispatch_queue_);
  dispatch_queue_->submit(query_launch_task, /*is_update_delete=*/false);
  auto result_future = query_launch_task->get_future();
  result_future.get();
  CHECK(result);
  return result;
}

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

  auto& cat = session_info_->getCatalog();

  std::shared_ptr<ExecutionResult> result;
  auto query_launch_task = std::make_shared<QueryDispatchQueue::Task>(
      [&cat,
       &query_str,
       &co,
       explain_type = this->explain_type_,
       &eo,
       &query_state,
       &result,
       parent_thread_local_ids = logger::thread_local_ids()](const size_t worker_id) {
        logger::LocalIdsScopeGuard lisg = parent_thread_local_ids.setNewThreadId();
        auto executor = Executor::getExecutor(worker_id);
        // TODO The next line should be deleted since it overwrites co, but then
        // NycTaxiTest.RunSelectsEncodingDictWhereGreater fails due to co not getting
        // reset to its default values.
        co = CompilationOptions::defaults(co.device_type);
        co.explain_type = explain_type;
        auto calcite_mgr = cat.getCalciteMgr();
        const auto calciteQueryParsingOption =
            calcite_mgr->getCalciteQueryParsingOption(true, false, true);
        const auto calciteOptimizationOption = calcite_mgr->getCalciteOptimizationOption(
            g_enable_calcite_view_optimize, g_enable_watchdog, {}, false);
        const auto query_ra = calcite_mgr
                                  ->process(query_state->createQueryStateProxy(),
                                            pg_shim(query_str),
                                            calciteQueryParsingOption,
                                            calciteOptimizationOption)
                                  .plan_result;
        auto ra_executor = RelAlgExecutor(executor.get(), cat, query_ra);
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
  return runSelectQuery(query_str,
                        std::move(co),
                        defaultExecutionOptionsForRunSQL(allow_loop_joins, just_explain));
}

ExtractedQueryPlanDag QueryRunner::extractQueryPlanDag(const std::string& query_str) {
  auto query_dag_info = getQueryInfoForDataRecyclerTest(query_str);
  auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID).get();
  auto extracted_dag_info = QueryPlanDagExtractor::extractQueryPlanDag(
      query_dag_info.root_node.get(), executor);
  return extracted_dag_info;
}

std::unique_ptr<RelAlgDag> QueryRunner::getRelAlgDag(const std::string& query_str) {
  CHECK(session_info_);
  CHECK(!Catalog_Namespace::SysCatalog::instance().isAggregator());
  auto query_state = create_query_state(session_info_, query_str);
  auto stdlog = STDLOG(query_state);
  auto& cat = session_info_->getCatalog();

  std::unique_ptr<RelAlgDag> rel_alg_dag;
  auto query_launch_task = std::make_shared<QueryDispatchQueue::Task>(
      [&cat,
       &query_str,
       &query_state,
       &rel_alg_dag,
       parent_thread_local_ids = logger::thread_local_ids()](const size_t worker_id) {
        logger::LocalIdsScopeGuard lisg = parent_thread_local_ids.setNewThreadId();
        auto executor = Executor::getExecutor(worker_id);
        auto eo = ExecutionOptions::defaults();
        auto calcite_mgr = cat.getCalciteMgr();
        const auto calciteQueryParsingOption =
            calcite_mgr->getCalciteQueryParsingOption(true, false, true);
        const auto calciteOptimizationOption = calcite_mgr->getCalciteOptimizationOption(
            g_enable_calcite_view_optimize, g_enable_watchdog, {}, false);
        const auto query_ra = calcite_mgr
                                  ->process(query_state->createQueryStateProxy(),
                                            pg_shim(query_str),
                                            calciteQueryParsingOption,
                                            calciteOptimizationOption)
                                  .plan_result;
        auto ra_executor = RelAlgExecutor(executor.get(), cat, query_ra);
        rel_alg_dag = ra_executor.getOwnedRelAlgDag();
      });
  CHECK(dispatch_queue_);
  dispatch_queue_->submit(query_launch_task, /*is_update_delete=*/false);
  auto result_future = query_launch_task->get_future();
  result_future.get();
  CHECK(rel_alg_dag);
  return rel_alg_dag;
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

size_t QueryRunner::getNumberOfCachedItem(CacheItemStatus item_status,
                                          CacheItemType hash_table_type,
                                          bool with_overlaps_tuning_param) const {
  auto get_num_cached_auto_tuner_param = [&item_status]() {
    auto auto_tuner_cache = OverlapsJoinHashTable::getOverlapsTuningParamCache();
    CHECK(auto_tuner_cache);
    switch (item_status) {
      case CacheItemStatus::ALL: {
        return auto_tuner_cache->getCurrentNumCachedItems(
            CacheItemType::OVERLAPS_AUTO_TUNER_PARAM,
            DataRecyclerUtil::CPU_DEVICE_IDENTIFIER);
      }
      case CacheItemStatus::CLEAN_ONLY: {
        return auto_tuner_cache->getCurrentNumCleanCachedItems(
            CacheItemType::OVERLAPS_AUTO_TUNER_PARAM,
            DataRecyclerUtil::CPU_DEVICE_IDENTIFIER);
      }
      case CacheItemStatus::DIRTY_ONLY: {
        return auto_tuner_cache->getCurrentNumDirtyCachedItems(
            CacheItemType::OVERLAPS_AUTO_TUNER_PARAM,
            DataRecyclerUtil::CPU_DEVICE_IDENTIFIER);
      }
      default: {
        UNREACHABLE();
        return static_cast<size_t>(0);
      }
    }
  };

  auto get_num_cached_hashtable =
      [&item_status,
       &hash_table_type,
       &with_overlaps_tuning_param,
       &get_num_cached_auto_tuner_param](HashtableRecycler* hash_table_cache) {
        switch (item_status) {
          case CacheItemStatus::ALL: {
            if (with_overlaps_tuning_param) {
              // we assume additional consideration of turing param cache is only valid
              // for overlaps join hashtable
              CHECK_EQ(hash_table_type, CacheItemType::OVERLAPS_HT);
              return hash_table_cache->getCurrentNumCachedItems(
                         hash_table_type, DataRecyclerUtil::CPU_DEVICE_IDENTIFIER) +
                     get_num_cached_auto_tuner_param();
            }
            return hash_table_cache->getCurrentNumCachedItems(
                hash_table_type, DataRecyclerUtil::CPU_DEVICE_IDENTIFIER);
          }
          case CacheItemStatus::CLEAN_ONLY: {
            if (with_overlaps_tuning_param) {
              CHECK_EQ(hash_table_type, CacheItemType::OVERLAPS_HT);
              return hash_table_cache->getCurrentNumCleanCachedItems(
                         hash_table_type, DataRecyclerUtil::CPU_DEVICE_IDENTIFIER) +
                     get_num_cached_auto_tuner_param();
            }
            return hash_table_cache->getCurrentNumCleanCachedItems(
                hash_table_type, DataRecyclerUtil::CPU_DEVICE_IDENTIFIER);
          }
          case CacheItemStatus::DIRTY_ONLY: {
            if (with_overlaps_tuning_param) {
              CHECK_EQ(hash_table_type, CacheItemType::OVERLAPS_HT);
              return hash_table_cache->getCurrentNumDirtyCachedItems(
                         hash_table_type, DataRecyclerUtil::CPU_DEVICE_IDENTIFIER) +
                     get_num_cached_auto_tuner_param();
            }
            return hash_table_cache->getCurrentNumDirtyCachedItems(
                hash_table_type, DataRecyclerUtil::CPU_DEVICE_IDENTIFIER);
          }
          default: {
            UNREACHABLE();
            return static_cast<size_t>(0);
          }
        }
      };

  switch (hash_table_type) {
    case CacheItemType::PERFECT_HT: {
      auto hash_table_cache = PerfectJoinHashTable::getHashTableCache();
      CHECK(hash_table_cache);
      return get_num_cached_hashtable(hash_table_cache);
    }
    case CacheItemType::BASELINE_HT: {
      auto hash_table_cache = BaselineJoinHashTable::getHashTableCache();
      CHECK(hash_table_cache);
      return get_num_cached_hashtable(hash_table_cache);
    }
    case CacheItemType::OVERLAPS_HT: {
      auto hash_table_cache = OverlapsJoinHashTable::getHashTableCache();
      CHECK(hash_table_cache);
      return get_num_cached_hashtable(hash_table_cache);
    }
    case CacheItemType::OVERLAPS_AUTO_TUNER_PARAM: {
      return get_num_cached_auto_tuner_param();
    }
    default: {
      UNREACHABLE();
      return 0;
    }
  }
  return 0;
}

void QueryRunner::reset() {
  qr_instance_->query_engine_.reset();
  qr_instance_.reset(nullptr);
  calcite_shutdown_handler();
}

ImportDriver::ImportDriver(std::shared_ptr<Catalog_Namespace::Catalog> cat,
                           const Catalog_Namespace::UserMetadata& user,
                           const ExecutorDeviceType dt,
                           const std::string session_id)
    : QueryRunner(
          std::make_unique<Catalog_Namespace::SessionInfo>(cat, user, dt, session_id)) {}

void ImportDriver::importGeoTable(const std::string& file_path,
                                  const std::string& table_name,
                                  const bool compression,
                                  const bool create_table,
                                  const bool explode_collections) {
  using namespace import_export;

  static constexpr bool kIsGeoRaster{false};

  CHECK(session_info_);

  CopyParams copy_params;
  copy_params.source_type = import_export::SourceType::kGeoFile;
  if (compression) {
    copy_params.geo_coords_encoding = EncodingType::kENCODING_GEOINT;
    copy_params.geo_coords_comp_param = 32;
  } else {
    copy_params.geo_coords_encoding = EncodingType::kENCODING_NONE;
    copy_params.geo_coords_comp_param = 0;
  }
  copy_params.geo_assign_render_groups = false;
  copy_params.geo_explode_collections = explode_collections;

  std::map<std::string, std::string> colname_to_src;
  auto& cat = session_info_->getCatalog();
  auto cds = Importer::gdalToColumnDescriptors(
      file_path, kIsGeoRaster, Geospatial::kGeoColumnName, copy_params);

  for (auto& cd : cds) {
    const auto col_name_sanitized = ImportHelpers::sanitize_name(cd.columnName);
    const auto ret =
        colname_to_src.insert(std::make_pair(col_name_sanitized, cd.columnName));
    CHECK(ret.second);
    cd.columnName = col_name_sanitized;
  }

  if (create_table) {
    const auto td = cat.getMetadataForTable(table_name);
    if (td != nullptr) {
      throw std::runtime_error(
          "Error: Table " + table_name +
          " already exists. Possible failure to correctly re-create " +
          shared::kDataDirectoryName + " directory.");
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
  auto ms = measure<>::execution(
      [&]() { importer.importGDAL(colname_to_src, session_info_.get(), kIsGeoRaster); });
  LOG(INFO) << "Import Time for " << table_name << ": " << (double)ms / 1000.0 << " s";
}

}  // namespace QueryRunner
