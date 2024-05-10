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

/**
 * @file   DBHandler.cpp
 * @brief
 *
 */

#include "DBHandler.h"
#include "DistributedLoader.h"
#include "TokenCompletionHints.h"

#ifdef HAVE_PROFILER
#include <gperftools/heap-profiler.h>
#endif  // HAVE_PROFILER

#include "MapDRelease.h"

#include "Calcite/Calcite.h"
#include "gen-cpp/CalciteServer.h"

#include "QueryEngine/ErrorHandling.h"
#include "QueryEngine/RelAlgExecutor.h"

#include "Catalog/Catalog.h"
#include "Catalog/DdlCommandExecutor.h"
#include "DataMgr/ForeignStorage/ArrowForeignStorage.h"
#include "DataMgr/ForeignStorage/DummyForeignStorage.h"
#include "DataMgr/ForeignStorage/PassThroughBuffer.h"
#include "DistributedHandler.h"
#include "Fragmenter/InsertOrderFragmenter.h"
#include "Geospatial/ColumnNames.h"
#include "Geospatial/Compression.h"
#include "Geospatial/GDAL.h"
#include "Geospatial/Types.h"
#include "ImportExport/Importer.h"
#include "LockMgr/LockMgr.h"
#include "OSDependent/heavyai_hostname.h"
#include "Parser/ParserWrapper.h"
#include "Parser/ReservedKeywords.h"
#include "QueryEngine/ArrowResultSet.h"
#include "QueryEngine/CalciteAdapter.h"
#include "QueryEngine/Execute.h"
#include "QueryEngine/ExtensionFunctionsWhitelist.h"
#include "QueryEngine/GpuMemUtils.h"
#include "QueryEngine/JoinFilterPushDown.h"
#include "QueryEngine/JsonAccessors.h"
#include "QueryEngine/QueryDispatchQueue.h"
#include "QueryEngine/ResultSetBuilder.h"
#include "QueryEngine/TableFunctions/TableFunctionsFactory.h"
#include "QueryEngine/TableOptimizer.h"
#include "QueryEngine/ThriftSerializers.h"
#include "RequestInfo.h"
#ifdef HAVE_RUNTIME_LIBS
#include "RuntimeLibManager/RuntimeLibManager.h"
#endif
#include "Shared/ArrowUtil.h"
#include "Shared/DateTimeParser.h"
#include "Shared/StringTransform.h"
#include "Shared/SysDefinitions.h"
#include "Shared/file_path_util.h"
#include "Shared/heavyai_shared_mutex.h"
#include "Shared/import_helpers.h"
#include "Shared/measure.h"
#include "Shared/misc.h"
#include "Shared/scope.h"
#include "UdfCompiler/UdfCompiler.h"

#ifdef HAVE_AWS_S3
#include <aws/core/auth/AWSCredentialsProviderChain.h>
#endif
#include <fcntl.h>
#include <picosha2.h>
#include <sys/types.h>
#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/make_shared.hpp>
#include <boost/process/search_path.hpp>
#include <boost/program_options.hpp>
#include <boost/tokenizer.hpp>
#include <chrono>
#include <cmath>
#include <csignal>
#include <fstream>
#include <future>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <typeinfo>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>

#include "Shared/ArrowUtil.h"
#include "Shared/distributed.h"

#ifdef ENABLE_IMPORT_PARQUET
extern bool g_enable_parquet_import_fsi;
#endif

#ifdef HAVE_AWS_S3
extern bool g_allow_s3_server_privileges;
#endif

extern bool g_enable_system_tables;
bool g_allow_system_dashboard_update{false};
bool g_uniform_request_ids_per_thrift_call{true};
extern bool g_allow_memory_status_log;

using Catalog_Namespace::Catalog;
using Catalog_Namespace::SysCatalog;

#define INVALID_SESSION_ID ""

#define SET_REQUEST_ID(parent_request_id)                         \
  if (g_uniform_request_ids_per_thrift_call && parent_request_id) \
    logger::set_request_id(parent_request_id);                    \
  else if (logger::set_new_request_id(); parent_request_id)       \
  LOG(INFO) << "This request has parent request_id(" << parent_request_id << ')'

#define THROW_DB_EXCEPTION(errstr) \
  {                                \
    TDBException ex;               \
    ex.error_msg = errstr;         \
    LOG(ERROR) << ex.error_msg;    \
    throw ex;                      \
  }

thread_local std::string TrackingProcessor::client_address;
thread_local ClientProtocol TrackingProcessor::client_protocol;

namespace {

bool dashboard_exists(const Catalog_Namespace::Catalog& cat,
                      const int32_t user_id,
                      const std::string& dashboard_name) {
  return (cat.getMetadataForDashboard(std::to_string(user_id), dashboard_name));
}

struct ForceDisconnect : public std::runtime_error {
  ForceDisconnect(const std::string& cause) : std::runtime_error(cause) {}
};

}  // namespace

#ifdef ENABLE_GEOS
// from Geospatial/GeosValidation.cpp
extern std::unique_ptr<std::string> g_libgeos_so_filename;
#endif

DBHandler::DBHandler(const std::vector<LeafHostInfo>& db_leaves,
                     const std::vector<LeafHostInfo>& string_leaves,
                     const std::string& base_data_path,
                     const bool allow_multifrag,
                     const bool jit_debug,
                     const bool intel_jit_profile,
                     const bool read_only,
                     const bool allow_loop_joins,
                     const bool enable_rendering,
                     const bool renderer_prefer_igpu,
                     const unsigned renderer_vulkan_timeout_ms,
                     const bool renderer_use_parallel_executors,
                     const bool enable_auto_clear_render_mem,
                     const int render_oom_retry_threshold,
                     const size_t render_mem_bytes,
                     const size_t max_concurrent_render_sessions,
                     const size_t reserved_gpu_mem,
                     const bool render_compositor_use_last_gpu,
                     const bool renderer_enable_slab_allocation,
                     const size_t num_reader_threads,
                     const AuthMetadata& authMetadata,
                     SystemParameters& system_parameters,
                     const bool legacy_syntax,
                     const int idle_session_duration,
                     const int max_session_duration,
                     const std::string& udf_filename,
                     const std::string& clang_path,
                     const std::vector<std::string>& clang_options,
#ifdef ENABLE_GEOS
                     const std::string& libgeos_so_filename,
#endif
#ifdef HAVE_TORCH_TFS
                     const std::string& torch_lib_path,
#endif
                     const File_Namespace::DiskCacheConfig& disk_cache_config,
                     const bool is_new_db)
    : leaf_aggregator_(db_leaves)
    , db_leaves_(db_leaves)
    , string_leaves_(string_leaves)
    , base_data_path_(base_data_path)
    , random_gen_(std::random_device{}())
    , session_id_dist_(0, INT32_MAX)
    , jit_debug_(jit_debug)
    , intel_jit_profile_(intel_jit_profile)
    , allow_multifrag_(allow_multifrag)
    , read_only_(read_only)
    , allow_loop_joins_(allow_loop_joins)
    , authMetadata_(authMetadata)
    , system_parameters_(system_parameters)
    , legacy_syntax_(legacy_syntax)
    , dispatch_queue_(
          std::make_unique<QueryDispatchQueue>(system_parameters.num_executors))
    , super_user_rights_(false)
    , idle_session_duration_(idle_session_duration * 60)
    , max_session_duration_(max_session_duration * 60)
    , enable_rendering_(enable_rendering)
    , renderer_prefer_igpu_(renderer_prefer_igpu)
    , renderer_vulkan_timeout_(renderer_vulkan_timeout_ms)
    , renderer_use_parallel_executors_(renderer_use_parallel_executors)
    , enable_auto_clear_render_mem_(enable_auto_clear_render_mem)
    , render_oom_retry_threshold_(render_oom_retry_threshold)
    , render_mem_bytes_(render_mem_bytes)
    , max_concurrent_render_sessions_(max_concurrent_render_sessions)
    , reserved_gpu_mem_(reserved_gpu_mem)
    , render_compositor_use_last_gpu_(render_compositor_use_last_gpu)
    , renderer_enable_slab_allocation_{renderer_enable_slab_allocation}
    , num_reader_threads_(num_reader_threads)
#ifdef ENABLE_GEOS
    , libgeos_so_filename_(libgeos_so_filename)
#endif
#ifdef HAVE_TORCH_TFS
    , torch_lib_path_(torch_lib_path)
#endif
    , disk_cache_config_(disk_cache_config)
    , udf_filename_(udf_filename)
    , clang_path_(clang_path)
    , clang_options_(clang_options)
    , max_num_sessions_(-1) {
  LOG(INFO) << "HeavyDB Server " << MAPD_RELEASE;
  initialize(is_new_db);
  resetSessionsStore();
}

void DBHandler::init_executor_resource_mgr() {
  size_t num_cpu_slots{0};
  size_t num_gpu_slots{0};
  size_t cpu_result_mem{0};
  size_t cpu_buffer_pool_mem{0};
  size_t gpu_buffer_pool_mem{0};
  LOG(INFO) << "Initializing Executor Resource Manager";

  if (g_cpu_threads_override != 0) {
    LOG(INFO) << "\tSetting Executor resource pool avaiable CPU threads/slots to "
                 "user-specified value of "
              << g_cpu_threads_override << ".";
    num_cpu_slots = g_cpu_threads_override;
  } else {
    LOG(INFO) << "\tSetting Executor resource pool avaiable CPU threads/slots to default "
                 "value of "
              << cpu_threads() << ".";
    // Setting the number of CPU slots to cpu_threads() will cause the ExecutorResourceMgr
    // to set the logical number of available cpu slots to mirror the number of threads in
    // the tbb thread pool and used elsewhere in the system, but we may want to consider a
    // capability to allow the executor resource pool number of threads to be set
    // independently as some fraction of the what cpu_threads() will return, to give some
    // breathing room for all the other processes in the system that use CPU threadds
    num_cpu_slots = cpu_threads();
  }
  LOG(INFO) << "\tSetting max per-query CPU threads to ratio of "
            << g_executor_resource_mgr_per_query_max_cpu_slots_ratio << " of "
            << num_cpu_slots << " available threads, or "
            << static_cast<size_t>(g_executor_resource_mgr_per_query_max_cpu_slots_ratio *
                                   num_cpu_slots)
            << " threads.";

  // system_parameters_.num_gpus will be -1 if there are no GPUs enabled so we need to
  // guard against this
  num_gpu_slots = system_parameters_.num_gpus < 0 ? static_cast<size_t>(0)
                                                  : system_parameters_.num_gpus;

  cpu_buffer_pool_mem = data_mgr_->getCpuBufferPoolSize();
  if (g_executor_resource_mgr_cpu_result_mem_bytes != Executor::auto_cpu_mem_bytes) {
    cpu_result_mem = g_executor_resource_mgr_cpu_result_mem_bytes;
  } else {
    const size_t system_mem_bytes = DataMgr::getTotalSystemMemory();
    CHECK_GT(system_mem_bytes, size_t(0));
    const size_t remaining_cpu_mem_bytes = system_mem_bytes >= cpu_buffer_pool_mem
                                               ? system_mem_bytes - cpu_buffer_pool_mem
                                               : 0UL;
    cpu_result_mem =
        std::max(static_cast<size_t>(remaining_cpu_mem_bytes *
                                     g_executor_resource_mgr_cpu_result_mem_ratio),
                 static_cast<size_t>(1UL << 32));
  }
  // Below gets total combined size of all gpu buffer pools
  // Likely will move to per device pool resource management,
  // but keeping simple for now
  gpu_buffer_pool_mem = data_mgr_->getGpuBufferPoolSize();

  // When we move to using the BufferMgrs directly in
  // ExecutorResourcePool, there won't be a need for
  // the buffer_pool_max_occupancy variable - a
  // safety "fudge" factor as what the resource pool sees
  // and what the BufferMgrs see will be exactly the same.

  // However we need to ensure we can quickly access
  // chunk state of BufferMgrs without going through coarse lock
  // before we do this, so use this fudge ratio for now

  // Note that if we are not conservative enough with the below and
  // overshoot, the error will still be caught and if on GPU, the query
  // can be re-run on CPU

  constexpr double buffer_pool_max_occupancy{0.95};
  const size_t conservative_cpu_buffer_pool_mem =
      static_cast<size_t>(cpu_buffer_pool_mem * buffer_pool_max_occupancy);
  const size_t conservative_gpu_buffer_pool_mem =
      static_cast<size_t>(gpu_buffer_pool_mem * buffer_pool_max_occupancy);

  LOG(INFO)
      << "\tSetting Executor resource pool reserved space for CPU buffer pool memory to "
      << format_num_bytes(conservative_cpu_buffer_pool_mem) << ".";
  if (gpu_buffer_pool_mem > 0UL) {
    LOG(INFO) << "\tSetting Executor resource pool reserved space for GPU buffer pool "
                 "memory to "
              << format_num_bytes(conservative_gpu_buffer_pool_mem) << ".";
  }
  LOG(INFO) << "\tSetting Executor resource pool reserved space for CPU result memory to "
            << format_num_bytes(cpu_result_mem) << ".";

  Executor::init_resource_mgr(
      num_cpu_slots,
      num_gpu_slots,
      cpu_result_mem,
      conservative_cpu_buffer_pool_mem,
      conservative_gpu_buffer_pool_mem,
      g_executor_resource_mgr_per_query_max_cpu_slots_ratio,
      g_executor_resource_mgr_per_query_max_cpu_result_mem_ratio,
      g_executor_resource_mgr_allow_cpu_kernel_concurrency,
      g_executor_resource_mgr_allow_cpu_gpu_kernel_concurrency,
      g_executor_resource_mgr_allow_cpu_slot_oversubscription_concurrency,
      g_executor_resource_mgr_allow_cpu_result_mem_oversubscription_concurrency,
      g_executor_resource_mgr_max_available_resource_use_ratio);
}

void DBHandler::validate_configurations() {
#ifndef _WIN32
  size_t temp;
  CHECK(!__builtin_mul_overflow(g_num_tuple_threshold_switch_to_baseline,
                                g_ratio_num_hash_entry_to_num_tuple_switch_to_baseline,
                                &temp))
      << "The product of g_num_tuple_threshold_switch_to_baseline and "
         "g_ratio_num_hash_entry_to_num_tuple_switch_to_baseline exceeds 64 bits.";
#endif
}

void DBHandler::resetSessionsStore() {
  if (sessions_store_) {
    // Disconnect any existing sessions.
    auto sessions = sessions_store_->getAllSessions();
    for (auto session : sessions) {
      sessions_store_->disconnect(session->get_session_id());
    }
  }
  sessions_store_ = Catalog_Namespace::SessionsStore::create(
      base_data_path_,
      1,
      idle_session_duration_,
      max_session_duration_,
      max_num_sessions_,
      [this](auto& session_ptr) { disconnect_impl(session_ptr); });
}

void DBHandler::initialize(const bool is_new_db) {
  if (!initialized_) {
    initialized_ = true;
  } else {
    THROW_DB_EXCEPTION(
        "Server already initialized; service restart required to activate any new "
        "entitlements.");
    return;
  }

  if (system_parameters_.cpu_only || system_parameters_.num_gpus == 0) {
    executor_device_type_ = ExecutorDeviceType::CPU;
    cpu_mode_only_ = true;
  } else {
#ifdef HAVE_CUDA
    executor_device_type_ = ExecutorDeviceType::GPU;
    cpu_mode_only_ = false;
#else
    executor_device_type_ = ExecutorDeviceType::CPU;
    LOG(WARNING) << "This build isn't CUDA enabled, will run on CPU";
    cpu_mode_only_ = true;
#endif
  }

  bool is_rendering_enabled = enable_rendering_;
  if (system_parameters_.num_gpus == 0) {
    is_rendering_enabled = false;
  }

  const auto data_path =
      boost::filesystem::path(base_data_path_) / shared::kDataDirectoryName;
  // calculate the total amount of memory we need to reserve from each gpu that the Buffer
  // manage cannot ask for
  size_t total_reserved = reserved_gpu_mem_;
  if (is_rendering_enabled) {
    total_reserved += render_mem_bytes_;
  }

  std::unique_ptr<CudaMgr_Namespace::CudaMgr> cuda_mgr;
#ifdef HAVE_CUDA
  if (!cpu_mode_only_ || is_rendering_enabled) {
    try {
      cuda_mgr = std::make_unique<CudaMgr_Namespace::CudaMgr>(
          system_parameters_.num_gpus, system_parameters_.start_gpu);
      if (system_parameters_.num_gpus < 0) {
        system_parameters_.num_gpus = cuda_mgr->getDeviceCount();
      } else {
        system_parameters_.num_gpus =
            std::min(system_parameters_.num_gpus, cuda_mgr->getDeviceCount());
      }
    } catch (const std::exception& e) {
      LOG(ERROR) << "Unable to instantiate CudaMgr, falling back to CPU-only mode. "
                 << e.what();
      executor_device_type_ = ExecutorDeviceType::CPU;
      cpu_mode_only_ = true;
      is_rendering_enabled = false;
    }
  }
#endif  // HAVE_CUDA

  validate_configurations();

  try {
    data_mgr_.reset(new Data_Namespace::DataMgr(data_path.string(),
                                                system_parameters_,
                                                std::move(cuda_mgr),
                                                !cpu_mode_only_,
                                                total_reserved,
                                                num_reader_threads_,
                                                disk_cache_config_));
  } catch (const std::exception& e) {
    LOG(FATAL) << "Failed to initialize data manager: " << e.what();
  }
  if (g_enable_executor_resource_mgr) {
    init_executor_resource_mgr();
  }

  std::string udf_ast_filename("");

  try {
    if (!udf_filename_.empty()) {
      const auto cuda_mgr = data_mgr_->getCudaMgr();
      const CudaMgr_Namespace::NvidiaDeviceArch device_arch =
          cuda_mgr ? cuda_mgr->getDeviceArch()
                   : CudaMgr_Namespace::NvidiaDeviceArch::Kepler;
      UdfCompiler compiler(device_arch, clang_path_, clang_options_);

      const auto [cpu_udf_ir_file, cuda_udf_ir_file] = compiler.compileUdf(udf_filename_);
      Executor::addUdfIrToModule(cpu_udf_ir_file, /*is_cuda_ir=*/false);
      if (!cuda_udf_ir_file.empty()) {
        Executor::addUdfIrToModule(cuda_udf_ir_file, /*is_cuda_ir=*/true);
      }
      udf_ast_filename = compiler.getAstFileName(udf_filename_);
    }
  } catch (const std::exception& e) {
    LOG(FATAL) << "Failed to initialize UDF compiler: " << e.what();
  }

  try {
    calcite_ =
        std::make_shared<Calcite>(system_parameters_, base_data_path_, udf_ast_filename);
  } catch (const std::exception& e) {
    LOG(FATAL) << "Failed to initialize Calcite server: " << e.what();
  }

  try {
    ExtensionFunctionsWhitelist::add(calcite_->getExtensionFunctionWhitelist());
    if (!udf_filename_.empty()) {
      ExtensionFunctionsWhitelist::addUdfs(calcite_->getUserDefinedFunctionWhitelist());
    }
  } catch (const std::exception& e) {
    LOG(FATAL) << "Failed to initialize extension functions: " << e.what();
  }

  try {
    table_functions::init_table_functions();
  } catch (const std::exception& e) {
    LOG(FATAL) << "Failed to initialize table functions factory: " << e.what();
  }

#ifdef HAVE_RUNTIME_LIBS
  try {
#ifdef HAVE_TORCH_TFS
    RuntimeLibManager::loadRuntimeLibs(torch_lib_path_);
#else
    RuntimeLibManager::loadRuntimeLibs();
#endif
  } catch (const std::exception& e) {
    LOG(ERROR) << "Failed to load runtime libraries: " << e.what();
    LOG(ERROR) << "Support for runtime library table functions is disabled.";
  }
#endif

  try {
    auto udtfs = ThriftSerializers::to_thrift(
        table_functions::TableFunctionsFactory::get_table_funcs(/*is_runtime=*/false));
    std::vector<TUserDefinedFunction> udfs = {};
    calcite_->setRuntimeExtensionFunctions(udfs, udtfs, /*is_runtime=*/false);
  } catch (const std::exception& e) {
    LOG(FATAL) << "Failed to register compile-time table functions: " << e.what();
  }

  if (!data_mgr_->gpusPresent() && !cpu_mode_only_) {
    executor_device_type_ = ExecutorDeviceType::CPU;
    LOG(ERROR) << "No GPUs detected, falling back to CPU mode";
    cpu_mode_only_ = true;
  }

  LOG(INFO) << "Started in " << executor_device_type_ << " mode.";

  try {
    g_base_path = base_data_path_;
    SysCatalog::instance().init(base_data_path_,
                                data_mgr_,
                                authMetadata_,
                                calcite_,
                                is_new_db,
                                !db_leaves_.empty(),
                                string_leaves_);
  } catch (const std::exception& e) {
    LOG(FATAL) << "Failed to initialize system catalog: " << e.what();
  }

  import_path_ = boost::filesystem::path(base_data_path_) / shared::kDefaultImportDirName;
  start_time_ = std::time(nullptr);

  if (is_rendering_enabled) {
    try {
      render_handler_.reset(new RenderHandler(this,
                                              render_mem_bytes_,
                                              max_concurrent_render_sessions_,
                                              render_compositor_use_last_gpu_,
                                              false,
                                              false,
                                              renderer_prefer_igpu_,
                                              renderer_vulkan_timeout_,
                                              renderer_use_parallel_executors_,
                                              system_parameters_,
                                              renderer_enable_slab_allocation_));
    } catch (const std::exception& e) {
      LOG(ERROR) << "Backend rendering disabled: " << e.what();
    }
  }

  query_engine_ = QueryEngine::createInstance(data_mgr_->getCudaMgr(), cpu_mode_only_);

#ifdef ENABLE_GEOS
  if (!libgeos_so_filename_.empty()) {
    g_libgeos_so_filename.reset(new std::string(libgeos_so_filename_));
    LOG(INFO) << "Overriding default geos library with '" + *g_libgeos_so_filename + "'";
  }
#endif
}

DBHandler::~DBHandler() {
  shutdown();
}

void DBHandler::check_read_only(const std::string& str) {
  if (DBHandler::read_only_) {
    THROW_DB_EXCEPTION(str + " disabled: server running in read-only mode.");
  }
}

std::string const DBHandler::createInMemoryCalciteSession(
    const std::shared_ptr<Catalog_Namespace::Catalog>& catalog_ptr) {
  // We would create an in memory session for calcite with super user privileges which
  // would be used for getting all tables metadata when a user runs the query. The
  // session would be under the name of a proxy user/password which would only persist
  // till server's lifetime or execution of calcite query(in memory) whichever is the
  // earliest.
  heavyai::lock_guard<heavyai::shared_mutex> lg(calcite_sessions_mtx_);
  std::string session_id;
  do {
    session_id = generate_random_string(Catalog_Namespace::CALCITE_SESSION_ID_LENGTH);
  } while (calcite_sessions_.find(session_id) != calcite_sessions_.end());
  Catalog_Namespace::UserMetadata user_meta(-1,
                                            calcite_->getInternalSessionProxyUserName(),
                                            calcite_->getInternalSessionProxyPassword(),
                                            true,
                                            -1,
                                            true,
                                            false);
  const auto emplace_ret = calcite_sessions_.emplace(
      session_id,
      std::make_shared<Catalog_Namespace::SessionInfo>(
          catalog_ptr, user_meta, executor_device_type_, session_id));
  CHECK(emplace_ret.second);
  return session_id;
}

void DBHandler::removeInMemoryCalciteSession(const std::string& session_id) {
  // Remove InMemory calcite Session.
  heavyai::lock_guard<heavyai::shared_mutex> lg(calcite_sessions_mtx_);
  CHECK(calcite_sessions_.erase(session_id)) << session_id;
}

// internal connection for connections with no password
void DBHandler::internal_connect(TSessionId& session_id,
                                 const std::string& username,
                                 const std::string& dbname) {
  logger::set_new_request_id();
  auto stdlog = STDLOG();            // session_id set by connect_impl()
  std::string username2 = username;  // login() may reset username given as argument
  std::string dbname2 = dbname;      // login() may reset dbname given as argument
  Catalog_Namespace::UserMetadata user_meta;
  std::shared_ptr<Catalog> cat = nullptr;
  try {
    cat =
        SysCatalog::instance().login(dbname2, username2, std::string(), user_meta, false);
  } catch (std::exception& e) {
    THROW_DB_EXCEPTION(e.what());
  }

  DBObject dbObject(dbname2, DatabaseDBObjectType);
  dbObject.loadKey(*cat);
  dbObject.setPrivileges(AccessPrivileges::ACCESS);
  std::vector<DBObject> dbObjects;
  dbObjects.push_back(dbObject);
  if (!SysCatalog::instance().checkPrivileges(user_meta, dbObjects)) {
    THROW_DB_EXCEPTION("Unauthorized Access: user " + user_meta.userLoggable() +
                       " is not allowed to access database " + dbname2 + ".");
  }
  connect_impl(session_id, std::string(), dbname2, user_meta, cat, stdlog);
}

bool DBHandler::isAggregator() const {
  return leaf_aggregator_.leafCount() > 0;
}

void DBHandler::krb5_connect(TKrb5Session& session,
                             const std::string& inputToken,
                             const std::string& dbname) {
  THROW_DB_EXCEPTION("Unauthrorized Access. Kerberos login not supported");
}

void DBHandler::connect(TSessionId& session_id,
                        const std::string& username,
                        const std::string& passwd,
                        const std::string& dbname) {
  logger::set_new_request_id();
  auto stdlog = STDLOG();  // session_info set by connect_impl()
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  std::string username2 = username;  // login() may reset username given as argument
  std::string dbname2 = dbname;      // login() may reset dbname given as argument
  Catalog_Namespace::UserMetadata user_meta;
  std::shared_ptr<Catalog> cat = nullptr;
  try {
    cat = SysCatalog::instance().login(
        dbname2, username2, passwd, user_meta, !super_user_rights_);
  } catch (std::exception& e) {
    stdlog.appendNameValuePairs("user", username, "db", dbname, "exception", e.what());
    THROW_DB_EXCEPTION(e.what());
  }

  DBObject dbObject(dbname2, DatabaseDBObjectType);
  dbObject.loadKey(*cat);
  dbObject.setPrivileges(AccessPrivileges::ACCESS);
  std::vector<DBObject> dbObjects;
  dbObjects.push_back(dbObject);
  if (!SysCatalog::instance().checkPrivileges(user_meta, dbObjects)) {
    stdlog.appendNameValuePairs(
        "user", username, "db", dbname, "exception", "Missing Privileges");
    THROW_DB_EXCEPTION("Unauthorized Access: user " + user_meta.userLoggable() +
                       " is not allowed to access database " + dbname2 + ".");
  }
  connect_impl(session_id, passwd, dbname2, user_meta, cat, stdlog);

  // if pki auth session_id will come back encrypted with user pubkey
  SysCatalog::instance().check_for_session_encryption(passwd, session_id);
}

void DBHandler::connect_impl(TSessionId& session_id,
                             const std::string& passwd,
                             const std::string& dbname,
                             const Catalog_Namespace::UserMetadata& user_meta,
                             std::shared_ptr<Catalog> cat,
                             query_state::StdLog& stdlog) {
  // TODO(sy): Is there any reason to have dbname as a parameter
  // here when the cat parameter already provides cat->name()?
  // Should dbname and cat->name() ever differ?
  auto session_ptr = sessions_store_->add(user_meta, cat, executor_device_type_);
  session_id = session_ptr->get_session_id();
  LOG(INFO) << "User " << user_meta.userLoggable() << " connected to database " << dbname;
  stdlog.setSessionInfo(session_ptr);
  session_ptr->set_connection_info(getConnectionInfo().toString());
  if (!super_user_rights_) {  // no need to connect to leaf_aggregator_ at this time
    // while doing warmup
  }
  auto const roles =
      stdlog.getConstSessionInfo()->get_currentUser().isSuper
          ? std::vector<std::string>{{"super"}}
          : SysCatalog::instance().getRoles(
                false, false, stdlog.getConstSessionInfo()->get_currentUser().userName);
  stdlog.appendNameValuePairs("roles", boost::algorithm::join(roles, ","));
}

void DBHandler::disconnect(const TSessionId& session_id_or_json) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  auto session_ptr = get_session_ptr(request_info.sessionId());
  auto stdlog = STDLOG(session_ptr, "client", getConnectionInfo().toString());
  sessions_store_->disconnect(request_info.sessionId());
}

void DBHandler::disconnect_impl(Catalog_Namespace::SessionInfoPtr& session_ptr) {
  const auto session_id = session_ptr->get_session_id();
  std::exception_ptr leaf_exception = nullptr;
  try {
    if (leaf_aggregator_.leafCount() > 0) {
      leaf_aggregator_.disconnect(session_id);
    }
  } catch (...) {
    leaf_exception = std::current_exception();
  }

  if (render_handler_) {
    render_handler_->disconnect(session_id);
  }

  if (leaf_exception) {
    std::rethrow_exception(leaf_exception);
  }
}

void DBHandler::switch_database(const TSessionId& session_id_or_json,
                                const std::string& dbname) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  auto session_ptr = get_session_ptr(request_info.sessionId());
  auto stdlog = STDLOG(session_ptr);
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  std::string dbname2 = dbname;  // switchDatabase() may reset dbname given as argument
  try {
    std::shared_ptr<Catalog> cat = SysCatalog::instance().switchDatabase(
        dbname2, session_ptr->get_currentUser().userName);
    session_ptr->set_catalog_ptr(cat);
    if (leaf_aggregator_.leafCount() > 0) {
      leaf_aggregator_.switch_database(request_info.sessionId(), dbname);
      return;
    }
  } catch (std::exception& e) {
    THROW_DB_EXCEPTION(e.what());
  }
}

void DBHandler::clone_session(TSessionId& session2_id,
                              const TSessionId& session1_id_or_json) {
  heavyai::RequestInfo const request_info(session1_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  auto session1_ptr = get_session_ptr(request_info.sessionId());
  auto stdlog = STDLOG(session1_ptr);
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());

  try {
    const Catalog_Namespace::UserMetadata& user_meta = session1_ptr->get_currentUser();
    std::shared_ptr<Catalog> cat = session1_ptr->get_catalog_ptr();
    auto session2_ptr = sessions_store_->add(user_meta, cat, executor_device_type_);
    session2_id = session2_ptr->get_session_id();
    LOG(INFO) << "User " << user_meta.userLoggable() << " connected to database "
              << cat->name();
    if (leaf_aggregator_.leafCount() > 0) {
      leaf_aggregator_.clone_session(request_info.sessionId(), session2_id);
      return;
    }
  } catch (std::exception& e) {
    THROW_DB_EXCEPTION(e.what());
  }
}

void DBHandler::interrupt(const TSessionId& query_session_id_or_json,
                          const TSessionId& interrupt_session_id_or_json) {
  // if this is for distributed setting, query_session becomes a parent session (agg)
  // and the interrupt session is one of existing session in the leaf node (leaf)
  // so we can think there exists a logical mapping
  // between query_session (agg) and interrupt_session (leaf)
  heavyai::RequestInfo const query_request_info(query_session_id_or_json);
  heavyai::RequestInfo const interrupt_request_info(interrupt_session_id_or_json);
  SET_REQUEST_ID(interrupt_request_info.requestId());
  auto session_ptr = get_session_ptr(interrupt_request_info.sessionId());
  auto& cat = session_ptr->getCatalog();
  auto stdlog = STDLOG(session_ptr);
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  const auto allow_query_interrupt =
      g_enable_runtime_query_interrupt || g_enable_non_kernel_time_query_interrupt;
  if (g_enable_dynamic_watchdog || allow_query_interrupt) {
    const auto dbname = cat.getCurrentDB().dbName;
    auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID,
                                          jit_debug_ ? "/tmp" : "",
                                          jit_debug_ ? "mapdquery" : "",
                                          system_parameters_);
    CHECK(executor);

    if (leaf_aggregator_.leafCount() > 0) {
      leaf_aggregator_.interrupt(query_request_info.sessionId(),
                                 interrupt_request_info.sessionId());
    }
    auto target_executor_ids =
        executor->getExecutorIdsRunningQuery(query_request_info.sessionId());
    if (target_executor_ids.empty()) {
      heavyai::shared_lock<heavyai::shared_mutex> session_read_lock(
          executor->getSessionLock());
      if (executor->checkIsQuerySessionEnrolled(query_request_info.sessionId(),
                                                session_read_lock)) {
        session_read_lock.unlock();
        VLOG(1) << "Received interrupt: "
                << "User " << session_ptr->get_currentUser().userLoggable()
                << ", Database " << dbname << std::endl;
        executor->interrupt(query_request_info.sessionId(),
                            interrupt_request_info.sessionId());
      }
    } else {
      for (auto& executor_id : target_executor_ids) {
        VLOG(1) << "Received interrupt: "
                << "Executor " << executor_id << ", User "
                << session_ptr->get_currentUser().userLoggable() << ", Database "
                << dbname << std::endl;
        auto target_executor = Executor::getExecutor(executor_id);
        target_executor->interrupt(query_request_info.sessionId(),
                                   interrupt_request_info.sessionId());
      }
    }

    LOG(INFO) << "User " << session_ptr->get_currentUser().userName
              << " interrupted session with database " << dbname << std::endl;
  }
}

TRole::type DBHandler::getServerRole() const {
  if (g_cluster) {
    if (leaf_aggregator_.leafCount() > 0) {
      return TRole::type::AGGREGATOR;
    }
    return TRole::type::LEAF;
  }
  return TRole::type::SERVER;
}
void DBHandler::get_server_status(TServerStatus& _return,
                                  const TSessionId& session_id_or_json) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  auto stdlog = STDLOG(get_session_ptr(request_info.sessionId()));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  const auto rendering_enabled = bool(render_handler_);
  _return.read_only = read_only_;
  _return.version = MAPD_RELEASE;
  _return.rendering_enabled = rendering_enabled;
  _return.start_time = start_time_;
  _return.edition = MAPD_EDITION;
  _return.host_name = heavyai::get_hostname();
  _return.poly_rendering_enabled = rendering_enabled;
  _return.role = getServerRole();
  _return.renderer_status_json =
      render_handler_ ? render_handler_->get_renderer_status_json() : "";
}

void DBHandler::get_status(std::vector<TServerStatus>& _return,
                           const TSessionId& session_id_or_json) {
  //
  // get_status() is now called locally at startup on the aggregator
  // in order to validate that all nodes of a cluster are running the
  // same software version and the same renderer status
  //
  // In that context, it is called with the InvalidSessionID, and
  // with the local super-user flag set.
  //
  // Hence, we allow this session-less mode only in distributed mode, and
  // then on a leaf (always), or on the aggregator (only in super-user mode)
  //
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  auto const allow_invalid_session = g_cluster && (!isAggregator() || super_user_rights_);

  if (!allow_invalid_session || request_info.sessionId() != getInvalidSessionId()) {
    auto stdlog = STDLOG(get_session_ptr(request_info.sessionId()));
    stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  } else {
    LOG(INFO) << "get_status() called in session-less mode";
  }
  const auto rendering_enabled = bool(render_handler_);
  TServerStatus ret;
  ret.read_only = read_only_;
  ret.version = MAPD_RELEASE;
  ret.rendering_enabled = rendering_enabled;
  ret.start_time = start_time_;
  ret.edition = MAPD_EDITION;
  ret.host_name = heavyai::get_hostname();
  ret.poly_rendering_enabled = rendering_enabled;
  ret.role = getServerRole();
  ret.renderer_status_json =
      render_handler_ ? render_handler_->get_renderer_status_json() : "";
  ret.host_id = "";

  _return.push_back(ret);
  if (leaf_aggregator_.leafCount() > 0) {
    std::vector<TServerStatus> leaf_status =
        leaf_aggregator_.getLeafStatus(request_info.sessionId());
    _return.insert(_return.end(), leaf_status.begin(), leaf_status.end());
  }
}

void DBHandler::get_hardware_info(TClusterHardwareInfo& _return,
                                  const TSessionId& session_id_or_json) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  auto stdlog = STDLOG(get_session_ptr(request_info.sessionId()));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  THardwareInfo ret;
  const auto cuda_mgr = data_mgr_->getCudaMgr();
  if (cuda_mgr) {
    ret.num_gpu_hw = cuda_mgr->getDeviceCount();
    ret.start_gpu = cuda_mgr->getStartGpu();
    if (ret.start_gpu >= 0) {
      ret.num_gpu_allocated = cuda_mgr->getDeviceCount() - cuda_mgr->getStartGpu();
      // ^ This will break as soon as we allow non contiguous GPU allocations to MapD
    }
    for (int16_t device_id = 0; device_id < ret.num_gpu_hw; device_id++) {
      TGpuSpecification gpu_spec;
      auto deviceProperties = cuda_mgr->getDeviceProperties(device_id);
      gpu_spec.num_sm = deviceProperties->numMPs;
      gpu_spec.clock_frequency_kHz = deviceProperties->clockKhz;
      gpu_spec.memory = deviceProperties->globalMem;
      gpu_spec.compute_capability_major = deviceProperties->computeMajor;
      gpu_spec.compute_capability_minor = deviceProperties->computeMinor;
      ret.gpu_info.push_back(gpu_spec);
    }
  }

  // start  hardware/OS dependent code
  ret.num_cpu_hw = std::thread::hardware_concurrency();
  // ^ This might return diffrent results in case of hyper threading
  // end hardware/OS dependent code

  _return.hardware_info.push_back(ret);
}

void DBHandler::get_session_info(TSessionInfo& _return,
                                 const TSessionId& session_id_or_json) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  auto session_ptr = get_session_ptr(request_info.sessionId());
  CHECK(session_ptr);
  auto stdlog = STDLOG(session_ptr);
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  auto user_metadata = session_ptr->get_currentUser();
  _return.user = user_metadata.userName;
  _return.database = session_ptr->getCatalog().getCurrentDB().dbName;
  _return.start_time = session_ptr->get_start_time();
  _return.is_super = user_metadata.isSuper;
}

void DBHandler::set_leaf_info(const TSessionId& session, const TLeafInfo& info) {
  g_distributed_leaf_idx = info.leaf_id;
  g_distributed_num_leaves = info.num_leaves;
}

void DBHandler::value_to_thrift_column(const TargetValue& tv,
                                       const SQLTypeInfo& ti,
                                       TColumn& column) {
  if (ti.is_array()) {
    CHECK(!ti.get_elem_type().get_notnull())
        << "element types of arrays should always be nullable";
    TColumn tColumn;
    const auto array_tv = boost::get<ArrayTargetValue>(&tv);
    CHECK(array_tv);
    bool is_null = !array_tv->is_initialized();
    if (!is_null) {
      const auto& vec = array_tv->get();
      for (const auto& elem_tv : vec) {
        value_to_thrift_column(elem_tv, ti.get_elem_type(), tColumn);
      }
    }
    column.data.arr_col.push_back(tColumn);
    column.nulls.push_back(is_null && !ti.get_notnull());
  } else if (ti.is_geometry()) {
    const auto scalar_tv = boost::get<ScalarTargetValue>(&tv);
    if (scalar_tv) {
      auto s_n = boost::get<NullableString>(scalar_tv);
      auto s = boost::get<std::string>(s_n);
      if (s) {
        column.data.str_col.push_back(*s);
      } else {
        column.data.str_col.emplace_back("");  // null string
        auto null_p = boost::get<void*>(s_n);
        CHECK(null_p && !*null_p);
      }
      column.nulls.push_back(!s && !ti.get_notnull());
    } else {
      const auto array_tv = boost::get<ArrayTargetValue>(&tv);
      CHECK(array_tv);
      bool is_null = !array_tv->is_initialized();
      if (!is_null) {
        auto elem_type = SQLTypeInfo(kDOUBLE, false);
        TColumn tColumn;
        const auto& vec = array_tv->get();
        for (const auto& elem_tv : vec) {
          value_to_thrift_column(elem_tv, elem_type, tColumn);
        }
        column.data.arr_col.push_back(tColumn);
        column.nulls.push_back(false);
      } else {
        TColumn tColumn;
        column.data.arr_col.push_back(tColumn);
        column.nulls.push_back(is_null && !ti.get_notnull());
      }
    }
  } else {
    CHECK(!ti.is_column());
    const auto scalar_tv = boost::get<ScalarTargetValue>(&tv);
    CHECK(scalar_tv);
    if (boost::get<int64_t>(scalar_tv)) {
      int64_t data = *(boost::get<int64_t>(scalar_tv));

      if (ti.is_decimal()) {
        double val = static_cast<double>(data);
        if (ti.get_scale() > 0) {
          val /= pow(10.0, std::abs(ti.get_scale()));
        }
        column.data.real_col.push_back(val);
      } else {
        column.data.int_col.push_back(data);
      }

      switch (ti.get_type()) {
        case kBOOLEAN:
          column.nulls.push_back(data == NULL_BOOLEAN && !ti.get_notnull());
          break;
        case kTINYINT:
          column.nulls.push_back(data == NULL_TINYINT && !ti.get_notnull());
          break;
        case kSMALLINT:
          column.nulls.push_back(data == NULL_SMALLINT && !ti.get_notnull());
          break;
        case kINT:
          column.nulls.push_back(data == NULL_INT && !ti.get_notnull());
          break;
        case kNUMERIC:
        case kDECIMAL:
        case kBIGINT:
          column.nulls.push_back(data == NULL_BIGINT && !ti.get_notnull());
          break;
        case kTIME:
        case kTIMESTAMP:
        case kDATE:
        case kINTERVAL_DAY_TIME:
        case kINTERVAL_YEAR_MONTH:
          column.nulls.push_back(data == NULL_BIGINT && !ti.get_notnull());
          break;
        default:
          column.nulls.push_back(false);
      }
    } else if (boost::get<double>(scalar_tv)) {
      double data = *(boost::get<double>(scalar_tv));
      column.data.real_col.push_back(data);
      if (ti.get_type() == kFLOAT) {
        column.nulls.push_back(data == NULL_FLOAT && !ti.get_notnull());
      } else {
        column.nulls.push_back(data == NULL_DOUBLE && !ti.get_notnull());
      }
    } else if (boost::get<float>(scalar_tv)) {
      CHECK_EQ(kFLOAT, ti.get_type());
      float data = *(boost::get<float>(scalar_tv));
      column.data.real_col.push_back(data);
      column.nulls.push_back(data == NULL_FLOAT && !ti.get_notnull());
    } else if (boost::get<NullableString>(scalar_tv)) {
      auto s_n = boost::get<NullableString>(scalar_tv);
      auto s = boost::get<std::string>(s_n);
      if (s) {
        column.data.str_col.push_back(*s);
      } else {
        column.data.str_col.emplace_back("");  // null string
        auto null_p = boost::get<void*>(s_n);
        CHECK(null_p && !*null_p);
      }
      column.nulls.push_back(!s && !ti.get_notnull());
    } else {
      CHECK(false);
    }
  }
}

TDatum DBHandler::value_to_thrift(const TargetValue& tv, const SQLTypeInfo& ti) {
  TDatum datum;
  const auto scalar_tv = boost::get<ScalarTargetValue>(&tv);
  if (!scalar_tv) {
    CHECK(ti.is_array());
    CHECK(!ti.get_elem_type().get_notnull())
        << "element types of arrays should always be nullable";
    const auto array_tv = boost::get<ArrayTargetValue>(&tv);
    CHECK(array_tv);
    if (array_tv->is_initialized()) {
      const auto& vec = array_tv->get();
      for (const auto& elem_tv : vec) {
        const auto scalar_col_val = value_to_thrift(elem_tv, ti.get_elem_type());
        datum.val.arr_val.push_back(scalar_col_val);
      }
      // Datum is not null, at worst it's an empty array Datum
      datum.is_null = false;
    } else {
      datum.is_null = true;
    }
    return datum;
  }
  if (boost::get<int64_t>(scalar_tv)) {
    int64_t data = *(boost::get<int64_t>(scalar_tv));

    if (ti.is_decimal()) {
      double val = static_cast<double>(data);
      if (ti.get_scale() > 0) {
        val /= pow(10.0, std::abs(ti.get_scale()));
      }
      datum.val.real_val = val;
    } else {
      datum.val.int_val = data;
    }

    switch (ti.get_type()) {
      case kBOOLEAN:
        datum.is_null = (datum.val.int_val == NULL_BOOLEAN);
        break;
      case kTINYINT:
        datum.is_null = (datum.val.int_val == NULL_TINYINT);
        break;
      case kSMALLINT:
        datum.is_null = (datum.val.int_val == NULL_SMALLINT);
        break;
      case kINT:
        datum.is_null = (datum.val.int_val == NULL_INT);
        break;
      case kDECIMAL:
      case kNUMERIC:
      case kBIGINT:
        datum.is_null = (datum.val.int_val == NULL_BIGINT);
        break;
      case kTIME:
      case kTIMESTAMP:
      case kDATE:
      case kINTERVAL_DAY_TIME:
      case kINTERVAL_YEAR_MONTH:
        datum.is_null = (datum.val.int_val == NULL_BIGINT);
        break;
      default:
        datum.is_null = false;
    }
  } else if (boost::get<double>(scalar_tv)) {
    datum.val.real_val = *(boost::get<double>(scalar_tv));
    if (ti.get_type() == kFLOAT) {
      datum.is_null = (datum.val.real_val == NULL_FLOAT);
    } else {
      datum.is_null = (datum.val.real_val == NULL_DOUBLE);
    }
  } else if (boost::get<float>(scalar_tv)) {
    CHECK_EQ(kFLOAT, ti.get_type());
    datum.val.real_val = *(boost::get<float>(scalar_tv));
    datum.is_null = (datum.val.real_val == NULL_FLOAT);
  } else if (boost::get<NullableString>(scalar_tv)) {
    auto s_n = boost::get<NullableString>(scalar_tv);
    auto s = boost::get<std::string>(s_n);
    if (s) {
      datum.val.str_val = *s;
    } else {
      auto null_p = boost::get<void*>(s_n);
      CHECK(null_p && !*null_p);
    }
    datum.is_null = !s;
  } else {
    CHECK(false);
  }
  return datum;
}

void DBHandler::sql_execute_local(
    TQueryResult& _return,
    const QueryStateProxy& query_state_proxy,
    const std::shared_ptr<Catalog_Namespace::SessionInfo> session_ptr,
    const std::string& query_str,
    const bool column_format,
    const std::string& nonce,
    const int32_t first_n,
    const int32_t at_most_n,
    const bool use_calcite) {
  _return.total_time_ms = 0;
  _return.nonce = nonce;
  ParserWrapper pw{query_str};
  switch (pw.getQueryType()) {
    case ParserWrapper::QueryType::Read: {
      _return.query_type = TQueryType::READ;
      VLOG(1) << "query type: READ";
      break;
    }
    case ParserWrapper::QueryType::Write: {
      _return.query_type = TQueryType::WRITE;
      VLOG(1) << "query type: WRITE";
      break;
    }
    case ParserWrapper::QueryType::SchemaRead: {
      _return.query_type = TQueryType::SCHEMA_READ;
      VLOG(1) << "query type: SCHEMA READ";
      break;
    }
    case ParserWrapper::QueryType::SchemaWrite: {
      _return.query_type = TQueryType::SCHEMA_WRITE;
      VLOG(1) << "query type: SCHEMA WRITE";
      break;
    }
    default: {
      _return.query_type = TQueryType::UNKNOWN;
      LOG(WARNING) << "query type: UNKNOWN";
      break;
    }
  }

  ExecutionResult result;
  lockmgr::LockedTableDescriptors locks;
  _return.total_time_ms += measure<>::execution([&]() {
    DBHandler::sql_execute_impl(result,
                                query_state_proxy,
                                column_format,
                                session_ptr->get_executor_device_type(),
                                first_n,
                                at_most_n,
                                use_calcite,
                                locks);
    DBHandler::convertData(
        _return, result, query_state_proxy, column_format, first_n, at_most_n);
  });
}

void DBHandler::convertData(TQueryResult& _return,
                            ExecutionResult& result,
                            const QueryStateProxy& query_state_proxy,
                            const bool column_format,
                            const int32_t first_n,
                            const int32_t at_most_n) {
  _return.execution_time_ms += result.getExecutionTime();
  if (result.empty()) {
    return;
  }

  switch (result.getResultType()) {
    case ExecutionResult::QueryResult:
      convertRows(_return,
                  query_state_proxy,
                  result.getTargetsMeta(),
                  *result.getRows(),
                  column_format,
                  first_n,
                  at_most_n);
      break;
    case ExecutionResult::SimpleResult:
      convertResult(_return, *result.getRows(), true);
      break;
    case ExecutionResult::Explanation:
      convertExplain(_return, *result.getRows(), true);
      break;
    case ExecutionResult::CalciteDdl:
      convertRows(_return,
                  query_state_proxy,
                  result.getTargetsMeta(),
                  *result.getRows(),
                  column_format,
                  -1,
                  -1);
      break;
  }
}

void DBHandler::sql_execute(TQueryResult& _return,
                            const TSessionId& session_id_or_json,
                            const std::string& query_str,
                            const bool column_format,
                            const std::string& nonce,
                            const int32_t first_n,
                            const int32_t at_most_n) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  const std::string exec_ra_prefix = "execute relalg";
  const bool use_calcite = !boost::starts_with(query_str, exec_ra_prefix);
  auto actual_query =
      use_calcite ? query_str : boost::trim_copy(query_str.substr(exec_ra_prefix.size()));
  auto session_ptr = get_session_ptr(request_info.sessionId());
  auto query_state = create_query_state(session_ptr, actual_query);
  auto stdlog = STDLOG(session_ptr, query_state);
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  stdlog.appendNameValuePairs("nonce", nonce);
  auto timer = DEBUG_TIMER(__func__);
  try {
    ScopeGuard reset_was_deferred_copy_from = [this, &session_ptr] {
      deferred_copy_from_sessions.remove(session_ptr->get_session_id());
    };

    if (first_n >= 0 && at_most_n >= 0) {
      THROW_DB_EXCEPTION(std::string("At most one of first_n and at_most_n can be set"));
    }

    if (leaf_aggregator_.leafCount() > 0) {
      if (!agg_handler_) {
        THROW_DB_EXCEPTION("Distributed support is disabled.");
      }
      _return.total_time_ms = measure<>::execution([&]() {
        agg_handler_->cluster_execute(_return,
                                      query_state->createQueryStateProxy(),
                                      query_state->getQueryStr(),
                                      column_format,
                                      nonce,
                                      first_n,
                                      at_most_n,
                                      system_parameters_);
      });
      _return.nonce = nonce;
    } else {
      sql_execute_local(_return,
                        query_state->createQueryStateProxy(),
                        session_ptr,
                        actual_query,
                        column_format,
                        nonce,
                        first_n,
                        at_most_n,
                        use_calcite);
    }
    _return.total_time_ms += process_deferred_copy_from(request_info.sessionId());
    std::string debug_json = timer.stopAndGetJson();
    if (!debug_json.empty()) {
      _return.__set_debug(std::move(debug_json));
    }
    stdlog.appendNameValuePairs(
        "execution_time_ms",
        _return.execution_time_ms,
        "total_time_ms",  // BE-3420 - Redundant with duration field
        stdlog.duration<std::chrono::milliseconds>());
    VLOG(1) << "Table Schema Locks:\n" << lockmgr::TableSchemaLockMgr::instance();
    VLOG(1) << "Table Data Locks:\n" << lockmgr::TableDataLockMgr::instance();
  } catch (const std::exception& e) {
    if (strstr(e.what(), "java.lang.NullPointerException")) {
      THROW_DB_EXCEPTION("query failed from broken view or other schema related issue");
    } else if (strstr(e.what(), "SQL Error: Encountered \";\"")) {
      THROW_DB_EXCEPTION("multiple SQL statements not allowed");
    } else if (strstr(e.what(), "SQL Error: Encountered \"<EOF>\" at line 0, column 0")) {
      THROW_DB_EXCEPTION("empty SQL statment not allowed");
    } else {
      THROW_DB_EXCEPTION(e.what());
    }
  }
}

void DBHandler::sql_execute(ExecutionResult& _return,
                            const TSessionId& session_id_or_json,
                            const std::string& query_str,
                            const bool column_format,
                            const int32_t first_n,
                            const int32_t at_most_n,
                            lockmgr::LockedTableDescriptors& locks) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  const std::string exec_ra_prefix = "execute relalg";
  const bool use_calcite = !boost::starts_with(query_str, exec_ra_prefix);
  auto actual_query =
      use_calcite ? query_str : boost::trim_copy(query_str.substr(exec_ra_prefix.size()));

  auto session_ptr = get_session_ptr(request_info.sessionId());
  CHECK(session_ptr);
  auto query_state = create_query_state(session_ptr, actual_query);
  auto stdlog = STDLOG(session_ptr, query_state);
  auto timer = DEBUG_TIMER(__func__);

  try {
    ScopeGuard reset_was_deferred_copy_from = [this, &session_ptr] {
      deferred_copy_from_sessions.remove(session_ptr->get_session_id());
    };

    if (first_n >= 0 && at_most_n >= 0) {
      THROW_DB_EXCEPTION(std::string("At most one of first_n and at_most_n can be set"));
    }
    auto total_time_ms = measure<>::execution([&]() {
      DBHandler::sql_execute_impl(_return,
                                  query_state->createQueryStateProxy(),
                                  column_format,
                                  session_ptr->get_executor_device_type(),
                                  first_n,
                                  at_most_n,
                                  use_calcite,
                                  locks);
    });

    _return.setExecutionTime(total_time_ms +
                             process_deferred_copy_from(request_info.sessionId()));

    stdlog.appendNameValuePairs(
        "execution_time_ms",
        _return.getExecutionTime(),
        "total_time_ms",  // BE-3420 - Redundant with duration field
        stdlog.duration<std::chrono::milliseconds>());
    VLOG(1) << "Table Schema Locks:\n" << lockmgr::TableSchemaLockMgr::instance();
    VLOG(1) << "Table Data Locks:\n" << lockmgr::TableDataLockMgr::instance();
  } catch (const std::exception& e) {
    if (strstr(e.what(), "java.lang.NullPointerException")) {
      THROW_DB_EXCEPTION("query failed from broken view or other schema related issue");
    } else if (strstr(e.what(), "SQL Error: Encountered \";\"")) {
      THROW_DB_EXCEPTION("multiple SQL statements not allowed");
    } else if (strstr(e.what(), "SQL Error: Encountered \"<EOF>\" at line 0, column 0")) {
      THROW_DB_EXCEPTION("empty SQL statment not allowed");
    } else {
      THROW_DB_EXCEPTION(e.what());
    }
  }
}

int64_t DBHandler::process_deferred_copy_from(const TSessionId& session_id) {
  int64_t total_time_ms(0);
  // if the SQL statement we just executed was a geo COPY FROM, the import
  // parameters were captured, and this flag set, so we do the actual import here
  if (auto deferred_copy_from_state = deferred_copy_from_sessions(session_id)) {
    // import_geo_table() calls create_table() which calls this function to
    // do the work, so reset the flag now to avoid executing this part a
    // second time at the end of that, which would fail as the table was
    // already created! Also reset the flag with a ScopeGuard on exiting
    // this function any other way, such as an exception from the code above!
    deferred_copy_from_sessions.remove(session_id);

    // create table as replicated?
    TCreateParams create_params;
    if (deferred_copy_from_state->partitions == "REPLICATED") {
      create_params.is_replicated = true;
    }

    // now do (and time) the import
    total_time_ms = measure<>::execution([&]() {
      importGeoTableGlobFilterSort(session_id,
                                   deferred_copy_from_state->table,
                                   deferred_copy_from_state->file_name,
                                   deferred_copy_from_state->copy_params,
                                   TRowDescriptor(),
                                   create_params);
    });
  }
  return total_time_ms;
}

void DBHandler::sql_execute_df(TDataFrame& _return,
                               const TSessionId& session_id_or_json,
                               const std::string& query_str,
                               const TDeviceType::type results_device_type,
                               const int32_t device_id,
                               const int32_t first_n,
                               const TArrowTransport::type transport_method) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  auto session_ptr = get_session_ptr(request_info.sessionId());
  CHECK(session_ptr);
  auto query_state = create_query_state(session_ptr, query_str);
  auto stdlog = STDLOG(session_ptr, query_state);

  const auto executor_device_type = session_ptr->get_executor_device_type();

  if (results_device_type == TDeviceType::GPU) {
    if (executor_device_type != ExecutorDeviceType::GPU) {
      THROW_DB_EXCEPTION(std::string("GPU mode is not allowed in this session"));
    }
    if (!data_mgr_->gpusPresent()) {
      THROW_DB_EXCEPTION(std::string("No GPU is available in this server"));
    }
    if (device_id < 0 || device_id >= data_mgr_->getCudaMgr()->getDeviceCount()) {
      THROW_DB_EXCEPTION(
          std::string("Invalid device_id or unavailable GPU with this ID"));
    }
  }
  ParserWrapper pw{query_str};
  if (pw.getQueryType() != ParserWrapper::QueryType::Read) {
    THROW_DB_EXCEPTION(std::string(
        "Only read queries supported for the Arrow sql_execute_df endpoint."));
  }
  if (ExplainInfo(query_str).isCalciteExplain()) {
    THROW_DB_EXCEPTION(std::string(
        "Explain is currently unsupported by the Arrow sql_execute_df endpoint."));
  }

  ExecutionResult execution_result;
  lockmgr::LockedTableDescriptors locks;
  sql_execute_impl(execution_result,
                   query_state->createQueryStateProxy(),
                   true, /* column_format - does this do anything? */
                   executor_device_type,
                   first_n,
                   -1, /* at_most_n */
                   true,
                   locks);

  const auto result_set = execution_result.getRows();
  const auto executor_results_device_type = results_device_type == TDeviceType::CPU
                                                ? ExecutorDeviceType::CPU
                                                : ExecutorDeviceType::GPU;
  _return.execution_time_ms =
      execution_result.getExecutionTime() - result_set->getQueueTime();
  const auto converter = std::make_unique<ArrowResultSetConverter>(
      result_set,
      data_mgr_,
      executor_results_device_type,
      device_id,
      getTargetNames(execution_result.getTargetsMeta()),
      first_n,
      ArrowTransport(transport_method));
  ArrowResult arrow_result;
  _return.arrow_conversion_time_ms +=
      measure<>::execution([&] { arrow_result = converter->getArrowResult(); });
  _return.sm_handle =
      std::string(arrow_result.sm_handle.begin(), arrow_result.sm_handle.end());
  _return.sm_size = arrow_result.sm_size;
  _return.df_handle =
      std::string(arrow_result.df_handle.begin(), arrow_result.df_handle.end());
  _return.df_buffer =
      std::string(arrow_result.df_buffer.begin(), arrow_result.df_buffer.end());
  if (executor_results_device_type == ExecutorDeviceType::GPU) {
    std::lock_guard<std::mutex> map_lock(handle_to_dev_ptr_mutex_);
    CHECK(!ipc_handle_to_dev_ptr_.count(_return.df_handle));
    ipc_handle_to_dev_ptr_.insert(
        std::make_pair(_return.df_handle, arrow_result.serialized_cuda_handle));
  }
  _return.df_size = arrow_result.df_size;
}

void DBHandler::sql_execute_gdf(TDataFrame& _return,
                                const TSessionId& session_id_or_json,
                                const std::string& query_str,
                                const int32_t device_id,
                                const int32_t first_n) {
  heavyai::RequestInfo request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  auto stdlog = STDLOG(get_session_ptr(request_info.sessionId()));
  request_info.setRequestId(logger::request_id());
  sql_execute_df(_return,
                 request_info.json(),
                 query_str,
                 TDeviceType::GPU,
                 device_id,
                 first_n,
                 TArrowTransport::SHARED_MEMORY);
}

// For now we have only one user of a data frame in all cases.
void DBHandler::deallocate_df(const TSessionId& session_id_or_json,
                              const TDataFrame& df,
                              const TDeviceType::type device_type,
                              const int32_t device_id) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  auto stdlog = STDLOG(get_session_ptr(request_info.sessionId()));
  std::string serialized_cuda_handle = "";
  if (device_type == TDeviceType::GPU) {
    std::lock_guard<std::mutex> map_lock(handle_to_dev_ptr_mutex_);
    if (ipc_handle_to_dev_ptr_.count(df.df_handle) != size_t(1)) {
      TDBException ex;
      ex.error_msg = std::string(
          "Current data frame handle is not bookkept or been inserted "
          "twice");
      LOG(ERROR) << ex.error_msg;
      throw ex;
    }
    serialized_cuda_handle = ipc_handle_to_dev_ptr_[df.df_handle];
    ipc_handle_to_dev_ptr_.erase(df.df_handle);
  }
  std::vector<char> sm_handle(df.sm_handle.begin(), df.sm_handle.end());
  std::vector<char> df_handle(df.df_handle.begin(), df.df_handle.end());
  ArrowResult result{
      sm_handle, df.sm_size, df_handle, df.df_size, serialized_cuda_handle};
  ArrowResultSet::deallocateArrowResultBuffer(
      result,
      device_type == TDeviceType::CPU ? ExecutorDeviceType::CPU : ExecutorDeviceType::GPU,
      device_id,
      data_mgr_);
}

void DBHandler::sql_validate(TRowDescriptor& _return,
                             const TSessionId& session_id_or_json,
                             const std::string& query_str) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  try {
    auto stdlog = STDLOG(get_session_ptr(request_info.sessionId()));
    stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
    auto query_state = create_query_state(stdlog.getSessionInfo(), query_str);
    stdlog.setQueryState(query_state);

    ParserWrapper pw{query_str};
    if (ExplainInfo(query_str).isExplain() || pw.is_ddl || pw.is_update_dml) {
      throw std::runtime_error("Can only validate SELECT statements.");
    }

    const auto execute_read_lock = legacylockmgr::getExecuteReadLock();

    TPlanResult parse_result;
    lockmgr::LockedTableDescriptors locks;
    std::tie(parse_result, locks) = parse_to_ra(query_state->createQueryStateProxy(),
                                                query_state->getQueryStr(),
                                                {},
                                                true,
                                                system_parameters_,
                                                /*check_privileges=*/true);
    const auto query_ra = parse_result.plan_result;
    _return = validateRelAlg(query_ra, query_state->createQueryStateProxy());
  } catch (const std::exception& e) {
    THROW_DB_EXCEPTION(std::string(e.what()));
  }
}

namespace {

struct ProjectionTokensForCompletion {
  std::unordered_set<std::string> uc_column_names;
  std::unordered_set<std::string> uc_column_table_qualifiers;
};

// Extract what looks like a (qualified) identifier from the partial query.
// The results will be used to rank the auto-completion results: tables which
// contain at least one of the identifiers first.
ProjectionTokensForCompletion extract_projection_tokens_for_completion(
    const std::string& sql) {
  boost::regex id_regex{R"(([[:alnum:]]|_|\.)+)",
                        boost::regex::extended | boost::regex::icase};
  boost::sregex_token_iterator tok_it(sql.begin(), sql.end(), id_regex, 0);
  boost::sregex_token_iterator end;
  std::unordered_set<std::string> uc_column_names;
  std::unordered_set<std::string> uc_column_table_qualifiers;
  for (; tok_it != end; ++tok_it) {
    std::string column_name = *tok_it;
    std::vector<std::string> column_tokens;
    boost::split(column_tokens, column_name, boost::is_any_of("."));
    if (column_tokens.size() == 2) {
      // If the column name is qualified, take user's word.
      uc_column_table_qualifiers.insert(to_upper(column_tokens.front()));
    } else {
      uc_column_names.insert(to_upper(column_name));
    }
  }
  return {uc_column_names, uc_column_table_qualifiers};
}

}  // namespace

void DBHandler::get_completion_hints(std::vector<TCompletionHint>& hints,
                                     const TSessionId& session_id_or_json,
                                     const std::string& sql,
                                     const int cursor) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  auto stdlog = STDLOG(get_session_ptr(request_info.sessionId()));
  std::vector<std::string> visible_tables;  // Tables allowed for the given session.
  get_completion_hints_unsorted(hints, visible_tables, stdlog, sql, cursor);
  const auto proj_tokens = extract_projection_tokens_for_completion(sql);
  auto compatible_table_names = get_uc_compatible_table_names_by_column(
      proj_tokens.uc_column_names, visible_tables, stdlog);
  // Add the table qualifiers explicitly specified by the user.
  compatible_table_names.insert(proj_tokens.uc_column_table_qualifiers.begin(),
                                proj_tokens.uc_column_table_qualifiers.end());
  // Sort the hints by category, from COLUMN (most specific) to KEYWORD.
  std::sort(
      hints.begin(),
      hints.end(),
      [&compatible_table_names](const TCompletionHint& lhs, const TCompletionHint& rhs) {
        if (lhs.type == TCompletionHintType::TABLE &&
            rhs.type == TCompletionHintType::TABLE) {
          // Between two tables, one which is compatible with the specified
          // projections and one which isn't, pick the one which is compatible.
          if (compatible_table_names.find(to_upper(lhs.hints.back())) !=
                  compatible_table_names.end() &&
              compatible_table_names.find(to_upper(rhs.hints.back())) ==
                  compatible_table_names.end()) {
            return true;
          }
        }
        return lhs.type < rhs.type;
      });
}

void DBHandler::get_completion_hints_unsorted(std::vector<TCompletionHint>& hints,
                                              std::vector<std::string>& visible_tables,
                                              query_state::StdLog& stdlog,
                                              const std::string& sql,
                                              const int cursor) {
  const auto& session_info = *stdlog.getConstSessionInfo();
  try {
    get_tables_impl(visible_tables, session_info, GET_PHYSICAL_TABLES_AND_VIEWS);

    // Filter out keywords suggested by Calcite which we don't support.
    hints = just_whitelisted_keyword_hints(
        calcite_->getCompletionHints(session_info, visible_tables, sql, cursor));
  } catch (const std::exception& e) {
    TDBException ex;
    ex.error_msg = std::string(e.what());
    LOG(ERROR) << ex.error_msg;
    throw ex;
  }
  boost::regex from_expr{R"(\s+from\s+)", boost::regex::extended | boost::regex::icase};
  const size_t length_to_cursor =
      cursor < 0 ? sql.size() : std::min(sql.size(), static_cast<size_t>(cursor));
  // Trust hints from Calcite after the FROM keyword.
  if (boost::regex_search(sql.cbegin(), sql.cbegin() + length_to_cursor, from_expr)) {
    return;
  }
  // Before FROM, the query is too incomplete for context-sensitive completions.
  get_token_based_completions(hints, stdlog, visible_tables, sql, cursor);
}

void DBHandler::get_token_based_completions(std::vector<TCompletionHint>& hints,
                                            query_state::StdLog& stdlog,
                                            std::vector<std::string>& visible_tables,
                                            const std::string& sql,
                                            const int cursor) {
  const auto last_word =
      find_last_word_from_cursor(sql, cursor < 0 ? sql.size() : cursor);
  boost::regex select_expr{R"(\s*select\s+)",
                           boost::regex::extended | boost::regex::icase};
  const size_t length_to_cursor =
      cursor < 0 ? sql.size() : std::min(sql.size(), static_cast<size_t>(cursor));
  // After SELECT but before FROM, look for all columns in all tables which match the
  // prefix.
  if (boost::regex_search(sql.cbegin(), sql.cbegin() + length_to_cursor, select_expr)) {
    const auto column_names_by_table = fill_column_names_by_table(visible_tables, stdlog);
    // Trust the fully qualified columns the most.
    if (get_qualified_column_hints(hints, last_word, column_names_by_table)) {
      return;
    }
    // Not much information to use, just retrieve column names which match the prefix.
    if (should_suggest_column_hints(sql)) {
      get_column_hints(hints, last_word, column_names_by_table);
      return;
    }
    const std::string kFromKeyword{"FROM"};
    if (boost::istarts_with(kFromKeyword, last_word)) {
      TCompletionHint keyword_hint;
      keyword_hint.type = TCompletionHintType::KEYWORD;
      keyword_hint.replaced = last_word;
      keyword_hint.hints.emplace_back(kFromKeyword);
      hints.push_back(keyword_hint);
    }
  } else {
    const std::string kSelectKeyword{"SELECT"};
    if (boost::istarts_with(kSelectKeyword, last_word)) {
      TCompletionHint keyword_hint;
      keyword_hint.type = TCompletionHintType::KEYWORD;
      keyword_hint.replaced = last_word;
      keyword_hint.hints.emplace_back(kSelectKeyword);
      hints.push_back(keyword_hint);
    }
  }
}

std::unordered_map<std::string, std::unordered_set<std::string>>
DBHandler::fill_column_names_by_table(std::vector<std::string>& table_names,
                                      query_state::StdLog& stdlog) {
  std::unordered_map<std::string, std::unordered_set<std::string>> column_names_by_table;
  for (auto it = table_names.begin(); it != table_names.end();) {
    TTableDetails table_details;
    try {
      get_table_details_impl(table_details, stdlog, *it, false, false);
    } catch (const TDBException& e) {
      // Remove the corrupted Table/View name from the list for further processing.
      it = table_names.erase(it);
      continue;
    }
    for (const auto& column_type : table_details.row_desc) {
      column_names_by_table[*it].emplace(column_type.col_name);
    }
    ++it;
  }
  return column_names_by_table;
}

ConnectionInfo DBHandler::getConnectionInfo() const {
  return ConnectionInfo{TrackingProcessor::client_address,
                        TrackingProcessor::client_protocol};
}

std::unordered_set<std::string> DBHandler::get_uc_compatible_table_names_by_column(
    const std::unordered_set<std::string>& uc_column_names,
    std::vector<std::string>& table_names,
    query_state::StdLog& stdlog) {
  std::unordered_set<std::string> compatible_table_names_by_column;
  for (auto it = table_names.begin(); it != table_names.end();) {
    TTableDetails table_details;
    try {
      get_table_details_impl(table_details, stdlog, *it, false, false);
    } catch (const TDBException& e) {
      // Remove the corrupted Table/View name from the list for further processing.
      it = table_names.erase(it);
      continue;
    }
    for (const auto& column_type : table_details.row_desc) {
      if (uc_column_names.find(to_upper(column_type.col_name)) != uc_column_names.end()) {
        compatible_table_names_by_column.emplace(to_upper(*it));
        break;
      }
    }
    ++it;
  }
  return compatible_table_names_by_column;
}

void DBHandler::dispatch_query_task(std::shared_ptr<QueryDispatchQueue::Task> query_task,
                                    const bool is_update_delete) {
  CHECK(dispatch_queue_);
  dispatch_queue_->submit(std::move(query_task), is_update_delete);
}

TRowDescriptor DBHandler::validateRelAlg(const std::string& query_ra,
                                         QueryStateProxy query_state_proxy) {
  TQueryResult query_result;
  ExecutionResult execution_result;
  auto execute_rel_alg_task = std::make_shared<QueryDispatchQueue::Task>(
      [this,
       &execution_result,
       query_state_proxy,
       &query_ra,
       parent_thread_local_ids =
           logger::thread_local_ids()](const size_t executor_index) {
        logger::LocalIdsScopeGuard lisg = parent_thread_local_ids.setNewThreadId();
        execute_rel_alg(execution_result,
                        query_state_proxy,
                        query_ra,
                        true,
                        ExecutorDeviceType::CPU,
                        -1,
                        -1,
                        /*just_validate=*/true,
                        /*find_filter_push_down_candidates=*/false,
                        ExplainInfo(),
                        executor_index);
      });
  dispatch_query_task(execute_rel_alg_task, /*is_update_delete=*/false);
  auto result_future = execute_rel_alg_task->get_future();
  result_future.get();
  DBHandler::convertData(query_result, execution_result, query_state_proxy, true, -1, -1);

  const auto& row_desc = query_result.row_set.row_desc;
  const auto& targets_meta = execution_result.getTargetsMeta();
  CHECK_EQ(row_desc.size(), targets_meta.size());

  // TODO: Below fixup logic should no longer be needed after the comp_param refactor
  TRowDescriptor fixedup_row_desc;
  for (size_t i = 0; i < row_desc.size(); i++) {
    const auto& col_desc = row_desc[i];
    auto fixedup_col_desc = col_desc;
    if (col_desc.col_type.encoding == TEncodingType::DICT &&
        col_desc.col_type.comp_param > 0) {
      const auto& type_info = targets_meta[i].get_type_info();
      CHECK_EQ(type_info.get_compression(), kENCODING_DICT);
      const auto cat = Catalog_Namespace::SysCatalog::instance().getCatalog(
          type_info.getStringDictKey().db_id);
      const auto dd = cat->getMetadataForDict(col_desc.col_type.comp_param, false);
      CHECK(dd);
      fixedup_col_desc.col_type.comp_param = dd->dictNBits;
    }
    fixedup_row_desc.push_back(fixedup_col_desc);
  }
  return fixedup_row_desc;
}

void DBHandler::get_roles(std::vector<std::string>& roles,
                          const TSessionId& session_id_or_json) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  auto stdlog = STDLOG(get_session_ptr(request_info.sessionId()));
  auto session_ptr = stdlog.getConstSessionInfo();
  if (!session_ptr->get_currentUser().isSuper) {
    // WARNING: This appears to not include roles a user is a member of,
    // if the role has no permissions granted to it.
    roles =
        SysCatalog::instance().getRoles(session_ptr->get_currentUser().userName,
                                        session_ptr->getCatalog().getCurrentDB().dbId);
  } else {
    roles = SysCatalog::instance().getRoles(
        false, true, session_ptr->get_currentUser().userName);
  }
}

bool DBHandler::has_role(const TSessionId& session_id_or_json,
                         const std::string& granteeName,
                         const std::string& roleName) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  const auto session_ptr = get_session_ptr(request_info.sessionId());
  const auto stdlog = STDLOG(session_ptr);
  const auto current_user = session_ptr->get_currentUser();
  if (!current_user.isSuper) {
    if (const auto* user = SysCatalog::instance().getUserGrantee(granteeName);
        user && current_user.userName != granteeName) {
      THROW_DB_EXCEPTION("Only super users can check other user's roles.");
    } else if (!SysCatalog::instance().isRoleGrantedToGrantee(
                   current_user.userName, granteeName, true)) {
      THROW_DB_EXCEPTION(
          "Only super users can check roles assignment that have not been directly "
          "granted to a user.");
    }
  }
  return SysCatalog::instance().isRoleGrantedToGrantee(granteeName, roleName, false);
}

static TDBObject serialize_db_object(const std::string& roleName,
                                     const DBObject& inObject) {
  TDBObject outObject;
  outObject.objectName = inObject.getName();
  outObject.grantee = roleName;
  outObject.objectId = inObject.getObjectKey().objectId;
  const auto ap = inObject.getPrivileges();
  switch (inObject.getObjectKey().permissionType) {
    case DatabaseDBObjectType:
      outObject.privilegeObjectType = TDBObjectType::DatabaseDBObjectType;
      outObject.privs.push_back(ap.hasPermission(DatabasePrivileges::CREATE_DATABASE));
      outObject.privs.push_back(ap.hasPermission(DatabasePrivileges::DROP_DATABASE));
      outObject.privs.push_back(ap.hasPermission(DatabasePrivileges::VIEW_SQL_EDITOR));
      outObject.privs.push_back(ap.hasPermission(DatabasePrivileges::ACCESS));

      break;
    case TableDBObjectType:
      outObject.privilegeObjectType = TDBObjectType::TableDBObjectType;
      outObject.privs.push_back(ap.hasPermission(TablePrivileges::CREATE_TABLE));
      outObject.privs.push_back(ap.hasPermission(TablePrivileges::DROP_TABLE));
      outObject.privs.push_back(ap.hasPermission(TablePrivileges::SELECT_FROM_TABLE));
      outObject.privs.push_back(ap.hasPermission(TablePrivileges::INSERT_INTO_TABLE));
      outObject.privs.push_back(ap.hasPermission(TablePrivileges::UPDATE_IN_TABLE));
      outObject.privs.push_back(ap.hasPermission(TablePrivileges::DELETE_FROM_TABLE));
      outObject.privs.push_back(ap.hasPermission(TablePrivileges::TRUNCATE_TABLE));
      outObject.privs.push_back(ap.hasPermission(TablePrivileges::ALTER_TABLE));

      break;
    case DashboardDBObjectType:
      outObject.privilegeObjectType = TDBObjectType::DashboardDBObjectType;
      outObject.privs.push_back(ap.hasPermission(DashboardPrivileges::CREATE_DASHBOARD));
      outObject.privs.push_back(ap.hasPermission(DashboardPrivileges::DELETE_DASHBOARD));
      outObject.privs.push_back(ap.hasPermission(DashboardPrivileges::VIEW_DASHBOARD));
      outObject.privs.push_back(ap.hasPermission(DashboardPrivileges::EDIT_DASHBOARD));

      break;
    case ViewDBObjectType:
      outObject.privilegeObjectType = TDBObjectType::ViewDBObjectType;
      outObject.privs.push_back(ap.hasPermission(ViewPrivileges::CREATE_VIEW));
      outObject.privs.push_back(ap.hasPermission(ViewPrivileges::DROP_VIEW));
      outObject.privs.push_back(ap.hasPermission(ViewPrivileges::SELECT_FROM_VIEW));
      outObject.privs.push_back(ap.hasPermission(ViewPrivileges::INSERT_INTO_VIEW));
      outObject.privs.push_back(ap.hasPermission(ViewPrivileges::UPDATE_IN_VIEW));
      outObject.privs.push_back(ap.hasPermission(ViewPrivileges::DELETE_FROM_VIEW));

      break;
    case ServerDBObjectType:
      outObject.privilegeObjectType = TDBObjectType::ServerDBObjectType;
      outObject.privs.push_back(ap.hasPermission(ServerPrivileges::CREATE_SERVER));
      outObject.privs.push_back(ap.hasPermission(ServerPrivileges::DROP_SERVER));
      outObject.privs.push_back(ap.hasPermission(ServerPrivileges::ALTER_SERVER));
      outObject.privs.push_back(ap.hasPermission(ServerPrivileges::SERVER_USAGE));

      break;
    default:
      CHECK(false);
  }
  const int type_val = static_cast<int>(inObject.getType());
  CHECK(type_val >= 0 && type_val < 6);
  outObject.objectType = static_cast<TDBObjectType::type>(type_val);
  return outObject;
}

bool DBHandler::has_database_permission(const AccessPrivileges& privs,
                                        const TDBObjectPermissions& permissions) {
  if (!permissions.__isset.database_permissions_) {
    THROW_DB_EXCEPTION("Database permissions not set for check.")
  }
  auto perms = permissions.database_permissions_;
  if ((perms.create_ && !privs.hasPermission(DatabasePrivileges::CREATE_DATABASE)) ||
      (perms.delete_ && !privs.hasPermission(DatabasePrivileges::DROP_DATABASE)) ||
      (perms.view_sql_editor_ &&
       !privs.hasPermission(DatabasePrivileges::VIEW_SQL_EDITOR)) ||
      (perms.access_ && !privs.hasPermission(DatabasePrivileges::ACCESS))) {
    return false;
  } else {
    return true;
  }
}

bool DBHandler::has_table_permission(const AccessPrivileges& privs,
                                     const TDBObjectPermissions& permissions) {
  if (!permissions.__isset.table_permissions_) {
    THROW_DB_EXCEPTION("Table permissions not set for check.")
  }
  auto perms = permissions.table_permissions_;
  if ((perms.create_ && !privs.hasPermission(TablePrivileges::CREATE_TABLE)) ||
      (perms.drop_ && !privs.hasPermission(TablePrivileges::DROP_TABLE)) ||
      (perms.select_ && !privs.hasPermission(TablePrivileges::SELECT_FROM_TABLE)) ||
      (perms.insert_ && !privs.hasPermission(TablePrivileges::INSERT_INTO_TABLE)) ||
      (perms.update_ && !privs.hasPermission(TablePrivileges::UPDATE_IN_TABLE)) ||
      (perms.delete_ && !privs.hasPermission(TablePrivileges::DELETE_FROM_TABLE)) ||
      (perms.truncate_ && !privs.hasPermission(TablePrivileges::TRUNCATE_TABLE)) ||
      (perms.alter_ && !privs.hasPermission(TablePrivileges::ALTER_TABLE))) {
    return false;
  } else {
    return true;
  }
}

bool DBHandler::has_dashboard_permission(const AccessPrivileges& privs,
                                         const TDBObjectPermissions& permissions) {
  if (!permissions.__isset.dashboard_permissions_) {
    THROW_DB_EXCEPTION("Dashboard permissions not set for check.")
  }
  auto perms = permissions.dashboard_permissions_;
  if ((perms.create_ && !privs.hasPermission(DashboardPrivileges::CREATE_DASHBOARD)) ||
      (perms.delete_ && !privs.hasPermission(DashboardPrivileges::DELETE_DASHBOARD)) ||
      (perms.view_ && !privs.hasPermission(DashboardPrivileges::VIEW_DASHBOARD)) ||
      (perms.edit_ && !privs.hasPermission(DashboardPrivileges::EDIT_DASHBOARD))) {
    return false;
  } else {
    return true;
  }
}

bool DBHandler::has_view_permission(const AccessPrivileges& privs,
                                    const TDBObjectPermissions& permissions) {
  if (!permissions.__isset.view_permissions_) {
    THROW_DB_EXCEPTION("View permissions not set for check.")
  }
  auto perms = permissions.view_permissions_;
  if ((perms.create_ && !privs.hasPermission(ViewPrivileges::CREATE_VIEW)) ||
      (perms.drop_ && !privs.hasPermission(ViewPrivileges::DROP_VIEW)) ||
      (perms.select_ && !privs.hasPermission(ViewPrivileges::SELECT_FROM_VIEW)) ||
      (perms.insert_ && !privs.hasPermission(ViewPrivileges::INSERT_INTO_VIEW)) ||
      (perms.update_ && !privs.hasPermission(ViewPrivileges::UPDATE_IN_VIEW)) ||
      (perms.delete_ && !privs.hasPermission(ViewPrivileges::DELETE_FROM_VIEW))) {
    return false;
  } else {
    return true;
  }
}

bool DBHandler::has_server_permission(const AccessPrivileges& privs,
                                      const TDBObjectPermissions& permissions) {
  CHECK(permissions.__isset.server_permissions_);
  auto perms = permissions.server_permissions_;
  if ((perms.create_ && !privs.hasPermission(ServerPrivileges::CREATE_SERVER)) ||
      (perms.drop_ && !privs.hasPermission(ServerPrivileges::DROP_SERVER)) ||
      (perms.alter_ && !privs.hasPermission(ServerPrivileges::ALTER_SERVER)) ||
      (perms.usage_ && !privs.hasPermission(ServerPrivileges::SERVER_USAGE))) {
    return false;
  } else {
    return true;
  }
}

bool DBHandler::has_object_privilege(const TSessionId& session_id_or_json,
                                     const std::string& granteeName,
                                     const std::string& objectName,
                                     const TDBObjectType::type objectType,
                                     const TDBObjectPermissions& permissions) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  auto session_ptr = get_session_ptr(request_info.sessionId());
  auto stdlog = STDLOG(session_ptr);
  auto const& cat = session_ptr->getCatalog();
  auto const& current_user = session_ptr->get_currentUser();
  if (!current_user.isSuper && !SysCatalog::instance().isRoleGrantedToGrantee(
                                   current_user.userName, granteeName, false)) {
    THROW_DB_EXCEPTION(
        "Users except superusers can only check privileges for self or roles granted "
        "to "
        "them.")
  }
  Catalog_Namespace::UserMetadata user_meta;
  if (SysCatalog::instance().getMetadataForUser(granteeName, user_meta) &&
      user_meta.isSuper) {
    return true;
  }
  Grantee* grnt = SysCatalog::instance().getGrantee(granteeName);
  if (!grnt) {
    THROW_DB_EXCEPTION("User or Role " + granteeName + " does not exist.")
  }
  DBObjectType type;
  std::string func_name;
  switch (objectType) {
    case TDBObjectType::DatabaseDBObjectType:
      type = DBObjectType::DatabaseDBObjectType;
      func_name = "database";
      break;
    case TDBObjectType::TableDBObjectType:
      type = DBObjectType::TableDBObjectType;
      func_name = "table";
      break;
    case TDBObjectType::DashboardDBObjectType:
      type = DBObjectType::DashboardDBObjectType;
      func_name = "dashboard";
      break;
    case TDBObjectType::ViewDBObjectType:
      type = DBObjectType::ViewDBObjectType;
      func_name = "view";
      break;
    case TDBObjectType::ServerDBObjectType:
      type = DBObjectType::ServerDBObjectType;
      func_name = "server";
      break;
    default:
      THROW_DB_EXCEPTION("Invalid object type (" + std::to_string(objectType) + ").");
  }
  DBObject req_object(objectName, type);
  req_object.loadKey(cat);

  auto grantee_object = grnt->findDbObject(req_object.getObjectKey(), false);
  if (grantee_object) {
    // if grantee has privs on the object
    return permissionFuncMap_[func_name](grantee_object->getPrivileges(), permissions);
  } else {
    // no privileges on that object
    return false;
  }
}

void DBHandler::get_db_objects_for_grantee(std::vector<TDBObject>& TDBObjectsForRole,
                                           const TSessionId& session_id_or_json,
                                           const std::string& roleName) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  auto session_ptr = get_session_ptr(request_info.sessionId());
  auto stdlog = STDLOG(session_ptr);
  auto const& user = session_ptr->get_currentUser();
  if (!user.isSuper &&
      !SysCatalog::instance().isRoleGrantedToGrantee(user.userName, roleName, false)) {
    return;
  }
  auto* rl = SysCatalog::instance().getGrantee(roleName);
  if (rl) {
    auto dbId = session_ptr->getCatalog().getCurrentDB().dbId;
    for (auto& dbObject : *rl->getDbObjects(true)) {
      if (dbObject.first.dbId != dbId) {
        // TODO (max): it doesn't scale well in case we have many DBs (not a typical
        // usecase for now, though)
        continue;
      }
      TDBObject tdbObject = serialize_db_object(roleName, *dbObject.second);
      TDBObjectsForRole.push_back(tdbObject);
    }
  } else {
    THROW_DB_EXCEPTION("User or role " + roleName + " does not exist.");
  }
}

void DBHandler::get_db_object_privs(std::vector<TDBObject>& TDBObjects,
                                    const TSessionId& session_id_or_json,
                                    const std::string& objectName,
                                    const TDBObjectType::type type) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  auto session_ptr = get_session_ptr(request_info.sessionId());
  auto stdlog = STDLOG(session_ptr);
  const auto& cat = session_ptr->getCatalog();
  DBObjectType object_type;
  switch (type) {
    case TDBObjectType::DatabaseDBObjectType:
      object_type = DBObjectType::DatabaseDBObjectType;
      break;
    case TDBObjectType::TableDBObjectType:
      object_type = DBObjectType::TableDBObjectType;
      break;
    case TDBObjectType::DashboardDBObjectType:
      object_type = DBObjectType::DashboardDBObjectType;
      break;
    case TDBObjectType::ViewDBObjectType:
      object_type = DBObjectType::ViewDBObjectType;
      break;
    case TDBObjectType::ServerDBObjectType:
      object_type = DBObjectType::ServerDBObjectType;
      break;
    default:
      THROW_DB_EXCEPTION("Failed to get object privileges for " + objectName +
                         ": unknown object type (" + std::to_string(type) + ").");
  }
  DBObject object_to_find(objectName, object_type);

  // TODO(adb): Use DatabaseLock to protect method
  try {
    if (object_type == DashboardDBObjectType) {
      if (objectName == "") {
        object_to_find = DBObject(-1, object_type);
      } else {
        object_to_find = DBObject(std::stoi(objectName), object_type);
      }
    } else if ((object_type == TableDBObjectType || object_type == ViewDBObjectType) &&
               !objectName.empty()) {
      // special handling for view / table
      auto td = cat.getMetadataForTable(objectName, false);
      if (td) {
        object_type = td->isView ? ViewDBObjectType : TableDBObjectType;
        object_to_find = DBObject(objectName, object_type);
      }
    }
    object_to_find.loadKey(cat);
  } catch (const std::exception&) {
    THROW_DB_EXCEPTION("Object with name " + objectName + " does not exist.");
  }

  // object type on database level
  DBObject object_to_find_dblevel("", object_type);
  object_to_find_dblevel.loadKey(cat);
  // if user is superuser respond with a full priv
  if (session_ptr->get_currentUser().isSuper) {
    // using ALL_TABLE here to set max permissions
    DBObject dbObj{object_to_find.getObjectKey(),
                   AccessPrivileges::ALL_TABLE,
                   session_ptr->get_currentUser().userId};
    dbObj.setName("super");
    TDBObjects.push_back(
        serialize_db_object(session_ptr->get_currentUser().userName, dbObj));
  };

  std::vector<std::string> grantees =
      SysCatalog::instance().getRoles(true,
                                      session_ptr->get_currentUser().isSuper,
                                      session_ptr->get_currentUser().userName);
  for (const auto& grantee : grantees) {
    DBObject* object_found;
    auto* gr = SysCatalog::instance().getGrantee(grantee);
    if (gr && (object_found = gr->findDbObject(object_to_find.getObjectKey(), true))) {
      TDBObjects.push_back(serialize_db_object(grantee, *object_found));
    }
    // check object permissions on Database level
    if (gr &&
        (object_found = gr->findDbObject(object_to_find_dblevel.getObjectKey(), true))) {
      TDBObjects.push_back(serialize_db_object(grantee, *object_found));
    }
  }
}

void DBHandler::getAllRolesForUserImpl(
    std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr,
    std::vector<std::string>& roles,
    const std::string& granteeName,
    bool effective) {
  auto* grantee = SysCatalog::instance().getGrantee(granteeName);
  if (grantee) {
    if (session_ptr->get_currentUser().isSuper) {
      roles = grantee->getRoles(/*only_direct=*/!effective);
    } else if (grantee->isUser()) {
      if (session_ptr->get_currentUser().userName == granteeName) {
        roles = grantee->getRoles(/*only_direct=*/!effective);
      } else {
        THROW_DB_EXCEPTION(
            "Only a superuser is authorized to request list of roles granted to another "
            "user.");
      }
    } else {
      CHECK(!grantee->isUser());
      // granteeName is actually a roleName here and we can check a role
      // only if it is granted to us
      if (SysCatalog::instance().isRoleGrantedToGrantee(
              session_ptr->get_currentUser().userName, granteeName, false)) {
        roles = grantee->getRoles(/*only_direct=*/!effective);
      } else {
        THROW_DB_EXCEPTION("A user can check only roles granted to him.");
      }
    }
  } else {
    THROW_DB_EXCEPTION("Grantee " + granteeName + " does not exist.");
  }
}

void DBHandler::get_all_roles_for_user(std::vector<std::string>& roles,
                                       const TSessionId& session_id_or_json,
                                       const std::string& granteeName) {
  // WARNING: This function only returns directly granted roles.
  // See also: get_all_effective_roles_for_user() for all of a user's roles.
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  auto stdlog = STDLOG(get_session_ptr(request_info.sessionId()));
  auto session_ptr = stdlog.getConstSessionInfo();
  getAllRolesForUserImpl(session_ptr, roles, granteeName, /*effective=*/false);
}

void DBHandler::get_all_effective_roles_for_user(std::vector<std::string>& roles,
                                                 const TSessionId& session_id_or_json,
                                                 const std::string& granteeName) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  auto stdlog = STDLOG(get_session_ptr(request_info.sessionId()));
  auto session_ptr = stdlog.getConstSessionInfo();
  getAllRolesForUserImpl(session_ptr, roles, granteeName, /*effective=*/true);
}

namespace {
std::string dump_table_col_names(
    const std::map<std::string, std::vector<std::string>>& table_col_names) {
  std::ostringstream oss;
  for (const auto& [table_name, col_names] : table_col_names) {
    oss << ":" << table_name;
    for (const auto& col_name : col_names) {
      oss << "," << col_name;
    }
  }
  return oss.str();
}
}  // namespace

void DBHandler::get_result_row_for_pixel(
    TPixelTableRowResult& _return,
    const TSessionId& session_id_or_json,
    const int64_t widget_id,
    const TPixel& pixel,
    const std::map<std::string, std::vector<std::string>>& table_col_names,
    const bool column_format,
    const int32_t pixel_radius,
    const std::string& nonce) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  auto session_ptr = get_session_ptr(request_info.sessionId());
  auto stdlog = STDLOG(session_ptr,
                       "widget_id",
                       widget_id,
                       "pixel.x",
                       pixel.x,
                       "pixel.y",
                       pixel.y,
                       "column_format",
                       column_format,
                       "pixel_radius",
                       pixel_radius,
                       "table_col_names",
                       dump_table_col_names(table_col_names),
                       "nonce",
                       nonce);
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  if (!render_handler_) {
    THROW_DB_EXCEPTION("Backend rendering is disabled.");
  }

  try {
    render_handler_->get_result_row_for_pixel(_return,
                                              session_ptr,
                                              widget_id,
                                              pixel,
                                              table_col_names,
                                              column_format,
                                              pixel_radius,
                                              nonce);
  } catch (std::exception& e) {
    THROW_DB_EXCEPTION(e.what());
  }
}

TColumnType DBHandler::populateThriftColumnType(const Catalog* cat,
                                                const ColumnDescriptor* cd) {
  TColumnType col_type;
  col_type.col_name = cd->columnName;
  col_type.src_name = cd->sourceName;
  col_type.col_id = cd->columnId;
  col_type.col_type.type = type_to_thrift(cd->columnType);
  col_type.col_type.encoding = encoding_to_thrift(cd->columnType);
  col_type.col_type.nullable = !cd->columnType.get_notnull();
  col_type.col_type.is_array = cd->columnType.get_type() == kARRAY;
  if (col_type.col_type.is_array || cd->columnType.get_type() == kDATE) {
    col_type.col_type.size = cd->columnType.get_size();  // only for arrays and dates
  }
  if (IS_GEO(cd->columnType.get_type())) {
    ThriftSerializers::fixup_geo_column_descriptor(
        col_type, cd->columnType.get_subtype(), cd->columnType.get_output_srid());
  } else {
    col_type.col_type.precision = cd->columnType.get_precision();
    col_type.col_type.scale = cd->columnType.get_scale();
  }
  col_type.is_system = cd->isSystemCol;
  if (cd->columnType.get_compression() == EncodingType::kENCODING_DICT &&
      cat != nullptr) {
    // have to get the actual size of the encoding from the dictionary definition
    const int dict_id = cd->columnType.get_comp_param();
    if (!cat->getMetadataForDict(dict_id, false)) {
      col_type.col_type.comp_param = 0;
      return col_type;
    }
    auto dd = cat->getMetadataForDict(dict_id, false);
    if (!dd) {
      THROW_DB_EXCEPTION("Dictionary doesn't exist");
    }
    col_type.col_type.comp_param = dd->dictNBits;
  } else {
    col_type.col_type.comp_param =
        (cd->columnType.is_date_in_days() && cd->columnType.get_comp_param() == 0)
            ? 32
            : cd->columnType.get_comp_param();
  }
  col_type.is_reserved_keyword = ImportHelpers::is_reserved_name(col_type.col_name);
  if (cd->default_value.has_value()) {
    col_type.__set_default_value(cd->getDefaultValueLiteral());
  }
  return col_type;
}

void DBHandler::get_internal_table_details(TTableDetails& _return,
                                           const TSessionId& session_id_or_json,
                                           const std::string& table_name,
                                           const bool include_system_columns) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  auto stdlog =
      STDLOG(get_session_ptr(request_info.sessionId()), "table_name", table_name);
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  get_table_details_impl(_return, stdlog, table_name, include_system_columns, false);
}

void DBHandler::get_internal_table_details_for_database(
    TTableDetails& _return,
    const TSessionId& session_id_or_json,
    const std::string& table_name,
    const std::string& database_name) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  auto stdlog =
      STDLOG(get_session_ptr(request_info.sessionId()), "table_name", table_name);
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  get_table_details_impl(_return, stdlog, table_name, true, false, database_name);
}

void DBHandler::get_table_details(TTableDetails& _return,
                                  const TSessionId& session_id_or_json,
                                  const std::string& table_name) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  auto stdlog =
      STDLOG(get_session_ptr(request_info.sessionId()), "table_name", table_name);
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());

  auto execute_read_lock = legacylockmgr::getExecuteReadLock();
  get_table_details_impl(_return, stdlog, table_name, false, false);
}

void DBHandler::get_table_details_for_database(TTableDetails& _return,
                                               const TSessionId& session_id_or_json,
                                               const std::string& table_name,
                                               const std::string& database_name) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  auto stdlog =
      STDLOG(get_session_ptr(request_info.sessionId()), "table_name", table_name);
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());

  auto execute_read_lock = legacylockmgr::getExecuteReadLock();
  get_table_details_impl(_return, stdlog, table_name, false, false, database_name);
}

namespace {
TTableRefreshInfo get_refresh_info(const TableDescriptor* td) {
  CHECK(td->isForeignTable());
  auto foreign_table = dynamic_cast<const foreign_storage::ForeignTable*>(td);
  CHECK(foreign_table);
  TTableRefreshInfo refresh_info;
  const auto& update_type =
      foreign_table->getOption(foreign_storage::ForeignTable::REFRESH_UPDATE_TYPE_KEY);
  CHECK(update_type.has_value());
  if (update_type.value() == foreign_storage::ForeignTable::ALL_REFRESH_UPDATE_TYPE) {
    refresh_info.update_type = TTableRefreshUpdateType::ALL;
  } else if (update_type.value() ==
             foreign_storage::ForeignTable::APPEND_REFRESH_UPDATE_TYPE) {
    refresh_info.update_type = TTableRefreshUpdateType::APPEND;
  } else {
    UNREACHABLE() << "Unexpected refresh update type: " << update_type.value();
  }

  const auto& timing_type =
      foreign_table->getOption(foreign_storage::ForeignTable::REFRESH_TIMING_TYPE_KEY);
  CHECK(timing_type.has_value());
  if (timing_type.value() == foreign_storage::ForeignTable::MANUAL_REFRESH_TIMING_TYPE) {
    refresh_info.timing_type = TTableRefreshTimingType::MANUAL;
    refresh_info.interval_count = -1;
  } else if (timing_type.value() ==
             foreign_storage::ForeignTable::SCHEDULE_REFRESH_TIMING_TYPE) {
    refresh_info.timing_type = TTableRefreshTimingType::SCHEDULED;
    const auto& start_date_time = foreign_table->getOption(
        foreign_storage::ForeignTable::REFRESH_START_DATE_TIME_KEY);
    CHECK(start_date_time.has_value());
    auto start_date_time_epoch = dateTimeParse<kTIMESTAMP>(start_date_time.value(), 0);
    refresh_info.start_date_time =
        shared::convert_temporal_to_iso_format({kTIMESTAMP}, start_date_time_epoch);
    const auto& interval =
        foreign_table->getOption(foreign_storage::ForeignTable::REFRESH_INTERVAL_KEY);
    CHECK(interval.has_value());
    const auto& interval_str = interval.value();
    refresh_info.interval_count =
        std::stoi(interval_str.substr(0, interval_str.length() - 1));
    auto interval_type = std::toupper(interval_str[interval_str.length() - 1]);
    if (interval_type == 'H') {
      refresh_info.interval_type = TTableRefreshIntervalType::HOUR;
    } else if (interval_type == 'D') {
      refresh_info.interval_type = TTableRefreshIntervalType::DAY;
    } else if (interval_type == 'S') {
      // This use case is for development only.
      refresh_info.interval_type = TTableRefreshIntervalType::NONE;
    } else {
      UNREACHABLE() << "Unexpected interval type: " << interval_str;
    }
  } else {
    UNREACHABLE() << "Unexpected refresh timing type: " << timing_type.value();
  }
  if (foreign_table->last_refresh_time !=
      foreign_storage::ForeignTable::NULL_REFRESH_TIME) {
    refresh_info.last_refresh_time = shared::convert_temporal_to_iso_format(
        {kTIMESTAMP}, foreign_table->last_refresh_time);
  }
  if (foreign_table->next_refresh_time !=
      foreign_storage::ForeignTable::NULL_REFRESH_TIME) {
    refresh_info.next_refresh_time = shared::convert_temporal_to_iso_format(
        {kTIMESTAMP}, foreign_table->next_refresh_time);
  }
  return refresh_info;
}
}  // namespace

void DBHandler::get_table_details_impl(TTableDetails& _return,
                                       query_state::StdLog& stdlog,
                                       const std::string& table_name,
                                       const bool get_system,
                                       const bool get_physical,
                                       const std::string& database_name) {
  try {
    auto session_info = stdlog.getSessionInfo();
    auto cat = (database_name.empty())
                   ? &session_info->getCatalog()
                   : SysCatalog::instance().getCatalog(database_name).get();
    if (!cat) {
      THROW_DB_EXCEPTION("Database " + database_name + " does not exist.");
    }
    const auto td_with_lock =
        lockmgr::TableSchemaLockContainer<lockmgr::ReadLock>::acquireTableDescriptor(
            *cat, table_name, false);
    const auto td = td_with_lock();
    CHECK(td);

    bool have_privileges_on_view_sources = true;
    if (td->isView) {
      auto query_state = create_query_state(session_info, td->viewSQL);
      stdlog.setQueryState(query_state);
      try {
        if (hasTableAccessPrivileges(td, *session_info)) {
          const auto [query_ra, locks] = parse_to_ra(query_state->createQueryStateProxy(),
                                                     query_state->getQueryStr(),
                                                     {},
                                                     true,
                                                     system_parameters_,
                                                     false);
          try {
            calcite_->checkAccessedObjectsPrivileges(query_state->createQueryStateProxy(),
                                                     query_ra);
          } catch (const std::runtime_error&) {
            have_privileges_on_view_sources = false;
          }

          _return.row_desc =
              validateRelAlg(query_ra.plan_result, query_state->createQueryStateProxy());
        } else {
          throw std::runtime_error(
              "Unable to access view " + table_name +
              ". The view may not exist, or the logged in user may not "
              "have permission to access the view.");
        }
      } catch (const std::exception& e) {
        throw std::runtime_error("View '" + table_name +
                                 "' query has failed with an error: '" +
                                 std::string(e.what()) +
                                 "'.\nThe view must be dropped and re-created to "
                                 "resolve the error. \nQuery:\n" +
                                 query_state->getQueryStr());
      }
    } else {
      if (hasTableAccessPrivileges(td, *session_info)) {
        const auto col_descriptors = cat->getAllColumnMetadataForTable(
            td->tableId, get_system, true, get_physical);
        const auto deleted_cd = cat->getDeletedColumn(td);
        for (const auto cd : col_descriptors) {
          if (cd == deleted_cd) {
            continue;
          }
          _return.row_desc.push_back(populateThriftColumnType(cat, cd));
        }
      } else {
        throw std::runtime_error(
            "Unable to access table " + table_name +
            ". The table may not exist, or the logged in user may not "
            "have permission to access the table.");
      }
    }
    _return.fragment_size = td->maxFragRows;
    _return.page_size = td->fragPageSize;
    _return.max_rows = td->maxRows;
    _return.view_sql =
        (have_privileges_on_view_sources ? td->viewSQL
                                         : "[Not enough privileges to see the view SQL]");
    _return.shard_count = td->nShards * std::max(g_leaf_count, size_t(1));
    if (td->nShards > 0) {
      auto cd = cat->getMetadataForColumn(td->tableId, td->shardedColumnId);
      CHECK(cd);
      _return.sharded_column_name = cd->columnName;
    }
    _return.key_metainfo = td->keyMetainfo;
    _return.is_temporary = td->persistenceLevel == Data_Namespace::MemoryLevel::CPU_LEVEL;
    _return.partition_detail =
        td->partitions.empty()
            ? TPartitionDetail::DEFAULT
            : (table_is_replicated(td)
                   ? TPartitionDetail::REPLICATED
                   : (td->partitions == "SHARDED" ? TPartitionDetail::SHARDED
                                                  : TPartitionDetail::OTHER));
    if (td->isView) {
      _return.table_type = TTableType::VIEW;
    } else if (td->isTemporaryTable()) {
      _return.table_type = TTableType::TEMPORARY;
    } else if (td->isForeignTable()) {
      _return.table_type = TTableType::FOREIGN;
      _return.refresh_info = get_refresh_info(td);
    } else {
      _return.table_type = TTableType::DEFAULT;
    }

  } catch (const std::runtime_error& e) {
    THROW_DB_EXCEPTION(std::string(e.what()));
  }
}

void DBHandler::get_link_view(TFrontendView& _return,
                              const TSessionId& session_id_or_json,
                              const std::string& link) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  auto session_ptr = get_session_ptr(request_info.sessionId());
  auto stdlog = STDLOG(session_ptr);
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  auto const& cat = session_ptr->getCatalog();
  auto ld = cat.getMetadataForLink(std::to_string(cat.getCurrentDB().dbId) + link);
  if (!ld) {
    THROW_DB_EXCEPTION("Link " + link + " is not valid.");
  }
  _return.view_state = ld->viewState;
  _return.view_name = ld->link;
  _return.update_time = ld->updateTime;
  _return.view_metadata = ld->viewMetadata;
}

bool DBHandler::hasTableAccessPrivileges(
    const TableDescriptor* td,
    const Catalog_Namespace::SessionInfo& session_info) {
  auto& cat = session_info.getCatalog();
  auto user_metadata = session_info.get_currentUser();

  if (user_metadata.isSuper) {
    return true;
  }

  DBObject dbObject(td->tableName, td->isView ? ViewDBObjectType : TableDBObjectType);
  dbObject.loadKey(cat);
  std::vector<DBObject> privObjects = {dbObject};

  return SysCatalog::instance().hasAnyPrivileges(user_metadata, privObjects);
}

void DBHandler::get_tables_impl(std::vector<std::string>& table_names,
                                const Catalog_Namespace::SessionInfo& session_info,
                                const GetTablesType get_tables_type,
                                const std::string& database_name) {
  if (database_name.empty()) {
    table_names = session_info.getCatalog().getTableNamesForUser(
        session_info.get_currentUser(), get_tables_type);
  } else {
    auto request_cat = SysCatalog::instance().getCatalog(database_name);
    if (!request_cat) {
      THROW_DB_EXCEPTION("Database " + database_name + " does not exist.");
    }
    table_names = request_cat->getTableNamesForUser(session_info.get_currentUser(),
                                                    get_tables_type);
  }
}

void DBHandler::get_tables(std::vector<std::string>& table_names,
                           const TSessionId& session_id_or_json) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  auto stdlog = STDLOG(get_session_ptr(request_info.sessionId()));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  get_tables_impl(
      table_names, *stdlog.getConstSessionInfo(), GET_PHYSICAL_TABLES_AND_VIEWS);
}

void DBHandler::get_tables_for_database(std::vector<std::string>& table_names,
                                        const TSessionId& session_id_or_json,
                                        const std::string& database_name) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  auto stdlog = STDLOG(get_session_ptr(request_info.sessionId()));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());

  get_tables_impl(table_names,
                  *stdlog.getConstSessionInfo(),
                  GET_PHYSICAL_TABLES_AND_VIEWS,
                  database_name);
}

void DBHandler::get_physical_tables(std::vector<std::string>& table_names,
                                    const TSessionId& session_id_or_json) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  auto stdlog = STDLOG(get_session_ptr(request_info.sessionId()));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  get_tables_impl(table_names, *stdlog.getConstSessionInfo(), GET_PHYSICAL_TABLES);
}

void DBHandler::get_views(std::vector<std::string>& table_names,
                          const TSessionId& session_id_or_json) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  auto stdlog = STDLOG(get_session_ptr(request_info.sessionId()));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  get_tables_impl(table_names, *stdlog.getConstSessionInfo(), GET_VIEWS);
}

void DBHandler::get_tables_meta_impl(std::vector<TTableMeta>& _return,
                                     QueryStateProxy query_state_proxy,
                                     const Catalog_Namespace::SessionInfo& session_info,
                                     const bool with_table_locks) {
  const auto& cat = session_info.getCatalog();
  // Get copies of table descriptors here in order to avoid possible use of dangling
  // pointers, if tables are concurrently dropped.
  const auto tables = cat.getAllTableMetadataCopy();
  _return.reserve(tables.size());

  for (const auto& td : tables) {
    if (td.shard >= 0) {
      // skip shards, they're not standalone tables
      continue;
    }
    if (!hasTableAccessPrivileges(&td, session_info)) {
      // skip table, as there are no privileges to access it
      continue;
    }

    TTableMeta ret;
    ret.table_name = td.tableName;
    ret.is_view = td.isView;
    ret.is_replicated = table_is_replicated(&td);
    ret.shard_count = td.nShards;
    ret.max_rows = td.maxRows;
    ret.table_id = td.tableId;

    std::vector<TTypeInfo> col_types;
    std::vector<std::string> col_names;
    size_t num_cols = 0;
    if (td.isView) {
      try {
        TPlanResult parse_result;
        lockmgr::LockedTableDescriptors locks;
        std::tie(parse_result, locks) = parse_to_ra(
            query_state_proxy, td.viewSQL, {}, with_table_locks, system_parameters_);
        const auto query_ra = parse_result.plan_result;

        ExecutionResult ex_result;
        execute_rel_alg(ex_result,
                        query_state_proxy,
                        query_ra,
                        true,
                        ExecutorDeviceType::CPU,
                        -1,
                        -1,
                        /*just_validate=*/true,
                        /*find_push_down_candidates=*/false,
                        ExplainInfo());
        TQueryResult result;
        DBHandler::convertData(result, ex_result, query_state_proxy, true, -1, -1);
        num_cols = result.row_set.row_desc.size();
        for (const auto& col : result.row_set.row_desc) {
          if (col.is_physical) {
            num_cols--;
            continue;
          }
          col_types.push_back(col.col_type);
          col_names.push_back(col.col_name);
        }
      } catch (std::exception& e) {
        LOG(WARNING) << "get_tables_meta: Ignoring broken view: " << td.tableName;
      }
    } else {
      try {
        if (hasTableAccessPrivileges(&td, session_info)) {
          const auto col_descriptors =
              cat.getAllColumnMetadataForTable(td.tableId, false, true, false);
          const auto deleted_cd = cat.getDeletedColumn(&td);
          for (const auto cd : col_descriptors) {
            if (cd == deleted_cd) {
              continue;
            }
            col_types.push_back(ThriftSerializers::type_info_to_thrift(cd->columnType));
            col_names.push_back(cd->columnName);
          }
          num_cols = col_descriptors.size();
        } else {
          continue;
        }
      } catch (const std::runtime_error& e) {
        THROW_DB_EXCEPTION(e.what());
      }
    }

    ret.num_cols = num_cols;
    std::copy(col_types.begin(), col_types.end(), std::back_inserter(ret.col_types));
    std::copy(col_names.begin(), col_names.end(), std::back_inserter(ret.col_names));

    _return.push_back(ret);
  }
}

void DBHandler::get_tables_meta(std::vector<TTableMeta>& _return,
                                const TSessionId& session_id_or_json) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  auto stdlog = STDLOG(get_session_ptr(request_info.sessionId()));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  auto session_ptr = stdlog.getConstSessionInfo();
  auto query_state = create_query_state(session_ptr, "");
  stdlog.setQueryState(query_state);

  auto execute_read_lock = legacylockmgr::getExecuteReadLock();

  try {
    get_tables_meta_impl(_return, query_state->createQueryStateProxy(), *session_ptr);
  } catch (const std::exception& e) {
    THROW_DB_EXCEPTION(e.what());
  }
}

void DBHandler::get_users(std::vector<std::string>& user_names,
                          const TSessionId& session_id_or_json) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  auto stdlog = STDLOG(get_session_ptr(request_info.sessionId()));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  auto session_ptr = stdlog.getConstSessionInfo();
  std::list<Catalog_Namespace::UserMetadata> user_list;

  if (!session_ptr->get_currentUser().isSuper) {
    user_list = SysCatalog::instance().getAllUserMetadata(
        session_ptr->getCatalog().getCurrentDB().dbId);
  } else {
    user_list = SysCatalog::instance().getAllUserMetadata();
  }
  for (auto u : user_list) {
    user_names.push_back(u.userName);
  }
}

void DBHandler::get_version(std::string& version) {
  version = MAPD_RELEASE;
}

namespace {

ScopeGuard pause_and_resume_executor_queue() {
  if (g_enable_executor_resource_mgr) {
    Executor::pause_executor_queue();
    return [] {
      // we need to resume erm queue if we throw any exception
      // that heavydb server can handle w/o shutting it down
      Executor::resume_executor_queue();
    };
  }
  return [] {};
}

}  // namespace

void DBHandler::clear_gpu_memory(const TSessionId& session_id_or_json) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  auto stdlog = STDLOG(get_session_ptr(request_info.sessionId()));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  auto session_ptr = stdlog.getConstSessionInfo();
  if (!session_ptr->get_currentUser().isSuper) {
    THROW_DB_EXCEPTION("Superuser privilege is required to run clear_gpu_memory");
  }
  auto resume_executor_queue = pause_and_resume_executor_queue();
  // clear renderer memory first
  // this will block until any running render finishes
  if (render_handler_) {
    render_handler_->clear_gpu_memory();
  }
  // then clear the QE memory
  // the renderer will have disconnected from any QE memory
  try {
    Executor::clearMemory(Data_Namespace::MemoryLevel::GPU_LEVEL);
  } catch (const std::exception& e) {
    THROW_DB_EXCEPTION(e.what());
  }
}

void DBHandler::clear_cpu_memory(const TSessionId& session_id_or_json) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  auto stdlog = STDLOG(get_session_ptr(request_info.sessionId()));
  auto session_ptr = stdlog.getConstSessionInfo();
  if (!session_ptr->get_currentUser().isSuper) {
    THROW_DB_EXCEPTION("Superuser privilege is required to run clear_cpu_memory");
  }
  auto resume_executor_queue = pause_and_resume_executor_queue();
  // clear renderer memory first
  // this will block until any running render finishes
  if (render_handler_) {
    render_handler_->clear_cpu_memory();
  }
  // then clear the QE memory
  // the renderer will have disconnected from any QE memory
  try {
    Executor::clearMemory(Data_Namespace::MemoryLevel::CPU_LEVEL);
  } catch (const std::exception& e) {
    THROW_DB_EXCEPTION(e.what());
  }
}

void DBHandler::clearRenderMemory(const TSessionId& session_id_or_json) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  auto stdlog = STDLOG(get_session_ptr(request_info.sessionId()));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  auto session_ptr = stdlog.getConstSessionInfo();
  if (!session_ptr->get_currentUser().isSuper) {
    THROW_DB_EXCEPTION("Superuser privilege is required to run clear_render_memory");
  }
  if (render_handler_) {
    auto resume_executor_queue = pause_and_resume_executor_queue();
    render_handler_->clear_cpu_memory();
    render_handler_->clear_gpu_memory();
  }
}

void DBHandler::pause_executor_queue(const TSessionId& session) {
  auto stdlog = STDLOG(get_session_ptr(session));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  auto session_ptr = stdlog.getConstSessionInfo();
  if (!session_ptr->get_currentUser().isSuper) {
    THROW_DB_EXCEPTION("Superuser privilege is required to run PAUSE EXECUTOR QUEUE");
  }
  try {
    Executor::pause_executor_queue();
  } catch (const std::exception& e) {
    THROW_DB_EXCEPTION(e.what());
  }
}

void DBHandler::resume_executor_queue(const TSessionId& session) {
  auto stdlog = STDLOG(get_session_ptr(session));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  auto session_ptr = stdlog.getConstSessionInfo();
  if (!session_ptr->get_currentUser().isSuper) {
    THROW_DB_EXCEPTION("Superuser privilege is required to run RESUME EXECUTOR QUEUE");
  }
  try {
    Executor::resume_executor_queue();
  } catch (const std::exception& e) {
    THROW_DB_EXCEPTION(e.what());
  }
}

void DBHandler::set_cur_session(const TSessionId& parent_session_id_or_json,
                                const TSessionId& leaf_session_id_or_json,
                                const std::string& start_time_str,
                                const std::string& label,
                                bool for_running_query_kernel) {
  // internal API to manage query interruption in distributed mode
  heavyai::RequestInfo const parent_request_info(parent_session_id_or_json);
  heavyai::RequestInfo const leaf_request_info(leaf_session_id_or_json);
  SET_REQUEST_ID(leaf_request_info.requestId());
  auto stdlog = STDLOG(get_session_ptr(leaf_request_info.sessionId()));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  auto session_ptr = stdlog.getConstSessionInfo();

  auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID).get();
  executor->enrollQuerySession(parent_request_info.sessionId(),
                               label,
                               start_time_str,
                               Executor::UNITARY_EXECUTOR_ID,
                               for_running_query_kernel
                                   ? QuerySessionStatus::QueryStatus::RUNNING_QUERY_KERNEL
                                   : QuerySessionStatus::QueryStatus::RUNNING_IMPORTER);
}

void DBHandler::invalidate_cur_session(const TSessionId& parent_session_id_or_json,
                                       const TSessionId& leaf_session_id_or_json,
                                       const std::string& start_time_str,
                                       const std::string& label,
                                       bool for_running_query_kernel) {
  // internal API to manage query interruption in distributed mode
  heavyai::RequestInfo const parent_request_info(parent_session_id_or_json);
  heavyai::RequestInfo const leaf_request_info(leaf_session_id_or_json);
  SET_REQUEST_ID(leaf_request_info.requestId());
  auto stdlog = STDLOG(get_session_ptr(leaf_request_info.sessionId()));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID).get();
  executor->clearQuerySessionStatus(parent_request_info.sessionId(), start_time_str);
}

TSessionId DBHandler::getInvalidSessionId() const {
  return INVALID_SESSION_ID;
}

void DBHandler::get_memory(std::vector<TNodeMemoryInfo>& _return,
                           const TSessionId& session_id_or_json,
                           const std::string& memory_level) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  auto stdlog = STDLOG(get_session_ptr(request_info.sessionId()));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  std::vector<Data_Namespace::MemoryInfo> internal_memory;
  if (!memory_level.compare("gpu")) {
    internal_memory =
        SysCatalog::instance().getDataMgr().getMemoryInfo(MemoryLevel::GPU_LEVEL);
  } else {
    internal_memory =
        SysCatalog::instance().getDataMgr().getMemoryInfo(MemoryLevel::CPU_LEVEL);
  }

  for (auto memInfo : internal_memory) {
    TNodeMemoryInfo nodeInfo;
    nodeInfo.page_size = memInfo.pageSize;
    nodeInfo.max_num_pages = memInfo.maxNumPages;
    nodeInfo.num_pages_allocated = memInfo.numPageAllocated;
    nodeInfo.is_allocation_capped = memInfo.isAllocationCapped;
    for (auto gpu : memInfo.nodeMemoryData) {
      TMemoryData md;
      md.slab = gpu.slabNum;
      md.start_page = gpu.startPage;
      md.num_pages = gpu.numPages;
      md.touch = gpu.touch;
      md.chunk_key.insert(md.chunk_key.end(), gpu.chunk_key.begin(), gpu.chunk_key.end());
      md.is_free = gpu.memStatus == Buffer_Namespace::MemStatus::FREE;
      nodeInfo.node_memory_data.push_back(md);
    }
    _return.push_back(nodeInfo);
  }
}

void DBHandler::get_databases(std::vector<TDBInfo>& dbinfos,
                              const TSessionId& session_id_or_json) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  auto stdlog = STDLOG(get_session_ptr(request_info.sessionId()));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  auto session_ptr = stdlog.getConstSessionInfo();
  const auto& user = session_ptr->get_currentUser();
  Catalog_Namespace::DBSummaryList dbs =
      SysCatalog::instance().getDatabaseListForUser(user);
  for (auto& db : dbs) {
    TDBInfo dbinfo;
    dbinfo.db_name = std::move(db.dbName);
    dbinfo.db_owner = std::move(db.dbOwnerName);
    dbinfos.push_back(std::move(dbinfo));
  }
}

TExecuteMode::type DBHandler::getExecutionMode(const TSessionId& session_id) {
  auto executor = get_session_ptr(session_id)->get_executor_device_type();
  switch (executor) {
    case ExecutorDeviceType::CPU:
      return TExecuteMode::CPU;
    case ExecutorDeviceType::GPU:
      return TExecuteMode::GPU;
    default:
      UNREACHABLE();
  }
  UNREACHABLE();
  return TExecuteMode::CPU;
}
void DBHandler::set_execution_mode(const TSessionId& session_id_or_json,
                                   const TExecuteMode::type mode) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  auto session_ptr = get_session_ptr(request_info.sessionId());
  auto stdlog = STDLOG(session_ptr);
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  DBHandler::set_execution_mode_nolock(session_ptr.get(), mode);
}

namespace {

void check_table_not_sharded(const TableDescriptor* td) {
  if (td && td->nShards) {
    throw std::runtime_error("Cannot import a sharded table directly to a leaf");
  }
}

void check_valid_column_names(const std::list<const ColumnDescriptor*>& descs,
                              const std::vector<std::string>& column_names) {
  std::unordered_set<std::string> unique_names;
  for (const auto& name : column_names) {
    auto lower_name = to_lower(name);
    if (unique_names.find(lower_name) != unique_names.end()) {
      THROW_DB_EXCEPTION("Column " + name + " is mentioned multiple times");
    } else {
      unique_names.insert(lower_name);
    }
  }
  for (const auto& cd : descs) {
    auto iter = unique_names.find(to_lower(cd->columnName));
    if (iter != unique_names.end()) {
      unique_names.erase(iter);
    }
  }
  if (!unique_names.empty()) {
    THROW_DB_EXCEPTION("Column " + *unique_names.begin() + " does not exist");
  }
}

// Return vector of IDs mapping column descriptors to the list of comumn names.
// The size of the vector is the number of actual columns (geophisical columns excluded).
// ID is either a position in column_names matching the descriptor, or -1 if the column
// is missing from the column_names
std::vector<int> column_ids_by_names(const std::list<const ColumnDescriptor*>& descs,
                                     const std::vector<std::string>& column_names) {
  std::vector<int> desc_to_column_ids;
  if (column_names.empty()) {
    int col_idx = 0;
    for (const auto& cd : descs) {
      if (!cd->isGeoPhyCol) {
        desc_to_column_ids.push_back(col_idx);
        ++col_idx;
      }
    }
  } else {
    for (const auto& cd : descs) {
      if (!cd->isGeoPhyCol) {
        bool found = false;
        for (size_t j = 0; j < column_names.size(); ++j) {
          if (to_lower(cd->columnName) == to_lower(column_names[j])) {
            found = true;
            desc_to_column_ids.push_back(j);
            break;
          }
        }
        if (!found) {
          if (!cd->columnType.get_notnull()) {
            desc_to_column_ids.push_back(-1);
          } else {
            THROW_DB_EXCEPTION("Column '" + cd->columnName +
                               "' cannot be omitted due to NOT NULL constraint");
          }
        }
      }
    }
  }
  return desc_to_column_ids;
}

void log_cache_size(const Catalog_Namespace::Catalog& cat) {
  std::ostringstream oss;
  oss << "Cache size information {";
  auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID);
  // 1. Data recycler
  // 1.a Resultset Recycler
  auto resultset_cache_size =
      executor->getResultSetRecyclerHolder()
          .getResultSetRecycler()
          ->getResultSetRecyclerMetricTracker()
          .getCurrentCacheSize(DataRecyclerUtil::CPU_DEVICE_IDENTIFIER);
  if (resultset_cache_size) {
    oss << "\"query_resultset\": " << *resultset_cache_size << " bytes, ";
  }

  // 1.b Join Hash Table Recycler
  auto perfect_join_ht_cache_size =
      PerfectJoinHashTable::getHashTableCache()->getCurrentCacheSizeForDevice(
          CacheItemType::PERFECT_HT, DataRecyclerUtil::CPU_DEVICE_IDENTIFIER);
  auto baseline_join_ht_cache_size =
      BaselineJoinHashTable::getHashTableCache()->getCurrentCacheSizeForDevice(
          CacheItemType::BASELINE_HT, DataRecyclerUtil::CPU_DEVICE_IDENTIFIER);
  auto bbox_intersect_ht_cache_size =
      BoundingBoxIntersectJoinHashTable::getHashTableCache()
          ->getCurrentCacheSizeForDevice(CacheItemType::BBOX_INTERSECT_HT,
                                         DataRecyclerUtil::CPU_DEVICE_IDENTIFIER);
  auto bbox_intersect_ht_tuner_cache_size =
      BoundingBoxIntersectJoinHashTable::getBoundingBoxIntersectTuningParamCache()
          ->getCurrentCacheSizeForDevice(CacheItemType::BBOX_INTERSECT_AUTO_TUNER_PARAM,
                                         DataRecyclerUtil::CPU_DEVICE_IDENTIFIER);
  auto sum_hash_table_cache_size =
      perfect_join_ht_cache_size + baseline_join_ht_cache_size +
      bbox_intersect_ht_cache_size + bbox_intersect_ht_tuner_cache_size;
  oss << "\"hash_tables\": " << sum_hash_table_cache_size << " bytes, ";

  // 1.c Chunk Metadata Recycler
  auto chunk_metadata_cache_size =
      executor->getResultSetRecyclerHolder()
          .getChunkMetadataRecycler()
          ->getCurrentCacheSizeForDevice(CacheItemType::CHUNK_METADATA,
                                         DataRecyclerUtil::CPU_DEVICE_IDENTIFIER);
  oss << "\"chunk_metadata\": " << chunk_metadata_cache_size << " bytes, ";

  // 2. Query Plan Dag
  auto query_plan_dag_cache_size =
      executor->getQueryPlanDagCache().getCurrentNodeMapSize();
  oss << "\"query_plan_dag\": " << query_plan_dag_cache_size << " bytes, ";

  // 3. Compiled (GPU) Code
  oss << "\"compiled_GPU code\": "
      << QueryEngine::getInstance()->gpu_code_accessor->getCacheSize() << " bytes, ";

  // 4. String Dictionary
  oss << "\"string_dictionary\": " << cat.getTotalMemorySizeForDictionariesForDatabase()
      << " bytes";
  oss << "}";
  LOG(INFO) << oss.str();
}

void log_system_cpu_memory_status(std::string const& query,
                                  const Catalog_Namespace::Catalog& cat) {
  if (g_allow_memory_status_log) {
    std::ostringstream oss;
    oss << query << "\n" << cat.getDataMgr().getSystemMemoryUsage();
    LOG(INFO) << oss.str();
    log_cache_size(cat);
  }
}
}  // namespace

void DBHandler::fillGeoColumns(
    const TSessionId& session_id,
    const Catalog& catalog,
    std::vector<std::unique_ptr<import_export::TypedImportBuffer>>& import_buffers,
    const ColumnDescriptor* cd,
    size_t& col_idx,
    size_t num_rows,
    const std::string& table_name) {
  auto geo_col_idx = col_idx - 1;
  const auto wkt_or_wkb_hex_column = import_buffers[geo_col_idx]->getGeoStringBuffer();
  std::vector<std::vector<double>> coords_column, bounds_column;
  std::vector<std::vector<int>> ring_sizes_column, poly_rings_column;
  SQLTypeInfo ti = cd->columnType;
  const bool validate_with_geos_if_available = false;
  if (num_rows != wkt_or_wkb_hex_column->size() ||
      !Geospatial::GeoTypesFactory::getGeoColumns(wkt_or_wkb_hex_column,
                                                  ti,
                                                  coords_column,
                                                  bounds_column,
                                                  ring_sizes_column,
                                                  poly_rings_column,
                                                  validate_with_geos_if_available)) {
    std::ostringstream oss;
    oss << "Invalid geometry in column " << cd->columnName;
    THROW_DB_EXCEPTION(oss.str());
  }

  // Populate physical columns, advance col_idx
  import_export::Importer::set_geo_physical_import_buffer_columnar(catalog,
                                                                   cd,
                                                                   import_buffers,
                                                                   col_idx,
                                                                   coords_column,
                                                                   bounds_column,
                                                                   ring_sizes_column,
                                                                   poly_rings_column);
}

void DBHandler::fillMissingBuffers(
    const TSessionId& session_id,
    const Catalog& catalog,
    std::vector<std::unique_ptr<import_export::TypedImportBuffer>>& import_buffers,
    const std::list<const ColumnDescriptor*>& cds,
    const std::vector<int>& desc_id_to_column_id,
    size_t num_rows,
    const std::string& table_name) {
  size_t skip_physical_cols = 0;
  size_t col_idx = 0, import_idx = 0;
  for (const auto& cd : cds) {
    if (skip_physical_cols > 0) {
      CHECK(cd->isGeoPhyCol);
      skip_physical_cols--;
      continue;
    } else if (cd->columnType.is_geometry()) {
      skip_physical_cols = cd->columnType.get_physical_cols();
    }
    if (desc_id_to_column_id[import_idx] == -1) {
      import_buffers[col_idx]->addDefaultValues(cd, num_rows);
      col_idx++;
      if (cd->columnType.is_geometry()) {
        fillGeoColumns(
            session_id, catalog, import_buffers, cd, col_idx, num_rows, table_name);
      }
    } else {
      col_idx++;
      col_idx += skip_physical_cols;
    }
    import_idx++;
  }
}

namespace {
std::string get_load_tag(const std::string& load_tag, const std::string& table_name) {
  std::ostringstream oss;
  oss << load_tag << "(" << table_name << ")";
  return oss.str();
}

std::string get_import_tag(const std::string& import_tag,
                           const std::string& table_name,
                           const std::string& file_path) {
  std::ostringstream oss;
  oss << import_tag << "(" << table_name << ", file_path:" << file_path << ")";
  return oss.str();
}
}  // namespace

void DBHandler::load_table_binary(const TSessionId& session_id_or_json,
                                  const std::string& table_name,
                                  const std::vector<TRow>& rows,
                                  const std::vector<std::string>& column_names) {
  try {
    heavyai::RequestInfo const request_info(session_id_or_json);
    SET_REQUEST_ID(request_info.requestId());
    auto stdlog =
        STDLOG(get_session_ptr(request_info.sessionId()), "table_name", table_name);
    stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
    auto session_ptr = stdlog.getConstSessionInfo();

    if (rows.empty()) {
      THROW_DB_EXCEPTION("No rows to insert");
    }

    const auto execute_read_lock = legacylockmgr::getExecuteReadLock();
    std::unique_ptr<import_export::Loader> loader;
    std::vector<std::unique_ptr<import_export::TypedImportBuffer>> import_buffers;
    auto schema_read_lock = prepare_loader_generic(*session_ptr,
                                                   table_name,
                                                   rows.front().cols.size(),
                                                   &loader,
                                                   &import_buffers,
                                                   column_names,
                                                   "load_table_binary");

    auto col_descs = loader->get_column_descs();
    auto desc_id_to_column_id = column_ids_by_names(col_descs, column_names);

    size_t rows_completed = 0;
    auto const load_tag = get_load_tag("load_table_binary", table_name);
    log_system_cpu_memory_status("start_" + load_tag, session_ptr->getCatalog());
    ScopeGuard cleanup = [&load_tag, &session_ptr]() {
      log_system_cpu_memory_status("finish_" + load_tag, session_ptr->getCatalog());
    };
    for (auto const& row : rows) {
      size_t col_idx = 0;
      try {
        for (auto cd : col_descs) {
          auto mapped_idx = desc_id_to_column_id[col_idx];
          if (mapped_idx != -1) {
            import_buffers[col_idx]->add_value(
                cd, row.cols[mapped_idx], row.cols[mapped_idx].is_null);
          }
          col_idx++;
        }
        rows_completed++;
      } catch (const std::exception& e) {
        for (size_t col_idx_to_pop = 0; col_idx_to_pop < col_idx; ++col_idx_to_pop) {
          import_buffers[col_idx_to_pop]->pop_value();
        }
        LOG(ERROR) << "Input exception thrown: " << e.what()
                   << ". Row discarded, issue at column : " << (col_idx + 1)
                   << " data :" << row;
      }
    }
    fillMissingBuffers(request_info.sessionId(),
                       session_ptr->getCatalog(),
                       import_buffers,
                       col_descs,
                       desc_id_to_column_id,
                       rows_completed,
                       table_name);
    auto insert_data_lock = lockmgr::InsertDataLockMgr::getWriteLockForTable(
        session_ptr->getCatalog(), table_name);
    if (!loader->load(import_buffers, rows.size(), session_ptr.get())) {
      THROW_DB_EXCEPTION(loader->getErrorMessage());
    }
  } catch (const std::exception& e) {
    THROW_DB_EXCEPTION(std::string(e.what()));
  }
}

std::unique_ptr<lockmgr::AbstractLockContainer<const TableDescriptor*>>
DBHandler::prepare_loader_generic(
    const Catalog_Namespace::SessionInfo& session_info,
    const std::string& table_name,
    size_t num_cols,
    std::unique_ptr<import_export::Loader>* loader,
    std::vector<std::unique_ptr<import_export::TypedImportBuffer>>* import_buffers,
    const std::vector<std::string>& column_names,
    std::string load_type) {
  if (num_cols == 0) {
    THROW_DB_EXCEPTION("No columns to insert");
  }
  check_read_only(load_type);
  auto& cat = session_info.getCatalog();
  auto td_with_lock =
      std::make_unique<lockmgr::TableSchemaLockContainer<lockmgr::ReadLock>>(
          lockmgr::TableSchemaLockContainer<lockmgr::ReadLock>::acquireTableDescriptor(
              cat, table_name, true));
  const auto td = (*td_with_lock)();
  CHECK(td);

  if (g_cluster && !leaf_aggregator_.leafCount()) {
    // Sharded table rows need to be routed to the leaf by an aggregator.
    check_table_not_sharded(td);
  }
  check_table_load_privileges(session_info, table_name);

  loader->reset(new import_export::Loader(cat, td));

  auto col_descs = (*loader)->get_column_descs();
  check_valid_column_names(col_descs, column_names);
  if (column_names.empty()) {
    // TODO(andrew): nColumns should be number of non-virtual/non-system columns.
    //               Subtracting 1 (rowid) until TableDescriptor is updated.
    auto geo_physical_cols = std::count_if(
        col_descs.begin(), col_descs.end(), [](auto cd) { return cd->isGeoPhyCol; });
    const auto num_table_cols = static_cast<size_t>(td->nColumns) - geo_physical_cols -
                                (td->hasDeletedCol ? 2 : 1);
    if (num_cols != num_table_cols) {
      throw std::runtime_error("Number of columns to load (" + std::to_string(num_cols) +
                               ") does not match number of columns in table " +
                               td->tableName + " (" + std::to_string(num_table_cols) +
                               ")");
    }
  } else if (num_cols != column_names.size()) {
    THROW_DB_EXCEPTION(
        "Number of columns specified does not match the "
        "number of columns given (" +
        std::to_string(num_cols) + " vs " + std::to_string(column_names.size()) + ")");
  }

  *import_buffers = import_export::setup_column_loaders(td, loader->get());
  return std::move(td_with_lock);
}
namespace {

size_t get_column_size(const TColumn& column) {
  if (!column.nulls.empty()) {
    return column.nulls.size();
  } else {
    // it is a very bold estimate but later we check it against REAL data
    // and if this function returns a wrong result (e.g. both int and string
    // vectors are filled with values), we get an error
    return column.data.int_col.size() + column.data.arr_col.size() +
           column.data.real_col.size() + column.data.str_col.size();
  }
}

}  // namespace

void DBHandler::load_table_binary_columnar(const TSessionId& session_id_or_json,
                                           const std::string& table_name,
                                           const std::vector<TColumn>& cols,
                                           const std::vector<std::string>& column_names) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  auto stdlog =
      STDLOG(get_session_ptr(request_info.sessionId()), "table_name", table_name);
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  auto session_ptr = stdlog.getConstSessionInfo();

  const auto execute_read_lock = legacylockmgr::getExecuteReadLock();
  std::unique_ptr<import_export::Loader> loader;
  std::vector<std::unique_ptr<import_export::TypedImportBuffer>> import_buffers;
  auto schema_read_lock = prepare_loader_generic(*session_ptr,
                                                 table_name,
                                                 cols.size(),
                                                 &loader,
                                                 &import_buffers,
                                                 column_names,
                                                 "load_table_binary_columnar");

  auto desc_id_to_column_id =
      column_ids_by_names(loader->get_column_descs(), column_names);
  size_t num_rows = get_column_size(cols.front());
  size_t import_idx = 0;  // index into the TColumn vector being loaded
  size_t col_idx = 0;     // index into column description vector
  auto const load_tag = get_load_tag("load_table_binary_columnar", table_name);
  log_system_cpu_memory_status("start_" + load_tag, session_ptr->getCatalog());
  ScopeGuard cleanup = [&load_tag, &session_ptr]() {
    log_system_cpu_memory_status("finish_" + load_tag, session_ptr->getCatalog());
  };
  try {
    size_t skip_physical_cols = 0;
    for (auto cd : loader->get_column_descs()) {
      if (skip_physical_cols > 0) {
        CHECK(cd->isGeoPhyCol);
        skip_physical_cols--;
        continue;
      }
      auto mapped_idx = desc_id_to_column_id[import_idx];
      if (mapped_idx != -1) {
        size_t col_rows = import_buffers[col_idx]->add_values(cd, cols[mapped_idx]);
        if (col_rows != num_rows) {
          std::ostringstream oss;
          oss << "load_table_binary_columnar: Inconsistent number of rows in column "
              << cd->columnName << " ,  expecting " << num_rows << " rows, column "
              << col_idx << " has " << col_rows << " rows";
          THROW_DB_EXCEPTION(oss.str());
        }
        // Advance to the next column in the table
        col_idx++;
        // For geometry columns: process WKT strings and fill physical columns
        if (cd->columnType.is_geometry()) {
          fillGeoColumns(request_info.sessionId(),
                         session_ptr->getCatalog(),
                         import_buffers,
                         cd,
                         col_idx,
                         num_rows,
                         table_name);
          skip_physical_cols = cd->columnType.get_physical_cols();
        }
      } else {
        col_idx++;
        if (cd->columnType.is_geometry()) {
          skip_physical_cols = cd->columnType.get_physical_cols();
          col_idx += skip_physical_cols;
        }
      }
      // Advance to the next column of values being loaded
      import_idx++;
    }
  } catch (const std::exception& e) {
    std::ostringstream oss;
    oss << "load_table_binary_columnar: Input exception thrown: " << e.what()
        << ". Issue at column : " << (col_idx + 1) << ". Import aborted";
    THROW_DB_EXCEPTION(oss.str());
  }
  fillMissingBuffers(request_info.sessionId(),
                     session_ptr->getCatalog(),
                     import_buffers,
                     loader->get_column_descs(),
                     desc_id_to_column_id,
                     num_rows,
                     table_name);
  auto insert_data_lock = lockmgr::InsertDataLockMgr::getWriteLockForTable(
      session_ptr->getCatalog(), table_name);
  if (!loader->load(import_buffers, num_rows, session_ptr.get())) {
    THROW_DB_EXCEPTION(loader->getErrorMessage());
  }
}

using RecordBatchVector = std::vector<std::shared_ptr<arrow::RecordBatch>>;

#define ARROW_THRIFT_THROW_NOT_OK(s) \
  do {                               \
    ::arrow::Status _s = (s);        \
    if (UNLIKELY(!_s.ok())) {        \
      TDBException ex;               \
      ex.error_msg = _s.ToString();  \
      LOG(ERROR) << s.ToString();    \
      throw ex;                      \
    }                                \
  } while (0)

namespace {

RecordBatchVector loadArrowStream(const std::string& stream) {
  RecordBatchVector batches;
  try {
    // TODO(wesm): Make this simpler in general, see ARROW-1600
    auto stream_buffer =
        std::make_shared<arrow::Buffer>(reinterpret_cast<const uint8_t*>(stream.c_str()),
                                        static_cast<int64_t>(stream.size()));

    arrow::io::BufferReader buf_reader(stream_buffer);
    std::shared_ptr<arrow::RecordBatchReader> batch_reader;
    ARROW_ASSIGN_OR_THROW(batch_reader,
                          arrow::ipc::RecordBatchStreamReader::Open(&buf_reader));

    while (true) {
      std::shared_ptr<arrow::RecordBatch> batch;
      // Read batch (zero-copy) from the stream
      ARROW_THRIFT_THROW_NOT_OK(batch_reader->ReadNext(&batch));
      if (batch == nullptr) {
        break;
      }
      batches.emplace_back(std::move(batch));
    }
  } catch (const std::exception& e) {
    LOG(ERROR) << "Error parsing Arrow stream: " << e.what() << ". Import aborted";
  }
  return batches;
}

}  // namespace

void DBHandler::load_table_binary_arrow(const TSessionId& session_id_or_json,
                                        const std::string& table_name,
                                        const std::string& arrow_stream,
                                        const bool use_column_names) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  auto stdlog =
      STDLOG(get_session_ptr(request_info.sessionId()), "table_name", table_name);
  auto session_ptr = stdlog.getConstSessionInfo();

  RecordBatchVector batches = loadArrowStream(arrow_stream);
  // Assuming have one batch for now
  if (batches.size() != 1) {
    THROW_DB_EXCEPTION("Expected a single Arrow record batch. Import aborted");
  }

  std::shared_ptr<arrow::RecordBatch> batch = batches[0];
  std::unique_ptr<import_export::Loader> loader;
  std::vector<std::unique_ptr<import_export::TypedImportBuffer>> import_buffers;
  std::vector<std::string> column_names;
  if (use_column_names) {
    column_names = batch->schema()->field_names();
  }
  const auto execute_read_lock = legacylockmgr::getExecuteReadLock();
  auto schema_read_lock =
      prepare_loader_generic(*session_ptr,
                             table_name,
                             static_cast<size_t>(batch->num_columns()),
                             &loader,
                             &import_buffers,
                             column_names,
                             "load_table_binary_arrow");

  auto desc_id_to_column_id =
      column_ids_by_names(loader->get_column_descs(), column_names);
  size_t num_rows = 0;

  // col_idx indexes "desc_id_to_column_id"
  size_t col_idx = 0;
  auto const load_tag = get_load_tag("load_table_binary_arrow", table_name);
  log_system_cpu_memory_status("start_" + load_tag, session_ptr->getCatalog());
  ScopeGuard cleanup = [&load_tag, &session_ptr]() {
    log_system_cpu_memory_status("finish_" + load_tag, session_ptr->getCatalog());
  };
  try {
    for (auto cd : loader->get_column_descs()) {
      if (cd->isGeoPhyCol) {
        // Skip in the case of "cd" being a physical cols, as they are generated
        // in fillGeoColumns:
        //  * Point: coords col
        //  * MultiPoint/LineString: coords/bounds cols
        //  etc...
        continue;
      }
      auto mapped_idx = desc_id_to_column_id[col_idx];
      if (mapped_idx != -1) {
        auto& array = *batch->column(mapped_idx);
        import_export::ArraySliceRange row_slice(0, array.length());

        // col_id indexes "import_buffers"
        size_t col_id = cd->columnId;

        // When importing a buffer with "add_arrow_values", the index in
        // "importing_buffers" is given by the "columnId" attribute of a ColumnDescriptor.
        // This index will differ from "col_idx" if any of the importing columns is a
        // geometry column as they have physical columns for other properties (i.e. a
        // LineString also has "coords" and "bounds").
        num_rows = import_buffers[col_id - 1]->add_arrow_values(
            cd, array, true, row_slice, nullptr);
        // For geometry columns: process WKT strings and fill physical columns
        if (cd->columnType.is_geometry()) {
          fillGeoColumns(request_info.sessionId(),
                         session_ptr->getCatalog(),
                         import_buffers,
                         cd,
                         col_id,
                         num_rows,
                         table_name);
        }
      }
      // Advance to the next column in the table
      col_idx++;
    }
  } catch (const std::exception& e) {
    LOG(ERROR) << "Input exception thrown: " << e.what()
               << ". Issue at column : " << (col_idx + 1) << ". Import aborted";
    // TODO(tmostak): Go row-wise on binary columnar import to be consistent with our
    // other import paths
    THROW_DB_EXCEPTION(e.what());
  }
  fillMissingBuffers(request_info.sessionId(),
                     session_ptr->getCatalog(),
                     import_buffers,
                     loader->get_column_descs(),
                     desc_id_to_column_id,
                     num_rows,
                     table_name);
  auto insert_data_lock = lockmgr::InsertDataLockMgr::getWriteLockForTable(
      session_ptr->getCatalog(), table_name);
  if (!loader->load(import_buffers, num_rows, session_ptr.get())) {
    THROW_DB_EXCEPTION(loader->getErrorMessage());
  }
}

void DBHandler::load_table(const TSessionId& session_id_or_json,
                           const std::string& table_name,
                           const std::vector<TStringRow>& rows,
                           const std::vector<std::string>& column_names) {
  try {
    heavyai::RequestInfo const request_info(session_id_or_json);
    SET_REQUEST_ID(request_info.requestId());
    auto stdlog =
        STDLOG(get_session_ptr(request_info.sessionId()), "table_name", table_name);
    stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
    auto session_ptr = stdlog.getConstSessionInfo();

    if (rows.empty()) {
      THROW_DB_EXCEPTION("No rows to insert");
    }
    auto const load_tag = get_load_tag("load_table", table_name);
    log_system_cpu_memory_status("start_" + load_tag, session_ptr->getCatalog());
    ScopeGuard cleanup = [&load_tag, &session_ptr]() {
      log_system_cpu_memory_status("finish_" + load_tag, session_ptr->getCatalog());
    };
    const auto execute_read_lock = legacylockmgr::getExecuteReadLock();
    std::unique_ptr<import_export::Loader> loader;
    std::vector<std::unique_ptr<import_export::TypedImportBuffer>> import_buffers;
    auto schema_read_lock =
        prepare_loader_generic(*session_ptr,
                               table_name,
                               static_cast<size_t>(rows.front().cols.size()),
                               &loader,
                               &import_buffers,
                               column_names,
                               "load_table");

    auto col_descs = loader->get_column_descs();
    auto desc_id_to_column_id = column_ids_by_names(col_descs, column_names);
    import_export::CopyParams copy_params;
    size_t rows_completed = 0;
    for (auto const& row : rows) {
      size_t import_idx = 0;  // index into the TStringRow being loaded
      size_t col_idx = 0;     // index into column description vector
      try {
        size_t skip_physical_cols = 0;
        for (auto cd : col_descs) {
          if (skip_physical_cols > 0) {
            CHECK(cd->isGeoPhyCol);
            skip_physical_cols--;
            continue;
          }
          auto mapped_idx = desc_id_to_column_id[import_idx];
          if (mapped_idx != -1) {
            import_buffers[col_idx]->add_value(cd,
                                               row.cols[mapped_idx].str_val,
                                               row.cols[mapped_idx].is_null,
                                               copy_params);
          }
          col_idx++;
          if (cd->columnType.is_geometry()) {
            // physical geo columns will be filled separately lately
            skip_physical_cols = cd->columnType.get_physical_cols();
            col_idx += skip_physical_cols;
          }
          // Advance to the next field within the row
          import_idx++;
        }
        rows_completed++;
      } catch (const std::exception& e) {
        LOG(ERROR) << "Input exception thrown: " << e.what()
                   << ". Row discarded, issue at column : " << (col_idx + 1)
                   << " data :" << row;
        THROW_DB_EXCEPTION(std::string("Exception: ") + e.what());
      }
    }
    // do batch filling of geo columns separately
    if (rows.size() != 0) {
      const auto& row = rows[0];
      size_t col_idx = 0;  // index into column description vector
      try {
        size_t import_idx = 0;
        size_t skip_physical_cols = 0;
        for (auto cd : col_descs) {
          if (skip_physical_cols > 0) {
            skip_physical_cols--;
            continue;
          }
          auto mapped_idx = desc_id_to_column_id[import_idx];
          col_idx++;
          if (cd->columnType.is_geometry()) {
            skip_physical_cols = cd->columnType.get_physical_cols();
            if (mapped_idx != -1) {
              fillGeoColumns(request_info.sessionId(),
                             session_ptr->getCatalog(),
                             import_buffers,
                             cd,
                             col_idx,
                             rows_completed,
                             table_name);
            } else {
              col_idx += skip_physical_cols;
            }
          }
          import_idx++;
        }
      } catch (const std::exception& e) {
        LOG(ERROR) << "Input exception thrown: " << e.what()
                   << ". Row discarded, issue at column : " << (col_idx + 1)
                   << " data :" << row;
        THROW_DB_EXCEPTION(e.what());
      }
    }
    fillMissingBuffers(request_info.sessionId(),
                       session_ptr->getCatalog(),
                       import_buffers,
                       col_descs,
                       desc_id_to_column_id,
                       rows_completed,
                       table_name);
    auto insert_data_lock = lockmgr::InsertDataLockMgr::getWriteLockForTable(
        session_ptr->getCatalog(), table_name);
    if (!loader->load(import_buffers, rows_completed, session_ptr.get())) {
      THROW_DB_EXCEPTION(loader->getErrorMessage());
    }

  } catch (const std::exception& e) {
    THROW_DB_EXCEPTION(std::string(e.what()));
  }
}

char DBHandler::unescape_char(std::string str) {
  char out = str[0];
  if (str.size() == 2 && str[0] == '\\') {
    if (str[1] == 't') {
      out = '\t';
    } else if (str[1] == 'n') {
      out = '\n';
    } else if (str[1] == '0') {
      out = '\0';
    } else if (str[1] == '\'') {
      out = '\'';
    } else if (str[1] == '\\') {
      out = '\\';
    }
  }
  return out;
}

import_export::CopyParams DBHandler::thrift_to_copyparams(const TCopyParams& cp) {
  import_export::CopyParams copy_params;
  switch (cp.has_header) {
    case TImportHeaderRow::AUTODETECT:
      copy_params.has_header = import_export::ImportHeaderRow::kAutoDetect;
      break;
    case TImportHeaderRow::NO_HEADER:
      copy_params.has_header = import_export::ImportHeaderRow::kNoHeader;
      break;
    case TImportHeaderRow::HAS_HEADER:
      copy_params.has_header = import_export::ImportHeaderRow::kHasHeader;
      break;
    default:
      CHECK(false);
  }
  copy_params.quoted = cp.quoted;
  if (cp.delimiter.length() > 0) {
    copy_params.delimiter = unescape_char(cp.delimiter);
  } else {
    copy_params.delimiter = '\0';
  }
  if (cp.null_str.length() > 0) {
    copy_params.null_str = cp.null_str;
  }
  if (cp.quote.length() > 0) {
    copy_params.quote = unescape_char(cp.quote);
  }
  if (cp.escape.length() > 0) {
    copy_params.escape = unescape_char(cp.escape);
  }
  if (cp.line_delim.length() > 0) {
    copy_params.line_delim = unescape_char(cp.line_delim);
  }
  if (cp.array_delim.length() > 0) {
    copy_params.array_delim = unescape_char(cp.array_delim);
  }
  if (cp.array_begin.length() > 0) {
    copy_params.array_begin = unescape_char(cp.array_begin);
  }
  if (cp.array_end.length() > 0) {
    copy_params.array_end = unescape_char(cp.array_end);
  }
  if (cp.threads != 0) {
    copy_params.threads = cp.threads;
  }
  if (cp.s3_access_key.length() > 0) {
    copy_params.s3_access_key = cp.s3_access_key;
  }
  if (cp.s3_secret_key.length() > 0) {
    copy_params.s3_secret_key = cp.s3_secret_key;
  }
  if (cp.s3_session_token.length() > 0) {
    copy_params.s3_session_token = cp.s3_session_token;
  }
  if (cp.s3_region.length() > 0) {
    copy_params.s3_region = cp.s3_region;
  }
  if (cp.s3_endpoint.length() > 0) {
    copy_params.s3_endpoint = cp.s3_endpoint;
  }
#ifdef HAVE_AWS_S3
  if (g_allow_s3_server_privileges && cp.s3_access_key.length() == 0 &&
      cp.s3_secret_key.length() == 0 && cp.s3_session_token.length() == 0) {
    const auto& server_credentials =
        Aws::Auth::DefaultAWSCredentialsProviderChain().GetAWSCredentials();
    copy_params.s3_access_key = server_credentials.GetAWSAccessKeyId();
    copy_params.s3_secret_key = server_credentials.GetAWSSecretKey();
    copy_params.s3_session_token = server_credentials.GetSessionToken();
  }
#endif

  switch (cp.source_type) {
    case TSourceType::DELIMITED_FILE:
      copy_params.source_type = import_export::SourceType::kDelimitedFile;
      break;
    case TSourceType::GEO_FILE:
      copy_params.source_type = import_export::SourceType::kGeoFile;
      break;
    case TSourceType::PARQUET_FILE:
#ifdef ENABLE_IMPORT_PARQUET
      copy_params.source_type = import_export::SourceType::kParquetFile;
      break;
#else
      THROW_DB_EXCEPTION("Parquet not supported");
#endif
    case TSourceType::ODBC:
      THROW_DB_EXCEPTION("ODBC source not supported");
    case TSourceType::RASTER_FILE:
      copy_params.source_type = import_export::SourceType::kRasterFile;
      break;
    default:
      CHECK(false);
  }

  switch (cp.geo_coords_encoding) {
    case TEncodingType::GEOINT:
      copy_params.geo_coords_encoding = kENCODING_GEOINT;
      break;
    case TEncodingType::NONE:
      copy_params.geo_coords_encoding = kENCODING_NONE;
      break;
    default:
      THROW_DB_EXCEPTION("Invalid geo_coords_encoding in TCopyParams: " +
                         std::to_string((int)cp.geo_coords_encoding));
  }
  copy_params.geo_coords_comp_param = cp.geo_coords_comp_param;
  switch (cp.geo_coords_type) {
    case TDatumType::GEOGRAPHY:
      copy_params.geo_coords_type = kGEOGRAPHY;
      break;
    case TDatumType::GEOMETRY:
      copy_params.geo_coords_type = kGEOMETRY;
      break;
    default:
      THROW_DB_EXCEPTION("Invalid geo_coords_type in TCopyParams: " +
                         std::to_string((int)cp.geo_coords_type));
  }
  switch (cp.geo_coords_srid) {
    case 4326:
    case 3857:
    case 900913:
      copy_params.geo_coords_srid = cp.geo_coords_srid;
      break;
    default:
      THROW_DB_EXCEPTION("Invalid geo_coords_srid in TCopyParams (" +
                         std::to_string((int)cp.geo_coords_srid));
  }
  copy_params.sanitize_column_names = cp.sanitize_column_names;
  copy_params.geo_layer_name = cp.geo_layer_name;
  copy_params.geo_explode_collections = cp.geo_explode_collections;
  copy_params.source_srid = cp.source_srid;
  switch (cp.raster_point_type) {
    case TRasterPointType::NONE:
      copy_params.raster_point_type = import_export::RasterPointType::kNone;
      break;
    case TRasterPointType::AUTO:
      copy_params.raster_point_type = import_export::RasterPointType::kAuto;
      break;
    case TRasterPointType::SMALLINT:
      copy_params.raster_point_type = import_export::RasterPointType::kSmallInt;
      break;
    case TRasterPointType::INT:
      copy_params.raster_point_type = import_export::RasterPointType::kInt;
      break;
    case TRasterPointType::FLOAT:
      copy_params.raster_point_type = import_export::RasterPointType::kFloat;
      break;
    case TRasterPointType::DOUBLE:
      copy_params.raster_point_type = import_export::RasterPointType::kDouble;
      break;
    case TRasterPointType::POINT:
      copy_params.raster_point_type = import_export::RasterPointType::kPoint;
      break;
    default:
      CHECK(false);
  }
  copy_params.raster_import_bands = cp.raster_import_bands;
  if (cp.raster_scanlines_per_thread < 0) {
    THROW_DB_EXCEPTION("Invalid raster_scanlines_per_thread in TCopyParams (" +
                       std::to_string((int)cp.raster_scanlines_per_thread));
  } else {
    copy_params.raster_scanlines_per_thread = cp.raster_scanlines_per_thread;
  }
  switch (cp.raster_point_transform) {
    case TRasterPointTransform::NONE:
      copy_params.raster_point_transform = import_export::RasterPointTransform::kNone;
      break;
    case TRasterPointTransform::AUTO:
      copy_params.raster_point_transform = import_export::RasterPointTransform::kAuto;
      break;
    case TRasterPointTransform::FILE:
      copy_params.raster_point_transform = import_export::RasterPointTransform::kFile;
      break;
    case TRasterPointTransform::WORLD:
      copy_params.raster_point_transform = import_export::RasterPointTransform::kWorld;
      break;
    default:
      CHECK(false);
  }
  copy_params.raster_point_compute_angle = cp.raster_point_compute_angle;
  copy_params.raster_import_dimensions = cp.raster_import_dimensions;
  copy_params.dsn = cp.odbc_dsn;
  copy_params.connection_string = cp.odbc_connection_string;
  copy_params.sql_select = cp.odbc_sql_select;
  copy_params.sql_order_by = cp.odbc_sql_order_by;
  copy_params.username = cp.odbc_username;
  copy_params.password = cp.odbc_password;
  copy_params.credential_string = cp.odbc_credential_string;
  copy_params.add_metadata_columns = cp.add_metadata_columns;
  copy_params.trim_spaces = cp.trim_spaces;
  copy_params.geo_validate_geometry = cp.geo_validate_geometry;
  copy_params.raster_drop_if_all_null = cp.raster_drop_if_all_null;
  return copy_params;
}

TCopyParams DBHandler::copyparams_to_thrift(const import_export::CopyParams& cp) {
  TCopyParams copy_params;
  copy_params.delimiter = cp.delimiter;
  copy_params.null_str = cp.null_str;
  switch (cp.has_header) {
    case import_export::ImportHeaderRow::kAutoDetect:
      copy_params.has_header = TImportHeaderRow::AUTODETECT;
      break;
    case import_export::ImportHeaderRow::kNoHeader:
      copy_params.has_header = TImportHeaderRow::NO_HEADER;
      break;
    case import_export::ImportHeaderRow::kHasHeader:
      copy_params.has_header = TImportHeaderRow::HAS_HEADER;
      break;
    default:
      CHECK(false);
  }
  copy_params.quoted = cp.quoted;
  copy_params.quote = cp.quote;
  copy_params.escape = cp.escape;
  copy_params.line_delim = cp.line_delim;
  copy_params.array_delim = cp.array_delim;
  copy_params.array_begin = cp.array_begin;
  copy_params.array_end = cp.array_end;
  copy_params.threads = cp.threads;
  copy_params.s3_access_key = cp.s3_access_key;
  copy_params.s3_secret_key = cp.s3_secret_key;
  copy_params.s3_session_token = cp.s3_session_token;
  copy_params.s3_region = cp.s3_region;
  copy_params.s3_endpoint = cp.s3_endpoint;
  switch (cp.source_type) {
    case import_export::SourceType::kDelimitedFile:
      copy_params.source_type = TSourceType::DELIMITED_FILE;
      break;
    case import_export::SourceType::kGeoFile:
      copy_params.source_type = TSourceType::GEO_FILE;
      break;
    case import_export::SourceType::kParquetFile:
      copy_params.source_type = TSourceType::PARQUET_FILE;
      break;
    case import_export::SourceType::kRasterFile:
      copy_params.source_type = TSourceType::RASTER_FILE;
      break;
    case import_export::SourceType::kOdbc:
      copy_params.source_type = TSourceType::ODBC;
      break;
    default:
      CHECK(false);
  }
  switch (cp.geo_coords_encoding) {
    case kENCODING_GEOINT:
      copy_params.geo_coords_encoding = TEncodingType::GEOINT;
      break;
    default:
      copy_params.geo_coords_encoding = TEncodingType::NONE;
      break;
  }
  copy_params.geo_coords_comp_param = cp.geo_coords_comp_param;
  switch (cp.geo_coords_type) {
    case kGEOGRAPHY:
      copy_params.geo_coords_type = TDatumType::GEOGRAPHY;
      break;
    case kGEOMETRY:
      copy_params.geo_coords_type = TDatumType::GEOMETRY;
      break;
    default:
      CHECK(false);
  }
  copy_params.geo_coords_srid = cp.geo_coords_srid;
  copy_params.sanitize_column_names = cp.sanitize_column_names;
  copy_params.geo_layer_name = cp.geo_layer_name;
  copy_params.geo_assign_render_groups = false;
  copy_params.geo_explode_collections = cp.geo_explode_collections;
  copy_params.source_srid = cp.source_srid;
  switch (cp.raster_point_type) {
    case import_export::RasterPointType::kNone:
      copy_params.raster_point_type = TRasterPointType::NONE;
      break;
    case import_export::RasterPointType::kAuto:
      copy_params.raster_point_type = TRasterPointType::AUTO;
      break;
    case import_export::RasterPointType::kSmallInt:
      copy_params.raster_point_type = TRasterPointType::SMALLINT;
      break;
    case import_export::RasterPointType::kInt:
      copy_params.raster_point_type = TRasterPointType::INT;
      break;
    case import_export::RasterPointType::kFloat:
      copy_params.raster_point_type = TRasterPointType::FLOAT;
      break;
    case import_export::RasterPointType::kDouble:
      copy_params.raster_point_type = TRasterPointType::DOUBLE;
      break;
    case import_export::RasterPointType::kPoint:
      copy_params.raster_point_type = TRasterPointType::POINT;
      break;
    default:
      CHECK(false);
  }
  copy_params.raster_import_bands = cp.raster_import_bands;
  copy_params.raster_scanlines_per_thread = cp.raster_scanlines_per_thread;
  switch (cp.raster_point_transform) {
    case import_export::RasterPointTransform::kNone:
      copy_params.raster_point_transform = TRasterPointTransform::NONE;
      break;
    case import_export::RasterPointTransform::kAuto:
      copy_params.raster_point_transform = TRasterPointTransform::AUTO;
      break;
    case import_export::RasterPointTransform::kFile:
      copy_params.raster_point_transform = TRasterPointTransform::FILE;
      break;
    case import_export::RasterPointTransform::kWorld:
      copy_params.raster_point_transform = TRasterPointTransform::WORLD;
      break;
    default:
      CHECK(false);
  }
  copy_params.raster_point_compute_angle = cp.raster_point_compute_angle;
  copy_params.raster_import_dimensions = cp.raster_import_dimensions;
  copy_params.odbc_dsn = cp.dsn;
  copy_params.odbc_connection_string = cp.connection_string;
  copy_params.odbc_sql_select = cp.sql_select;
  copy_params.odbc_sql_order_by = cp.sql_order_by;
  copy_params.odbc_username = cp.username;
  copy_params.odbc_password = cp.password;
  copy_params.odbc_credential_string = cp.credential_string;
  copy_params.add_metadata_columns = cp.add_metadata_columns;
  copy_params.trim_spaces = cp.trim_spaces;
  copy_params.geo_validate_geometry = cp.geo_validate_geometry;
  copy_params.raster_drop_if_all_null = cp.raster_drop_if_all_null;
  return copy_params;
}

namespace {
void add_vsi_network_prefix(std::string& path) {
  // do we support network file access?
  bool gdal_network = Geospatial::GDAL::supportsNetworkFileAccess();

  // modify head of filename based on source location
  if (boost::istarts_with(path, "http://") || boost::istarts_with(path, "https://")) {
    if (!gdal_network) {
      THROW_DB_EXCEPTION(
          "HTTP geo file import not supported! Update to GDAL 2.2 or later!");
    }
    // invoke GDAL CURL virtual file reader
    path = "/vsicurl/" + path;
  } else if (boost::istarts_with(path, "s3://")) {
    if (!gdal_network) {
      THROW_DB_EXCEPTION(
          "S3 geo file import not supported! Update to GDAL 2.2 or later!");
    }
    // invoke GDAL S3 virtual file reader
    boost::replace_first(path, "s3://", "/vsis3/");
  }
}

void add_vsi_geo_prefix(std::string& path) {
  // single gzip'd file (not an archive)?
  if (boost::iends_with(path, ".gz") && !boost::iends_with(path, ".tar.gz")) {
    path = "/vsigzip/" + path;
  }
}

void add_vsi_archive_prefix(std::string& path) {
  // check for compressed file or file bundle
  if (boost::iends_with(path, ".zip")) {
    // zip archive
    path = "/vsizip/" + path;
  } else if (boost::iends_with(path, ".tar") || boost::iends_with(path, ".tgz") ||
             boost::iends_with(path, ".tar.gz")) {
    // tar archive (compressed or uncompressed)
    path = "/vsitar/" + path;
  }
}

std::string remove_vsi_prefixes(const std::string& path_in) {
  std::string path(path_in);

  // these will be first
  if (boost::istarts_with(path, "/vsizip/")) {
    boost::replace_first(path, "/vsizip/", "");
  } else if (boost::istarts_with(path, "/vsitar/")) {
    boost::replace_first(path, "/vsitar/", "");
  } else if (boost::istarts_with(path, "/vsigzip/")) {
    boost::replace_first(path, "/vsigzip/", "");
  }

  // then these
  if (boost::istarts_with(path, "/vsicurl/")) {
    boost::replace_first(path, "/vsicurl/", "");
  } else if (boost::istarts_with(path, "/vsis3/")) {
    boost::replace_first(path, "/vsis3/", "s3://");
  }

  return path;
}

bool path_is_relative(const std::string& path) {
  if (boost::istarts_with(path, "s3://") || boost::istarts_with(path, "http://") ||
      boost::istarts_with(path, "https://")) {
    return false;
  }
  return !boost::filesystem::path(path).is_absolute();
}

bool path_has_valid_filename(const std::string& path) {
  auto filename = boost::filesystem::path(path).filename().string();
  if (filename.size() == 0 || filename[0] == '.' || filename[0] == '/') {
    return false;
  }
  return true;
}

bool is_a_supported_geo_file(const std::string& path) {
  if (!path_has_valid_filename(path)) {
    return false;
  }
  // this is now just for files that we want to recognize
  // as geo when inside an archive (see below)
  // @TODO(se) make this more flexible?
  if (boost::iends_with(path, ".shp") || boost::iends_with(path, ".geojson") ||
      boost::iends_with(path, ".json") || boost::iends_with(path, ".kml") ||
      boost::iends_with(path, ".kmz") || boost::iends_with(path, ".gdb") ||
      boost::iends_with(path, ".gdb.zip") || boost::iends_with(path, ".fgb")) {
    return true;
  }
  return false;
}

bool is_a_supported_archive_file(const std::string& path) {
  if (!path_has_valid_filename(path)) {
    return false;
  }
  if (boost::iends_with(path, ".zip") && !boost::iends_with(path, ".gdb.zip")) {
    return true;
  } else if (boost::iends_with(path, ".tar") || boost::iends_with(path, ".tgz") ||
             boost::iends_with(path, ".tar.gz")) {
    return true;
  }
  return false;
}

std::string find_first_geo_file_in_archive(const std::string& archive_path,
                                           const import_export::CopyParams& copy_params) {
  // get the recursive list of all files in the archive
  std::vector<std::string> files =
      import_export::Importer::gdalGetAllFilesInArchive(archive_path, copy_params);

  // report the list
  LOG(INFO) << "Found " << files.size() << " files in Archive "
            << remove_vsi_prefixes(archive_path);
  for (const auto& file : files) {
    LOG(INFO) << "  " << file;
  }

  // scan the list for the first candidate file
  bool found_suitable_file = false;
  std::string file_name;
  for (const auto& file : files) {
    if (is_a_supported_geo_file(file)) {
      file_name = file;
      found_suitable_file = true;
      break;
    }
  }

  // if we didn't find anything
  if (!found_suitable_file) {
    LOG(INFO) << "Failed to find any supported geo files in Archive: " +
                     remove_vsi_prefixes(archive_path);
    file_name.clear();
  }

  // done
  return file_name;
}

bool is_local_file(const std::string& file_path) {
  return (!boost::istarts_with(file_path, "s3://") &&
          !boost::istarts_with(file_path, "http://") &&
          !boost::istarts_with(file_path, "https://"));
}

void validate_import_file_path_if_local(const std::string& file_path) {
  if (is_local_file(file_path)) {
    ddl_utils::validate_allowed_file_path(
        file_path, ddl_utils::DataTransferType::IMPORT, true);
  }
}
}  // namespace

void DBHandler::detect_column_types(TDetectResult& _return,
                                    const TSessionId& session_id_or_json,
                                    const std::string& file_name_in,
                                    const TCopyParams& cp) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  auto stdlog = STDLOG(get_session_ptr(request_info.sessionId()));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  check_read_only("detect_column_types");

  bool is_raster = false;
  boost::filesystem::path file_path;
  import_export::CopyParams copy_params = thrift_to_copyparams(cp);
  if (copy_params.source_type != import_export::SourceType::kOdbc) {
    std::string file_name{file_name_in};
    if (path_is_relative(file_name)) {
      // assume relative paths are relative to data_path / import / <session>
      auto temp_file_path = import_path_ /
                            picosha2::hash256_hex_string(request_info.sessionId()) /
                            boost::filesystem::path(file_name).filename();
      file_name = temp_file_path.string();
    }
    validate_import_file_path_if_local(file_name);

    if ((copy_params.source_type == import_export::SourceType::kGeoFile ||
         copy_params.source_type == import_export::SourceType::kRasterFile) &&
        is_local_file(file_name)) {
      const shared::FilePathOptions options{copy_params.regex_path_filter,
                                            copy_params.file_sort_order_by,
                                            copy_params.file_sort_regex};
      auto file_paths = shared::local_glob_filter_sort_files(file_name, options, false);
      // For geo and raster detect, pick the first file, if multiple files are provided
      // (e.g. through file globbing).
      CHECK(!file_paths.empty());
      file_name = file_paths[0];
    }

    // if it's a geo or raster import, handle alternative paths (S3, HTTP, archive etc.)
    if (copy_params.source_type == import_export::SourceType::kGeoFile) {
      if (is_a_supported_archive_file(file_name)) {
        // find the archive file
        add_vsi_network_prefix(file_name);
        if (!import_export::Importer::gdalFileExists(file_name, copy_params)) {
          THROW_DB_EXCEPTION("Archive does not exist: " + file_name_in);
        }
        // find geo file in archive
        add_vsi_archive_prefix(file_name);
        std::string geo_file = find_first_geo_file_in_archive(file_name, copy_params);
        // prepare to detect that geo file
        if (geo_file.size()) {
          file_name = file_name + std::string("/") + geo_file;
        }
      } else {
        // prepare to detect geo file directly
        add_vsi_network_prefix(file_name);
        add_vsi_geo_prefix(file_name);
      }
    } else if (copy_params.source_type == import_export::SourceType::kRasterFile) {
      // prepare to detect raster file directly
      add_vsi_network_prefix(file_name);
      add_vsi_geo_prefix(file_name);
      is_raster = true;
    }

    file_path = boost::filesystem::path(file_name);
    // can be a s3 url
    if (!boost::istarts_with(file_name, "s3://")) {
      if (!boost::filesystem::path(file_name).is_absolute()) {
        file_path = import_path_ /
                    picosha2::hash256_hex_string(request_info.sessionId()) /
                    boost::filesystem::path(file_name).filename();
        file_name = file_path.string();
      }

      if (copy_params.source_type == import_export::SourceType::kGeoFile ||
          copy_params.source_type == import_export::SourceType::kRasterFile) {
        // check for geo or raster file
        if (!import_export::Importer::gdalFileOrDirectoryExists(file_name, copy_params)) {
          THROW_DB_EXCEPTION("File or directory \"" + file_path.string() +
                             "\" does not exist.")
        }
      } else {
        // check for regular file
        if (!shared::file_or_glob_path_exists(file_path.string())) {
          THROW_DB_EXCEPTION("File or directory \"" + file_path.string() +
                             "\" does not exist.");
        }
      }
    }
  }

  try {
    if (copy_params.source_type == import_export::SourceType::kDelimitedFile
#ifdef ENABLE_IMPORT_PARQUET
        || (copy_params.source_type == import_export::SourceType::kParquetFile)
#endif
    ) {
      import_export::Detector detector(file_path, copy_params);
      auto best_types = detector.getBestColumnTypes();
      std::vector<std::string> headers = detector.get_headers();
      copy_params = detector.get_copy_params();

      _return.copy_params = copyparams_to_thrift(copy_params);
      _return.row_set.row_desc.resize(best_types.size());
      for (size_t col_idx = 0; col_idx < best_types.size(); col_idx++) {
        TColumnType col;
        auto& ti = best_types[col_idx];
        col.col_type.precision = ti.get_precision();
        col.col_type.scale = ti.get_scale();
        col.col_type.comp_param = ti.get_comp_param();
        if (ti.is_geometry()) {
          // set this so encoding_to_thrift does the right thing
          ti.set_compression(copy_params.geo_coords_encoding);
          // fill in these directly
          col.col_type.precision = static_cast<int>(copy_params.geo_coords_type);
          col.col_type.scale = copy_params.geo_coords_srid;
          col.col_type.comp_param = copy_params.geo_coords_comp_param;
        }
        col.col_type.type = type_to_thrift(ti);
        col.col_type.encoding = encoding_to_thrift(ti);
        if (ti.is_array()) {
          col.col_type.is_array = true;
        }
        if (copy_params.sanitize_column_names) {
          col.col_name = ImportHelpers::sanitize_name(headers[col_idx]);
        } else {
          col.col_name = headers[col_idx];
        }
        col.is_reserved_keyword = ImportHelpers::is_reserved_name(col.col_name);
        _return.row_set.row_desc[col_idx] = col;
      }
      auto sample_data = detector.get_sample_rows(shared::kDefaultSampleRowsCount);

      TRow sample_row;
      for (auto row : sample_data) {
        sample_row.cols.clear();
        for (const auto& s : row) {
          TDatum td;
          td.val.str_val = s;
          td.is_null = s.empty();
          sample_row.cols.push_back(td);
        }
        _return.row_set.rows.push_back(sample_row);
      }
    } else if (copy_params.source_type == import_export::SourceType::kGeoFile ||
               copy_params.source_type == import_export::SourceType::kRasterFile) {
      check_geospatial_files(file_path, copy_params);
      std::list<ColumnDescriptor> cds = import_export::Importer::gdalToColumnDescriptors(
          file_path.string(), is_raster, Geospatial::kGeoColumnName, copy_params);
      for (auto cd : cds) {
        if (copy_params.sanitize_column_names) {
          cd.columnName = ImportHelpers::sanitize_name(cd.columnName);
        }
        _return.row_set.row_desc.push_back(populateThriftColumnType(nullptr, &cd));
      }
      if (!is_raster) {
        // @TODO(se) support for raster?
        std::map<std::string, std::vector<std::string>> sample_data;
        import_export::Importer::readMetadataSampleGDAL(file_path.string(),
                                                        Geospatial::kGeoColumnName,
                                                        sample_data,
                                                        shared::kDefaultSampleRowsCount,
                                                        copy_params);
        if (sample_data.size() > 0) {
          for (size_t i = 0; i < sample_data.begin()->second.size(); i++) {
            TRow sample_row;
            for (auto cd : cds) {
              TDatum td;
              td.val.str_val = sample_data[cd.sourceName].at(i);
              td.is_null = td.val.str_val.empty();
              sample_row.cols.push_back(td);
            }
            _return.row_set.rows.push_back(sample_row);
          }
        }
      }
      _return.copy_params = copyparams_to_thrift(copy_params);
    }
  } catch (const std::exception& e) {
    THROW_DB_EXCEPTION("detect_column_types error: " + std::string(e.what()));
  }
}

void DBHandler::render_vega(TRenderResult& _return,
                            const TSessionId& session_id_or_json,
                            const int64_t widget_id,
                            const std::string& vega_json,
                            const int compression_level,
                            const std::string& nonce) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  auto stdlog = STDLOG(get_session_ptr(request_info.sessionId()),
                       "widget_id",
                       widget_id,
                       "compression_level",
                       compression_level,
                       "vega_json",
                       vega_json,
                       "nonce",
                       nonce);
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  stdlog.appendNameValuePairs("nonce", nonce);
  if (!render_handler_) {
    THROW_DB_EXCEPTION("Backend rendering is disabled.");
  }

  // cast away const-ness of incoming Thrift string ref
  // to allow it to be passed down as an r-value and
  // ultimately std::moved into the RenderSession
  auto& non_const_vega_json = const_cast<std::string&>(vega_json);

  _return.total_time_ms = measure<>::execution([&]() {
    try {
      render_handler_->render_vega(_return,
                                   stdlog.getSessionInfo(),
                                   widget_id,
                                   std::move(non_const_vega_json),
                                   compression_level,
                                   nonce);
    } catch (std::exception& e) {
      THROW_DB_EXCEPTION(e.what());
    }
  });
}

static bool is_allowed_on_dashboard(const Catalog_Namespace::SessionInfo& session_info,
                                    int32_t dashboard_id,
                                    AccessPrivileges requestedPermissions) {
  DBObject object(dashboard_id, DashboardDBObjectType);
  auto& catalog = session_info.getCatalog();
  auto& user = session_info.get_currentUser();
  object.loadKey(catalog);
  object.setPrivileges(requestedPermissions);
  std::vector<DBObject> privs = {object};
  return SysCatalog::instance().checkPrivileges(user, privs);
}

// custom expressions
namespace {
using Catalog_Namespace::CustomExpression;
using Catalog_Namespace::DataSourceType;

std::unique_ptr<Catalog_Namespace::CustomExpression> create_custom_expr_from_thrift_obj(
    const TCustomExpression& t_custom_expr,
    const Catalog& catalog) {
  if (t_custom_expr.data_source_name.empty()) {
    THROW_DB_EXCEPTION("Custom expression data source name cannot be empty.")
  }
  CHECK(t_custom_expr.data_source_type == TDataSourceType::type::TABLE)
      << "Unexpected data source type: "
      << static_cast<int>(t_custom_expr.data_source_type);
  auto td = catalog.getMetadataForTable(t_custom_expr.data_source_name, false);
  if (!td) {
    THROW_DB_EXCEPTION("Custom expression references a table \"" +
                       t_custom_expr.data_source_name + "\" that does not exist.")
  }
  DataSourceType data_source_type = DataSourceType::TABLE;
  return std::make_unique<CustomExpression>(
      t_custom_expr.name, t_custom_expr.expression_json, data_source_type, td->tableId);
}

TCustomExpression create_thrift_obj_from_custom_expr(const CustomExpression& custom_expr,
                                                     const Catalog& catalog) {
  TCustomExpression t_custom_expr;
  t_custom_expr.id = custom_expr.id;
  t_custom_expr.name = custom_expr.name;
  t_custom_expr.expression_json = custom_expr.expression_json;
  t_custom_expr.data_source_id = custom_expr.data_source_id;
  t_custom_expr.is_deleted = custom_expr.is_deleted;
  CHECK(custom_expr.data_source_type == DataSourceType::TABLE)
      << "Unexpected data source type: "
      << static_cast<int>(custom_expr.data_source_type);
  t_custom_expr.data_source_type = TDataSourceType::type::TABLE;
  auto td = catalog.getMetadataForTable(custom_expr.data_source_id, false);
  if (td) {
    t_custom_expr.data_source_name = td->tableName;
  } else {
    LOG(WARNING)
        << "Custom expression references a deleted data source. Custom expression id: "
        << custom_expr.id << ", name: " << custom_expr.name;
  }
  return t_custom_expr;
}
}  // namespace

int32_t DBHandler::create_custom_expression(const TSessionId& session_id_or_json,
                                            const TCustomExpression& t_custom_expr) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  auto stdlog = STDLOG(get_session_ptr(request_info.sessionId()));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  check_read_only("create_custom_expression");

  auto session_ptr = stdlog.getConstSessionInfo();
  if (!session_ptr->get_currentUser().isSuper) {
    THROW_DB_EXCEPTION("Custom expressions can only be created by super users.")
  }
  auto& catalog = session_ptr->getCatalog();
  heavyai::unique_lock<heavyai::shared_mutex> write_lock(custom_expressions_mutex_);
  return catalog.createCustomExpression(
      create_custom_expr_from_thrift_obj(t_custom_expr, catalog));
}

void DBHandler::get_custom_expressions(std::vector<TCustomExpression>& _return,
                                       const TSessionId& session_id_or_json) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  auto stdlog = STDLOG(get_session_ptr(request_info.sessionId()));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());

  auto session_ptr = stdlog.getConstSessionInfo();
  auto& catalog = session_ptr->getCatalog();
  heavyai::shared_lock<heavyai::shared_mutex> read_lock(custom_expressions_mutex_);
  auto custom_expressions =
      catalog.getCustomExpressionsForUser(session_ptr->get_currentUser());
  for (const auto& custom_expression : custom_expressions) {
    _return.emplace_back(create_thrift_obj_from_custom_expr(*custom_expression, catalog));
  }
}

void DBHandler::update_custom_expression(const TSessionId& session_id_or_json,
                                         const int32_t id,
                                         const std::string& expression_json) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  auto stdlog = STDLOG(get_session_ptr(request_info.sessionId()));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  check_read_only("update_custom_expression");

  auto session_ptr = stdlog.getConstSessionInfo();
  if (!session_ptr->get_currentUser().isSuper) {
    THROW_DB_EXCEPTION("Custom expressions can only be updated by super users.")
  }
  auto& catalog = session_ptr->getCatalog();
  heavyai::unique_lock<heavyai::shared_mutex> write_lock(custom_expressions_mutex_);
  catalog.updateCustomExpression(id, expression_json);
}

void DBHandler::delete_custom_expressions(
    const TSessionId& session_id_or_json,
    const std::vector<int32_t>& custom_expression_ids,
    const bool do_soft_delete) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  auto stdlog = STDLOG(get_session_ptr(request_info.sessionId()));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  check_read_only("delete_custom_expressions");

  auto session_ptr = stdlog.getConstSessionInfo();
  if (!session_ptr->get_currentUser().isSuper) {
    THROW_DB_EXCEPTION("Custom expressions can only be deleted by super users.")
  }
  auto& catalog = session_ptr->getCatalog();
  heavyai::unique_lock<heavyai::shared_mutex> write_lock(custom_expressions_mutex_);
  catalog.deleteCustomExpressions(custom_expression_ids, do_soft_delete);
}

// dashboards
void DBHandler::get_dashboard(TDashboard& dashboard,
                              const TSessionId& session_id_or_json,
                              const int32_t dashboard_id) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  auto stdlog = STDLOG(get_session_ptr(request_info.sessionId()));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  auto session_ptr = stdlog.getConstSessionInfo();
  auto const& cat = session_ptr->getCatalog();
  Catalog_Namespace::UserMetadata user_meta;
  auto dash = cat.getMetadataForDashboard(dashboard_id);
  if (!dash) {
    THROW_DB_EXCEPTION("Dashboard with dashboard id " + std::to_string(dashboard_id) +
                       " doesn't exist");
  }
  if (!is_allowed_on_dashboard(
          *session_ptr, dash->dashboardId, AccessPrivileges::VIEW_DASHBOARD)) {
    THROW_DB_EXCEPTION("User has no view privileges for the dashboard with id " +
                       std::to_string(dashboard_id));
  }
  user_meta.userName = "";
  SysCatalog::instance().getMetadataForUserById(dash->userId, user_meta);
  dashboard = get_dashboard_impl(session_ptr, user_meta, dash);
}

void DBHandler::get_dashboards(std::vector<TDashboard>& dashboards,
                               const TSessionId& session_id_or_json) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  auto stdlog = STDLOG(get_session_ptr(request_info.sessionId()));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  auto session_ptr = stdlog.getConstSessionInfo();
  auto const& cat = session_ptr->getCatalog();
  Catalog_Namespace::UserMetadata user_meta;
  const auto dashes = cat.getAllDashboardsMetadata();
  user_meta.userName = "";
  for (const auto dash : dashes) {
    if (is_allowed_on_dashboard(
            *session_ptr, dash->dashboardId, AccessPrivileges::VIEW_DASHBOARD)) {
      // dashboardState is intentionally not populated here
      // for payload reasons
      // use get_dashboard call to get state
      dashboards.push_back(get_dashboard_impl(session_ptr, user_meta, dash, false));
    }
  }
}

TDashboard DBHandler::get_dashboard_impl(
    const std::shared_ptr<Catalog_Namespace::SessionInfo const>& session_ptr,
    Catalog_Namespace::UserMetadata& user_meta,
    const DashboardDescriptor* dash,
    const bool populate_state) {
  auto const& cat = session_ptr->getCatalog();
  SysCatalog::instance().getMetadataForUserById(dash->userId, user_meta);
  auto objects_list = SysCatalog::instance().getMetadataForObject(
      cat.getCurrentDB().dbId,
      static_cast<int>(DBObjectType::DashboardDBObjectType),
      dash->dashboardId);
  TDashboard dashboard;
  dashboard.dashboard_name = dash->dashboardName;
  if (populate_state) {
    dashboard.dashboard_state = dash->dashboardState;
  }
  dashboard.image_hash = dash->imageHash;
  dashboard.update_time = dash->updateTime;
  dashboard.dashboard_metadata = dash->dashboardMetadata;
  dashboard.dashboard_id = dash->dashboardId;
  dashboard.dashboard_owner = dash->user;
  TDashboardPermissions perms;
  // Super user has all permissions.
  if (session_ptr->get_currentUser().isSuper) {
    perms.create_ = true;
    perms.delete_ = true;
    perms.edit_ = true;
    perms.view_ = true;
  } else {
    // Collect all grants on current user
    // add them to the permissions.
    auto obj_to_find =
        DBObject(dashboard.dashboard_id, DBObjectType::DashboardDBObjectType);
    obj_to_find.loadKey(cat);
    std::vector<std::string> grantees =
        SysCatalog::instance().getRoles(true,
                                        session_ptr->get_currentUser().isSuper,
                                        session_ptr->get_currentUser().userName);
    for (const auto& grantee : grantees) {
      DBObject* object_found;
      auto* gr = SysCatalog::instance().getGrantee(grantee);
      if (gr && (object_found = gr->findDbObject(obj_to_find.getObjectKey(), true))) {
        const auto obj_privs = object_found->getPrivileges();
        perms.create_ |= obj_privs.hasPermission(DashboardPrivileges::CREATE_DASHBOARD);
        perms.delete_ |= obj_privs.hasPermission(DashboardPrivileges::DELETE_DASHBOARD);
        perms.edit_ |= obj_privs.hasPermission(DashboardPrivileges::EDIT_DASHBOARD);
        perms.view_ |= obj_privs.hasPermission(DashboardPrivileges::VIEW_DASHBOARD);
      }
    }
  }
  dashboard.dashboard_permissions = perms;
  if (objects_list.empty() ||
      (objects_list.size() == 1 && objects_list[0]->roleName == user_meta.userName)) {
    dashboard.is_dash_shared = false;
  } else {
    dashboard.is_dash_shared = true;
  }
  return dashboard;
}

namespace dbhandler {
bool is_info_schema_db(const std::string& db_name) {
  return (db_name == shared::kInfoSchemaDbName &&
          SysCatalog::instance().hasExecutedMigration(shared::kInfoSchemaMigrationName));
}

void check_not_info_schema_db(const std::string& db_name, bool throw_db_exception) {
  if (is_info_schema_db(db_name)) {
    std::string error_message{"Write requests/queries are not allowed in the " +
                              shared::kInfoSchemaDbName + " database."};
    if (throw_db_exception) {
      THROW_DB_EXCEPTION(error_message)
    } else {
      throw std::runtime_error(error_message);
    }
  }
}
}  // namespace dbhandler

int32_t DBHandler::create_dashboard(const TSessionId& session_id_or_json,
                                    const std::string& dashboard_name,
                                    const std::string& dashboard_state,
                                    const std::string& image_hash,
                                    const std::string& dashboard_metadata) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  auto stdlog = STDLOG(get_session_ptr(request_info.sessionId()));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  auto session_ptr = stdlog.getConstSessionInfo();
  CHECK(session_ptr);
  check_read_only("create_dashboard");
  auto& cat = session_ptr->getCatalog();
  if (!g_allow_system_dashboard_update) {
    dbhandler::check_not_info_schema_db(cat.name(), true);
  }

  if (!session_ptr->checkDBAccessPrivileges(DBObjectType::DashboardDBObjectType,
                                            AccessPrivileges::CREATE_DASHBOARD)) {
    THROW_DB_EXCEPTION("Not enough privileges to create a dashboard.");
  }

  if (dashboard_exists(cat, session_ptr->get_currentUser().userId, dashboard_name)) {
    THROW_DB_EXCEPTION("Dashboard with name: " + dashboard_name + " already exists.");
  }

  DashboardDescriptor dd;
  dd.dashboardName = dashboard_name;
  dd.dashboardState = dashboard_state;
  dd.imageHash = image_hash;
  dd.dashboardMetadata = dashboard_metadata;
  dd.userId = session_ptr->get_currentUser().userId;
  dd.user = session_ptr->get_currentUser().userName;

  try {
    auto id = cat.createDashboard(dd);
    // TODO: transactionally unsafe
    SysCatalog::instance().createDBObject(
        session_ptr->get_currentUser(), dashboard_name, DashboardDBObjectType, cat, id);
    return id;
  } catch (const std::exception& e) {
    THROW_DB_EXCEPTION(e.what());
  }
}

void DBHandler::replace_dashboard(const TSessionId& session_id_or_json,
                                  const int32_t dashboard_id,
                                  const std::string& dashboard_name,
                                  const std::string& dashboard_owner,
                                  const std::string& dashboard_state,
                                  const std::string& image_hash,
                                  const std::string& dashboard_metadata) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  auto stdlog = STDLOG(get_session_ptr(request_info.sessionId()));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  auto session_ptr = stdlog.getConstSessionInfo();
  CHECK(session_ptr);
  check_read_only("replace_dashboard");
  auto& cat = session_ptr->getCatalog();
  if (!g_allow_system_dashboard_update) {
    dbhandler::check_not_info_schema_db(cat.name(), true);
  }

  if (!is_allowed_on_dashboard(
          *session_ptr, dashboard_id, AccessPrivileges::EDIT_DASHBOARD)) {
    THROW_DB_EXCEPTION("Not enough privileges to replace a dashboard.");
  }

  if (auto dash = cat.getMetadataForDashboard(
          std::to_string(session_ptr->get_currentUser().userId), dashboard_name)) {
    if (dash->dashboardId != dashboard_id) {
      THROW_DB_EXCEPTION("Dashboard with name: " + dashboard_name + " already exists.");
    }
  }

  DashboardDescriptor dd;
  dd.dashboardName = dashboard_name;
  dd.dashboardState = dashboard_state;
  dd.imageHash = image_hash;
  dd.dashboardMetadata = dashboard_metadata;
  Catalog_Namespace::UserMetadata user;
  if (!SysCatalog::instance().getMetadataForUser(dashboard_owner, user)) {
    THROW_DB_EXCEPTION(std::string("Dashboard owner ") + dashboard_owner +
                       " does not exist");
  }
  dd.userId = user.userId;
  dd.user = dashboard_owner;
  dd.dashboardId = dashboard_id;

  try {
    cat.replaceDashboard(dd);
  } catch (const std::exception& e) {
    THROW_DB_EXCEPTION(e.what());
  }
}

void DBHandler::delete_dashboard(const TSessionId& session_id_or_json,
                                 const int32_t dashboard_id) {
  delete_dashboards(session_id_or_json, {dashboard_id});
}

void DBHandler::delete_dashboards(const TSessionId& session_id_or_json,
                                  const std::vector<int32_t>& dashboard_ids) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  auto stdlog = STDLOG(get_session_ptr(request_info.sessionId()));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  auto session_ptr = stdlog.getConstSessionInfo();
  check_read_only("delete_dashboards");
  auto& cat = session_ptr->getCatalog();
  if (!g_allow_system_dashboard_update) {
    dbhandler::check_not_info_schema_db(cat.name(), true);
  }
  // Checks will be performed in catalog
  try {
    cat.deleteMetadataForDashboards(dashboard_ids, session_ptr->get_currentUser());
  } catch (const std::exception& e) {
    THROW_DB_EXCEPTION(e.what());
  }
}

std::vector<std::string> DBHandler::get_valid_groups(const TSessionId& session_id_or_json,
                                                     int32_t dashboard_id,
                                                     std::vector<std::string> groups) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  const auto session_info = get_session_copy(request_info.sessionId());
  auto& cat = session_info.getCatalog();
  auto dash = cat.getMetadataForDashboard(dashboard_id);
  if (!dash) {
    THROW_DB_EXCEPTION("Dashboard id " + std::to_string(dashboard_id) +
                       " does not exist");
  } else if (session_info.get_currentUser().userId != dash->userId &&
             !session_info.get_currentUser().isSuper) {
    throw std::runtime_error(
        "User should be either owner of dashboard or super user to share/unshare it");
  }
  std::vector<std::string> valid_groups;
  Catalog_Namespace::UserMetadata user_meta;
  for (auto& group : groups) {
    user_meta.isSuper = false;  // initialize default flag
    if (!SysCatalog::instance().getGrantee(group)) {
      THROW_DB_EXCEPTION("User/Role " + group + " does not exist");
    } else if (!user_meta.isSuper) {
      valid_groups.push_back(group);
    }
  }
  return valid_groups;
}

void DBHandler::validateGroups(const std::vector<std::string>& groups) {
  for (auto const& group : groups) {
    if (!SysCatalog::instance().getGrantee(group)) {
      THROW_DB_EXCEPTION("User/Role '" + group + "' does not exist");
    }
  }
}

void DBHandler::validateDashboardIdsForSharing(
    const Catalog_Namespace::SessionInfo& session_info,
    const std::vector<int32_t>& dashboard_ids) {
  auto& cat = session_info.getCatalog();
  std::map<std::string, std::list<int32_t>> errors;
  for (auto const& dashboard_id : dashboard_ids) {
    auto dashboard = cat.getMetadataForDashboard(dashboard_id);
    if (!dashboard) {
      errors["Dashboard id does not exist"].push_back(dashboard_id);
    } else if (session_info.get_currentUser().userId != dashboard->userId &&
               !session_info.get_currentUser().isSuper) {
      errors["User should be either owner of dashboard or super user to share/unshare it"]
          .push_back(dashboard_id);
    }
  }
  if (!errors.empty()) {
    std::stringstream error_stream;
    error_stream << "Share/Unshare dashboard(s) failed with error(s)\n";
    for (const auto& [error, id_list] : errors) {
      error_stream << "Dashboard ids " << join(id_list, ", ") << ": " << error << "\n";
    }
    THROW_DB_EXCEPTION(error_stream.str());
  }
}

void DBHandler::shareOrUnshareDashboards(const TSessionId& session_id,
                                         const std::vector<int32_t>& dashboard_ids,
                                         const std::vector<std::string>& groups,
                                         const TDashboardPermissions& permissions,
                                         const bool do_share) {
  auto stdlog = STDLOG(get_session_ptr(session_id));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  check_read_only(do_share ? "share_dashboards" : "unshare_dashboards");
  if (!permissions.create_ && !permissions.delete_ && !permissions.edit_ &&
      !permissions.view_) {
    THROW_DB_EXCEPTION("At least one privilege should be assigned for " +
                       std::string(do_share ? "grants" : "revokes"));
  }
  auto session_ptr = stdlog.getConstSessionInfo();
  auto const& catalog = session_ptr->getCatalog();
  auto& sys_catalog = SysCatalog::instance();
  validateGroups(groups);
  validateDashboardIdsForSharing(*session_ptr, dashboard_ids);
  std::vector<DBObject> batch_objects;
  for (auto const& dashboard_id : dashboard_ids) {
    DBObject object(dashboard_id, DBObjectType::DashboardDBObjectType);
    AccessPrivileges privs;
    if (permissions.delete_) {
      privs.add(AccessPrivileges::DELETE_DASHBOARD);
    }
    if (permissions.create_) {
      privs.add(AccessPrivileges::CREATE_DASHBOARD);
    }
    if (permissions.edit_) {
      privs.add(AccessPrivileges::EDIT_DASHBOARD);
    }
    if (permissions.view_) {
      privs.add(AccessPrivileges::VIEW_DASHBOARD);
    }
    object.setPrivileges(privs);
    batch_objects.push_back(object);
  }
  if (do_share) {
    sys_catalog.grantDBObjectPrivilegesBatch(groups, batch_objects, catalog);
  } else {
    sys_catalog.revokeDBObjectPrivilegesBatch(groups, batch_objects, catalog);
  }
}

void DBHandler::share_dashboards(const TSessionId& session_id_or_json,
                                 const std::vector<int32_t>& dashboard_ids,
                                 const std::vector<std::string>& groups,
                                 const TDashboardPermissions& permissions) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  shareOrUnshareDashboards(
      request_info.sessionId(), dashboard_ids, groups, permissions, true);
}

// NOOP: Grants not available for objects as of now
void DBHandler::share_dashboard(const TSessionId& session_id_or_json,
                                const int32_t dashboard_id,
                                const std::vector<std::string>& groups,
                                const std::vector<std::string>& objects,
                                const TDashboardPermissions& permissions,
                                const bool grant_role = false) {
  share_dashboards(session_id_or_json, {dashboard_id}, groups, permissions);
}

void DBHandler::unshare_dashboards(const TSessionId& session_id_or_json,
                                   const std::vector<int32_t>& dashboard_ids,
                                   const std::vector<std::string>& groups,
                                   const TDashboardPermissions& permissions) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  shareOrUnshareDashboards(
      request_info.sessionId(), dashboard_ids, groups, permissions, false);
}

void DBHandler::unshare_dashboard(const TSessionId& session_id_or_json,
                                  const int32_t dashboard_id,
                                  const std::vector<std::string>& groups,
                                  const std::vector<std::string>& objects,
                                  const TDashboardPermissions& permissions) {
  unshare_dashboards(session_id_or_json, {dashboard_id}, groups, permissions);
}

void DBHandler::get_dashboard_grantees(
    std::vector<TDashboardGrantees>& dashboard_grantees,
    const TSessionId& session_id_or_json,
    const int32_t dashboard_id) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  auto stdlog = STDLOG(get_session_ptr(request_info.sessionId()));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  auto session_ptr = stdlog.getConstSessionInfo();
  auto const& cat = session_ptr->getCatalog();
  Catalog_Namespace::UserMetadata user_meta;
  auto dash = cat.getMetadataForDashboard(dashboard_id);
  if (!dash) {
    THROW_DB_EXCEPTION("Dashboard id " + std::to_string(dashboard_id) +
                       " does not exist");
  } else if (session_ptr->get_currentUser().userId != dash->userId &&
             !session_ptr->get_currentUser().isSuper) {
    THROW_DB_EXCEPTION(
        "User should be either owner of dashboard or super user to access grantees");
  }
  std::vector<ObjectRoleDescriptor*> objectsList;
  objectsList = SysCatalog::instance().getMetadataForObject(
      cat.getCurrentDB().dbId,
      static_cast<int>(DBObjectType::DashboardDBObjectType),
      dashboard_id);  // By default objecttypecan be only dashabaords
  user_meta.userId = -1;
  user_meta.userName = "";
  SysCatalog::instance().getMetadataForUserById(dash->userId, user_meta);
  for (auto object : objectsList) {
    if (user_meta.userName == object->roleName) {
      // Mask owner
      continue;
    }
    TDashboardGrantees grantee;
    TDashboardPermissions perm;
    grantee.name = object->roleName;
    grantee.is_user = object->roleType;
    perm.create_ = object->privs.hasPermission(DashboardPrivileges::CREATE_DASHBOARD);
    perm.delete_ = object->privs.hasPermission(DashboardPrivileges::DELETE_DASHBOARD);
    perm.edit_ = object->privs.hasPermission(DashboardPrivileges::EDIT_DASHBOARD);
    perm.view_ = object->privs.hasPermission(DashboardPrivileges::VIEW_DASHBOARD);
    grantee.permissions = perm;
    dashboard_grantees.push_back(grantee);
  }
}

void DBHandler::create_link(std::string& _return,
                            const TSessionId& session_id_or_json,
                            const std::string& view_state,
                            const std::string& view_metadata) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  auto stdlog = STDLOG(get_session_ptr(request_info.sessionId()));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  auto session_ptr = stdlog.getConstSessionInfo();
  // check_read_only("create_link");
  auto& cat = session_ptr->getCatalog();

  LinkDescriptor ld;
  ld.userId = session_ptr->get_currentUser().userId;
  ld.viewState = view_state;
  ld.viewMetadata = view_metadata;

  try {
    _return = cat.createLink(ld, 6);
  } catch (const std::exception& e) {
    THROW_DB_EXCEPTION(e.what());
  }
}

TColumnType DBHandler::create_geo_column(const TDatumType::type type,
                                         const std::string& name,
                                         const bool is_array) {
  TColumnType ct;
  ct.col_name = name;
  ct.col_type.type = type;
  ct.col_type.is_array = is_array;
  return ct;
}

void DBHandler::check_geospatial_files(const boost::filesystem::path file_path,
                                       const import_export::CopyParams& copy_params) {
  const std::list<std::string> shp_ext{".shp", ".shx", ".dbf"};
  if (std::find(shp_ext.begin(),
                shp_ext.end(),
                boost::algorithm::to_lower_copy(file_path.extension().string())) !=
      shp_ext.end()) {
    for (auto ext : shp_ext) {
      auto aux_file = file_path;
      if (!import_export::Importer::gdalFileExists(
              aux_file.replace_extension(boost::algorithm::to_upper_copy(ext)).string(),
              copy_params) &&
          !import_export::Importer::gdalFileExists(
              aux_file.replace_extension(ext).string(), copy_params)) {
        throw std::runtime_error("required file for shapefile does not exist: " +
                                 aux_file.filename().string());
      }
    }
  }
}

void DBHandler::create_table(const TSessionId& session_id_or_json,
                             const std::string& table_name,
                             const TRowDescriptor& rd,
                             const TCreateParams& create_params) {
  heavyai::RequestInfo request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  auto stdlog = STDLOG("table_name", table_name);
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  check_read_only("create_table");

  if (ImportHelpers::is_reserved_name(table_name)) {
    THROW_DB_EXCEPTION("Invalid table name (reserved keyword): " + table_name);
  } else if (table_name != ImportHelpers::sanitize_name(table_name)) {
    THROW_DB_EXCEPTION("Invalid characters in table name: " + table_name);
  }

  auto rds = rd;

  std::string stmt{"CREATE TABLE " + table_name};
  std::vector<std::string> col_stmts;

  for (auto col : rds) {
    if (ImportHelpers::is_reserved_name(col.col_name)) {
      THROW_DB_EXCEPTION("Invalid column name (reserved keyword): " + col.col_name);
    } else if (col.col_name != ImportHelpers::sanitize_name(col.col_name)) {
      THROW_DB_EXCEPTION("Invalid characters in column name: " + col.col_name);
    }
    if (col.col_type.type == TDatumType::INTERVAL_DAY_TIME ||
        col.col_type.type == TDatumType::INTERVAL_YEAR_MONTH) {
      THROW_DB_EXCEPTION("Unsupported type: " + thrift_to_name(col.col_type) +
                         " for column: " + col.col_name);
    }

    if (col.col_type.type == TDatumType::DECIMAL) {
      // if no precision or scale passed in set to default 14,7
      if (col.col_type.precision == 0 && col.col_type.scale == 0) {
        col.col_type.precision = 14;
        col.col_type.scale = 7;
      }
    }

    std::string col_stmt;
    col_stmt.append(col.col_name + " " + thrift_to_name(col.col_type));
    if (col.__isset.default_value) {
      col_stmt.append(" DEFAULT " + col.default_value);
    }

    // As of 2016-06-27 the Immerse v1 frontend does not explicitly set the
    // `nullable` argument, leading this to default to false. Uncomment for v2.
    // if (!col.col_type.nullable) col_stmt.append(" NOT NULL");

    if (thrift_to_encoding(col.col_type.encoding) != kENCODING_NONE) {
      col_stmt.append(" ENCODING " + thrift_to_encoding_name(col.col_type));
      if (thrift_to_encoding(col.col_type.encoding) == kENCODING_DICT ||
          thrift_to_encoding(col.col_type.encoding) == kENCODING_FIXED ||
          thrift_to_encoding(col.col_type.encoding) == kENCODING_GEOINT ||
          thrift_to_encoding(col.col_type.encoding) == kENCODING_DATE_IN_DAYS) {
        col_stmt.append("(" + std::to_string(col.col_type.comp_param) + ")");
      }
    } else if (col.col_type.type == TDatumType::STR) {
      // non DICT encoded strings
      col_stmt.append(" ENCODING NONE");
    } else if (col.col_type.type == TDatumType::POINT ||
               col.col_type.type == TDatumType::MULTIPOINT ||
               col.col_type.type == TDatumType::LINESTRING ||
               col.col_type.type == TDatumType::MULTILINESTRING ||
               col.col_type.type == TDatumType::POLYGON ||
               col.col_type.type == TDatumType::MULTIPOLYGON) {
      // non encoded compressable geo
      if (col.col_type.scale == 4326) {
        col_stmt.append(" ENCODING NONE");
      }
    }
    col_stmts.push_back(col_stmt);
  }

  stmt.append(" (" + boost::algorithm::join(col_stmts, ", ") + ")");

  if (create_params.is_replicated) {
    stmt.append(" WITH (PARTITIONS = 'REPLICATED')");
  }

  stmt.append(";");

  TQueryResult ret;
  request_info.setRequestId(logger::request_id());
  sql_execute(ret, request_info.json(), stmt, true, "", -1, -1);
}

void DBHandler::import_table(const TSessionId& session_id_or_json,
                             const std::string& table_name,
                             const std::string& file_name_in,
                             const TCopyParams& cp) {
  try {
    heavyai::RequestInfo const request_info(session_id_or_json);
    SET_REQUEST_ID(request_info.requestId());
    auto stdlog =
        STDLOG(get_session_ptr(request_info.sessionId()), "table_name", table_name);
    stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
    auto session_ptr = stdlog.getConstSessionInfo();
    check_read_only("import_table");
    LOG(INFO) << "import_table " << table_name << " from " << file_name_in;

    const auto execute_read_lock = legacylockmgr::getExecuteReadLock();
    auto& cat = session_ptr->getCatalog();
    auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID);
    auto start_time = ::toString(std::chrono::system_clock::now());
    if (g_enable_non_kernel_time_query_interrupt) {
      executor->enrollQuerySession(request_info.sessionId(),
                                   "IMPORT_TABLE",
                                   start_time,
                                   Executor::UNITARY_EXECUTOR_ID,
                                   QuerySessionStatus::QueryStatus::RUNNING_IMPORTER);
    }

    ScopeGuard clearInterruptStatus = [executor, &request_info, &start_time] {
      // reset the runtime query interrupt status
      if (g_enable_non_kernel_time_query_interrupt) {
        executor->clearQuerySessionStatus(request_info.sessionId(), start_time);
      }
    };
    const auto td_with_lock =
        lockmgr::TableSchemaLockContainer<lockmgr::ReadLock>::acquireTableDescriptor(
            cat, table_name);
    const auto td = td_with_lock();
    CHECK(td);
    check_table_load_privileges(*session_ptr, table_name);

    std::string copy_from_source;
    import_export::CopyParams copy_params = thrift_to_copyparams(cp);
    if (copy_params.source_type == import_export::SourceType::kOdbc) {
      copy_from_source = copy_params.sql_select;
    } else {
      std::string file_name{file_name_in};
      auto file_path = boost::filesystem::path(file_name);
      if (!boost::istarts_with(file_name, "s3://")) {
        if (!boost::filesystem::path(file_name).is_absolute()) {
          file_path = import_path_ /
                      picosha2::hash256_hex_string(request_info.sessionId()) /
                      boost::filesystem::path(file_name).filename();
          file_name = file_path.string();
        }
        if (!shared::file_or_glob_path_exists(file_path.string())) {
          THROW_DB_EXCEPTION("File or directory \"" + file_path.string() +
                             "\" does not exist.");
        }
      }
      validate_import_file_path_if_local(file_name);

      // TODO(andrew): add delimiter detection to Importer
      if (copy_params.delimiter == '\0') {
        copy_params.delimiter = ',';
        if (boost::filesystem::path(file_path).extension() == ".tsv") {
          copy_params.delimiter = '\t';
        }
      }
      copy_from_source = file_path.string();
    }
    auto const load_tag = get_import_tag("import_table", table_name, copy_from_source);
    log_system_cpu_memory_status("start_" + load_tag, session_ptr->getCatalog());
    ScopeGuard cleanup = [&load_tag, &session_ptr]() {
      log_system_cpu_memory_status("finish_" + load_tag, session_ptr->getCatalog());
    };
    const auto insert_data_lock = lockmgr::InsertDataLockMgr::getWriteLockForTable(
        session_ptr->getCatalog(), table_name);
    std::unique_ptr<import_export::AbstractImporter> importer;
    importer = import_export::create_importer(cat, td, copy_from_source, copy_params);
    auto ms = measure<>::execution([&]() { importer->import(session_ptr.get()); });
    LOG(INFO) << "Total Import Time: " << (double)ms / 1000.0 << " Seconds.";
  } catch (const TDBException& e) {
    throw;
  } catch (const std::exception& e) {
    THROW_DB_EXCEPTION(std::string(e.what()));
  }
}

namespace {

// helper functions for error checking below
// these would usefully be added as methods of TDatumType
// but that's not possible as it's auto-generated by Thrift

bool TTypeInfo_IsGeo(const TDatumType::type& t) {
  return (t == TDatumType::POLYGON || t == TDatumType::MULTIPOLYGON ||
          t == TDatumType::LINESTRING || t == TDatumType::MULTILINESTRING ||
          t == TDatumType::POINT || t == TDatumType::MULTIPOINT);
}

std::string TTypeInfo_TypeToString(const TDatumType::type& t) {
  std::stringstream ss;
  ss << t;
  return ss.str();
}

std::string get_mismatch_attr_warning_text(const std::string& table_name,
                                           const std::string& file_path,
                                           const std::string& column_name,
                                           const std::string& attr,
                                           const std::string& got,
                                           const std::string& expected) {
  return "Issue encountered in geo/raster file '" + file_path +
         "' while appending to table '" + table_name + "'. Column '" + column_name +
         "' " + attr + " mismatch (got '" + got + "', expected '" + expected + "')";
}

}  // namespace

#define THROW_COLUMN_ATTR_MISMATCH_EXCEPTION(attr, got, expected)                        \
  THROW_DB_EXCEPTION("Could not append geo/raster file '" +                              \
                     file_path.filename().string() + "' to table '" + table_name +       \
                     "'. Column '" + cd->columnName + "' " + attr + " mismatch (got '" + \
                     got + "', expected '" + expected + "')");

void DBHandler::import_geo_table(const TSessionId& session_id_or_json,
                                 const std::string& table_name,
                                 const std::string& file_name,
                                 const TCopyParams& cp,
                                 const TRowDescriptor& row_desc,
                                 const TCreateParams& create_params) {
  // this is the direct Thrift endpoint
  // it does NOT support the separate FSI regex/filter/sort options
  // but it DOES support basic globbing specified in the filename itself
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  importGeoTableGlobFilterSort(request_info.sessionId(),
                               table_name,
                               file_name,
                               thrift_to_copyparams(cp),
                               row_desc,
                               create_params);
}

void DBHandler::importGeoTableGlobFilterSort(const TSessionId& session_id,
                                             const std::string& table_name,
                                             const std::string& file_name,
                                             const import_export::CopyParams& copy_params,
                                             const TRowDescriptor& row_desc,
                                             const TCreateParams& create_params) {
  // this is called by the above direct Thrift endpoint
  // and also for a deferred COPY FROM for geo/raster
  // it DOES support the full FSI regex/filter/sort options
  std::vector<std::string> file_names;
  try {
    const shared::FilePathOptions options{copy_params.regex_path_filter,
                                          copy_params.file_sort_order_by,
                                          copy_params.file_sort_regex};
    shared::validate_sort_options(options);
    file_names = shared::local_glob_filter_sort_files(file_name, options, false);
  } catch (const shared::FileNotFoundException& e) {
    // no files match, just try the original filename, might be remote
    file_names.push_back(file_name);
  }
  // import whatever we found
  for (auto const& file_name : file_names) {
    importGeoTableSingle(
        session_id, table_name, file_name, copy_params, row_desc, create_params);
  }
}

void DBHandler::importGeoTableSingle(const TSessionId& session_id,
                                     const std::string& table_name,
                                     const std::string& file_name_in,
                                     const import_export::CopyParams& copy_params,
                                     const TRowDescriptor& row_desc,
                                     const TCreateParams& create_params) {
  auto stdlog = STDLOG(get_session_ptr(session_id), "table_name", table_name);
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  auto session_ptr = stdlog.getConstSessionInfo();
  check_read_only("import_table");

  auto& cat = session_ptr->getCatalog();
  auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID);
  auto start_time = ::toString(std::chrono::system_clock::now());
  if (g_enable_non_kernel_time_query_interrupt) {
    executor->enrollQuerySession(session_id,
                                 "IMPORT_GEO_TABLE",
                                 start_time,
                                 Executor::UNITARY_EXECUTOR_ID,
                                 QuerySessionStatus::QueryStatus::RUNNING_IMPORTER);
  }

  ScopeGuard clearInterruptStatus = [executor, &session_id, &start_time] {
    // reset the runtime query interrupt status
    if (g_enable_non_kernel_time_query_interrupt) {
      executor->clearQuerySessionStatus(session_id, start_time);
    }
  };

  std::string file_name{file_name_in};

  if (path_is_relative(file_name)) {
    // assume relative paths are relative to data_path / import / <session>
    auto file_path = import_path_ / picosha2::hash256_hex_string(session_id) /
                     boost::filesystem::path(file_name).filename();
    file_name = file_path.string();
  }
  validate_import_file_path_if_local(file_name);

  bool is_raster = false;
  if (copy_params.source_type == import_export::SourceType::kGeoFile) {
    if (is_a_supported_archive_file(file_name)) {
      // find the archive file
      add_vsi_network_prefix(file_name);
      if (!import_export::Importer::gdalFileExists(file_name, copy_params)) {
        THROW_DB_EXCEPTION("Archive does not exist: " + file_name_in);
      }
      // find geo file in archive
      add_vsi_archive_prefix(file_name);
      std::string geo_file = find_first_geo_file_in_archive(file_name, copy_params);
      // prepare to load that geo file
      if (geo_file.size()) {
        file_name = file_name + std::string("/") + geo_file;
      }
    } else {
      // prepare to load geo file directly
      add_vsi_network_prefix(file_name);
      add_vsi_geo_prefix(file_name);
    }
  } else if (copy_params.source_type == import_export::SourceType::kRasterFile) {
    // prepare to load geo raster file directly
    add_vsi_network_prefix(file_name);
    add_vsi_geo_prefix(file_name);
    is_raster = true;
  } else {
    THROW_DB_EXCEPTION("import_geo_table called with file_type other than GEO or RASTER");
  }

  // log what we're about to try to do
  VLOG(1) << "import_geo_table: Original filename: " << file_name_in;
  VLOG(1) << "import_geo_table: Actual filename: " << file_name;
  VLOG(1) << "import_geo_table: Raster: " << is_raster;
  auto const load_tag = get_import_tag("import_geo_table", table_name, file_name);
  log_system_cpu_memory_status("start_" + load_tag, session_ptr->getCatalog());
  ScopeGuard cleanup = [&load_tag, &session_ptr]() {
    log_system_cpu_memory_status("finish_" + load_tag, session_ptr->getCatalog());
  };
  // use GDAL to check the primary file exists (even if on S3 and/or in archive)
  auto file_path = boost::filesystem::path(file_name);
  if (!import_export::Importer::gdalFileOrDirectoryExists(file_name, copy_params)) {
    THROW_DB_EXCEPTION("File does not exist: " + file_path.filename().string());
  }

  // use GDAL to check any dependent files exist (ditto)
  try {
    check_geospatial_files(file_path, copy_params);
  } catch (const std::exception& e) {
    THROW_DB_EXCEPTION("import_geo_table error: " + std::string(e.what()));
  }

  // get layer info and deconstruct
  // in general, we will get a combination of layers of these four types:
  //   EMPTY: no rows, report and skip
  //   GEO: create a geo table from this
  //   NON_GEO: create a regular table from this
  //   UNSUPPORTED_GEO: report and skip
  std::vector<import_export::Importer::GeoFileLayerInfo> layer_info;
  if (!is_raster) {
    try {
      layer_info =
          import_export::Importer::gdalGetLayersInGeoFile(file_name, copy_params);
    } catch (const std::exception& e) {
      THROW_DB_EXCEPTION("import_geo_table error: " + std::string(e.what()));
    }
  }

  // categorize the results
  using LayerNameToContentsMap =
      std::map<std::string, import_export::Importer::GeoFileLayerContents>;
  LayerNameToContentsMap load_layers;
  LOG_IF(INFO, layer_info.size() > 0)
      << "import_geo_table: Found the following layers in the geo file:";
  for (const auto& layer : layer_info) {
    switch (layer.contents) {
      case import_export::Importer::GeoFileLayerContents::GEO:
        LOG(INFO) << "import_geo_table:   '" << layer.name
                  << "' (will import as geo table)";
        load_layers[layer.name] = layer.contents;
        break;
      case import_export::Importer::GeoFileLayerContents::NON_GEO:
        LOG(INFO) << "import_geo_table:   '" << layer.name
                  << "' (will import as regular table)";
        load_layers[layer.name] = layer.contents;
        break;
      case import_export::Importer::GeoFileLayerContents::UNSUPPORTED_GEO:
        LOG(WARNING) << "import_geo_table:   '" << layer.name
                     << "' (will not import, unsupported geo type)";
        break;
      case import_export::Importer::GeoFileLayerContents::EMPTY:
        LOG(INFO) << "import_geo_table:   '" << layer.name << "' (ignoring, empty)";
        break;
      default:
        break;
    }
  }

  // if nothing is loadable, stop now
  if (!is_raster && load_layers.size() == 0) {
    THROW_DB_EXCEPTION("import_geo_table: No loadable layers found, aborting!");
  }

  // if we've been given an explicit layer name, check that it exists and is loadable
  // scan the original list, as it may exist but not have been gathered as loadable
  if (!is_raster && copy_params.geo_layer_name.size()) {
    bool found = false;
    for (const auto& layer : layer_info) {
      if (copy_params.geo_layer_name == layer.name) {
        if (layer.contents == import_export::Importer::GeoFileLayerContents::GEO ||
            layer.contents == import_export::Importer::GeoFileLayerContents::NON_GEO) {
          // forget all the other layers and just load this one
          load_layers.clear();
          load_layers[layer.name] = layer.contents;
          found = true;
          break;
        } else if (layer.contents ==
                   import_export::Importer::GeoFileLayerContents::UNSUPPORTED_GEO) {
          THROW_DB_EXCEPTION("import_geo_table: Explicit geo layer '" +
                             copy_params.geo_layer_name + "' has unsupported geo type!");
        } else if (layer.contents ==
                   import_export::Importer::GeoFileLayerContents::EMPTY) {
          THROW_DB_EXCEPTION("import_geo_table: Explicit geo layer '" +
                             copy_params.geo_layer_name + "' is empty!");
        }
      }
    }
    if (!found) {
      THROW_DB_EXCEPTION("import_geo_table: Explicit geo layer '" +
                         copy_params.geo_layer_name + "' not found!");
    }
  }

  // Immerse import of multiple layers is not yet supported
  // @TODO fix this!
  if (!is_raster && row_desc.size() > 0 && load_layers.size() > 1) {
    THROW_DB_EXCEPTION(
        "import_geo_table: Multi-layer geo import not yet supported from Immerse!");
  }

  // one definition of layer table name construction
  // we append the layer name if we're loading more than one table
  auto construct_layer_table_name = [&load_layers](const std::string& table_name,
                                                   const std::string& layer_name) {
    if (load_layers.size() > 1) {
      auto sanitized_layer_name = ImportHelpers::sanitize_name(layer_name);
      if (sanitized_layer_name != layer_name) {
        LOG(INFO) << "import_geo_table: Using sanitized layer name '"
                  << sanitized_layer_name << "' for table name";
      }
      return table_name + "_" + sanitized_layer_name;
    }
    return table_name;
  };

  // if we're importing multiple tables, then NONE of them must exist already
  if (!is_raster && load_layers.size() > 1) {
    for (const auto& layer : load_layers) {
      // construct table name
      auto this_table_name = construct_layer_table_name(table_name, layer.first);

      // table must not exist
      if (cat.getMetadataForTable(this_table_name)) {
        THROW_DB_EXCEPTION("import_geo_table: Table '" + this_table_name +
                           "' already exists, aborting!");
      }
    }
  }

  // prepare to gather errors that would otherwise be exceptions, as we can only throw
  // one
  std::vector<std::string> caught_exception_messages;

  // prepare to time multi-layer import
  double total_import_ms = 0.0;

  // for geo raster, we make a single dummy layer
  // the name is irrelevant, but set it to the filename so the log makes sense
  if (is_raster) {
    CHECK_EQ(load_layers.size(), 0u);
    load_layers.emplace(file_name, import_export::Importer::GeoFileLayerContents::GEO);
  }

  // now we're safe to start importing
  // we loop over the layers we're going to attempt to load
  for (const auto& layer : load_layers) {
    // unpack
    const auto& layer_name = layer.first;
    const auto& layer_contents = layer.second;
    bool is_geo_layer =
        (layer_contents == import_export::Importer::GeoFileLayerContents::GEO);

    // construct table name again
    auto this_table_name = construct_layer_table_name(table_name, layer_name);

    // report
    LOG(INFO) << "import_geo_table: Creating table: " << this_table_name;

    // we need a row descriptor
    TRowDescriptor rd;
    if (row_desc.size() > 0) {
      // we have a valid RowDescriptor
      // this is the case where Immerse has already detected and created
      // all we need to do is import and trust that the data will match
      // use the provided row descriptor
      // table must already exist (we check this below)
      rd = row_desc;
    } else {
      // we don't have a RowDescriptor
      // we have to detect the file ourselves
      TDetectResult cds;
      TCopyParams cp_copy = copyparams_to_thrift(copy_params);
      cp_copy.geo_layer_name = layer_name;
      try {
        detect_column_types(cds, session_id, file_name_in, cp_copy);
      } catch (const std::exception& e) {
        // capture the error and abort this layer
        caught_exception_messages.emplace_back("Column Type Detection failed for '" +
                                               layer_name + "':" + e.what());
        continue;
      }
      rd = cds.row_set.row_desc;

      // then, if the table does NOT already exist, create it
      const TableDescriptor* td = cat.getMetadataForTable(this_table_name);
      if (!td) {
        try {
          create_table(session_id, this_table_name, rd, create_params);
        } catch (const std::exception& e) {
          // capture the error and abort this layer
          caught_exception_messages.emplace_back("Failed to create table for Layer '" +
                                                 layer_name + "':" + e.what());
          continue;
        }
      }
    }

    // match locking sequence for CopyTableStmt::execute
    auto execute_read_lock = legacylockmgr::getExecuteReadLock();

    const TableDescriptor* td{nullptr};
    std::unique_ptr<lockmgr::TableSchemaLockContainer<lockmgr::ReadLock>> td_with_lock;
    std::unique_ptr<lockmgr::WriteLock> insert_data_lock;

    try {
      td_with_lock =
          std::make_unique<lockmgr::TableSchemaLockContainer<lockmgr::ReadLock>>(
              lockmgr::TableSchemaLockContainer<
                  lockmgr::ReadLock>::acquireTableDescriptor(cat, this_table_name));
      td = (*td_with_lock)();
      insert_data_lock = std::make_unique<lockmgr::WriteLock>(
          lockmgr::InsertDataLockMgr::getWriteLockForTable(cat, this_table_name));
    } catch (const std::runtime_error& e) {
      // capture the error and abort this layer
      std::string exception_message = "Could not import geo/raster file '" +
                                      file_path.filename().string() + "' to table '" +
                                      this_table_name +
                                      "'; table does not exist or failed to create.";
      caught_exception_messages.emplace_back(exception_message);
      continue;
    }
    CHECK(td);

    // then, we have to verify that the structure matches
    // get column descriptors (non-system, non-deleted, logical columns only)
    const auto col_descriptors =
        cat.getAllColumnMetadataForTable(td->tableId, false, false, false);

    // first, compare the column count
    if (col_descriptors.size() != rd.size()) {
      // capture the error and abort this layer
      std::string exception_message = "Could not append geo/raster file '" +
                                      file_path.filename().string() + "' to table '" +
                                      this_table_name + "'. Column count mismatch (got " +
                                      std::to_string(rd.size()) + ", expecting " +
                                      std::to_string(col_descriptors.size()) + ")";
      caught_exception_messages.emplace_back(exception_message);
      continue;
    }

    try {
      // validate column type match
      // also handle geo column name changes
      int rd_index = 0;
      for (auto const* cd : col_descriptors) {
        auto const cd_col_type = populateThriftColumnType(&cat, cd);

        // for types, all we care about is that the got and expected types are either both
        // geo or both non-geo, and if they're geo that the exact geo type matches
        auto const gtype = rd[rd_index].col_type.type;  // importer type
        auto const etype = cd_col_type.col_type.type;   // existing table type
        if (TTypeInfo_IsGeo(gtype) && TTypeInfo_IsGeo(etype)) {
          if (gtype != etype) {
            THROW_COLUMN_ATTR_MISMATCH_EXCEPTION(
                "type", TTypeInfo_TypeToString(gtype), TTypeInfo_TypeToString(etype));
          }
        } else if (TTypeInfo_IsGeo(gtype) != TTypeInfo_IsGeo(etype)) {
          THROW_COLUMN_ATTR_MISMATCH_EXCEPTION(
              "type", TTypeInfo_TypeToString(gtype), TTypeInfo_TypeToString(etype));
        }

        // for names, we keep the existing table geo column name (for example, to handle
        // the case where an existing table has a geo column with a legacy name), but all
        // other column names must match, otherwise the import will fail
        auto const gname = rd[rd_index].col_name;  // importer name
        auto const ename = cd->columnName;         // existing table name
        if (gname != ename) {
          if (TTypeInfo_IsGeo(gtype)) {
            LOG(INFO) << "import_geo_table: Renaming incoming geo column to match "
                         "existing table column name '"
                      << ename << "'";
            rd[rd_index].col_name = ename;
          } else {
            if (is_raster) {
              LOG(WARNING) << get_mismatch_attr_warning_text(
                  table_name,
                  file_path.filename().string(),
                  cd->columnName,
                  "name",
                  gname,
                  ename);
            } else {
              THROW_COLUMN_ATTR_MISMATCH_EXCEPTION("name", gname, ename);
            }
          }
        }
        rd_index++;
      }
    } catch (const std::exception& e) {
      // capture the error and abort this layer
      caught_exception_messages.emplace_back(e.what());
      continue;
    }

    std::map<std::string, std::string> colname_to_src;
    for (auto r : rd) {
      colname_to_src[r.col_name] =
          r.src_name.length() > 0 ? r.src_name : ImportHelpers::sanitize_name(r.src_name);
    }

    try {
      check_table_load_privileges(*session_ptr, this_table_name);
    } catch (const std::exception& e) {
      // capture the error and abort this layer
      caught_exception_messages.emplace_back(e.what());
      continue;
    }

    if (!is_raster && is_geo_layer) {
      // Final check to ensure that we have exactly one geo column
      // before doing the actual import, in case the user naively
      // overrode the types in Immerse Preview (which as of 6/17/21
      // it still allows you to do). We should make Immerse more
      // robust and disallow re-typing of columns to/from geo types
      // completely. Currently, if multiple columns are re-typed
      // such that there is still exactly one geo column (but it's
      // the wrong one) then this test will pass, but the import
      // will then reject some (or more likely all) of the rows.
      int num_geo_columns{0};
      for (auto const& col : rd) {
        if (TTypeInfo_IsGeo(col.col_type.type)) {
          num_geo_columns++;
        }
      }
      if (num_geo_columns != 1) {
        std::string exception_message =
            "Table '" + this_table_name +
            "' must have exactly one geo column. Import aborted!";
        caught_exception_messages.emplace_back(exception_message);
        continue;
      }
    }

    std::string layer_or_raster = is_raster ? "Raster" : "Layer";

    try {
      // import this layer only?
      import_export::CopyParams copy_params_copy = copy_params;
      copy_params_copy.geo_layer_name = layer_name;

      // create an importer
      std::unique_ptr<import_export::Importer> importer;
      importer.reset(
          new import_export::Importer(cat, td, file_path.string(), copy_params_copy));

      // import
      auto ms = measure<>::execution(
          [&]() { importer->importGDAL(colname_to_src, session_ptr.get(), is_raster); });
      LOG(INFO) << "Import of " << layer_or_raster << " '" << layer_name << "' took "
                << (double)ms / 1000.0 << "s";
      total_import_ms += ms;
    } catch (const std::exception& e) {
      std::string exception_message = "Import of " + layer_or_raster + " '" +
                                      this_table_name + "' failed: " + e.what();
      caught_exception_messages.emplace_back(exception_message);
      continue;
    }
  }

  // did we catch any exceptions?
  if (caught_exception_messages.size()) {
    // combine all the strings into one and throw a single Thrift exception
    std::string combined_exception_message = "Failed to import geo/raster file: ";
    bool comma{false};
    for (const auto& message : caught_exception_messages) {
      combined_exception_message += comma ? (", " + message) : message;
      comma = true;
    }
    THROW_DB_EXCEPTION(combined_exception_message);
  } else {
    // report success and total time
    LOG(INFO) << "Import Successful!";
    LOG(INFO) << "Total Import Time: " << total_import_ms / 1000.0 << "s";
  }
}

#undef THROW_COLUMN_ATTR_MISMATCH_EXCEPTION

void DBHandler::import_table_status(TImportStatus& _return,
                                    const TSessionId& session_id_or_json,
                                    const std::string& import_id) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  auto stdlog =
      STDLOG(get_session_ptr(request_info.sessionId()), "import_table_status", import_id);
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  auto is = import_export::Importer::get_import_status(import_id);
  _return.elapsed = is.elapsed.count();
  _return.rows_completed = is.rows_completed;
  _return.rows_estimated = is.rows_estimated;
  _return.rows_rejected = is.rows_rejected;
}

void DBHandler::get_first_geo_file_in_archive(std::string& _return,
                                              const TSessionId& session_id_or_json,
                                              const std::string& archive_path_in,
                                              const TCopyParams& copy_params) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  auto stdlog = STDLOG(get_session_ptr(request_info.sessionId()),
                       "get_first_geo_file_in_archive",
                       archive_path_in);
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());

  std::string archive_path(archive_path_in);

  if (path_is_relative(archive_path)) {
    // assume relative paths are relative to data_path / import / <session>
    auto file_path = import_path_ /
                     picosha2::hash256_hex_string(request_info.sessionId()) /
                     boost::filesystem::path(archive_path).filename();
    archive_path = file_path.string();
  }
  validate_import_file_path_if_local(archive_path);

  if (is_a_supported_archive_file(archive_path)) {
    // find the archive file
    add_vsi_network_prefix(archive_path);
    if (!import_export::Importer::gdalFileExists(archive_path,
                                                 thrift_to_copyparams(copy_params))) {
      THROW_DB_EXCEPTION("Archive does not exist: " + archive_path_in);
    }
    // find geo file in archive
    add_vsi_archive_prefix(archive_path);
    std::string geo_file =
        find_first_geo_file_in_archive(archive_path, thrift_to_copyparams(copy_params));
    // what did we get?
    if (geo_file.size()) {
      // prepend it with the original path
      _return = archive_path_in + std::string("/") + geo_file;
    } else {
      // just return the original path
      _return = archive_path_in;
    }
  } else {
    // just return the original path
    _return = archive_path_in;
  }
}

void DBHandler::get_all_files_in_archive(std::vector<std::string>& _return,
                                         const TSessionId& session_id_or_json,
                                         const std::string& archive_path_in,
                                         const TCopyParams& copy_params) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  auto stdlog = STDLOG(get_session_ptr(request_info.sessionId()),
                       "get_all_files_in_archive",
                       archive_path_in);
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());

  std::string archive_path(archive_path_in);
  if (path_is_relative(archive_path)) {
    // assume relative paths are relative to data_path / import / <session>
    auto file_path = import_path_ /
                     picosha2::hash256_hex_string(request_info.sessionId()) /
                     boost::filesystem::path(archive_path).filename();
    archive_path = file_path.string();
  }
  validate_import_file_path_if_local(archive_path);

  if (is_a_supported_archive_file(archive_path)) {
    // find the archive file
    add_vsi_network_prefix(archive_path);
    if (!import_export::Importer::gdalFileExists(archive_path,
                                                 thrift_to_copyparams(copy_params))) {
      THROW_DB_EXCEPTION("Archive does not exist: " + archive_path_in);
    }
    // find all files in archive
    add_vsi_archive_prefix(archive_path);
    _return = import_export::Importer::gdalGetAllFilesInArchive(
        archive_path, thrift_to_copyparams(copy_params));
    // prepend them all with original path
    for (auto& s : _return) {
      s = archive_path_in + '/' + s;
    }
  }
}

void DBHandler::get_layers_in_geo_file(std::vector<TGeoFileLayerInfo>& _return,
                                       const TSessionId& session_id_or_json,
                                       const std::string& file_name_in,
                                       const TCopyParams& cp) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  auto stdlog = STDLOG(
      get_session_ptr(request_info.sessionId()), "get_layers_in_geo_file", file_name_in);
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());

  std::string file_name(file_name_in);

  import_export::CopyParams copy_params = thrift_to_copyparams(cp);

  // handle relative paths
  if (path_is_relative(file_name)) {
    // assume relative paths are relative to data_path / import / <session>
    auto file_path = import_path_ /
                     picosha2::hash256_hex_string(request_info.sessionId()) /
                     boost::filesystem::path(file_name).filename();
    file_name = file_path.string();
  }
  validate_import_file_path_if_local(file_name);

  // archive or file?
  if (is_a_supported_archive_file(file_name)) {
    // find the archive file
    add_vsi_network_prefix(file_name);
    if (!import_export::Importer::gdalFileExists(file_name, copy_params)) {
      THROW_DB_EXCEPTION("Archive does not exist: " + file_name_in);
    }
    // find geo file in archive
    add_vsi_archive_prefix(file_name);
    std::string geo_file = find_first_geo_file_in_archive(file_name, copy_params);
    // prepare to load that geo file
    if (geo_file.size()) {
      file_name = file_name + std::string("/") + geo_file;
    }
  } else {
    // prepare to load geo file directly
    add_vsi_network_prefix(file_name);
    add_vsi_geo_prefix(file_name);
  }

  // check the file actually exists
  if (!import_export::Importer::gdalFileOrDirectoryExists(file_name, copy_params)) {
    THROW_DB_EXCEPTION("Geo file/archive does not exist: " + file_name_in);
  }

  // find all layers
  auto internal_layer_info =
      import_export::Importer::gdalGetLayersInGeoFile(file_name, copy_params);

  // convert to Thrift type
  for (const auto& internal_layer : internal_layer_info) {
    TGeoFileLayerInfo layer;
    layer.name = internal_layer.name;
    switch (internal_layer.contents) {
      case import_export::Importer::GeoFileLayerContents::EMPTY:
        layer.contents = TGeoFileLayerContents::EMPTY;
        break;
      case import_export::Importer::GeoFileLayerContents::GEO:
        layer.contents = TGeoFileLayerContents::GEO;
        break;
      case import_export::Importer::GeoFileLayerContents::NON_GEO:
        layer.contents = TGeoFileLayerContents::NON_GEO;
        break;
      case import_export::Importer::GeoFileLayerContents::UNSUPPORTED_GEO:
        layer.contents = TGeoFileLayerContents::UNSUPPORTED_GEO;
        break;
      default:
        CHECK(false);
    }
    _return.emplace_back(layer);  // no suitable constructor to just pass parameters
  }
}

void DBHandler::start_heap_profile(const TSessionId& session_id_or_json) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  auto stdlog = STDLOG(get_session_ptr(request_info.sessionId()));
#ifdef HAVE_PROFILER
  if (IsHeapProfilerRunning()) {
    THROW_DB_EXCEPTION("Profiler already started");
  }
  HeapProfilerStart("omnisci");
#else
  THROW_DB_EXCEPTION("Profiler not enabled");
#endif  // HAVE_PROFILER
}

void DBHandler::stop_heap_profile(const TSessionId& session_id_or_json) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  auto stdlog = STDLOG(get_session_ptr(request_info.sessionId()));
#ifdef HAVE_PROFILER
  if (!IsHeapProfilerRunning()) {
    THROW_DB_EXCEPTION("Profiler not running");
  }
  HeapProfilerStop();
#else
  THROW_DB_EXCEPTION("Profiler not enabled");
#endif  // HAVE_PROFILER
}

Catalog_Namespace::SessionInfoPtr DBHandler::findCalciteSession(
    TSessionId const& session_id) const {
  heavyai::lock_guard<heavyai::shared_mutex> lg(calcite_sessions_mtx_);
  auto const itr = calcite_sessions_.find(session_id);
  return itr == calcite_sessions_.end() ? nullptr : itr->second;
}

void DBHandler::get_heap_profile(std::string& profile,
                                 const TSessionId& session_id_or_json) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  auto stdlog = STDLOG(get_session_ptr(request_info.sessionId()));
#ifdef HAVE_PROFILER
  if (!IsHeapProfilerRunning()) {
    THROW_DB_EXCEPTION("Profiler not running");
  }
  auto profile_buff = GetHeapProfile();
  profile = profile_buff;
  free(profile_buff);
#else
  THROW_DB_EXCEPTION("Profiler not enabled");
#endif  // HAVE_PROFILER
}

Catalog_Namespace::SessionInfo DBHandler::get_session_copy(const TSessionId& session_id) {
  if (session_id.length() == Catalog_Namespace::CALCITE_SESSION_ID_LENGTH) {
    heavyai::shared_lock<heavyai::shared_mutex> lock(calcite_sessions_mtx_);
    if (auto it = calcite_sessions_.find(session_id); it != calcite_sessions_.end()) {
      return *it->second;
    }
    throw std::runtime_error("No session with id " + session_id);
  }
  return sessions_store_->getSessionCopy(session_id);
}

std::shared_ptr<Catalog_Namespace::SessionInfo> DBHandler::get_session_ptr(
    const TSessionId& session_id) {
  // Note(Wamsi): This method will give you a shared_ptr to master SessionInfo itself.
  // Should be used only when you need to make updates to original SessionInfo object.
  // Currently used by `update_session_last_used_duration`

  // 1) `session_id` will be empty during intial connect. 2)`sessionmapd iterator` will
  // be invalid during disconnect. SessionInfo will be erased from map by the time it
  // reaches here. In both the above cases, we would return `nullptr` and can skip
  // SessionInfo updates.
  if (session_id.empty()) {
    return nullptr;
  }
  auto ptr = session_id.length() == Catalog_Namespace::CALCITE_SESSION_ID_LENGTH
                 ? findCalciteSession(session_id)
                 : sessions_store_->get(session_id);
  if (!ptr) {
    THROW_DB_EXCEPTION("Session not valid or expired.");
  }
  return ptr;
}

void DBHandler::check_table_load_privileges(
    const Catalog_Namespace::SessionInfo& session_info,
    const std::string& table_name) {
  auto user_metadata = session_info.get_currentUser();
  auto& cat = session_info.getCatalog();
  DBObject dbObject(table_name, TableDBObjectType);
  dbObject.loadKey(cat);
  dbObject.setPrivileges(AccessPrivileges::INSERT_INTO_TABLE);
  std::vector<DBObject> privObjects;
  privObjects.push_back(dbObject);
  if (!SysCatalog::instance().checkPrivileges(user_metadata, privObjects)) {
    THROW_DB_EXCEPTION("Violation of access privileges: user " +
                       user_metadata.userLoggable() +
                       " has no insert privileges for table " + table_name + ".");
  }
}

void DBHandler::set_execution_mode_nolock(Catalog_Namespace::SessionInfo* session_ptr,
                                          const TExecuteMode::type mode) {
  const std::string user_name = session_ptr->get_currentUser().userLoggable();
  switch (mode) {
    case TExecuteMode::GPU:
      if (cpu_mode_only_) {
        TDBException e;
        e.error_msg = "Cannot switch to GPU mode in a server started in CPU-only mode.";
        throw e;
      }
      session_ptr->set_executor_device_type(ExecutorDeviceType::GPU);
      LOG(INFO) << "User " << user_name << " sets GPU mode.";
      break;
    case TExecuteMode::CPU:
      session_ptr->set_executor_device_type(ExecutorDeviceType::CPU);
      LOG(INFO) << "User " << user_name << " sets CPU mode.";
      break;
  }
}

std::vector<PushedDownFilterInfo> DBHandler::execute_rel_alg(
    ExecutionResult& _return,
    QueryStateProxy query_state_proxy,
    const std::string& query_ra,
    const bool column_format,
    const ExecutorDeviceType executor_device_type,
    const int32_t first_n,
    const int32_t at_most_n,
    const bool just_validate,
    const bool find_push_down_candidates,
    const ExplainInfo& explain_info,
    const std::optional<size_t> executor_index) const {
  query_state::Timer timer = query_state_proxy.createTimer(__func__);
  VLOG(1) << "Table Schema Locks:\n" << lockmgr::TableSchemaLockMgr::instance();
  VLOG(1) << "Table Data Locks:\n" << lockmgr::TableDataLockMgr::instance();
  auto executor = Executor::getExecutor(
      executor_index ? *executor_index : Executor::UNITARY_EXECUTOR_ID,
      jit_debug_ ? "/tmp" : "",
      jit_debug_ ? "mapdquery" : "",
      system_parameters_);
  RelAlgExecutor ra_executor(
      executor.get(), query_ra, query_state_proxy->shared_from_this());
  CompilationOptions co = {executor_device_type,
                           /*hoist_literals=*/true,
                           ExecutorOptLevel::Default,
                           g_enable_dynamic_watchdog,
                           /*allow_lazy_fetch=*/true,
                           /*filter_on_deleted_column=*/true,
                           explain_info.isOptimizedExplain()
                               ? ExecutorExplainType::Optimized
                               : ExecutorExplainType::Default,
                           intel_jit_profile_};
  auto validate_or_explain_query =
      explain_info.isJustExplain() || explain_info.isCalciteExplain() || just_validate;
  ExecutionOptions eo = {
      g_enable_columnar_output,
      false,
      allow_multifrag_,
      explain_info.isJustExplain(),
      allow_loop_joins_ || just_validate,
      g_enable_watchdog,
      jit_debug_,
      just_validate,
      g_enable_dynamic_watchdog,
      g_dynamic_watchdog_time_limit,
      find_push_down_candidates,
      explain_info.isCalciteExplain(),
      system_parameters_.gpu_input_mem_limit,
      g_enable_runtime_query_interrupt && !validate_or_explain_query &&
          !query_state_proxy->getConstSessionInfo()->get_session_id().empty(),
      g_running_query_interrupt_freq,
      g_pending_query_interrupt_freq,
      g_optimize_cuda_block_and_grid_sizes};
  auto execution_time_ms =
      _return.getExecutionTime() + measure<>::execution([&]() {
        _return = ra_executor.executeRelAlgQuery(
            co, eo, explain_info.isPlanExplain(), explain_info.isVerbose(), nullptr);
      });
  // reduce execution time by the time spent during queue waiting
  const auto rs = _return.getRows();
  if (rs) {
    execution_time_ms -= rs->getQueueTime();
  }
  _return.setExecutionTime(execution_time_ms);
  const auto& filter_push_down_info = _return.getPushedDownFilterInfo();
  if (!filter_push_down_info.empty()) {
    return filter_push_down_info;
  }
  if (explain_info.isJustExplain()) {
    _return.setResultType(ExecutionResult::Explanation);
  } else if (!explain_info.isCalciteExplain()) {
    _return.setResultType(ExecutionResult::QueryResult);
  }
  return {};
}

std::vector<TargetMetaInfo> DBHandler::getTargetMetaInfo(
    const std::vector<std::shared_ptr<Analyzer::TargetEntry>>& targets) const {
  std::vector<TargetMetaInfo> result;
  for (const auto& target : targets) {
    CHECK(target);
    CHECK(target->get_expr());
    result.emplace_back(target->get_resname(), target->get_expr()->get_type_info());
  }
  return result;
}

std::vector<std::string> DBHandler::getTargetNames(
    const std::vector<std::shared_ptr<Analyzer::TargetEntry>>& targets) const {
  std::vector<std::string> names;
  for (const auto& target : targets) {
    CHECK(target);
    CHECK(target->get_expr());
    names.push_back(target->get_resname());
  }
  return names;
}

std::vector<std::string> DBHandler::getTargetNames(
    const std::vector<TargetMetaInfo>& targets) const {
  std::vector<std::string> names;
  for (const auto& target : targets) {
    names.push_back(target.get_resname());
  }
  return names;
}

void DBHandler::convertRows(TQueryResult& _return,
                            QueryStateProxy query_state_proxy,
                            const std::vector<TargetMetaInfo>& targets,
                            const ResultSet& results,
                            const bool column_format,
                            const int32_t first_n,
                            const int32_t at_most_n) {
  query_state::Timer timer = query_state_proxy.createTimer(__func__);
  _return.row_set.row_desc = ThriftSerializers::target_meta_infos_to_thrift(targets);
  int32_t fetched{0};
  if (column_format) {
    _return.row_set.is_columnar = true;
    std::vector<TColumn> tcolumns(results.colCount());
    while (first_n == -1 || fetched < first_n) {
      const auto crt_row = results.getNextRow(true, true);
      if (crt_row.empty()) {
        break;
      }
      ++fetched;
      if (at_most_n >= 0 && fetched > at_most_n) {
        THROW_DB_EXCEPTION("The result contains more rows than the specified cap of " +
                           std::to_string(at_most_n));
      }
      for (size_t i = 0; i < results.colCount(); ++i) {
        const auto agg_result = crt_row[i];
        value_to_thrift_column(agg_result, targets[i].get_type_info(), tcolumns[i]);
      }
    }
    for (size_t i = 0; i < results.colCount(); ++i) {
      _return.row_set.columns.push_back(tcolumns[i]);
    }
  } else {
    _return.row_set.is_columnar = false;
    while (first_n == -1 || fetched < first_n) {
      const auto crt_row = results.getNextRow(true, true);
      if (crt_row.empty()) {
        break;
      }
      ++fetched;
      if (at_most_n >= 0 && fetched > at_most_n) {
        THROW_DB_EXCEPTION("The result contains more rows than the specified cap of " +
                           std::to_string(at_most_n));
      }
      TRow trow;
      trow.cols.reserve(results.colCount());
      for (size_t i = 0; i < results.colCount(); ++i) {
        const auto agg_result = crt_row[i];
        trow.cols.push_back(value_to_thrift(agg_result, targets[i].get_type_info()));
      }
      _return.row_set.rows.push_back(trow);
    }
  }
}

// create simple result set to return a single column result
void DBHandler::createSimpleResult(TQueryResult& _return,
                                   const ResultSet& results,
                                   const bool column_format,
                                   const std::string label) {
  CHECK_EQ(size_t(1), results.rowCount());
  TColumnType proj_info;
  proj_info.col_name = label;
  proj_info.col_type.type = TDatumType::STR;
  proj_info.col_type.nullable = false;
  proj_info.col_type.is_array = false;
  _return.row_set.row_desc.push_back(proj_info);
  const auto crt_row = results.getNextRow(true, true);
  const auto tv = crt_row[0];
  CHECK(results.getNextRow(true, true).empty());
  const auto scalar_tv = boost::get<ScalarTargetValue>(&tv);
  CHECK(scalar_tv);
  const auto s_n = boost::get<NullableString>(scalar_tv);
  CHECK(s_n);
  const auto s = boost::get<std::string>(s_n);
  CHECK(s);
  if (column_format) {
    TColumn tcol;
    tcol.data.str_col.push_back(*s);
    tcol.nulls.push_back(false);
    _return.row_set.is_columnar = true;
    _return.row_set.columns.push_back(tcol);
  } else {
    TDatum explanation;
    explanation.val.str_val = *s;
    explanation.is_null = false;
    TRow trow;
    trow.cols.push_back(explanation);
    _return.row_set.is_columnar = false;
    _return.row_set.rows.push_back(trow);
  }
}

void DBHandler::convertExplain(TQueryResult& _return,
                               const ResultSet& results,
                               const bool column_format) {
  createSimpleResult(_return, results, column_format, "Explanation");
}

void DBHandler::convertResult(TQueryResult& _return,
                              const ResultSet& results,
                              const bool column_format) {
  createSimpleResult(_return, results, column_format, "Result");
}

// this all should be moved out of here to catalog
bool DBHandler::user_can_access_table(const Catalog_Namespace::SessionInfo& session_info,
                                      const TableDescriptor* td,
                                      const AccessPrivileges access_priv) {
  CHECK(td);
  auto& cat = session_info.getCatalog();
  std::vector<DBObject> privObjects;
  DBObject dbObject(td->tableName, TableDBObjectType);
  dbObject.loadKey(cat);
  dbObject.setPrivileges(access_priv);
  privObjects.push_back(dbObject);
  return SysCatalog::instance().checkPrivileges(session_info.get_currentUser(),
                                                privObjects);
}

// TODO(max): usage of it was accidentally lost. Need to restore this check
void DBHandler::check_and_invalidate_sessions(Parser::DDLStmt* ddl) {
  if (const auto drop_db_stmt = dynamic_cast<Parser::DropDBStmt*>(ddl)) {
    sessions_store_->eraseByDB(*drop_db_stmt->getDatabaseName());
  } else if (const auto rename_db_stmt = dynamic_cast<Parser::RenameDBStmt*>(ddl)) {
    sessions_store_->eraseByDB(*rename_db_stmt->getPreviousDatabaseName());
  } else if (const auto drop_user_stmt = dynamic_cast<Parser::DropUserStmt*>(ddl)) {
    sessions_store_->eraseByUser(*drop_user_stmt->getUserName());
  } else if (const auto rename_user_stmt = dynamic_cast<Parser::RenameUserStmt*>(ddl)) {
    sessions_store_->eraseByUser(*rename_user_stmt->getOldUserName());
  }
}

void DBHandler::sql_execute_impl(ExecutionResult& _return,
                                 QueryStateProxy query_state_proxy,
                                 const bool column_format,
                                 const ExecutorDeviceType executor_device_type,
                                 const int32_t first_n,
                                 const int32_t at_most_n,
                                 const bool use_calcite,
                                 lockmgr::LockedTableDescriptors& locks) {
  if (leaf_handler_) {
    leaf_handler_->flush_queue();
  }
  auto const query_str = strip(query_state_proxy->getQueryStr());
  auto session_ptr = query_state_proxy->getConstSessionInfo();
  // Call to DistributedValidate() below may change cat.
  auto& cat = session_ptr->getCatalog();
  legacylockmgr::ExecutorWriteLock execute_write_lock;
  legacylockmgr::ExecutorReadLock execute_read_lock;

  ParserWrapper pw{query_str};
  auto [query_substr, post_fix] = ::substring(query_str, g_max_log_length);
  std::ostringstream oss;
  oss << query_substr << post_fix;
  auto const reduced_query_str = oss.str();
  bool show_cpu_memory_stat_after_finishing_query = false;
  ScopeGuard cpu_system_memory_logging = [&show_cpu_memory_stat_after_finishing_query,
                                          &cat,
                                          &reduced_query_str]() {
    if (show_cpu_memory_stat_after_finishing_query) {
      log_system_cpu_memory_status("Finish query execution: " + reduced_query_str, cat);
    }
  };
  auto log_cpu_memory_status =
      [&reduced_query_str, &cat, &show_cpu_memory_stat_after_finishing_query]() {
        log_system_cpu_memory_status("Start query execution: " + reduced_query_str, cat);
        show_cpu_memory_stat_after_finishing_query = true;
      };

  // test to see if db/catalog is writable before execution of a writable SQL/DDL command
  //   TODO: move to execute() (?)
  //      instead of pre-filtering here based upon incomplete info ?
  if (!pw.is_refresh && pw.getQueryType() != ParserWrapper::QueryType::Read &&
      pw.getQueryType() != ParserWrapper::QueryType::SchemaRead &&
      pw.getQueryType() != ParserWrapper::QueryType::Unknown) {
    dbhandler::check_not_info_schema_db(cat.name());
  }

  if (pw.is_itas) {
    // itas can attempt to execute here
    check_read_only("insert_into_table");

    std::string query_ra;
    _return.addExecutionTime(measure<>::execution([&]() {
      TPlanResult result;
      std::tie(result, locks) =
          parse_to_ra(query_state_proxy, query_str, {}, false, system_parameters_);
      query_ra = result.plan_result;
    }));
    rapidjson::Document ddl_query;
    ddl_query.Parse(query_ra);
    CHECK(ddl_query.HasMember("payload"));
    CHECK(ddl_query["payload"].IsObject());
    auto stmt = Parser::InsertIntoTableAsSelectStmt(ddl_query["payload"].GetObject());
    log_cpu_memory_status();
    _return.addExecutionTime(
        measure<>::execution([&]() { stmt.execute(*session_ptr, read_only_); }));
    return;

  } else if (pw.is_ctas) {
    // ctas can attempt to execute here
    check_read_only("create_table_as");

    std::string query_ra;
    _return.addExecutionTime(measure<>::execution([&]() {
      TPlanResult result;
      std::tie(result, locks) =
          parse_to_ra(query_state_proxy, query_str, {}, false, system_parameters_);
      query_ra = result.plan_result;
    }));
    if (query_ra.size()) {
      rapidjson::Document ddl_query;
      ddl_query.Parse(query_ra);
      CHECK(ddl_query.HasMember("payload"));
      CHECK(ddl_query["payload"].IsObject());
      auto stmt = Parser::CreateTableAsSelectStmt(ddl_query["payload"].GetObject());
      log_cpu_memory_status();
      _return.addExecutionTime(
          measure<>::execution([&]() { stmt.execute(*session_ptr, read_only_); }));
    }
    return;

  } else if (pw.getDMLType() == ParserWrapper::DMLType::Insert) {
    check_read_only("insert_into_table");
    std::string query_ra;
    _return.addExecutionTime(measure<>::execution([&]() {
      TPlanResult result;
      std::tie(result, locks) =
          parse_to_ra(query_state_proxy, query_str, {}, false, system_parameters_);
      query_ra = result.plan_result;
    }));
    rapidjson::Document ddl_query;
    ddl_query.Parse(query_ra);
    CHECK(ddl_query.HasMember("payload"));
    CHECK(ddl_query["payload"].IsObject());
    auto stmt = Parser::InsertValuesStmt(cat, ddl_query["payload"].GetObject());
    if (stmt.get_value_lists().size() > 1) {
      log_cpu_memory_status();
    }
    _return.addExecutionTime(
        measure<>::execution([&]() { stmt.execute(*session_ptr, read_only_); }));
    return;

  } else if (pw.is_validate) {
    // check user is superuser
    if (!session_ptr->get_currentUser().isSuper) {
      throw std::runtime_error("Superuser is required to run VALIDATE");
    }

    std::string query_ra;
    _return.addExecutionTime(measure<>::execution([&]() {
      TPlanResult result;
      std::tie(result, locks) =
          parse_to_ra(query_state_proxy, query_str, {}, false, system_parameters_);
      query_ra = result.plan_result;
    }));
    rapidjson::Document ddl_query;
    ddl_query.Parse(query_ra);
    CHECK(ddl_query.HasMember("payload"));
    CHECK(ddl_query["payload"].IsObject());
    auto validate_stmt = Parser::ValidateStmt(ddl_query["payload"].GetObject());
    _return.addExecutionTime(measure<>::execution([&]() {
      // Prevent any other query from running while doing validate
      execute_write_lock = legacylockmgr::getExecuteWriteLock();

      std::string output{"Result for validate"};
      if (g_cluster) {
        THROW_DB_EXCEPTION("Validate command should be executed on the aggregator.");
      } else {
        _return.addExecutionTime(measure<>::execution([&]() {
          const system_validator::SingleNodeValidator validator(validate_stmt.getType(),
                                                                cat);
          output = validator.validate();
        }));
      }
      _return.updateResultSet(output, ExecutionResult::SimpleResult);
    }));
    return;

  } else if (pw.is_copy && !pw.is_copy_to) {
    std::unique_ptr<Parser::Stmt> stmt =
        Parser::create_stmt_for_query(query_str, *session_ptr);
    const auto import_stmt = dynamic_cast<Parser::CopyTableStmt*>(stmt.get());
    if (import_stmt) {
      if (g_cluster && !leaf_aggregator_.leafCount()) {
        // Don't allow copy from imports directly on a leaf node
        throw std::runtime_error(
            "Cannot import on an individual leaf. Please import from the Aggregator.");
      } else if (leaf_aggregator_.leafCount() > 0) {
        _return.addExecutionTime(measure<>::execution(
            [&]() { execute_distributed_copy_statement(import_stmt, *session_ptr); }));
      } else {
        log_cpu_memory_status();
        _return.addExecutionTime(measure<>::execution(
            [&]() { import_stmt->execute(*session_ptr, read_only_); }));
      }

      // Read response message
      _return.updateResultSet(*import_stmt->return_message.get(),
                              ExecutionResult::SimpleResult,
                              import_stmt->get_success());

      // get deferred_copy_from info
      if (import_stmt->was_deferred_copy_from()) {
        DeferredCopyFromState deferred_copy_from_state;
        import_stmt->get_deferred_copy_from_payload(deferred_copy_from_state.table,
                                                    deferred_copy_from_state.file_name,
                                                    deferred_copy_from_state.copy_params,
                                                    deferred_copy_from_state.partitions);
        deferred_copy_from_sessions.add(session_ptr->get_session_id(),
                                        deferred_copy_from_state);
      }

      // } else {
      //   possibly a failure case:
      //      CopyTableStmt failed to be created, or failed typecast
      //      but historically just returned
      // }
    }
    return;

  } else if (pw.is_ddl) {
    std::string query_ra;
    _return.addExecutionTime(measure<>::execution([&]() {
      TPlanResult result;
      std::tie(result, locks) =
          parse_to_ra(query_state_proxy, query_str, {}, false, system_parameters_);
      query_ra = result.plan_result;
    }));
    executeDdl(_return, query_ra, session_ptr);
    return;

  } else if (pw.is_other_explain) {
    // does nothing
    throw std::runtime_error("EXPLAIN not yet supported for DDL or DML commands.");
    return;

  } else {
    // includes:
    //    explain that is not 'other'
    //    copy_to
    //    DmlUpdate DmlDelete
    //    anything else that failed to match

    if (pw.getDMLType() != ParserWrapper::DMLType::NotDML) {
      check_read_only("modify");
    }

    execute_read_lock = legacylockmgr::getExecuteReadLock();

    std::string query_ra = query_str;
    if (use_calcite) {
      _return.addExecutionTime(measure<>::execution([&]() {
        TPlanResult result;
        std::tie(result, locks) =
            parse_to_ra(query_state_proxy, query_str, {}, true, system_parameters_);
        query_ra = result.plan_result;
      }));
    }
    std::string query_ra_calcite_explain;
    ExplainInfo explain(query_str);
    if (explain.isCalciteExplain()) {
      if (!g_enable_filter_push_down || g_cluster) {
        // return the ra as the result
        _return.updateResultSet(query_ra, ExecutionResult::Explanation);
        return;
      }
      CHECK(!locks.empty());
      query_ra_calcite_explain =
          parse_to_ra(
              query_state_proxy, explain.ActualQuery(), {}, false, system_parameters_)
              .first.plan_result;
    }
    std::vector<PushedDownFilterInfo> filter_push_down_requests;
    auto submitted_time_str = query_state_proxy->getQuerySubmittedTime();
    auto query_session = session_ptr ? session_ptr->get_session_id() : "";
    auto execute_rel_alg_task = std::make_shared<QueryDispatchQueue::Task>(
        [this,
         &filter_push_down_requests,
         &_return,
         query_state_proxy,
         &explain,
         &query_ra_calcite_explain,
         &query_ra,
         &query_str,
         &locks,
         column_format,
         executor_device_type,
         first_n,
         at_most_n,
         parent_thread_local_ids =
             logger::thread_local_ids()](const size_t executor_index) {
          // if we find proper filters we need to "re-execute" the query
          // with a modified query plan (i.e., which has pushdowned filter)
          // otherwise this trial just executes the query and keeps corresponding query
          // resultset in _return object
          logger::LocalIdsScopeGuard lisg = parent_thread_local_ids.setNewThreadId();
          filter_push_down_requests = execute_rel_alg(
              _return,
              query_state_proxy,
              explain.isCalciteExplain() ? query_ra_calcite_explain : query_ra,
              column_format,
              executor_device_type,
              first_n,
              at_most_n,
              /*just_validate=*/false,
              g_enable_filter_push_down && !g_cluster,
              explain,
              executor_index);
          if (explain.isCalciteExplain()) {
            if (filter_push_down_requests.empty()) {
              // we only reach here if filter push down was enabled, but no filter
              // push down candidate was found
              _return.updateResultSet(query_ra, ExecutionResult::Explanation);
            } else {
              CHECK(!locks.empty());
              std::vector<TFilterPushDownInfo> filter_push_down_info;
              for (const auto& req : filter_push_down_requests) {
                TFilterPushDownInfo filter_push_down_info_for_request;
                filter_push_down_info_for_request.input_prev = req.input_prev;
                filter_push_down_info_for_request.input_start = req.input_start;
                filter_push_down_info_for_request.input_next = req.input_next;
                filter_push_down_info.push_back(filter_push_down_info_for_request);
              }
              query_ra = parse_to_ra(query_state_proxy,
                                     query_str,
                                     filter_push_down_info,
                                     false,
                                     system_parameters_)
                             .first.plan_result;
              _return.updateResultSet(query_ra, ExecutionResult::Explanation);
            }
          } else {
            if (!filter_push_down_requests.empty()) {
              CHECK(!locks.empty());
              execute_rel_alg_with_filter_push_down(_return,
                                                    query_state_proxy,
                                                    query_ra,
                                                    column_format,
                                                    executor_device_type,
                                                    first_n,
                                                    at_most_n,
                                                    false,
                                                    false,
                                                    filter_push_down_requests);
            }
          }
        });
    CHECK(dispatch_queue_);
    auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID);
    if (g_enable_runtime_query_interrupt && !query_session.empty() &&
        !explain.isSelectExplain()) {
      executor->enrollQuerySession(query_session,
                                   query_str,
                                   submitted_time_str,
                                   Executor::UNITARY_EXECUTOR_ID,
                                   QuerySessionStatus::QueryStatus::PENDING_QUEUE);
      while (!dispatch_queue_->hasIdleWorker()) {
        try {
          executor->checkPendingQueryStatus(query_session);
        } catch (QueryExecutionError& e) {
          executor->clearQuerySessionStatus(query_session, submitted_time_str);
          if (e.hasErrorCode(ErrorCode::INTERRUPTED)) {
            throw std::runtime_error(
                "Query execution has been interrupted (pending query).");
          }
          throw e;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
    }
    log_cpu_memory_status();
    dispatch_queue_->submit(execute_rel_alg_task,
                            pw.getDMLType() == ParserWrapper::DMLType::Update ||
                                pw.getDMLType() == ParserWrapper::DMLType::Delete);
    auto result_future = execute_rel_alg_task->get_future();
    result_future.get();
    return;
  }
}

void DBHandler::execute_rel_alg_with_filter_push_down(
    ExecutionResult& _return,
    QueryStateProxy query_state_proxy,
    std::string& query_ra,
    const bool column_format,
    const ExecutorDeviceType executor_device_type,
    const int32_t first_n,
    const int32_t at_most_n,
    const bool just_explain,
    const bool is_calcite_explain,
    const std::vector<PushedDownFilterInfo>& filter_push_down_requests) {
  // collecting the selected filters' info to be sent to Calcite:
  std::vector<TFilterPushDownInfo> filter_push_down_info;
  for (const auto& req : filter_push_down_requests) {
    TFilterPushDownInfo filter_push_down_info_for_request;
    filter_push_down_info_for_request.input_prev = req.input_prev;
    filter_push_down_info_for_request.input_start = req.input_start;
    filter_push_down_info_for_request.input_next = req.input_next;
    filter_push_down_info.push_back(filter_push_down_info_for_request);
  }
  // deriving the new relational algebra plan with respect to the pushed down filters
  _return.addExecutionTime(measure<>::execution([&]() {
    query_ra = parse_to_ra(query_state_proxy,
                           query_state_proxy->getQueryStr(),
                           filter_push_down_info,
                           false,
                           system_parameters_)
                   .first.plan_result;
  }));

  // execute the new relational algebra plan:
  auto explain_info = ExplainInfo(ExplainInfo::ExplainType::None);
  execute_rel_alg(_return,
                  query_state_proxy,
                  query_ra,
                  column_format,
                  executor_device_type,
                  first_n,
                  at_most_n,
                  /*just_validate=*/false,
                  /*find_push_down_candidates=*/false,
                  explain_info);
}

void DBHandler::execute_distributed_copy_statement(
    Parser::CopyTableStmt* copy_stmt,
    const Catalog_Namespace::SessionInfo& session_info) {}

namespace {
bool check_and_reset_in_memory_system_table(const Catalog& catalog,
                                            const TableDescriptor& td) {
  if (td.is_in_memory_system_table) {
    if (g_enable_system_tables) {
      // Reset system table fragmenter in order to force chunk metadata refetch.
      auto table_schema_lock =
          lockmgr::TableSchemaLockMgr::getWriteLockForTable(catalog, td.tableName);
      auto table_data_lock =
          lockmgr::TableDataLockMgr::getWriteLockForTable(catalog, td.tableName);
      catalog.removeFragmenterForTable(td.tableId);
      catalog.getMetadataForTable(td.tableId, true);
      return true;
    } else {
      throw std::runtime_error(
          "Query cannot be executed because use of system tables is currently "
          "disabled.");
    }
  }
  return false;
}

void check_in_memory_system_table_query(
    const std::vector<std::vector<std::string>>& selected_tables) {
  const auto info_schema_catalog =
      Catalog_Namespace::SysCatalog::instance().getCatalog(shared::kInfoSchemaDbName);
  if (info_schema_catalog) {
    for (const auto& table : selected_tables) {
      if (table[1] == shared::kInfoSchemaDbName) {
        auto td = info_schema_catalog->getMetadataForTable(table[0], false);
        CHECK(td);
        check_and_reset_in_memory_system_table(*info_schema_catalog, *td);
      }
    }
  }
}
}  // namespace

TPlanResult DBHandler::processCalciteRequest(
    QueryStateProxy query_state_proxy,
    const std::shared_ptr<Catalog_Namespace::Catalog>& cat,
    const std::string& query_str,
    const std::vector<TFilterPushDownInfo>& filter_push_down_info,
    const SystemParameters& system_parameters,
    const bool check_privileges) {
  query_state::Timer timer = query_state_proxy.createTimer(__func__);

  heavyai::RequestInfo const request_info(createInMemoryCalciteSession(cat),
                                          logger::request_id());
  ScopeGuard cleanup = [&]() { removeInMemoryCalciteSession(request_info.sessionId()); };
  ExplainInfo explain(query_str);
  std::string const actual_query{explain.isSelectExplain() ? explain.ActualQuery()
                                                           : query_str};
  auto query_parsing_option =
      calcite_->getCalciteQueryParsingOption(legacy_syntax_,
                                             explain.isCalciteExplain(),
                                             check_privileges,
                                             explain.isCalciteExplainDetail());
  auto optimization_option = calcite_->getCalciteOptimizationOption(
      system_parameters.enable_calcite_view_optimize,
      g_enable_watchdog,
      filter_push_down_info,
      Catalog_Namespace::SysCatalog::instance().isAggregator());

  return calcite_->process(timer.createQueryStateProxy(),
                           legacy_syntax_ ? pg_shim(actual_query) : actual_query,
                           query_parsing_option,
                           optimization_option,
                           request_info.json());
}

std::pair<TPlanResult, lockmgr::LockedTableDescriptors> DBHandler::parse_to_ra(
    QueryStateProxy query_state_proxy,
    const std::string& query_str,
    const std::vector<TFilterPushDownInfo>& filter_push_down_info,
    const bool acquire_locks,
    const SystemParameters& system_parameters,
    bool check_privileges) {
  query_state::Timer timer = query_state_proxy.createTimer(__func__);
  ParserWrapper pw{query_str};
  TPlanResult result;
  lockmgr::LockedTableDescriptors locks;
  if (pw.is_ddl || (!pw.is_validate && !pw.is_other_explain)) {
    auto cat = query_state_proxy->getConstSessionInfo()->get_catalog_ptr();
    // Need to read lock the catalog while determining what table names are used by this
    // query, confirming the tables exist, checking the user's permissions, and finally
    // locking the individual tables. The catalog lock can be released once the query
    // begins running. The table locks will protect the running query.
    std::shared_lock<heavyai::DistributedSharedMutex> cat_lock;
    if (g_multi_instance) {
      cat_lock = std::shared_lock<heavyai::DistributedSharedMutex>(*cat->dcatalogMutex_);
    }
    result = processCalciteRequest(timer.createQueryStateProxy(),
                                   cat,
                                   query_str,
                                   filter_push_down_info,
                                   system_parameters,
                                   check_privileges);
    check_in_memory_system_table_query(
        result.resolved_accessed_objects.tables_selected_from);

    if (acquire_locks) {
      std::set<std::vector<std::string>> write_only_tables;
      std::vector<std::vector<std::string>> tables;

      tables.insert(tables.end(),
                    result.resolved_accessed_objects.tables_updated_in.begin(),
                    result.resolved_accessed_objects.tables_updated_in.end());
      tables.insert(tables.end(),
                    result.resolved_accessed_objects.tables_deleted_from.begin(),
                    result.resolved_accessed_objects.tables_deleted_from.end());

      // Collect the tables that need a write lock
      for (const auto& table : tables) {
        write_only_tables.insert(table);
      }

      tables.insert(tables.end(),
                    result.resolved_accessed_objects.tables_selected_from.begin(),
                    result.resolved_accessed_objects.tables_selected_from.end());
      tables.insert(tables.end(),
                    result.resolved_accessed_objects.tables_inserted_into.begin(),
                    result.resolved_accessed_objects.tables_inserted_into.end());

      // avoid deadlocks by enforcing a deterministic locking sequence
      // first, obtain table schema locks
      // then, obtain table data locks
      // force sort by database id and table id order in case of name change to
      // guarantee fixed order of mutex access
      std::sort(tables.begin(),
                tables.end(),
                [](const std::vector<std::string>& a, const std::vector<std::string>& b) {
                  if (a[1] != b[1]) {
                    const auto cat_a = SysCatalog::instance().getCatalog(a[1]);
                    const auto cat_b = SysCatalog::instance().getCatalog(b[1]);
                    return cat_a->getDatabaseId() < cat_b->getDatabaseId();
                  }
                  const auto cat = SysCatalog::instance().getCatalog(a[1]);
                  return cat->getMetadataForTable(a[0], false)->tableId <
                         cat->getMetadataForTable(b[0], false)->tableId;
                });

      // In the case of self-join and possibly other cases, we will
      // have duplicate tables. Ensure we only take one for locking below.
      tables.erase(unique(tables.begin(), tables.end()), tables.end());
      for (const auto& table : tables) {
        const auto cat = SysCatalog::instance().getCatalog(table[1]);
        CHECK(cat);
        locks.emplace_back(
            std::make_unique<lockmgr::TableSchemaLockContainer<lockmgr::ReadLock>>(
                lockmgr::TableSchemaLockContainer<
                    lockmgr::ReadLock>::acquireTableDescriptor(*cat, table[0])));
        if (write_only_tables.count(table)) {
          // Aquire an insert data lock for updates/deletes, consistent w/ insert. The
          // table data lock will be aquired in the fragmenter during checkpoint.
          locks.emplace_back(
              std::make_unique<lockmgr::TableInsertLockContainer<lockmgr::WriteLock>>(
                  lockmgr::TableInsertLockContainer<lockmgr::WriteLock>::acquire(
                      cat->getDatabaseId(), (*locks.back())())));
        } else {
          auto lock_td = (*locks.back())();
          if (lock_td->is_in_memory_system_table) {
            locks.emplace_back(
                std::make_unique<lockmgr::TableDataLockContainer<lockmgr::WriteLock>>(
                    lockmgr::TableDataLockContainer<lockmgr::WriteLock>::acquire(
                        cat->getDatabaseId(), lock_td)));
          } else {
            locks.emplace_back(
                std::make_unique<lockmgr::TableDataLockContainer<lockmgr::ReadLock>>(
                    lockmgr::TableDataLockContainer<lockmgr::ReadLock>::acquire(
                        cat->getDatabaseId(), lock_td)));
          }
        }
      }
    }
  }
  return std::make_pair(result, std::move(locks));
}

int64_t DBHandler::query_get_outer_fragment_count(const TSessionId& session_id_or_json,
                                                  const std::string& select_query) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  auto stdlog = STDLOG(get_session_ptr(request_info.sessionId()));
  if (!leaf_handler_) {
    THROW_DB_EXCEPTION("Distributed support is disabled.");
  }
  try {
    return leaf_handler_->query_get_outer_fragment_count(request_info.sessionId(),
                                                         select_query);
  } catch (std::exception& e) {
    THROW_DB_EXCEPTION(e.what());
  }
}

void DBHandler::check_table_consistency(TTableMeta& _return,
                                        const TSessionId& session_id_or_json,
                                        const int32_t table_id) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  auto stdlog = STDLOG(get_session_ptr(request_info.sessionId()));
  if (!leaf_handler_) {
    THROW_DB_EXCEPTION("Distributed support is disabled.");
  }
  try {
    leaf_handler_->check_table_consistency(_return, request_info.sessionId(), table_id);
  } catch (std::exception& e) {
    THROW_DB_EXCEPTION(e.what());
  }
}

void DBHandler::start_query(TPendingQuery& _return,
                            const TSessionId& leaf_session_id_or_json,
                            const TSessionId& parent_session_id_or_json,
                            const std::string& serialized_rel_alg_dag,
                            const std::string& start_time_str,
                            const bool just_explain,
                            const std::vector<int64_t>& outer_fragment_indices) {
  heavyai::RequestInfo const leaf_request_info(leaf_session_id_or_json);
  heavyai::RequestInfo const parent_request_info(parent_session_id_or_json);
  SET_REQUEST_ID(leaf_request_info.requestId());
  auto stdlog = STDLOG(get_session_ptr(leaf_request_info.sessionId()));
  auto session_ptr = stdlog.getConstSessionInfo();
  if (!leaf_handler_) {
    THROW_DB_EXCEPTION("Distributed support is disabled.");
  }
  LOG(INFO) << "start_query :" << *session_ptr << " :" << just_explain;
  auto time_ms = measure<>::execution([&]() {
    try {
      leaf_handler_->start_query(_return,
                                 leaf_request_info.sessionId(),
                                 parent_request_info.sessionId(),
                                 serialized_rel_alg_dag,
                                 start_time_str,
                                 just_explain,
                                 outer_fragment_indices);
    } catch (std::exception& e) {
      THROW_DB_EXCEPTION(e.what());
    }
  });
  LOG(INFO) << "start_query-COMPLETED " << time_ms << "ms "
            << "id is " << _return.id;
}

void DBHandler::execute_query_step(TStepResult& _return,
                                   const TPendingQuery& pending_query,
                                   const TSubqueryId subquery_id,
                                   const std::string& start_time_str) {
  SET_REQUEST_ID(0);  // No SessionID is available
  if (!leaf_handler_) {
    THROW_DB_EXCEPTION("Distributed support is disabled.");
  }
  LOG(INFO) << "execute_query_step :  id:" << pending_query.id;
  auto time_ms = measure<>::execution([&]() {
    try {
      leaf_handler_->execute_query_step(
          _return, pending_query, subquery_id, start_time_str);
    } catch (std::exception& e) {
      THROW_DB_EXCEPTION(e.what());
    }
  });
  LOG(INFO) << "execute_query_step-COMPLETED " << time_ms << "ms";
}

void DBHandler::broadcast_serialized_rows(const TSerializedRows& serialized_rows,
                                          const TRowDescriptor& row_desc,
                                          const TQueryId query_id,
                                          const TSubqueryId subquery_id,
                                          const bool is_final_subquery_result) {
  if (!leaf_handler_) {
    THROW_DB_EXCEPTION("Distributed support is disabled.");
  }
  LOG(INFO) << "BROADCAST-SERIALIZED-ROWS  id:" << query_id;
  auto time_ms = measure<>::execution([&]() {
    try {
      leaf_handler_->broadcast_serialized_rows(
          serialized_rows, row_desc, query_id, subquery_id, is_final_subquery_result);
    } catch (std::exception& e) {
      THROW_DB_EXCEPTION(e.what());
    }
  });
  LOG(INFO) << "BROADCAST-SERIALIZED-ROWS COMPLETED " << time_ms << "ms";
}

void DBHandler::insert_chunks(const TSessionId& session_id_or_json,
                              const TInsertChunks& thrift_insert_chunks) {
  try {
    heavyai::RequestInfo const request_info(session_id_or_json);
    SET_REQUEST_ID(request_info.requestId());
    auto stdlog = STDLOG(get_session_ptr(request_info.sessionId()));
    auto session_ptr = stdlog.getConstSessionInfo();
    auto const& cat = session_ptr->getCatalog();
    Fragmenter_Namespace::InsertChunks insert_chunks{thrift_insert_chunks.table_id,
                                                     thrift_insert_chunks.db_id};
    insert_chunks.valid_row_indices.resize(thrift_insert_chunks.valid_indices.size());
    std::copy(thrift_insert_chunks.valid_indices.begin(),
              thrift_insert_chunks.valid_indices.end(),
              insert_chunks.valid_row_indices.begin());

    auto columns =
        cat.getAllColumnMetadataForTable(insert_chunks.table_id, false, false, true);
    CHECK_EQ(columns.size(), thrift_insert_chunks.data.size());

    std::list<foreign_storage::PassThroughBuffer> pass_through_buffers;
    auto thrift_data_it = thrift_insert_chunks.data.begin();
    for (const auto col_desc : columns) {
      AbstractBuffer* data_buffer = nullptr;
      AbstractBuffer* index_buffer = nullptr;
      data_buffer = &pass_through_buffers.emplace_back(
          reinterpret_cast<const int8_t*>(thrift_data_it->data_buffer.data()),
          thrift_data_it->data_buffer.size());
      data_buffer->initEncoder(col_desc->columnType);
      data_buffer->getEncoder()->setNumElems(thrift_insert_chunks.num_rows);
      if (col_desc->columnType.is_varlen_indeed()) {
        CHECK(thrift_insert_chunks.num_rows == 0 ||
              thrift_data_it->index_buffer.size() > 0);
        index_buffer = &pass_through_buffers.emplace_back(
            reinterpret_cast<const int8_t*>(thrift_data_it->index_buffer.data()),
            thrift_data_it->index_buffer.size());
      }

      insert_chunks.chunks[col_desc->columnId] =
          Chunk_NS::Chunk::getChunk(col_desc, data_buffer, index_buffer, false);
      thrift_data_it++;
    }

    const ChunkKey lock_chunk_key{cat.getDatabaseId(),
                                  cat.getLogicalTableId(insert_chunks.table_id)};
    auto table_read_lock =
        lockmgr::TableSchemaLockMgr::getReadLockForTable(lock_chunk_key);
    const auto td = cat.getMetadataForTable(insert_chunks.table_id);
    CHECK(td);

    // this should have the same lock sequence as COPY FROM
    auto insert_data_lock =
        lockmgr::InsertDataLockMgr::getWriteLockForTable(lock_chunk_key);
    td->fragmenter->insertChunksNoCheckpoint(insert_chunks);

  } catch (const std::exception& e) {
    THROW_DB_EXCEPTION(std::string(e.what()));
  }
}

void DBHandler::insert_data(const TSessionId& session_id_or_json,
                            const TInsertData& thrift_insert_data) {
  try {
    heavyai::RequestInfo const request_info(session_id_or_json);
    SET_REQUEST_ID(request_info.requestId());
    auto stdlog = STDLOG(get_session_ptr(request_info.sessionId()));
    auto session_ptr = stdlog.getConstSessionInfo();
    CHECK_EQ(thrift_insert_data.column_ids.size(), thrift_insert_data.data.size());
    CHECK(thrift_insert_data.is_default.size() == 0 ||
          thrift_insert_data.is_default.size() == thrift_insert_data.column_ids.size());
    auto const& cat = session_ptr->getCatalog();
    Fragmenter_Namespace::InsertData insert_data;
    insert_data.databaseId = thrift_insert_data.db_id;
    insert_data.tableId = thrift_insert_data.table_id;
    insert_data.columnIds = thrift_insert_data.column_ids;
    insert_data.is_default = thrift_insert_data.is_default;
    insert_data.numRows = thrift_insert_data.num_rows;
    std::vector<std::unique_ptr<std::vector<std::string>>> none_encoded_string_columns;
    std::vector<std::unique_ptr<std::vector<ArrayDatum>>> array_columns;
    SQLTypeInfo geo_ti{kNULLT,
                       false};  // will be filled with the correct info if possible
    for (size_t col_idx = 0; col_idx < insert_data.columnIds.size(); ++col_idx) {
      const int column_id = insert_data.columnIds[col_idx];
      DataBlockPtr p;
      const auto cd = cat.getMetadataForColumn(insert_data.tableId, column_id);
      CHECK(cd);
      const auto& ti = cd->columnType;
      size_t rows_expected =
          !insert_data.is_default.empty() && insert_data.is_default[col_idx]
              ? 1ul
              : insert_data.numRows;
      if (ti.is_number() || ti.is_time() || ti.is_boolean()) {
        p.numbersPtr = (int8_t*)thrift_insert_data.data[col_idx].fixed_len_data.data();
      } else if (ti.is_string()) {
        if (ti.get_compression() == kENCODING_DICT) {
          p.numbersPtr = (int8_t*)thrift_insert_data.data[col_idx].fixed_len_data.data();
        } else {
          CHECK_EQ(kENCODING_NONE, ti.get_compression());
          none_encoded_string_columns.emplace_back(new std::vector<std::string>());
          auto& none_encoded_strings = none_encoded_string_columns.back();

          CHECK_EQ(rows_expected, thrift_insert_data.data[col_idx].var_len_data.size());
          for (const auto& varlen_str : thrift_insert_data.data[col_idx].var_len_data) {
            none_encoded_strings->push_back(varlen_str.payload);
          }
          p.stringsPtr = none_encoded_strings.get();
        }
      } else if (ti.is_geometry()) {
        none_encoded_string_columns.emplace_back(new std::vector<std::string>());
        auto& none_encoded_strings = none_encoded_string_columns.back();
        CHECK_EQ(rows_expected, thrift_insert_data.data[col_idx].var_len_data.size());
        for (const auto& varlen_str : thrift_insert_data.data[col_idx].var_len_data) {
          none_encoded_strings->push_back(varlen_str.payload);
        }
        p.stringsPtr = none_encoded_strings.get();

        // point geo type needs to mark null sentinel in its physical coord column
        // To recognize null sentinel for point, therefore, we keep the actual geo type
        // and needs to use it when constructing geo null point
        geo_ti = ti;
      } else {
        CHECK(ti.is_array());
        array_columns.emplace_back(new std::vector<ArrayDatum>());
        auto& array_column = array_columns.back();
        CHECK_EQ(rows_expected, thrift_insert_data.data[col_idx].var_len_data.size());
        for (const auto& t_arr_datum : thrift_insert_data.data[col_idx].var_len_data) {
          if (t_arr_datum.is_null) {
            if ((cd->columnName.find("_coords") != std::string::npos) &&
                geo_ti.get_type() == kPOINT) {
              // For geo point, we manually mark its null sentinel to coord buffer
              array_column->push_back(
                  import_export::ImporterUtils::composeNullPointCoords(ti, geo_ti));
            } else if (ti.get_size() > 0) {
              array_column->push_back(import_export::ImporterUtils::composeNullArray(ti));
            } else {
              array_column->emplace_back(0, nullptr, true);
            }
          } else {
            ArrayDatum arr_datum;
            arr_datum.length = t_arr_datum.payload.size();
            int8_t* ptr = (int8_t*)(t_arr_datum.payload.data());
            arr_datum.pointer = ptr;
            // In this special case, ArrayDatum does not handle freeing the underlying
            // memory
            arr_datum.data_ptr = std::shared_ptr<int8_t>(ptr, [](auto p) {});
            arr_datum.is_null = false;
            array_column->push_back(arr_datum);
          }
        }
        p.arraysPtr = array_column.get();
      }
      insert_data.data.push_back(p);
    }
    const ChunkKey lock_chunk_key{cat.getDatabaseId(),
                                  cat.getLogicalTableId(insert_data.tableId)};
    auto table_read_lock =
        lockmgr::TableSchemaLockMgr::getReadLockForTable(lock_chunk_key);
    const auto td = cat.getMetadataForTable(insert_data.tableId);
    CHECK(td);

    // this should have the same lock seq as COPY FROM
    auto insert_data_lock =
        lockmgr::InsertDataLockMgr::getWriteLockForTable(lock_chunk_key);
    auto data_memory_holder = import_export::fill_missing_columns(&cat, insert_data);
    td->fragmenter->insertDataNoCheckpoint(insert_data);
  } catch (const std::exception& e) {
    THROW_DB_EXCEPTION(std::string(e.what()));
  }
}

void DBHandler::start_render_query(TPendingRenderQuery& _return,
                                   const TSessionId& session_id_or_json,
                                   const int64_t widget_id,
                                   const int16_t node_idx,
                                   const std::string& vega_json) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  auto stdlog = STDLOG(get_session_ptr(request_info.sessionId()));
  auto session_ptr = stdlog.getConstSessionInfo();
  if (!render_handler_) {
    THROW_DB_EXCEPTION("Backend rendering is disabled.");
  }
  LOG(INFO) << "start_render_query :" << *session_ptr << " :widget_id:" << widget_id
            << ":vega_json:" << vega_json;

  // cast away const-ness of incoming Thrift string ref
  // to allow it to be passed down as an r-value and
  // ultimately std::moved into the RenderSession
  auto& non_const_vega_json = const_cast<std::string&>(vega_json);

  auto time_ms = measure<>::execution([&]() {
    try {
      render_handler_->start_render_query(_return,
                                          request_info.sessionId(),
                                          widget_id,
                                          node_idx,
                                          std::move(non_const_vega_json));
    } catch (std::exception& e) {
      THROW_DB_EXCEPTION(e.what());
    }
  });
  LOG(INFO) << "start_render_query-COMPLETED " << time_ms << "ms "
            << "id is " << _return.id;
}

void DBHandler::execute_next_render_step(TRenderStepResult& _return,
                                         const TPendingRenderQuery& pending_render,
                                         const TRenderAggDataMap& merged_data) {
  // No SessionID is available
  SET_REQUEST_ID(0);

  if (!render_handler_) {
    THROW_DB_EXCEPTION("Backend rendering is disabled.");
  }

  LOG(INFO) << "execute_next_render_step: id:" << pending_render.id;
  auto time_ms = measure<>::execution([&]() {
    try {
      render_handler_->execute_next_render_step(_return, pending_render, merged_data);
    } catch (std::exception& e) {
      THROW_DB_EXCEPTION(e.what());
    }
  });
  LOG(INFO) << "execute_next_render_step-COMPLETED id: " << pending_render.id
            << ", time: " << time_ms << "ms ";
}

void DBHandler::checkpoint(const TSessionId& session_id_or_json, const int32_t table_id) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  auto stdlog = STDLOG(get_session_ptr(request_info.sessionId()));
  auto session_ptr = stdlog.getConstSessionInfo();
  auto& cat = session_ptr->getCatalog();
  cat.checkpoint(table_id);
}

// check and reset epoch if a request has been made
void DBHandler::set_table_epoch(const TSessionId& session_id_or_json,
                                const int db_id,
                                const int table_id,
                                const int new_epoch) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  auto stdlog = STDLOG(get_session_ptr(request_info.sessionId()));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  auto session_ptr = stdlog.getConstSessionInfo();
  if (!session_ptr->get_currentUser().isSuper) {
    throw std::runtime_error("Only superuser can set_table_epoch");
  }
  const auto execute_read_lock = legacylockmgr::getExecuteReadLock();
  ChunkKey table_key{db_id, table_id};
  auto table_write_lock = lockmgr::TableSchemaLockMgr::getWriteLockForTable(table_key);
  auto table_data_write_lock = lockmgr::TableDataLockMgr::getWriteLockForTable(table_key);
  try {
    auto& cat = session_ptr->getCatalog();
    cat.setTableEpoch(db_id, table_id, new_epoch);
  } catch (const std::runtime_error& e) {
    THROW_DB_EXCEPTION(std::string(e.what()));
  }
}

// check and reset epoch if a request has been made
void DBHandler::set_table_epoch_by_name(const TSessionId& session_id_or_json,
                                        const std::string& table_name,
                                        const int new_epoch) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  auto stdlog =
      STDLOG(get_session_ptr(request_info.sessionId()), "table_name", table_name);
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  auto session_ptr = stdlog.getConstSessionInfo();
  if (!session_ptr->get_currentUser().isSuper) {
    throw std::runtime_error("Only superuser can set_table_epoch");
  }

  const auto execute_read_lock = legacylockmgr::getExecuteReadLock();
  auto& cat = session_ptr->getCatalog();
  auto table_write_lock =
      lockmgr::TableSchemaLockMgr::getWriteLockForTable(cat, table_name);
  auto table_data_write_lock =
      lockmgr::TableDataLockMgr::getWriteLockForTable(cat, table_name);
  auto td = cat.getMetadataForTable(
      table_name,
      false);  // don't populate fragmenter on this call since we only want metadata
  int32_t db_id = cat.getCurrentDB().dbId;
  try {
    cat.setTableEpoch(db_id, td->tableId, new_epoch);
  } catch (const std::runtime_error& e) {
    THROW_DB_EXCEPTION(std::string(e.what()));
  }
}

int32_t DBHandler::get_table_epoch(const TSessionId& session_id_or_json,
                                   const int32_t db_id,
                                   const int32_t table_id) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  auto stdlog = STDLOG(get_session_ptr(request_info.sessionId()));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  auto session_ptr = stdlog.getConstSessionInfo();

  const auto execute_read_lock = legacylockmgr::getExecuteReadLock();
  ChunkKey table_key{db_id, table_id};
  auto table_read_lock = lockmgr::TableSchemaLockMgr::getReadLockForTable(table_key);
  auto table_data_write_lock = lockmgr::TableDataLockMgr::getReadLockForTable(table_key);
  try {
    auto const& cat = session_ptr->getCatalog();
    return cat.getTableEpoch(db_id, table_id);
  } catch (const std::runtime_error& e) {
    THROW_DB_EXCEPTION(std::string(e.what()));
  }
}

int32_t DBHandler::get_table_epoch_by_name(const TSessionId& session_id_or_json,
                                           const std::string& table_name) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  auto stdlog =
      STDLOG(get_session_ptr(request_info.sessionId()), "table_name", table_name);
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  auto session_ptr = stdlog.getConstSessionInfo();

  const auto execute_read_lock = legacylockmgr::getExecuteReadLock();
  auto& cat = session_ptr->getCatalog();
  auto table_read_lock =
      lockmgr::TableSchemaLockMgr::getReadLockForTable(cat, table_name);
  auto table_data_read_lock =
      lockmgr::TableDataLockMgr::getReadLockForTable(cat, table_name);
  auto td = cat.getMetadataForTable(
      table_name,
      false);  // don't populate fragmenter on this call since we only want metadata
  int32_t db_id = cat.getCurrentDB().dbId;
  try {
    return cat.getTableEpoch(db_id, td->tableId);
  } catch (const std::runtime_error& e) {
    THROW_DB_EXCEPTION(std::string(e.what()));
  }
}

void DBHandler::get_table_epochs(std::vector<TTableEpochInfo>& _return,
                                 const TSessionId& session_id_or_json,
                                 const int32_t db_id,
                                 const int32_t table_id) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  auto stdlog = STDLOG(get_session_ptr(request_info.sessionId()));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  auto session_ptr = stdlog.getConstSessionInfo();

  const auto execute_read_lock = legacylockmgr::getExecuteReadLock();
  ChunkKey table_key{db_id, table_id};
  auto table_read_lock = lockmgr::TableSchemaLockMgr::getReadLockForTable(table_key);
  auto table_data_read_lock = lockmgr::TableDataLockMgr::getReadLockForTable(table_key);

  std::vector<Catalog_Namespace::TableEpochInfo> table_epochs;
  auto const& cat = session_ptr->getCatalog();
  table_epochs = cat.getTableEpochs(db_id, table_id);
  CHECK(!table_epochs.empty());

  for (const auto& table_epoch : table_epochs) {
    TTableEpochInfo table_epoch_info;
    table_epoch_info.table_id = table_epoch.table_id;
    table_epoch_info.table_epoch = table_epoch.table_epoch;
    table_epoch_info.leaf_index = table_epoch.leaf_index;
    _return.emplace_back(table_epoch_info);
  }
}

void DBHandler::set_table_epochs(const TSessionId& session_id_or_json,
                                 const int32_t db_id,
                                 const std::vector<TTableEpochInfo>& table_epochs) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  auto stdlog = STDLOG(get_session_ptr(request_info.sessionId()));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  auto session_ptr = stdlog.getConstSessionInfo();

  // Only super users are allowed to call this API on a single node instance
  // or aggregator (for distributed mode)
  if (!g_cluster || leaf_aggregator_.leafCount() > 0) {
    if (!session_ptr->get_currentUser().isSuper) {
      THROW_DB_EXCEPTION("Only super users can set table epochs");
    }
  }
  if (table_epochs.empty()) {
    return;
  }
  auto& cat = session_ptr->getCatalog();
  auto logical_table_id = cat.getLogicalTableId(table_epochs[0].table_id);
  std::vector<Catalog_Namespace::TableEpochInfo> table_epochs_vector;
  for (const auto& table_epoch : table_epochs) {
    if (logical_table_id != cat.getLogicalTableId(table_epoch.table_id)) {
      THROW_DB_EXCEPTION("Table epochs do not reference the same logical table");
    }
    table_epochs_vector.emplace_back(
        table_epoch.table_id, table_epoch.table_epoch, table_epoch.leaf_index);
  }

  const auto execute_read_lock = legacylockmgr::getExecuteReadLock();
  Executor::clearExternalCaches(
      true, cat.getMetadataForTable(logical_table_id, false), db_id);
  ChunkKey table_key{db_id, logical_table_id};
  auto table_write_lock = lockmgr::TableSchemaLockMgr::getWriteLockForTable(table_key);
  auto table_data_write_lock = lockmgr::TableDataLockMgr::getWriteLockForTable(table_key);
  cat.setTableEpochs(db_id, table_epochs_vector);
}

void DBHandler::set_license_key(TLicenseInfo& _return,
                                const TSessionId& session_id_or_json,
                                const std::string& key,
                                const std::string& nonce) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  check_read_only("set_license_key");
  THROW_DB_EXCEPTION(std::string("Licensing not supported."));
}

void DBHandler::get_license_claims(TLicenseInfo& _return,
                                   const TSessionId& session_id_or_json,
                                   const std::string& nonce) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  auto stdlog = STDLOG(get_session_ptr(request_info.sessionId()));
  _return.claims.emplace_back("");
}

void DBHandler::shutdown() {
  emergency_shutdown();

  Executor::clearExternalCaches(false, nullptr, -1);

  query_engine_.reset();

  if (render_handler_) {
    render_handler_->shutdown();
  }

  Catalog_Namespace::SysCatalog::destroy();
}

void DBHandler::emergency_shutdown() {
  if (calcite_) {
    calcite_->close_calcite_server(false);
  }
}

extern std::map<std::string, std::string> get_device_parameters(bool cpu_only);

#define EXPOSE_THRIFT_MAP(TYPENAME)                                             \
  {                                                                             \
    std::map<int, const char*>::const_iterator it =                             \
        _##TYPENAME##_VALUES_TO_NAMES.begin();                                  \
    while (it != _##TYPENAME##_VALUES_TO_NAMES.end()) {                         \
      _return.insert(std::pair<std::string, std::string>(                       \
          #TYPENAME "." + std::string(it->second), std::to_string(it->first))); \
      it++;                                                                     \
    }                                                                           \
  }

void DBHandler::get_device_parameters(std::map<std::string, std::string>& _return,
                                      const TSessionId& session_id_or_json) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  auto stdlog = STDLOG(get_session_ptr(request_info.sessionId()));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());
  auto params = ::get_device_parameters(cpu_mode_only_);
  for (auto item : params) {
    _return.insert(item);
  }
  EXPOSE_THRIFT_MAP(TDeviceType);
  EXPOSE_THRIFT_MAP(TDatumType);
  EXPOSE_THRIFT_MAP(TEncodingType);
  EXPOSE_THRIFT_MAP(TExtArgumentType);
  EXPOSE_THRIFT_MAP(TOutputBufferSizeType);
}

void DBHandler::register_runtime_extension_functions(
    const TSessionId& session_id_or_json,
    const std::vector<TUserDefinedFunction>& udfs,
    const std::vector<TUserDefinedTableFunction>& udtfs,
    const std::map<std::string, std::string>& device_ir_map) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  auto stdlog = STDLOG(get_session_ptr(request_info.sessionId()));
  stdlog.appendNameValuePairs("client", getConnectionInfo().toString());

  VLOG(1) << "register_runtime_extension_functions: # UDFs: " << udfs.size()
          << " # UDTFs: " << udtfs.size() << std::endl;

  if (system_parameters_.runtime_udf_registration_policy ==
      SystemParameters::RuntimeUdfRegistrationPolicy::DISALLOWED) {
    THROW_DB_EXCEPTION("Runtime UDF and UDTF function registration is disabled.");
  }

  if (system_parameters_.runtime_udf_registration_policy ==
      SystemParameters::RuntimeUdfRegistrationPolicy::ALLOWED_SUPERUSERS_ONLY) {
    auto session_ptr = stdlog.getConstSessionInfo();
    if (!session_ptr->get_currentUser().isSuper) {
      THROW_DB_EXCEPTION(
          "Server is configured to require superuser privilege to register UDFs and "
          "UDTFs.");
    }
  }
  CHECK(system_parameters_.runtime_udf_registration_policy !=
        SystemParameters::RuntimeUdfRegistrationPolicy::DISALLOWED);

  Executor::registerExtensionFunctions([&]() {
    auto it_cpu = device_ir_map.find(std::string{"cpu"});
    auto it_gpu = device_ir_map.find(std::string{"gpu"});
    if (it_cpu != device_ir_map.end() || it_gpu != device_ir_map.end()) {
      if (it_cpu != device_ir_map.end()) {
        Executor::extension_module_sources[Executor::ExtModuleKinds::rt_udf_cpu_module] =
            it_cpu->second;
      } else {
        Executor::extension_module_sources.erase(
            Executor::ExtModuleKinds::rt_udf_cpu_module);
      }
      if (it_gpu != device_ir_map.end()) {
        Executor::extension_module_sources[Executor::ExtModuleKinds::rt_udf_gpu_module] =
            it_gpu->second;
      } else {
        Executor::extension_module_sources.erase(
            Executor::ExtModuleKinds::rt_udf_gpu_module);
      }
    } /* else avoid locking compilation if registration does not change
         the rt_udf_cpu/gpu_module instances */

    VLOG(1) << "Registering runtime UDTFs:\n";

    table_functions::TableFunctionsFactory::reset();

    for (auto it = udtfs.begin(); it != udtfs.end(); it++) {
      VLOG(1) << "UDTF name=" << it->name << std::endl;
      table_functions::TableFunctionsFactory::add(
          it->name,
          table_functions::TableFunctionOutputRowSizer{
              ThriftSerializers::from_thrift(it->sizerType),
              static_cast<size_t>(it->sizerArgPos)},
          ThriftSerializers::from_thrift(it->inputArgTypes),
          ThriftSerializers::from_thrift(it->outputArgTypes),
          ThriftSerializers::from_thrift(it->sqlArgTypes),
          it->annotations,
          /*is_runtime =*/true);
    }
    /* Register extension functions with Calcite server */
    CHECK(calcite_);
    auto udtfs_ = ThriftSerializers::to_thrift(
        table_functions::TableFunctionsFactory::get_table_funcs(/*is_runtime=*/true));
    calcite_->setRuntimeExtensionFunctions(udfs, udtfs_, /*is_runtime =*/true);

    /* Update the extension function whitelist */
    std::string whitelist = calcite_->getRuntimeExtensionFunctionWhitelist();
    VLOG(1) << "Registering runtime extension functions with CodeGen using whitelist:\n"
            << whitelist;
    ExtensionFunctionsWhitelist::clearRTUdfs();
    ExtensionFunctionsWhitelist::addRTUdfs(whitelist);
  });
}

void DBHandler::get_function_names(std::vector<std::string>& _return,
                                   const TSessionId& session) {
  for (auto udf_name :
       ExtensionFunctionsWhitelist::get_udfs_name(/* is_runtime */ false)) {
    if (std::find(_return.begin(), _return.end(), udf_name) == _return.end()) {
      _return.emplace_back(udf_name);
    }
  }
}

void DBHandler::get_runtime_function_names(std::vector<std::string>& _return,
                                           const TSessionId& session) {
  for (auto udf_name :
       ExtensionFunctionsWhitelist::get_udfs_name(/* is_runtime */ true)) {
    if (std::find(_return.begin(), _return.end(), udf_name) == _return.end()) {
      _return.emplace_back(udf_name);
    }
  }
}

void DBHandler::get_function_details(std::vector<TUserDefinedFunction>& _return,
                                     const TSessionId& session,
                                     const std::vector<std::string>& udf_names) {
  for (const std::string& udf_name : udf_names) {
    for (auto udf : ExtensionFunctionsWhitelist::get_ext_funcs(udf_name)) {
      _return.emplace_back(ThriftSerializers::to_thrift(udf));
    }
  }
}

void DBHandler::get_table_function_names(std::vector<std::string>& _return,
                                         const TSessionId& session) {
  for (auto tf : table_functions::TableFunctionsFactory::get_table_funcs()) {
    const std::string& name = tf.getName(/* drop_suffix */ true, /* to_lower */ true);
    if (std::find(_return.begin(), _return.end(), name) == _return.end()) {
      _return.emplace_back(name);
    }
  }
}

void DBHandler::get_runtime_table_function_names(std::vector<std::string>& _return,
                                                 const TSessionId& session) {
  for (auto tf :
       table_functions::TableFunctionsFactory::get_table_funcs(/* is_runtime */ true)) {
    const std::string& name = tf.getName(/* drop_suffix */ true, /* to_lower */ true);
    if (std::find(_return.begin(), _return.end(), name) == _return.end()) {
      _return.emplace_back(name);
    }
  }
}

void DBHandler::get_table_function_details(
    std::vector<TUserDefinedTableFunction>& _return,
    const TSessionId& session,
    const std::vector<std::string>& udtf_names) {
  for (const std::string& udtf_name : udtf_names) {
    for (auto tf : table_functions::TableFunctionsFactory::get_table_funcs(udtf_name)) {
      _return.emplace_back(ThriftSerializers::to_thrift(tf));
    }
  }
}

void DBHandler::convertResultSet(ExecutionResult& result,
                                 const Catalog_Namespace::SessionInfo& session_info,
                                 const std::string& query_state_str,
                                 TQueryResult& _return) {
  // Stuff ResultSet into _return (which is a TQueryResult)
  // calls convertRows, but after some setup using session_info

  auto session_ptr = get_session_ptr(session_info.get_session_id());
  CHECK(session_ptr);
  auto qs = create_query_state(session_ptr, query_state_str);
  QueryStateProxy qsp = qs->createQueryStateProxy();

  // heavysql only accepts column format as being 'VALID",
  //   assume that heavydb should only return column format
  int32_t nRows = result.getDataPtr()->rowCount();

  convertData(_return,
              result,
              qsp,
              /*column_format=*/true,
              /*first_n=*/nRows,
              /*at_most_n=*/nRows);
}

static std::unique_ptr<RexLiteral> genLiteralStr(std::string val) {
  return std::unique_ptr<RexLiteral>(
      new RexLiteral(val, SQLTypes::kTEXT, SQLTypes::kTEXT, 0, 0, 0, 0));
}

ExecutionResult DBHandler::getUserSessions(
    std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr) {
  std::shared_ptr<ResultSet> rSet = nullptr;
  std::vector<TargetMetaInfo> label_infos;

  if (!session_ptr->get_currentUser().isSuper) {
    throw std::runtime_error(
        "SHOW USER SESSIONS failed, because it can only be executed by super user.");
  } else {
    // label_infos -> column labels
    std::vector<std::string> labels{
        "session_id", "login_name", "client_address", "db_name"};
    for (const auto& label : labels) {
      label_infos.emplace_back(label, SQLTypeInfo(kTEXT, true));
    }

    // logical_values -> table data
    std::vector<RelLogicalValues::RowValues> logical_values;
    auto sessions = sessions_store_->getAllSessions();
    for (const auto& session_ptr : sessions) {
      logical_values.emplace_back(RelLogicalValues::RowValues{});
      logical_values.back().emplace_back(
          genLiteralStr(session_ptr->get_public_session_id()));
      logical_values.back().emplace_back(
          genLiteralStr(session_ptr->get_currentUser().userName));
      logical_values.back().emplace_back(
          genLiteralStr(session_ptr->get_connection_info()));
      logical_values.back().emplace_back(
          genLiteralStr(session_ptr->getCatalog().getCurrentDB().dbName));
    }

    // Create ResultSet
    rSet = std::shared_ptr<ResultSet>(
        ResultSetLogicalValuesBuilder::create(label_infos, logical_values));
  }
  return ExecutionResult(rSet, label_infos);
}

ExecutionResult DBHandler::getQueries(
    std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr) {
  std::shared_ptr<ResultSet> rSet = nullptr;
  std::vector<TargetMetaInfo> label_infos;
  auto current_user_name = session_ptr->get_currentUser().userName;
  auto is_super_user = session_ptr->get_currentUser().isSuper.load();

  std::vector<std::string> labels{"query_session_id",
                                  "current_status",
                                  "executor_id",
                                  "submitted",
                                  "query_str",
                                  "login_name",
                                  "client_address",
                                  "db_name",
                                  "exec_device_type"};
  for (const auto& label : labels) {
    label_infos.emplace_back(label, SQLTypeInfo(kTEXT, true));
  }

  std::vector<RelLogicalValues::RowValues> logical_values;
  auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID,
                                        jit_debug_ ? "/tmp" : "",
                                        jit_debug_ ? "mapdquery" : "",
                                        system_parameters_);
  CHECK(executor);
  auto sessions = (is_super_user ? sessions_store_->getAllSessions()
                                 : sessions_store_->getUserSessions(current_user_name));
  for (const auto& query_session_ptr : sessions) {
    std::vector<QuerySessionStatus> query_infos;
    {
      heavyai::shared_lock<heavyai::shared_mutex> session_read_lock(
          executor->getSessionLock());
      query_infos = executor->getQuerySessionInfo(query_session_ptr->get_session_id(),
                                                  session_read_lock);
    }
    // if there exists query info fired from this session we report it to user
    const std::string getQueryStatusStr[] = {"UNDEFINED",
                                             "PENDING_QUEUE",
                                             "PENDING_EXECUTOR",
                                             "RUNNING_QUERY_KERNEL",
                                             "RUNNING_REDUCTION",
                                             "RUNNING_IMPORTER"};
    bool is_table_import_session = false;
    for (QuerySessionStatus& query_info : query_infos) {
      logical_values.emplace_back(RelLogicalValues::RowValues{});
      logical_values.back().emplace_back(
          genLiteralStr(query_session_ptr->get_public_session_id()));
      auto query_status = query_info.getQueryStatus();
      logical_values.back().emplace_back(genLiteralStr(getQueryStatusStr[query_status]));
      if (query_status == QuerySessionStatus::QueryStatus::RUNNING_IMPORTER) {
        is_table_import_session = true;
      }
      logical_values.back().emplace_back(
          genLiteralStr(::toString(query_info.getExecutorId())));
      logical_values.back().emplace_back(
          genLiteralStr(query_info.getQuerySubmittedTime()));
      logical_values.back().emplace_back(genLiteralStr(query_info.getQueryStr()));
      logical_values.back().emplace_back(
          genLiteralStr(query_session_ptr->get_currentUser().userName));
      logical_values.back().emplace_back(
          genLiteralStr(query_session_ptr->get_connection_info()));
      logical_values.back().emplace_back(
          genLiteralStr(query_session_ptr->getCatalog().getCurrentDB().dbName));
      if (query_session_ptr->get_executor_device_type() == ExecutorDeviceType::GPU &&
          !is_table_import_session) {
        logical_values.back().emplace_back(genLiteralStr("GPU"));
      } else {
        logical_values.back().emplace_back(genLiteralStr("CPU"));
      }
    }
  }

  rSet = std::shared_ptr<ResultSet>(
      ResultSetLogicalValuesBuilder::create(label_infos, logical_values));

  return ExecutionResult(rSet, label_infos);
}

void DBHandler::get_queries_info(std::vector<TQueryInfo>& _return,
                                 const TSessionId& session_id_or_json) {
  heavyai::RequestInfo const request_info(session_id_or_json);
  SET_REQUEST_ID(request_info.requestId());
  auto stdlog = STDLOG(get_session_ptr(request_info.sessionId()));
  auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID,
                                        jit_debug_ ? "/tmp" : "",
                                        jit_debug_ ? "mapdquery" : "",
                                        system_parameters_);
  CHECK(executor);
  auto sessions = sessions_store_->getAllSessions();
  for (const auto& query_session_ptr : sessions) {
    const auto query_session_user_name = query_session_ptr->get_currentUser().userName;
    std::vector<QuerySessionStatus> query_infos;
    {
      heavyai::shared_lock<heavyai::shared_mutex> session_read_lock(
          executor->getSessionLock());
      query_infos = executor->getQuerySessionInfo(query_session_ptr->get_session_id(),
                                                  session_read_lock);
    }
    // if there exists query info fired from this session we report it to user
    const std::string getQueryStatusStr[] = {"UNDEFINED",
                                             "PENDING_QUEUE",
                                             "PENDING_EXECUTOR",
                                             "RUNNING_QUERY_KERNEL",
                                             "RUNNING_REDUCTION",
                                             "RUNNING_IMPORTER"};
    TQueryInfo info;
    for (QuerySessionStatus& query_info : query_infos) {
      info.query_session_id = query_session_ptr->get_session_id();
      info.query_public_session_id = query_session_ptr->get_public_session_id();
      info.current_status = getQueryStatusStr[query_info.getQueryStatus()];
      info.query_str = query_info.getQueryStr();
      info.executor_id = query_info.getExecutorId();
      info.submitted = query_info.getQuerySubmittedTime();
      info.login_name = query_session_user_name;
      info.client_address = query_session_ptr->get_connection_info();
      info.db_name = query_session_ptr->getCatalog().getCurrentDB().dbName;
      if (query_session_ptr->get_executor_device_type() == ExecutorDeviceType::GPU) {
        info.exec_device_type = "GPU";
      } else {
        info.exec_device_type = "CPU";
      }
    }
    _return.push_back(info);
  }
}

void DBHandler::interruptQuery(const Catalog_Namespace::SessionInfo& session_info,
                               const std::string& target_session) {
  // capture the interrupt request from user and then pass to corresponding Executors
  // that queries fired by the given session are assigned
  // Basic-flow that each query session gets through:
  // Enroll --> Update (query session info / executor) --> Running -> Cleanup
  // 1. We have to separate 1) "target" query session to interrupt and 2) request session
  // Here, we have to focus on "target" session: all interruption management is based on
  // the "target" session
  // 2. Session info and its required data structures are global to Executor, so
  // we can send the interrupt request from UNITARY_EXECUTOR (note that the actual query
  // is processed by specific Executor but can also access the global data structure)
  // to the Executor that the session's query has been assigned
  // this means each Executor should handle the interrupt request, and then update its
  // the latest status to the global session map for the correctness
  // 3. Three target session's status: PENDING_QUEUE / PENDING_EXECUTOR / RUNNING
  // (for now we can interrupt a query at "PENDING_EXECUTOR" and "RUNNING")
  // 4. each session has 1) a list of queries that the session tries to initiate and
  // 2) a interrupt flag map that indicates whether the session is interrupted
  // If a session is interrupted, we turn the flag for the session on so as to Executor
  // can know about the user's interrupt request on the query (after all queries are
  // removed then the session's query list and its flag are also deleted). And those
  // info is managed by Executor's global data structure
  // 5. To interrupt queries at "PENDING_EXECUTOR", corresponding Executor regularly
  // checks the interrupt flag of the session, and throws an exception if got interrupted
  // For the case of running query, we also turn the flag in device memory on in async
  // manner so as to inform the query kernel about the latest interrupt flag status
  // (it also checks the flag regularly during the query kernel execution and
  // query threads return with the error code if necessary -->
  // for this we inject interrupt flag checking logic in the generated query kernel)
  // 6. Interruption are implemented by throwing runtime_error that contains a visible
  // error message like "Query has been interrupted"

  if (!g_enable_runtime_query_interrupt && !g_enable_non_kernel_time_query_interrupt) {
    // at least type of query interruption is enabled to allow kill query
    // if non-kernel query interrupt is enabled but tries to kill that type's query?
    // then the request is skipped
    // todo(yoonmin): improve kill query cmd under both types of query
    throw std::runtime_error(
        "Unable to interrupt running query. Query interrupt is disabled.");
  }

  CHECK_EQ(target_session.length(), static_cast<unsigned long>(8));
  auto target_query_session = sessions_store_->getByPublicID(target_session);
  if (!target_query_session) {
    throw std::runtime_error(
        "Unable to interrupt running query. An invalid query session is given.");
  }
  auto target_session_id = target_query_session->get_session_id();
  auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID,
                                        jit_debug_ ? "/tmp" : "",
                                        jit_debug_ ? "mapdquery" : "",
                                        system_parameters_);
  CHECK(executor);

  auto non_admin_interrupt_user = !session_info.get_currentUser().isSuper.load();
  auto interrupt_user_name = session_info.get_currentUser().userName;
  if (non_admin_interrupt_user) {
    auto target_user_name = target_query_session->get_currentUser().userName;
    if (target_user_name.compare(interrupt_user_name) != 0) {
      throw std::runtime_error("Unable to interrupt running query.");
    }
  }

  auto target_executor_ids = executor->getExecutorIdsRunningQuery(target_session_id);
  if (target_executor_ids.empty()) {
    heavyai::shared_lock<heavyai::shared_mutex> session_read_lock(
        executor->getSessionLock());
    if (executor->checkIsQuerySessionEnrolled(target_session_id, session_read_lock)) {
      session_read_lock.unlock();
      VLOG(1) << "Received interrupt: "
              << "User " << session_info.get_currentUser().userLoggable()
              << ", LeafCount " << leaf_aggregator_.leafCount() << ", Database "
              << session_info.getCatalog().getCurrentDB().dbName << std::endl;
      executor->interrupt(target_session_id, session_info.get_session_id());
    }
  } else {
    for (auto& executor_id : target_executor_ids) {
      VLOG(1) << "Received interrupt: "
              << "User " << session_info.get_currentUser().userLoggable() << ", Executor "
              << executor_id << ", LeafCount " << leaf_aggregator_.leafCount()
              << ", Database " << session_info.getCatalog().getCurrentDB().dbName
              << std::endl;
      auto target_executor = Executor::getExecutor(executor_id);
      target_executor->interrupt(target_session_id, session_info.get_session_id());
    }
  }
}

void DBHandler::alterSystemClear(const std::string& session_id,
                                 ExecutionResult& result,
                                 const std::string& cache_type,
                                 int64_t& execution_time_ms) {
  result = ExecutionResult();
  if (to_upper(cache_type) == "CPU") {
    execution_time_ms = measure<>::execution([&]() { clear_cpu_memory(session_id); });
  } else if (to_upper(cache_type) == "GPU") {
    execution_time_ms = measure<>::execution([&]() { clear_gpu_memory(session_id); });
  } else if (to_upper(cache_type) == "RENDER") {
    execution_time_ms = measure<>::execution([&]() { clearRenderMemory(session_id); });
  } else {
    throw std::runtime_error("Invalid cache type. Valid values are CPU,GPU or RENDER");
  }
}

void DBHandler::alterSession(const std::string& session_id,
                             ExecutionResult& result,
                             const std::pair<std::string, std::string>& session_parameter,
                             int64_t& execution_time_ms) {
  result = ExecutionResult();
  if (session_parameter.first == "EXECUTOR_DEVICE") {
    std::string parameter_value = to_upper(session_parameter.second);
    TExecuteMode::type executorType;
    if (parameter_value == "GPU") {
      executorType = TExecuteMode::type::GPU;
    } else if (parameter_value == "CPU") {
      executorType = TExecuteMode::type::CPU;
    } else {
      throw std::runtime_error("Cannot set the " + session_parameter.first + " to " +
                               session_parameter.second +
                               ". Valid options are CPU and GPU");
    }
    execution_time_ms =
        measure<>::execution([&]() { set_execution_mode(session_id, executorType); });
  } else if (session_parameter.first == "CURRENT_DATABASE") {
    execution_time_ms = measure<>::execution(
        [&]() { switch_database(session_id, session_parameter.second); });
  }
}

void DBHandler::executeDdl(
    TQueryResult& _return,
    const std::string& query_ra,
    std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr) {
  DdlCommandExecutor executor = DdlCommandExecutor(query_ra, session_ptr);
  std::string commandStr = executor.commandStr();

  if (executor.isKillQuery()) {
    interruptQuery(*session_ptr, executor.getTargetQuerySessionToKill());
  } else {
    ExecutionResult result;
    int64_t execution_time_ms;
    if (executor.isShowQueries()) {
      // getQueries still requires Thrift cannot be nested into DdlCommandExecutor
      _return.execution_time_ms +=
          measure<>::execution([&]() { result = getQueries(session_ptr); });
    } else if (executor.isShowUserSessions()) {
      // getUserSessions still requires Thrift cannot be nested into DdlCommandExecutor
      _return.execution_time_ms +=
          measure<>::execution([&]() { result = getUserSessions(session_ptr); });
    } else if (executor.isAlterSystemClear()) {
      alterSystemClear(session_ptr->get_session_id(),
                       result,
                       executor.returnCacheType(),
                       execution_time_ms);
      _return.execution_time_ms += execution_time_ms;

    } else if (executor.isAlterSessionSet()) {
      alterSession(session_ptr->get_session_id(),
                   result,
                   executor.getSessionParameter(),
                   execution_time_ms);
      _return.execution_time_ms += execution_time_ms;
    } else if (executor.isAlterSystemControlExecutorQueue()) {
      result = ExecutionResult();
      if (executor.returnQueueAction() == "PAUSE") {
        _return.execution_time_ms += measure<>::execution(
            [&]() { pause_executor_queue(session_ptr->get_session_id()); });
      } else if (executor.returnQueueAction() == "RESUME") {
        _return.execution_time_ms += measure<>::execution(
            [&]() { resume_executor_queue(session_ptr->get_session_id()); });
      } else {
        throw std::runtime_error("Unknown queue command.");
      }
    } else {
      _return.execution_time_ms +=
          measure<>::execution([&]() { result = executor.execute(read_only_); });
    }

    if (!result.empty()) {
      // reduce execution time by the time spent during queue waiting
      _return.execution_time_ms -= result.getRows()->getQueueTime();
      convertResultSet(result, *session_ptr, commandStr, _return);
    }
  }
}

void DBHandler::executeDdl(
    ExecutionResult& _return,
    const std::string& query_ra,
    std::shared_ptr<Catalog_Namespace::SessionInfo const> session_ptr) {
  DdlCommandExecutor executor = DdlCommandExecutor(query_ra, session_ptr);
  std::string commandStr = executor.commandStr();

  if (executor.isKillQuery()) {
    interruptQuery(*session_ptr, executor.getTargetQuerySessionToKill());
  } else {
    int64_t execution_time_ms;
    if (executor.isShowQueries()) {
      // getQueries still requires Thrift cannot be nested into DdlCommandExecutor
      execution_time_ms =
          measure<>::execution([&]() { _return = getQueries(session_ptr); });
    } else if (executor.isShowUserSessions()) {
      // getUserSessions still requires Thrift cannot be nested into DdlCommandExecutor
      execution_time_ms =
          measure<>::execution([&]() { _return = getUserSessions(session_ptr); });
    } else if (executor.isAlterSystemClear()) {
      alterSystemClear(session_ptr->get_session_id(),
                       _return,
                       executor.returnCacheType(),
                       execution_time_ms);
    } else if (executor.isAlterSessionSet()) {
      alterSession(session_ptr->get_session_id(),
                   _return,
                   executor.getSessionParameter(),
                   execution_time_ms);
    } else if (executor.isAlterSystemControlExecutorQueue()) {
      _return = ExecutionResult();
      if (executor.returnQueueAction() == "PAUSE") {
        execution_time_ms = measure<>::execution(
            [&]() { pause_executor_queue(session_ptr->get_session_id()); });
      } else if (executor.returnQueueAction() == "RESUME") {
        execution_time_ms = measure<>::execution(
            [&]() { resume_executor_queue(session_ptr->get_session_id()); });
      } else {
        throw std::runtime_error("Unknwon queue command.");
      }
    } else {
      execution_time_ms =
          measure<>::execution([&]() { _return = executor.execute(read_only_); });
    }
    _return.setExecutionTime(execution_time_ms);
  }
  if (_return.getResultType() == ExecutionResult::QueryResult) {
    // ResultType defaults to QueryResult => which can limit
    //   the number of lines output via ConvertRow... use CalciteDdl instead
    _return.setResultType(ExecutionResult::CalciteDdl);
  }
}

void DBHandler::resizeDispatchQueue(size_t queue_size) {
  dispatch_queue_ = std::make_unique<QueryDispatchQueue>(queue_size);
}

bool DBHandler::checkInMemorySystemTableQuery(
    const std::unordered_set<shared::TableKey>& selected_table_keys) const {
  bool is_in_memory_system_table_query{false};
  const auto info_schema_catalog =
      Catalog_Namespace::SysCatalog::instance().getCatalog(shared::kInfoSchemaDbName);
  if (info_schema_catalog) {
    for (const auto& table_key : selected_table_keys) {
      if (table_key.db_id == info_schema_catalog->getDatabaseId()) {
        auto td = info_schema_catalog->getMetadataForTable(table_key.table_id, false);
        CHECK(td);
        if (check_and_reset_in_memory_system_table(*info_schema_catalog, *td)) {
          is_in_memory_system_table_query = true;
        }
      }
    }
  }
  return is_in_memory_system_table_query;
}
