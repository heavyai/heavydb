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

/*
 * File:   MapDHandler.cpp
 * Author: michael
 *
 * Created on Jan 1, 2017, 12:40 PM
 */

#include "MapDHandler.h"
#include "DistributedLoader.h"
#include "MapDServer.h"
#include "TokenCompletionHints.h"
#ifdef HAVE_PROFILER
#include <gperftools/heap-profiler.h>
#endif  // HAVE_PROFILER
#include <thrift/concurrency/PlatformThreadFactory.h>
#include <thrift/concurrency/ThreadManager.h>
#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/protocol/TJSONProtocol.h>
#include <thrift/server/TThreadedServer.h>
#include <thrift/transport/TBufferTransports.h>
#include <thrift/transport/THttpServer.h>
#include <thrift/transport/TServerSocket.h>

#include "MapDRelease.h"

#include "Calcite/Calcite.h"

#include "QueryEngine/RelAlgExecutor.h"

#include "Catalog/Catalog.h"
#include "Fragmenter/InsertOrderFragmenter.h"
#include "Import/Importer.h"
#include "Parser/ParserWrapper.h"
#include "Parser/ReservedKeywords.h"
#include "Parser/parser.h"
#include "Planner/Planner.h"
#include "QueryEngine/CalciteAdapter.h"
#include "QueryEngine/Execute.h"
#include "QueryEngine/ExtensionFunctionsWhitelist.h"
#include "QueryEngine/GpuMemUtils.h"
#include "QueryEngine/JsonAccessors.h"
#include "Shared/geosupport.h"
#include "Shared/mapd_shared_mutex.h"
#include "Shared/measure.h"
#include "Shared/scope.h"
#include "Shared/StringTransform.h"
#include "Shared/MapDParameters.h"
#include "MapDRenderHandler.h"
#include "MapDDistributedHandler.h"

#include <fcntl.h>
#include <glog/logging.h>
#include <signal.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/filesystem.hpp>
#include <boost/make_shared.hpp>
#include <boost/program_options.hpp>
#include <boost/regex.hpp>
#include <boost/tokenizer.hpp>
#include <cmath>
#include <fstream>
#include <future>
#include <map>
#include <memory>
#include <random>
#include <regex>
#include <string>
#include <thread>
#include <typeinfo>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>

#include "QueryEngine/ArrowUtil.h"

#define INVALID_SESSION_ID ""

#define THROW_MAPD_EXCEPTION(errstr) \
  TMapDException ex;                 \
  ex.error_msg = errstr;             \
  LOG(ERROR) << ex.error_msg;        \
  throw ex;

MapDHandler::MapDHandler(const std::vector<LeafHostInfo>& db_leaves,
                         const std::vector<LeafHostInfo>& string_leaves,
                         const std::string& base_data_path,
                         const std::string& executor_device,
                         const bool allow_multifrag,
                         const bool jit_debug,
                         const bool read_only,
                         const bool allow_loop_joins,
                         const bool enable_rendering,
                         const size_t cpu_buffer_mem_bytes,
                         const size_t render_mem_bytes,
                         const int num_gpus,
                         const int start_gpu,
                         const size_t reserved_gpu_mem,
                         const size_t num_reader_threads,
                         const LdapMetadata ldapMetadata,
                         const MapDParameters& mapd_parameters,
                         const std::string& db_convert_dir,
                         const bool legacy_syntax,
                         const bool access_priv_check)
    : leaf_aggregator_(db_leaves),
      string_leaves_(string_leaves),
      base_data_path_(base_data_path),
      random_gen_(std::random_device{}()),
      session_id_dist_(0, INT32_MAX),
      jit_debug_(jit_debug),
      allow_multifrag_(allow_multifrag),
      read_only_(read_only),
      allow_loop_joins_(allow_loop_joins),
      mapd_parameters_(mapd_parameters),
      legacy_syntax_(legacy_syntax),
      super_user_rights_(false),
      access_priv_check_(access_priv_check) {
  LOG(INFO) << "MapD Server " << MAPD_RELEASE;
  if (executor_device == "gpu") {
#ifdef HAVE_CUDA
    executor_device_type_ = ExecutorDeviceType::GPU;
    cpu_mode_only_ = false;
#else
    executor_device_type_ = ExecutorDeviceType::CPU;
    LOG(ERROR) << "This build isn't CUDA enabled, will run on CPU";
    cpu_mode_only_ = true;
#endif  // HAVE_CUDA
  } else if (executor_device == "hybrid") {
    executor_device_type_ = ExecutorDeviceType::Hybrid;
    cpu_mode_only_ = false;
  } else {
    executor_device_type_ = ExecutorDeviceType::CPU;
    cpu_mode_only_ = true;
  }
  const auto data_path = boost::filesystem::path(base_data_path_) / "mapd_data";
  // calculate the total amount of memory we need to reserve from each gpu that the Buffer manage cannot ask for
  size_t total_reserved = reserved_gpu_mem;
  if (enable_rendering) {
    total_reserved += render_mem_bytes;
  }
  data_mgr_.reset(new Data_Namespace::DataMgr(data_path.string(),
                                              cpu_buffer_mem_bytes,
                                              !cpu_mode_only_,
                                              num_gpus,
                                              db_convert_dir,
                                              start_gpu,
                                              total_reserved,
                                              num_reader_threads));
  calcite_.reset(new Calcite(mapd_parameters.mapd_server_port,
                             mapd_parameters.calcite_port,
                             base_data_path_,
                             mapd_parameters_.calcite_max_mem));
  ExtensionFunctionsWhitelist::add(calcite_->getExtensionFunctionWhitelist());

  if (!data_mgr_->gpusPresent()) {
    executor_device_type_ = ExecutorDeviceType::CPU;
    LOG(ERROR) << "No GPUs detected, falling back to CPU mode";
    cpu_mode_only_ = true;
  }

  switch (executor_device_type_) {
    case ExecutorDeviceType::GPU:
      LOG(INFO) << "Started in GPU mode" << std::endl;
      break;
    case ExecutorDeviceType::CPU:
      LOG(INFO) << "Started in CPU mode" << std::endl;
      break;
    case ExecutorDeviceType::Hybrid:
      LOG(INFO) << "Started in Hybrid mode" << std::endl;
  }

  sys_cat_.reset(
      new Catalog_Namespace::SysCatalog(base_data_path_, data_mgr_, ldapMetadata, calcite_, false, access_priv_check_));
  import_path_ = boost::filesystem::path(base_data_path_) / "mapd_import";
  start_time_ = std::time(nullptr);

  if (enable_rendering) {
    try {
      render_handler_.reset(new MapDRenderHandler(this, render_mem_bytes, num_gpus, start_gpu));
    } catch (const std::exception& e) {
      LOG(ERROR) << "Backend rendering disabled: " << e.what();
    }
  }

  if (leaf_aggregator_.leafCount() > 0) {
    try {
      agg_handler_.reset(new MapDAggHandler(this));
    } catch (const std::exception& e) {
      LOG(ERROR) << "Distributed aggregator support disabled: " << e.what();
    }
  } else if (g_cluster) {
    try {
      leaf_handler_.reset(new MapDLeafHandler(this));
    } catch (const std::exception& e) {
      LOG(ERROR) << "Distributed leaf support disabled: " << e.what();
    }
  }
}

MapDHandler::~MapDHandler() {
  LOG(INFO) << "mapd_server exits." << std::endl;
}

void MapDHandler::check_read_only(const std::string& str) {
  if (MapDHandler::read_only_) {
    THROW_MAPD_EXCEPTION(str + " disabled: server running in read-only mode.");
  }
}

std::string generate_random_string(const size_t len) {
  static char charset[] =
      "0123456789"
      "abcdefghijklmnopqrstuvwxyz"
      "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

  static std::mt19937 prng{std::random_device{}()};
  static std::uniform_int_distribution<size_t> dist(0, strlen(charset) - 1);

  std::string str;
  str.reserve(len);
  for (size_t i = 0; i < len; i++) {
    str += charset[dist(prng)];
  }
  return str;
}

// internal connection for connections with no password
void MapDHandler::internal_connect(TSessionId& session, const std::string& user, const std::string& dbname) {
  mapd_lock_guard<mapd_shared_mutex> write_lock(sessions_mutex_);
  Catalog_Namespace::UserMetadata user_meta;
  if (!sys_cat_->getMetadataForUser(user, user_meta)) {
    THROW_MAPD_EXCEPTION(std::string("User ") + user + " does not exist.");
  }
  connectImpl(session, user, std::string(""), dbname, user_meta);
}

void MapDHandler::connect(TSessionId& session,
                          const std::string& user,
                          const std::string& passwd,
                          const std::string& dbname) {
  mapd_lock_guard<mapd_shared_mutex> write_lock(sessions_mutex_);
  Catalog_Namespace::UserMetadata user_meta;
  if (!sys_cat_->getMetadataForUser(user, user_meta)) {
    THROW_MAPD_EXCEPTION(std::string("User ") + user + " does not exist.");
  }
  if (!super_user_rights_) {
    if (!sys_cat_->checkPasswordForUser(passwd, user_meta)) {
      THROW_MAPD_EXCEPTION(std::string("Password for User ") + user + " is incorrect.");
    }
  }
  connectImpl(session, user, passwd, dbname, user_meta);
}

void MapDHandler::connectImpl(TSessionId& session,
                              const std::string& user,
                              const std::string& passwd,
                              const std::string& dbname,
                              Catalog_Namespace::UserMetadata& user_meta) {
  Catalog_Namespace::DBMetadata db_meta;
  if (!sys_cat_->getMetadataForDB(dbname, db_meta)) {
    THROW_MAPD_EXCEPTION(std::string("Database ") + dbname + " does not exist.");
  }
  if (!sys_cat_->isAccessPrivCheckEnabled()) {
    // insert privilege is being treated as access allowed for now
    Privileges privs;
    privs.insert_ = true;
    privs.select_ = false;
    // use old style check for DB object level privs code only to check user access to the database
    if (!sys_cat_->checkPrivileges(user_meta, db_meta, privs)) {
      THROW_MAPD_EXCEPTION(std::string("User ") + user + " is not authorized to access database " + dbname);
    }
  }
  session = INVALID_SESSION_ID;
  while (true) {
    session = generate_random_string(32);
    auto session_it = sessions_.find(session);
    if (session_it == sessions_.end())
      break;
  }
  auto cat_it = cat_map_.find(dbname);
  if (cat_it == cat_map_.end()) {
    Catalog_Namespace::Catalog* cat = new Catalog_Namespace::Catalog(
        base_data_path_, db_meta, data_mgr_, string_leaves_, calcite_, access_priv_check_);
    cat_map_[dbname].reset(cat);
    sessions_[session].reset(
        new Catalog_Namespace::SessionInfo(cat_map_[dbname], user_meta, executor_device_type_, session));
    sessions_[session]->setDatabaseCatalog(dbname, cat);
    if (dbname == MAPD_SYSTEM_DB) {
      auto mapd_session_ptr = sessions_[session];
      mapd_session_ptr->setSysCatalog(static_cast<Catalog_Namespace::SysCatalog*>(cat));
    }
  } else {
    sessions_[session].reset(
        new Catalog_Namespace::SessionInfo(cat_it->second, user_meta, executor_device_type_, session));
  }
  if (!super_user_rights_) {  // no need to connect to leaf_aggregator_ at this time while doing warmup
    if (leaf_aggregator_.leafCount() > 0) {
      const auto parent_session_info_ptr = sessions_[session];
      CHECK(parent_session_info_ptr);
      leaf_aggregator_.connect(*parent_session_info_ptr, user, passwd, dbname);
      return;
    }
  }
  LOG(INFO) << "User " << user << " connected to database " << dbname << std::endl;
}

void MapDHandler::disconnect(const TSessionId& session) {
  mapd_lock_guard<mapd_shared_mutex> write_lock(sessions_mutex_);
  if (leaf_aggregator_.leafCount() > 0) {
    leaf_aggregator_.disconnect(session);
  }
  auto session_it = MapDHandler::get_session_it(session);
  const auto dbname = session_it->second->get_catalog().get_currentDB().dbName;
  LOG(INFO) << "User " << session_it->second->get_currentUser().userName << " disconnected from database " << dbname
            << std::endl;
  sessions_.erase(session_it);
}

void MapDHandler::interrupt(const TSessionId& session) {
  if (g_enable_dynamic_watchdog) {
    mapd_lock_guard<mapd_shared_mutex> read_lock(sessions_mutex_);
    if (leaf_aggregator_.leafCount() > 0) {
      leaf_aggregator_.interrupt(session);
    }
    auto session_it = get_session_it(session);
    const auto dbname = session_it->second->get_catalog().get_currentDB().dbName;
    auto session_info_ptr = session_it->second.get();
    auto& cat = session_info_ptr->get_catalog();
    auto executor = Executor::getExecutor(
        cat.get_currentDB().dbId, jit_debug_ ? "/tmp" : "", jit_debug_ ? "mapdquery" : "", mapd_parameters_, nullptr);
    CHECK(executor);

    VLOG(1) << "Received interrupt: "
            << "Session " << session << ", Executor " << executor << ", leafCount " << leaf_aggregator_.leafCount()
            << ", User " << session_it->second->get_currentUser().userName << ", Database " << dbname << std::endl;

    executor->interrupt();

    LOG(INFO) << "User " << session_it->second->get_currentUser().userName << " interrupted session with database "
              << dbname << std::endl;
  }
}

void MapDHandler::get_server_status(TServerStatus& _return, const TSessionId& session) {
  _return.read_only = read_only_;
  _return.version = MAPD_RELEASE;
  _return.rendering_enabled = bool(render_handler_);
  _return.start_time = start_time_;
  _return.edition = MAPD_EDITION;
  _return.host_name = "aggregator";
}

void MapDHandler::get_status(std::vector<TServerStatus>& _return, const TSessionId& session) {
  TServerStatus ret;
  ret.read_only = read_only_;
  ret.version = MAPD_RELEASE;
  ret.rendering_enabled = bool(render_handler_);
  ret.start_time = start_time_;
  ret.edition = MAPD_EDITION;
  ret.host_name = "aggregator";
  _return.push_back(ret);
  if (leaf_aggregator_.leafCount() > 0) {
    std::vector<TServerStatus> leaf_status = leaf_aggregator_.getLeafStatus(session);
    _return.insert(_return.end(), leaf_status.begin(), leaf_status.end());
  }
}

void MapDHandler::get_hardware_info(TClusterHardwareInfo& _return, const TSessionId& session) {
  THardwareInfo ret;
  CudaMgr_Namespace::CudaMgr* cuda_mgr = data_mgr_->cudaMgr_;
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
  if (leaf_aggregator_.leafCount() > 0) {
    ret.host_name = "aggregator";
    TClusterHardwareInfo leaf_hardware = leaf_aggregator_.getHardwareInfo(session);
    _return.hardware_info.insert(
        _return.hardware_info.end(), leaf_hardware.hardware_info.begin(), leaf_hardware.hardware_info.end());
  }
}

void MapDHandler::value_to_thrift_column(const TargetValue& tv, const SQLTypeInfo& ti, TColumn& column) {
  const auto scalar_tv = boost::get<ScalarTargetValue>(&tv);
  if (!scalar_tv) {
    const auto list_tv = boost::get<std::vector<ScalarTargetValue>>(&tv);
    CHECK(list_tv);
    CHECK(ti.is_array());
    TColumn tColumn;
    for (const auto& elem_tv : *list_tv) {
      value_to_thrift_column(elem_tv, ti.get_elem_type(), tColumn);
    }
    column.data.arr_col.push_back(tColumn);
    column.nulls.push_back(list_tv->size() == 0);
  } else {
    if (boost::get<int64_t>(scalar_tv)) {
      int64_t data = *(boost::get<int64_t>(scalar_tv));
      column.data.int_col.push_back(data);
      switch (ti.get_type()) {
        case kBOOLEAN:
          column.nulls.push_back(data == NULL_BOOLEAN);
          break;
        case kSMALLINT:
          column.nulls.push_back(data == NULL_SMALLINT);
          break;
        case kINT:
          column.nulls.push_back(data == NULL_INT);
          break;
        case kBIGINT:
          column.nulls.push_back(data == NULL_BIGINT);
          break;
        case kTIME:
        case kTIMESTAMP:
        case kDATE:
        case kINTERVAL_DAY_TIME:
        case kINTERVAL_YEAR_MONTH:
          if (sizeof(time_t) == 4)
            column.nulls.push_back(data == NULL_INT);
          else
            column.nulls.push_back(data == NULL_BIGINT);
          break;
        default:
          column.nulls.push_back(false);
      }
    } else if (boost::get<double>(scalar_tv)) {
      double data = *(boost::get<double>(scalar_tv));
      column.data.real_col.push_back(data);
      if (ti.get_type() == kFLOAT) {
        column.nulls.push_back(data == NULL_FLOAT);
      } else {
        column.nulls.push_back(data == NULL_DOUBLE);
      }
    } else if (boost::get<float>(scalar_tv)) {
      CHECK_EQ(kFLOAT, ti.get_type());
      float data = *(boost::get<float>(scalar_tv));
      column.data.real_col.push_back(data);
      column.nulls.push_back(data == NULL_FLOAT);
    } else if (boost::get<NullableString>(scalar_tv)) {
      auto s_n = boost::get<NullableString>(scalar_tv);
      auto s = boost::get<std::string>(s_n);
      if (s) {
        column.data.str_col.push_back(*s);
      } else {
        column.data.str_col.push_back("");  // null string
        auto null_p = boost::get<void*>(s_n);
        CHECK(null_p && !*null_p);
      }
      column.nulls.push_back(!s);
    } else {
      CHECK(false);
    }
  }
}

TDatum MapDHandler::value_to_thrift(const TargetValue& tv, const SQLTypeInfo& ti) {
  TDatum datum;
  const auto scalar_tv = boost::get<ScalarTargetValue>(&tv);
  if (!scalar_tv) {
    const auto list_tv = boost::get<std::vector<ScalarTargetValue>>(&tv);
    CHECK(list_tv);
    CHECK(ti.is_array());
    for (const auto& elem_tv : *list_tv) {
      const auto scalar_col_val = value_to_thrift(elem_tv, ti.get_elem_type());
      datum.val.arr_val.push_back(scalar_col_val);
    }
    datum.is_null = datum.val.arr_val.empty();
    return datum;
  }
  if (boost::get<int64_t>(scalar_tv)) {
    datum.val.int_val = *(boost::get<int64_t>(scalar_tv));
    switch (ti.get_type()) {
      case kBOOLEAN:
        datum.is_null = (datum.val.int_val == NULL_BOOLEAN);
        break;
      case kSMALLINT:
        datum.is_null = (datum.val.int_val == NULL_SMALLINT);
        break;
      case kINT:
        datum.is_null = (datum.val.int_val == NULL_INT);
        break;
      case kBIGINT:
        datum.is_null = (datum.val.int_val == NULL_BIGINT);
        break;
      case kTIME:
      case kTIMESTAMP:
      case kDATE:
      case kINTERVAL_DAY_TIME:
      case kINTERVAL_YEAR_MONTH:
        if (sizeof(time_t) == 4)
          datum.is_null = (datum.val.int_val == NULL_INT);
        else
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

void MapDHandler::sql_execute(TQueryResult& _return,
                              const TSessionId& session,
                              const std::string& query_str,
                              const bool column_format,
                              const std::string& nonce,
                              const int32_t first_n,
                              const int32_t at_most_n) {
  if (first_n >= 0 && at_most_n >= 0) {
    THROW_MAPD_EXCEPTION(std::string("At most one of first_n and at_most_n can be set"));
  }
  const auto session_info = MapDHandler::get_session(session);
  LOG(INFO) << "sql_execute :" << session << ":query_str:" << query_str;
  if (leaf_aggregator_.leafCount() > 0) {
    if (!agg_handler_) {
      THROW_MAPD_EXCEPTION("Distributed support is disabled.");
    }
    _return.total_time_ms = measure<>::execution([&]() {
      try {
        agg_handler_->cluster_execute(_return, session_info, query_str, column_format, nonce, first_n, at_most_n);
      } catch (std::exception& e) {
        const auto mapd_exception = dynamic_cast<const TMapDException*>(&e);
        THROW_MAPD_EXCEPTION(mapd_exception ? mapd_exception->error_msg : (std::string("Exception: ") + e.what()));
      }
      _return.nonce = nonce;
    });
    LOG(INFO) << "sql_execute-COMPLETED Distributed Execute Time: " << _return.total_time_ms << " (ms)";
  } else {
    _return.total_time_ms = measure<>::execution([&]() {
      MapDHandler::sql_execute_impl(_return,
                                    session_info,
                                    query_str,
                                    column_format,
                                    nonce,
                                    session_info.get_executor_device_type(),
                                    first_n,
                                    at_most_n);
    });
    LOG(INFO) << "sql_execute-COMPLETED Total: " << _return.total_time_ms
              << " (ms), Execution: " << _return.execution_time_ms << " (ms)";
  }
}

void MapDHandler::sql_execute_df(TDataFrame& _return,
                                 const TSessionId& session,
                                 const std::string& query_str,
                                 const TDeviceType::type device_type,
                                 const int32_t device_id,
                                 const int32_t first_n) {
  const auto session_info = MapDHandler::get_session(session);
  int64_t execution_time_ms = 0;
  if (device_type == TDeviceType::GPU) {
    const auto executor_device_type = session_info.get_executor_device_type();
    if (executor_device_type != ExecutorDeviceType::GPU) {
      THROW_MAPD_EXCEPTION(std::string("Exception: GPU mode is not allowed in this session"));
    }
    if (!data_mgr_->gpusPresent()) {
      THROW_MAPD_EXCEPTION(std::string("Exception: no GPU is available in this server"));
    }
    if (device_id < 0 || device_id >= data_mgr_->cudaMgr_->getDeviceCount()) {
      THROW_MAPD_EXCEPTION(std::string("Exception: invalid device_id or unavailable GPU with this ID"));
    }
  }
  LOG(INFO) << query_str;
  int64_t total_time_ms = measure<>::execution([&]() {
    SQLParser parser;
    std::list<std::unique_ptr<Parser::Stmt>> parse_trees;
    std::string last_parsed;
    try {
      ParserWrapper pw{query_str};
      if (!pw.is_ddl && !pw.is_update_dml && !pw.is_other_explain) {
        std::string query_ra;
        execution_time_ms += measure<>::execution([&]() { query_ra = parse_to_ra(query_str, session_info); });
        if (pw.is_select_calcite_explain) {
          throw std::runtime_error("explain is not unsupported by current thrift API");
        }
        execute_rel_alg_df(_return,
                           query_ra,
                           session_info,
                           device_type == TDeviceType::CPU ? ExecutorDeviceType::CPU : ExecutorDeviceType::GPU,
                           static_cast<size_t>(device_id),
                           first_n);
        if (!_return.sm_size) {
          throw std::runtime_error("schema is missing in returned result");
        }
        return;
      }
      LOG(INFO) << "passing query to legacy processor";
    } catch (std::exception& e) {
      THROW_MAPD_EXCEPTION(std::string("Exception: ") + e.what());
    }
    THROW_MAPD_EXCEPTION("Exception: DDL or update DML are not unsupported by current thrift API");
  });
  LOG(INFO) << "Total: " << total_time_ms << " (ms), Execution: " << execution_time_ms << " (ms)";
}

void MapDHandler::sql_execute_gdf(TDataFrame& _return,
                                  const TSessionId& session,
                                  const std::string& query_str,
                                  const int32_t device_id,
                                  const int32_t first_n) {
  sql_execute_df(_return, session, query_str, TDeviceType::GPU, device_id, first_n);
}

// For now we have only one user of a data frame in all cases.
void MapDHandler::deallocate_df(const TSessionId& session,
                                const TDataFrame& df,
                                const TDeviceType::type device_type,
                                const int32_t device_id) {
  const auto session_info = get_session(session);
  int8_t* dev_ptr{0};
  if (device_type == TDeviceType::GPU) {
    std::lock_guard<std::mutex> map_lock(handle_to_dev_ptr_mutex_);
    if (ipc_handle_to_dev_ptr_.count(df.df_handle) != size_t(1)) {
      TMapDException ex;
      ex.error_msg = std::string("Exception: current data frame handle is not bookkept or been inserted twice");
      LOG(ERROR) << ex.error_msg;
      throw ex;
    }
    dev_ptr = ipc_handle_to_dev_ptr_[df.df_handle];
    ipc_handle_to_dev_ptr_.erase(df.df_handle);
  }
  std::vector<char> sm_handle(df.sm_handle.begin(), df.sm_handle.end());
  std::vector<char> df_handle(df.df_handle.begin(), df.df_handle.end());
  ArrowResult result{sm_handle, df.sm_size, df_handle, df.df_size, dev_ptr};
  deallocate_arrow_result(result,
                          device_type == TDeviceType::CPU ? ExecutorDeviceType::CPU : ExecutorDeviceType::GPU,
                          device_id,
                          data_mgr_.get());
}

std::string MapDHandler::apply_copy_to_shim(const std::string& query_str) {
  auto result = query_str;
  {
    // boost::regex copy_to{R"(COPY\s\((.*)\)\sTO\s(.*))", boost::regex::extended | boost::regex::icase};
    boost::regex copy_to{R"(COPY\s*\(([^#])(.+)\)\s+TO\s)", boost::regex::extended | boost::regex::icase};
    apply_shim(result, copy_to, [](std::string& result, const boost::smatch& what) {
      result.replace(what.position(), what.length(), "COPY (#~#" + what[1] + what[2] + "#~#) TO  ");
    });
  }
  return result;
}

void MapDHandler::sql_validate(TTableDescriptor& _return, const TSessionId& session, const std::string& query_str) {
  std::unique_ptr<const Planner::RootPlan> root_plan;
  const auto session_info = get_session(session);
  ParserWrapper pw{query_str};
  if (pw.is_select_explain || pw.is_other_explain || pw.is_ddl || pw.is_update_dml) {
    THROW_MAPD_EXCEPTION("Can only validate SELECT statements.");
  }
  MapDHandler::validate_rel_alg(_return, query_str, session_info);
}

void MapDHandler::get_completion_hints(std::vector<TCompletionHint>& hints,
                                       const TSessionId& session,
                                       const std::string& sql,
                                       const int cursor) {
  const auto last_word = find_last_word_from_cursor(sql, cursor < 0 ? sql.size() : cursor);
  const auto session_info = get_session(session);
  // Get the whitelisted keywords from Calcite. We should probably stop going to Calcite
  // for completions altogether, especially since we have to honor permissions.
  try {
    hints = just_whitelisted_keyword_hints(calcite_->getCompletionHints(session_info, sql, cursor));
  } catch (const std::exception& e) {
    TMapDException ex;
    ex.error_msg = "Exception: " + std::string(e.what());
    LOG(ERROR) << ex.error_msg;
    throw ex;
  }
  const auto column_names_by_table = fill_column_names_by_table(session);
  // Trust the fully qualified columns the most.
  if (get_qualified_column_hints(hints, last_word, column_names_by_table)) {
    return;
  }
  // Not much information to use, just retrieve all table and column names which match the prefix.
  get_column_hints(hints, last_word, column_names_by_table);
  get_table_hints(hints, last_word, column_names_by_table);
  if (hints.empty()) {
    hints = get_keyword_hints(last_word);
  }
}

std::unordered_map<std::string, std::unordered_set<std::string>> MapDHandler::fill_column_names_by_table(
    const TSessionId& session) {
  std::vector<std::string> table_names;
  get_tables(table_names, session);
  std::unordered_map<std::string, std::unordered_set<std::string>> column_names_by_table;
  for (const auto& table_name : table_names) {
    TTableDetails table_details;
    get_table_details(table_details, session, table_name);
    for (const auto& column_type : table_details.row_desc) {
      column_names_by_table[table_name].emplace(column_type.col_name);
    }
  }
  return column_names_by_table;
}

void MapDHandler::validate_rel_alg(TTableDescriptor& _return,
                                   const std::string& query_str,
                                   const Catalog_Namespace::SessionInfo& session_info) {
  try {
    const auto query_ra = parse_to_ra(query_str, session_info);
    TQueryResult result;
    MapDHandler::execute_rel_alg(result, query_ra, true, session_info, ExecutorDeviceType::CPU, -1, -1, false, true);
    const auto& row_desc = fixup_row_descriptor(result.row_set.row_desc, session_info.get_catalog());
    for (const auto& col_desc : row_desc) {
      const auto it_ok = _return.insert(std::make_pair(col_desc.col_name, col_desc));
      CHECK(it_ok.second);
    }
  } catch (std::exception& e) {
    THROW_MAPD_EXCEPTION(std::string("Exception: ") + e.what());
  }
}

void MapDHandler::get_role(std::vector<std::string>& roles,
                           const TSessionId& session,
                           const std::string& roleName,
                           bool userPrivateRole) {
  auto session_it = get_session_it(session);
  auto session_info_ptr = session_it->second.get();
  auto& cat = session_info_ptr->get_catalog();
  auto& sys_cat = static_cast<Catalog_Namespace::SysCatalog&>(cat);
  if (sys_cat.getRole(roleName, userPrivateRole)) {
    if ((session_info_ptr->get_currentUser().isSuper) ||
        (!session_info_ptr->get_currentUser().isSuper &&
         sys_cat.isRoleGrantedToUser(session_info_ptr->get_currentUser().userId, roleName))) {
      roles.push_back(roleName);
    }
  }
  return;
}

void MapDHandler::get_all_roles(std::vector<std::string>& roles, const TSessionId& session, bool userPrivateRole) {
  auto session_it = get_session_it(session);
  auto session_info_ptr = session_it->second.get();
  auto& cat = session_info_ptr->get_catalog();
  auto& sys_cat = static_cast<Catalog_Namespace::SysCatalog&>(cat);
  roles = sys_cat.getAllRoles(
      userPrivateRole, session_info_ptr->get_currentUser().isSuper, session_info_ptr->get_currentUser().userId);
}

void MapDHandler::get_db_object_privileges_for_role(std::vector<TAccessPrivileges>& TDBObjectPrivsForRole,
                                                    const TSessionId& session,
                                                    const std::string& roleName,
                                                    const int16_t objectType,
                                                    const std::string& objectName) {
  TAccessPrivileges tprivObject;
  TDBObjectPrivsForRole.push_back(tprivObject);
}

void MapDHandler::get_db_objects_for_role(std::vector<TDBObject>& TDBObjectsForRole,
                                          const TSessionId& session,
                                          const std::string& roleName) {
  auto session_it = get_session_it(session);
  auto session_info_ptr = session_it->second.get();
  auto& cat = session_info_ptr->get_catalog();
  auto& sys_cat = static_cast<Catalog_Namespace::SysCatalog&>(cat);
  Role* rl = sys_cat.getMetadataForRole(roleName);
  if (rl) {
    for (auto dbObjectIt = rl->getDbObject()->begin(); dbObjectIt != rl->getDbObject()->end(); ++dbObjectIt) {
      TDBObject tdbObject;
      tdbObject.objectName = dbObjectIt->second->getName();
      switch (dbObjectIt->second->getType()) {
        case (DatabaseDBObjectType): {
          tdbObject.objectType = TDBObjectType::DatabaseDBObjectType;
          break;
        }
        case (TableDBObjectType): {
          tdbObject.objectType = TDBObjectType::TableDBObjectType;
          break;
        }
        default: { CHECK(false); }
      }
      auto privs = dbObjectIt->second->getPrivileges();
      tdbObject.privs = {privs.select, privs.insert, privs.create, privs.truncate};
      TDBObjectsForRole.push_back(tdbObject);
    }
  } else {
    std::cout << "Role " << roleName << " does not exist." << std::endl;
  }
}

void MapDHandler::get_db_object_privs(std::vector<TDBObject>& TDBObjects,
                                      const TSessionId& session,
                                      const std::string& objectName) {
  throw std::runtime_error("MapDHandler::get_db_object_privs(..) api should not be used.");
}

void MapDHandler::get_all_roles_for_user(std::vector<std::string>& roles,
                                         const TSessionId& session,
                                         const std::string& userName) {
  auto session_it = get_session_it(session);
  auto session_info_ptr = session_it->second.get();
  auto& cat = session_info_ptr->get_catalog();
  auto& sys_cat = static_cast<Catalog_Namespace::SysCatalog&>(cat);
  Catalog_Namespace::UserMetadata user_meta;
  if (sys_cat.getMetadataForUser(userName, user_meta)) {
    bool get_roles = false;
    if (session_info_ptr->get_currentUser().isSuper) {
      get_roles = true;
    } else {
      if (session_info_ptr->get_currentUser().userId == user_meta.userId) {
        get_roles = true;
      } else {
        TMapDException ex;
        ex.error_msg = "Only superuser is authorized to request list of roles granted to another user.";
        LOG(ERROR) << ex.error_msg;
        throw ex;
      }
    }
    if (get_roles) {
      roles = sys_cat.getAllRolesForUser(user_meta.userId);
    }
  } else {
    TMapDException ex;
    ex.error_msg = "User " + userName + " does not exist.";
    LOG(ERROR) << ex.error_msg;
    throw ex;
  }
}

void MapDHandler::get_db_object_privileges_for_user(std::vector<TAccessPrivileges>& TDBObjectPrivsForUser,
                                                    const TSessionId& session,
                                                    const std::string& userName,
                                                    const int16_t objectType,
                                                    const std::string& objectName) {
  TAccessPrivileges tprivObject;
  TDBObjectPrivsForUser.push_back(tprivObject);
}

void MapDHandler::get_db_objects_for_user(std::vector<TDBObject>& TDBObjectsForUser,
                                          const TSessionId& session,
                                          const std::string& userName) {
  TDBObject tdbObject;
  TDBObjectsForUser.push_back(tdbObject);
}

std::string dump_table_col_names(const std::map<std::string, std::vector<std::string>>& table_col_names) {
  std::ostringstream oss;
  for (const auto table_col : table_col_names) {
    oss << ":" << table_col.first;
    for (const auto col : table_col.second) {
      oss << "," << col;
    }
  }
  return oss.str();
}

void MapDHandler::get_result_row_for_pixel(TPixelTableRowResult& _return,
                                           const TSessionId& session,
                                           const int64_t widget_id,
                                           const TPixel& pixel,
                                           const std::map<std::string, std::vector<std::string>>& table_col_names,
                                           const bool column_format,
                                           const int32_t pixel_radius,
                                           const std::string& nonce) {
  if (!render_handler_) {
    THROW_MAPD_EXCEPTION("Backend rendering is disabled.");
  }

  const auto session_info = MapDHandler::get_session(session);
  LOG(INFO) << "get_result_row_for_pixel :" << session << ":widget_id:" << widget_id << ":pixel.x:" << pixel.x
            << ":pixel.y:" << pixel.y << ":column_format:" << column_format << ":pixel_radius:" << pixel_radius
            << ":table_col_names" << dump_table_col_names(table_col_names) << ":nonce:" << nonce;

  auto time_ms = measure<>::execution([&]() {
    try {
      render_handler_->get_result_row_for_pixel(
          _return, session_info, widget_id, pixel, table_col_names, column_format, pixel_radius, nonce);
    } catch (std::exception& e) {
      THROW_MAPD_EXCEPTION(std::string("Exception: ") + e.what());
    }
  });

  LOG(INFO) << "get_result_row_for_pixel-COMPLETED nonce: " << nonce << ", Execute Time: " << time_ms << " (ms)";
}

TColumnType MapDHandler::populateThriftColumnType(const Catalog_Namespace::Catalog* cat, const ColumnDescriptor* cd) {
  TColumnType col_type;
  col_type.col_name = cd->columnName;
  col_type.src_name = cd->sourceName;
  col_type.col_type.type = type_to_thrift(cd->columnType);
  col_type.col_type.encoding = encoding_to_thrift(cd->columnType);
  col_type.col_type.nullable = !cd->columnType.get_notnull();
  col_type.col_type.is_array = cd->columnType.get_type() == kARRAY;
  col_type.col_type.precision = cd->columnType.get_precision();
  col_type.col_type.scale = cd->columnType.get_scale();
  col_type.is_system = cd->isSystemCol;
  if (cd->columnType.get_compression() == EncodingType::kENCODING_DICT && cat != nullptr) {
    // have to get the actual size of the encoding from the dictionary definition
    const int dict_id = cd->columnType.get_comp_param();
    if (!cat->getMetadataForDict(dict_id, false)) {
      col_type.col_type.comp_param = 0;
      return col_type;
    }
    auto dd = cat->getMetadataForDict(dict_id, false);
    if (!dd) {
      THROW_MAPD_EXCEPTION("Dictionary doesn't exist");
    }
    col_type.col_type.comp_param = dd->dictNBits;
  } else {
    col_type.col_type.comp_param = cd->columnType.get_comp_param();
  }
  return col_type;
}

// DEPRECATED(2017-04-17) - use get_table_details()
void MapDHandler::get_table_descriptor(TTableDescriptor& _return,
                                       const TSessionId& session,
                                       const std::string& table_name) {
  LOG(ERROR) << "get_table_descriptor is deprecated, please fix application";
}

void MapDHandler::get_internal_table_details(TTableDetails& _return,
                                             const TSessionId& session,
                                             const std::string& table_name) {
  get_table_details_impl(_return, session, table_name, true);
}

void MapDHandler::get_table_details(TTableDetails& _return, const TSessionId& session, const std::string& table_name) {
  get_table_details_impl(_return, session, table_name, false);
}

void MapDHandler::get_table_details_impl(TTableDetails& _return,
                                         const TSessionId& session,
                                         const std::string& table_name,
                                         const bool get_system) {
  const auto session_info = get_session(session);
  auto& cat = session_info.get_catalog();
  auto td =
      cat.getMetadataForTable(table_name, false);  // don't populate fragmenter on this call since we only want metadata
  if (!td) {
    THROW_MAPD_EXCEPTION("Table " + table_name + " doesn't exist");
  }
  if (td->isView) {
    try {
      const auto query_ra = parse_to_ra(td->viewSQL, session_info);
      TQueryResult result;
      execute_rel_alg(result, query_ra, true, session_info, ExecutorDeviceType::CPU, -1, -1, false, true);
      _return.row_desc = fixup_row_descriptor(result.row_set.row_desc, cat);
    } catch (std::exception& e) {
      THROW_MAPD_EXCEPTION(std::string("Exception: ") + e.what());
    }
  } else {
    if (!cat.isAccessPrivCheckEnabled() || (cat.isAccessPrivCheckEnabled() && hasTableAccessPrivileges(td, session))) {
      const auto col_descriptors = cat.getAllColumnMetadataForTable(td->tableId, get_system, true);
      for (const auto cd : col_descriptors) {
        _return.row_desc.push_back(populateThriftColumnType(&cat, cd));
      }
    } else {
      THROW_MAPD_EXCEPTION("User has no access privileges to table " + table_name);
    }
  }
  _return.fragment_size = td->maxFragRows;
  _return.page_size = td->fragPageSize;
  _return.max_rows = td->maxRows;
  _return.view_sql = td->viewSQL;
  _return.shard_count = td->nShards;
  _return.key_metainfo = td->keyMetainfo;
  _return.is_temporary = td->persistenceLevel == Data_Namespace::MemoryLevel::CPU_LEVEL;
}

// DEPRECATED(2017-04-17) - use get_table_details()
void MapDHandler::get_row_descriptor(TRowDescriptor& _return,
                                     const TSessionId& session,
                                     const std::string& table_name) {
  LOG(ERROR) << "get_row_descriptor is deprecated, please fix application";
}

void MapDHandler::get_frontend_view(TFrontendView& _return, const TSessionId& session, const std::string& view_name) {
  const auto session_info = get_session(session);
  auto& cat = session_info.get_catalog();
  auto vd = cat.getMetadataForFrontendView(std::to_string(session_info.get_currentUser().userId), view_name);
  if (!vd) {
    THROW_MAPD_EXCEPTION("View " + view_name + " doesn't exist");
  }
  _return.view_name = view_name;
  _return.view_state = vd->viewState;
  _return.image_hash = vd->imageHash;
  _return.update_time = vd->updateTime;
  _return.view_metadata = vd->viewMetadata;
}

void MapDHandler::get_link_view(TFrontendView& _return, const TSessionId& session, const std::string& link) {
  const auto session_info = get_session(session);
  auto& cat = session_info.get_catalog();
  auto ld = cat.getMetadataForLink(std::to_string(cat.get_currentDB().dbId) + link);
  if (!ld) {
    THROW_MAPD_EXCEPTION("Link " + link + " is not valid.");
  }
  _return.view_state = ld->viewState;
  _return.view_name = ld->link;
  _return.update_time = ld->updateTime;
  _return.view_metadata = ld->viewMetadata;
}

bool MapDHandler::isUserAuthorized(const Catalog_Namespace::SessionInfo& session_info, const std::string command_name) {
  bool is_user_authorized = true;
  if (session_info.get_catalog().isAccessPrivCheckEnabled() && !session_info.get_currentUser().isSuper) {
    is_user_authorized = false;
    TMapDException ex;
    ex.error_msg = "Only superuser is authorized to run command " + command_name;
    LOG(ERROR) << ex.error_msg;
    throw ex;
  }
  return is_user_authorized;
}

bool MapDHandler::hasTableAccessPrivileges(const TableDescriptor* td, const TSessionId& session) {
  bool hasAccessPrivs = false;
  const auto session_info = get_session(session);
  auto& cat = session_info.get_catalog();
  auto user_metadata = session_info.get_currentUser();
  auto& sys_cat = *session_info.getSysCatalog();
  DBObject dbObject(td->tableName, TableDBObjectType);
  sys_cat.populateDBObjectKey(dbObject, cat);
  std::vector<DBObject> privObjects;
  dbObject.setPrivileges(AccessPrivileges::SELECT);
  privObjects.push_back(dbObject);
  for (size_t i = 0; i < 4; i++) {
    if (sys_cat.checkPrivileges(user_metadata, privObjects)) {
      hasAccessPrivs = true;
      break;
    }
    switch (i) {
      case (0):
        dbObject.setPrivileges(AccessPrivileges::INSERT);
        break;
      case (1):
        dbObject.setPrivileges(AccessPrivileges::CREATE);
        break;
      case (2):
        dbObject.setPrivileges(AccessPrivileges::TRUNCATE);
        break;
      case (3):
        break;
      default:
        CHECK(false);
    }
    privObjects.pop_back();
    privObjects.push_back(dbObject);
  }
  return hasAccessPrivs;
}

void MapDHandler::get_tables_impl(std::vector<std::string>& table_names,
                                  const TSessionId& session,
                                  const GetTablesType get_tables_type) {
  const auto session_info = get_session(session);
  auto& cat = session_info.get_catalog();
  const auto tables = cat.getAllTableMetadata();
  for (const auto td : tables) {
    if (td->shard >= 0) {
      // skip shards, they're not standalone tables
      continue;
    }
    switch (get_tables_type) {
      case GET_PHYSICAL_TABLES: {
        if (td->isView) {
          continue;
        }
        break;
      }
      case GET_VIEWS: {
        if (!td->isView) {
          continue;
        }
      }
      default: { break; }
    }
    if (cat.isAccessPrivCheckEnabled() && !hasTableAccessPrivileges(td, session)) {
      // skip table, as there are no privileges to access it
      continue;
    }
    table_names.push_back(td->tableName);
  }
}

void MapDHandler::get_tables(std::vector<std::string>& table_names, const TSessionId& session) {
  get_tables_impl(table_names, session, GET_PHYSICAL_TABLES_AND_VIEWS);
}

void MapDHandler::get_physical_tables(std::vector<std::string>& table_names, const TSessionId& session) {
  get_tables_impl(table_names, session, GET_PHYSICAL_TABLES);
}

void MapDHandler::get_views(std::vector<std::string>& table_names, const TSessionId& session) {
  get_tables_impl(table_names, session, GET_VIEWS);
}

void MapDHandler::get_users(std::vector<std::string>& user_names, const TSessionId& session) {
  const auto session_info = get_session(session);
  if (!isUserAuthorized(session_info, std::string("get_users"))) {
    return;
  }
  std::list<Catalog_Namespace::UserMetadata> user_list = sys_cat_->getAllUserMetadata();
  for (auto u : user_list) {
    user_names.push_back(u.userName);
  }
}

void MapDHandler::get_version(std::string& version) {
  version = MAPD_RELEASE;
}

// TODO This need to be corrected for distributed they are only hitting aggr
void MapDHandler::clear_gpu_memory(const TSessionId& session) {
  const auto session_info = get_session(session);
  sys_cat_->get_dataMgr().clearMemory(MemoryLevel::GPU_LEVEL);
}

// TODO This need to be corrected for distributed they are only hitting aggr
void MapDHandler::clear_cpu_memory(const TSessionId& session) {
  const auto session_info = get_session(session);
  sys_cat_->get_dataMgr().clearMemory(MemoryLevel::CPU_LEVEL);
}

TSessionId MapDHandler::getInvalidSessionId() const {
  return INVALID_SESSION_ID;
}

void MapDHandler::get_memory(std::vector<TNodeMemoryInfo>& _return,
                             const TSessionId& session,
                             const std::string& memory_level) {
  const auto session_info = get_session(session);
  std::vector<Data_Namespace::MemoryInfo> internal_memory;
  Data_Namespace::MemoryLevel mem_level;
  if (!memory_level.compare("gpu")) {
    mem_level = Data_Namespace::MemoryLevel::GPU_LEVEL;
    internal_memory = sys_cat_->get_dataMgr().getMemoryInfo(MemoryLevel::GPU_LEVEL);
  } else {
    mem_level = Data_Namespace::MemoryLevel::CPU_LEVEL;
    internal_memory = sys_cat_->get_dataMgr().getMemoryInfo(MemoryLevel::CPU_LEVEL);
  }

  for (auto memInfo : internal_memory) {
    TNodeMemoryInfo nodeInfo;
    if (leaf_aggregator_.leafCount() > 0) {
      nodeInfo.host_name = "aggregator";
    }
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
      md.is_free = gpu.isFree == Buffer_Namespace::MemStatus::FREE;
      nodeInfo.node_memory_data.push_back(md);
    }
    _return.push_back(nodeInfo);
  }
  if (leaf_aggregator_.leafCount() > 0) {
    std::vector<TNodeMemoryInfo> leafSummary = leaf_aggregator_.getLeafMemoryInfo(session, mem_level);
    _return.insert(_return.begin(), leafSummary.begin(), leafSummary.end());
  }
}

void MapDHandler::get_databases(std::vector<TDBInfo>& dbinfos, const TSessionId& session) {
  const auto session_info = get_session(session);
  if (!isUserAuthorized(session_info, std::string("get_databases"))) {
    return;
  }
  std::list<Catalog_Namespace::DBMetadata> db_list = sys_cat_->getAllDBMetadata();
  std::list<Catalog_Namespace::UserMetadata> user_list = sys_cat_->getAllUserMetadata();
  for (auto d : db_list) {
    TDBInfo dbinfo;
    dbinfo.db_name = d.dbName;
    for (auto u : user_list) {
      if (d.dbOwner == u.userId) {
        dbinfo.db_owner = u.userName;
        break;
      }
    }
    dbinfos.push_back(dbinfo);
  }
}

void MapDHandler::get_frontend_views(std::vector<TFrontendView>& view_names, const TSessionId& session) {
  const auto session_info = get_session(session);
  auto& cat = session_info.get_catalog();
  const auto views = cat.getAllFrontendViewMetadata();
  for (const auto vd : views) {
    if (vd->userId == session_info.get_currentUser().userId) {
      TFrontendView fv;
      fv.view_name = vd->viewName;
      fv.image_hash = vd->imageHash;
      fv.update_time = vd->updateTime;
      fv.view_metadata = vd->viewMetadata;
      view_names.push_back(fv);
    }
  }
}

void MapDHandler::set_execution_mode(const TSessionId& session, const TExecuteMode::type mode) {
  mapd_lock_guard<mapd_shared_mutex> write_lock(sessions_mutex_);
  auto session_it = get_session_it(session);
  if (leaf_aggregator_.leafCount() > 0) {
    leaf_aggregator_.set_execution_mode(session, mode);
    try {
      MapDHandler::set_execution_mode_nolock(session_it->second.get(), mode);
    } catch (const TMapDException& e) {
      LOG(INFO) << "Aggregator failed to set execution mode: " << e.error_msg;
    }
    return;
  }
  MapDHandler::set_execution_mode_nolock(session_it->second.get(), mode);
}

namespace {

void check_table_not_sharded(const Catalog_Namespace::Catalog& cat, const std::string& table_name) {
  const auto td = cat.getMetadataForTable(table_name);
  if (td && td->nShards) {
    throw std::runtime_error("Cannot import a sharded table directly to a leaf");
  }
}

}  // namespace

void MapDHandler::load_table_binary(const TSessionId& session,
                                    const std::string& table_name,
                                    const std::vector<TRow>& rows) {
  check_read_only("load_table_binary");
  const auto session_info = get_session(session);
  auto& cat = session_info.get_catalog();
  if (g_cluster && !leaf_aggregator_.leafCount()) {
    // Sharded table rows need to be routed to the leaf by an aggregator.
    check_table_not_sharded(cat, table_name);
  }
  const TableDescriptor* td = cat.getMetadataForTable(table_name);
  if (td == nullptr) {
    THROW_MAPD_EXCEPTION("Table " + table_name + " does not exist.");
  }
  check_table_load_privileges(session, table_name);
  std::unique_ptr<Importer_NS::Loader> loader;
  if (leaf_aggregator_.leafCount() > 0) {
    loader.reset(new DistributedLoader(session_info, td, &leaf_aggregator_));
  } else {
    loader.reset(new Importer_NS::Loader(cat, td));
  }
  // TODO(andrew): nColumns should be number of non-virtual/non-system columns.
  //               Subtracting 1 (rowid) until TableDescriptor is updated.
  if (rows.front().cols.size() != static_cast<size_t>(td->nColumns) - 1) {
    THROW_MAPD_EXCEPTION("Wrong number of columns to load into Table " + table_name);
  }
  auto col_descs = loader->get_column_descs();
  std::vector<std::unique_ptr<Importer_NS::TypedImportBuffer>> import_buffers;
  for (auto cd : col_descs) {
    import_buffers.push_back(std::unique_ptr<Importer_NS::TypedImportBuffer>(
        new Importer_NS::TypedImportBuffer(cd, loader->get_string_dict(cd))));
  }
  for (auto const& row : rows) {
    size_t col_idx = 0;
    try {
      for (auto cd : col_descs) {
        import_buffers[col_idx]->add_value(cd, row.cols[col_idx], row.cols[col_idx].is_null);
        col_idx++;
      }
    } catch (const std::exception& e) {
      for (size_t col_idx_to_pop = 0; col_idx_to_pop < col_idx; ++col_idx_to_pop) {
        import_buffers[col_idx_to_pop]->pop_value();
      }
      LOG(ERROR) << "Input exception thrown: " << e.what() << ". Row discarded, issue at column : " << (col_idx + 1)
                 << " data :" << row;
    }
  }
  loader->load(import_buffers, rows.size());
}

void MapDHandler::prepare_columnar_loader(
    const TSessionId& session,
    const std::string& table_name,
    size_t num_cols,
    std::unique_ptr<Importer_NS::Loader>* loader,
    std::vector<std::unique_ptr<Importer_NS::TypedImportBuffer>>* import_buffers) {
  const auto session_info = get_session(session);
  auto& cat = session_info.get_catalog();
  if (g_cluster && !leaf_aggregator_.leafCount()) {
    // Sharded table rows need to be routed to the leaf by an aggregator.
    check_table_not_sharded(cat, table_name);
  }
  const TableDescriptor* td = cat.getMetadataForTable(table_name);
  if (td == nullptr) {
    THROW_MAPD_EXCEPTION("Table " + table_name + " does not exist.");
  }
  check_table_load_privileges(session, table_name);
  if (leaf_aggregator_.leafCount() > 0) {
    loader->reset(new DistributedLoader(session_info, td, &leaf_aggregator_));
  } else {
    loader->reset(new Importer_NS::Loader(cat, td));
  }
  // TODO(andrew): nColumns should be number of non-virtual/non-system columns.
  //               Subtracting 1 (rowid) until TableDescriptor is updated.
  if (num_cols != static_cast<size_t>(td->nColumns) - 1 || num_cols < 1) {
    THROW_MAPD_EXCEPTION("Wrong number of columns to load into Table " + table_name);
  }
  auto col_descs = (*loader)->get_column_descs();
  for (auto cd : col_descs) {
    import_buffers->push_back(std::unique_ptr<Importer_NS::TypedImportBuffer>(
        new Importer_NS::TypedImportBuffer(cd, (*loader)->get_string_dict(cd))));
  }
}

void MapDHandler::load_table_binary_columnar(const TSessionId& session,
                                             const std::string& table_name,
                                             const std::vector<TColumn>& cols) {
  check_read_only("load_table_binary_columnar");

  std::unique_ptr<Importer_NS::Loader> loader;
  std::vector<std::unique_ptr<Importer_NS::TypedImportBuffer>> import_buffers;
  prepare_columnar_loader(session, table_name, cols.size(), &loader, &import_buffers);

  size_t numRows = 0;
  size_t col_idx = 0;
  try {
    for (auto cd : loader->get_column_descs()) {
      size_t colRows = import_buffers[col_idx]->add_values(cd, cols[col_idx]);
      if (col_idx == 0) {
        numRows = colRows;
      } else {
        if (colRows != numRows) {
          std::ostringstream oss;
          oss << "load_table_binary_columnar: Inconsistent number of rows in request,  was " << numRows << " column "
              << col_idx << " has " << colRows;
          THROW_MAPD_EXCEPTION(oss.str());
        }
      }
      col_idx++;
    }
  } catch (const std::exception& e) {
    std::ostringstream oss;
    oss << "load_table_binary_columnar: Input exception thrown: " << e.what() << ". Issue at column : " << (col_idx + 1)
        << ". Import aborted";
    THROW_MAPD_EXCEPTION(oss.str());
  }
  loader->load(import_buffers, numRows);
}

using RecordBatchVector = std::vector<std::shared_ptr<arrow::RecordBatch>>;

#define ARROW_THRIFT_THROW_NOT_OK(s) \
  do {                               \
    ::arrow::Status _s = (s);        \
    if (UNLIKELY(!_s.ok())) {        \
      TMapDException ex;             \
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
    auto stream_buffer = std::make_shared<arrow::Buffer>(reinterpret_cast<const uint8_t*>(stream.c_str()),
                                                         static_cast<int64_t>(stream.size()));

    arrow::io::BufferReader buf_reader(stream_buffer);
    std::shared_ptr<arrow::RecordBatchReader> batch_reader;
    ARROW_THRIFT_THROW_NOT_OK(arrow::ipc::RecordBatchStreamReader::Open(&buf_reader, &batch_reader));

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

void MapDHandler::load_table_binary_arrow(const TSessionId& session,
                                          const std::string& table_name,
                                          const std::string& arrow_stream) {
  check_read_only("load_table_binary_arrow");

  RecordBatchVector batches = loadArrowStream(arrow_stream);

  // Assuming have one batch for now
  if (batches.size() != 1) {
    THROW_MAPD_EXCEPTION("Expected a single Arrow record batch. Import aborted");
  }

  std::shared_ptr<arrow::RecordBatch> batch = batches[0];

  std::unique_ptr<Importer_NS::Loader> loader;
  std::vector<std::unique_ptr<Importer_NS::TypedImportBuffer>> import_buffers;
  prepare_columnar_loader(session, table_name, static_cast<size_t>(batch->num_columns()), &loader, &import_buffers);

  size_t numRows = 0;
  size_t col_idx = 0;
  try {
    for (auto cd : loader->get_column_descs()) {
      numRows = import_buffers[col_idx]->add_arrow_values(cd, *batch->column(col_idx));
      col_idx++;
    }
  } catch (const std::exception& e) {
    LOG(ERROR) << "Input exception thrown: " << e.what() << ". Issue at column : " << (col_idx + 1)
               << ". Import aborted";
    // TODO(tmostak): Go row-wise on binary columnar import to be consistent with our other import paths
    THROW_MAPD_EXCEPTION(std::string("Exception: ") + e.what());
  }
  loader->load(import_buffers, numRows);
}

void MapDHandler::load_table(const TSessionId& session,
                             const std::string& table_name,
                             const std::vector<TStringRow>& rows) {
  check_read_only("load_table");
  const auto session_info = get_session(session);
  auto& cat = session_info.get_catalog();
  if (g_cluster && !leaf_aggregator_.leafCount()) {
    // Sharded table rows need to be routed to the leaf by an aggregator.
    check_table_not_sharded(cat, table_name);
  }
  const TableDescriptor* td = cat.getMetadataForTable(table_name);
  if (td == nullptr) {
    THROW_MAPD_EXCEPTION("Table " + table_name + " does not exist.");
  }
  check_table_load_privileges(session, table_name);
  std::unique_ptr<Importer_NS::Loader> loader;
  if (leaf_aggregator_.leafCount() > 0) {
    loader.reset(new DistributedLoader(session_info, td, &leaf_aggregator_));
  } else {
    loader.reset(new Importer_NS::Loader(cat, td));
  }
  Importer_NS::CopyParams copy_params;
  // TODO(andrew): nColumns should be number of non-virtual/non-system columns.
  //               Subtracting 1 (rowid) until TableDescriptor is updated.
  if (rows.front().cols.size() != static_cast<size_t>(td->nColumns) - 1) {
    THROW_MAPD_EXCEPTION("Wrong number of columns to load into Table " + table_name + " (" +
                         std::to_string(rows.front().cols.size()) + " vs " + std::to_string(td->nColumns - 1) + ")");
  }
  auto col_descs = loader->get_column_descs();
  std::vector<std::unique_ptr<Importer_NS::TypedImportBuffer>> import_buffers;
  for (auto cd : col_descs) {
    import_buffers.push_back(std::unique_ptr<Importer_NS::TypedImportBuffer>(
        new Importer_NS::TypedImportBuffer(cd, loader->get_string_dict(cd))));
  }
  size_t rows_completed = 0;
  size_t col_idx = 0;
  for (auto const& row : rows) {
    try {
      col_idx = 0;
      for (auto cd : col_descs) {
        import_buffers[col_idx]->add_value(cd, row.cols[col_idx].str_val, row.cols[col_idx].is_null, copy_params);
        col_idx++;
      }
      rows_completed++;
    } catch (const std::exception& e) {
      for (size_t col_idx_to_pop = 0; col_idx_to_pop < col_idx; ++col_idx_to_pop) {
        import_buffers[col_idx_to_pop]->pop_value();
      }
      LOG(ERROR) << "Input exception thrown: " << e.what() << ". Row discarded, issue at column : " << (col_idx + 1)
                 << " data :" << row;
    }
  }
  loader->load(import_buffers, rows_completed);
}

char MapDHandler::unescape_char(std::string str) {
  char out = str[0];
  if (str.size() == 2 && str[0] == '\\') {
    if (str[1] == 't')
      out = '\t';
    else if (str[1] == 'n')
      out = '\n';
    else if (str[1] == '0')
      out = '\0';
    else if (str[1] == '\'')
      out = '\'';
    else if (str[1] == '\\')
      out = '\\';
  }
  return out;
}

Importer_NS::CopyParams MapDHandler::thrift_to_copyparams(const TCopyParams& cp) {
  Importer_NS::CopyParams copy_params;
  copy_params.has_header = cp.has_header;
  copy_params.quoted = cp.quoted;
  if (cp.delimiter.length() > 0)
    copy_params.delimiter = unescape_char(cp.delimiter);
  else
    copy_params.delimiter = '\0';
  if (cp.null_str.length() > 0)
    copy_params.null_str = cp.null_str;
  if (cp.quote.length() > 0)
    copy_params.quote = unescape_char(cp.quote);
  if (cp.escape.length() > 0)
    copy_params.escape = unescape_char(cp.escape);
  if (cp.line_delim.length() > 0)
    copy_params.line_delim = unescape_char(cp.line_delim);
  if (cp.array_delim.length() > 0)
    copy_params.array_delim = unescape_char(cp.array_delim);
  if (cp.array_begin.length() > 0)
    copy_params.array_begin = unescape_char(cp.array_begin);
  if (cp.array_end.length() > 0)
    copy_params.array_end = unescape_char(cp.array_end);
  if (cp.threads != 0)
    copy_params.threads = cp.threads;
  switch (cp.table_type) {
    case TTableType::POLYGON:
      copy_params.table_type = Importer_NS::TableType::POLYGON;
      break;
    default:
      copy_params.table_type = Importer_NS::TableType::DELIMITED;
      break;
  }
  return copy_params;
}

TCopyParams MapDHandler::copyparams_to_thrift(const Importer_NS::CopyParams& cp) {
  TCopyParams copy_params;
  copy_params.delimiter = cp.delimiter;
  copy_params.null_str = cp.null_str;
  copy_params.has_header = cp.has_header;
  copy_params.quoted = cp.quoted;
  copy_params.quote = cp.quote;
  copy_params.escape = cp.escape;
  copy_params.line_delim = cp.line_delim;
  copy_params.array_delim = cp.array_delim;
  copy_params.array_begin = cp.array_begin;
  copy_params.array_end = cp.array_end;
  copy_params.threads = cp.threads;
  switch (cp.table_type) {
    case Importer_NS::TableType::POLYGON:
      copy_params.table_type = TTableType::POLYGON;
      break;
    default:
      copy_params.table_type = TTableType::DELIMITED;
      break;
  }
  return copy_params;
}

void MapDHandler::detect_column_types(TDetectResult& _return,
                                      const TSessionId& session,
                                      const std::string& file_name_in,
                                      const TCopyParams& cp) {
  check_read_only("detect_column_types");
  get_session(session);

  // Assume relative paths are relative to data_path / mapd_import / <session>
  std::string file_name{file_name_in};
  auto file_path = boost::filesystem::path(file_name);
  if (!boost::filesystem::path(file_name).is_absolute()) {
    file_path = import_path_ / session / boost::filesystem::path(file_name).filename();
    file_name = file_path.string();
  }

  if (!boost::filesystem::exists(file_path)) {
    THROW_MAPD_EXCEPTION("File does not exist: " + file_path.string());
  }

  Importer_NS::CopyParams copy_params = thrift_to_copyparams(cp);

  try {
    if (copy_params.table_type == Importer_NS::TableType::DELIMITED) {
      Importer_NS::Detector detector(file_path, copy_params);
      std::vector<SQLTypes> best_types = detector.best_sqltypes;
      std::vector<EncodingType> best_encodings = detector.best_encodings;
      std::vector<std::string> headers = detector.get_headers();
      copy_params = detector.get_copy_params();

      _return.copy_params = copyparams_to_thrift(copy_params);
      _return.row_set.row_desc.resize(best_types.size());
      TColumnType col;
      for (size_t col_idx = 0; col_idx < best_types.size(); col_idx++) {
        SQLTypes t = best_types[col_idx];
        EncodingType encodingType = best_encodings[col_idx];
        SQLTypeInfo ti(t, false, encodingType);
        col.col_type.type = type_to_thrift(ti);
        col.col_type.encoding = encoding_to_thrift(ti);
        col.col_name = headers[col_idx];
        col.is_reserved_keyword =
            reserved_keywords.find(boost::to_upper_copy<std::string>(col.col_name)) != reserved_keywords.end();
        _return.row_set.row_desc[col_idx] = col;
      }
      size_t num_samples = 100;
      auto sample_data = detector.get_sample_rows(num_samples);

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
    } else if (copy_params.table_type == Importer_NS::TableType::POLYGON) {
      check_geospatial_files(file_path);
      std::list<ColumnDescriptor> cds = Importer_NS::Importer::gdalToColumnDescriptors(file_path.string());
      for (auto cd : cds) {
        cd.columnName = sanitize_name(cd.columnName);
        _return.row_set.row_desc.push_back(populateThriftColumnType(nullptr, &cd));
      }
      std::map<std::string, std::vector<std::string>> sample_data;
      Importer_NS::Importer::readMetadataSampleGDAL(file_path.string(), sample_data, 100);
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
  } catch (const std::exception& e) {
    THROW_MAPD_EXCEPTION("detect_column_types error: " + std::string(e.what()));
  }
}

Planner::RootPlan* MapDHandler::parse_to_plan_legacy(const std::string& query_str,
                                                     const Catalog_Namespace::SessionInfo& session_info,
                                                     const std::string& action /* render or validate */) {
  auto& cat = session_info.get_catalog();
  LOG(INFO) << action << ": " << query_str;
  SQLParser parser;
  std::list<std::unique_ptr<Parser::Stmt>> parse_trees;
  std::string last_parsed;
  int num_parse_errors = 0;
  try {
    num_parse_errors = parser.parse(query_str, parse_trees, last_parsed);
  } catch (std::exception& e) {
    THROW_MAPD_EXCEPTION(std::string("Exception: ") + e.what());
  }
  if (num_parse_errors > 0) {
    THROW_MAPD_EXCEPTION("Syntax error at: " + last_parsed);
  }
  if (parse_trees.size() != 1) {
    THROW_MAPD_EXCEPTION("Can only " + action + " a single query at a time.");
  }
  Parser::Stmt* stmt = parse_trees.front().get();
  Parser::DDLStmt* ddl = dynamic_cast<Parser::DDLStmt*>(stmt);
  if (ddl != nullptr) {
    THROW_MAPD_EXCEPTION("Can only " + action + " SELECT statements.");
  }
  auto dml = static_cast<Parser::DMLStmt*>(stmt);
  Analyzer::Query query;
  dml->analyze(cat, query);
  Planner::Optimizer optimizer(query, cat);
  return optimizer.optimize();
}

void MapDHandler::render_vega(TRenderResult& _return,
                              const TSessionId& session,
                              const int64_t widget_id,
                              const std::string& vega_json,
                              const int compression_level,
                              const std::string& nonce) {
  if (!render_handler_) {
    THROW_MAPD_EXCEPTION("Backend rendering is disabled.");
  }

  const auto session_info = MapDHandler::get_session(session);
  LOG(INFO) << "render_vega :" << session << ":widget_id:" << widget_id << ":compression_level:" << compression_level
            << ":vega_json:" << vega_json << ":nonce:" << nonce;

  _return.total_time_ms = measure<>::execution([&]() {
    try {
      render_handler_->render_vega(_return, session_info, widget_id, vega_json, compression_level, nonce);
    } catch (std::exception& e) {
      THROW_MAPD_EXCEPTION(std::string("Exception: ") + e.what());
    }
  });
  LOG(INFO) << "render_vega-COMPLETED nonce: " << nonce << " Total: " << _return.total_time_ms
            << " (ms), Total Execution: " << _return.execution_time_ms
            << " (ms), Total Render: " << _return.render_time_ms << " (ms)";
}

void MapDHandler::create_frontend_view(const TSessionId& session,
                                       const std::string& view_name,
                                       const std::string& view_state,
                                       const std::string& image_hash,
                                       const std::string& view_metadata) {
  check_read_only("create_frontend_view");
  const auto session_info = get_session(session);
  auto& cat = session_info.get_catalog();
  FrontendViewDescriptor vd;
  vd.viewName = view_name;
  vd.viewState = view_state;
  vd.imageHash = image_hash;
  vd.viewMetadata = view_metadata;
  vd.userId = session_info.get_currentUser().userId;

  try {
    cat.createFrontendView(vd);
  } catch (const std::exception& e) {
    THROW_MAPD_EXCEPTION(std::string("Exception: ") + e.what());
  }
}

void MapDHandler::delete_frontend_view(const TSessionId& session, const std::string& view_name) {
  check_read_only("delete_frontend_view");
  const auto session_info = get_session(session);
  auto& cat = session_info.get_catalog();
  auto vd = cat.getMetadataForFrontendView(std::to_string(session_info.get_currentUser().userId), view_name);
  if (!vd) {
    THROW_MAPD_EXCEPTION("View " + view_name + " doesn't exist");
  }
  try {
    cat.deleteMetadataForFrontendView(std::to_string(session_info.get_currentUser().userId), view_name);
  } catch (const std::exception& e) {
    THROW_MAPD_EXCEPTION(std::string("Exception: ") + e.what());
  }
}

void MapDHandler::create_link(std::string& _return,
                              const TSessionId& session,
                              const std::string& view_state,
                              const std::string& view_metadata) {
  // check_read_only("create_link");
  const auto session_info = get_session(session);
  auto& cat = session_info.get_catalog();

  LinkDescriptor ld;
  ld.userId = session_info.get_currentUser().userId;
  ld.viewState = view_state;
  ld.viewMetadata = view_metadata;

  try {
    _return = cat.createLink(ld, 6);
  } catch (const std::exception& e) {
    THROW_MAPD_EXCEPTION(std::string("Exception: ") + e.what());
  }
}

std::string MapDHandler::sanitize_name(const std::string& name) {
  boost::regex invalid_chars{R"([^0-9a-z_])", boost::regex::extended | boost::regex::icase};

  std::string col_name = boost::regex_replace(name, invalid_chars, "");
  if (reserved_keywords.find(boost::to_upper_copy<std::string>(col_name)) != reserved_keywords.end()) {
    col_name += "_";
  }
  return col_name;
}

TColumnType MapDHandler::create_array_column(const TDatumType::type type, const std::string& name) {
  TColumnType ct;
  ct.col_name = name;
  ct.col_type.type = type;
  ct.col_type.is_array = true;
  return ct;
}

void MapDHandler::check_geospatial_files(const boost::filesystem::path file_path) {
  const std::list<std::string> shp_ext{".shp", ".shx", ".dbf"};
  if (std::find(shp_ext.begin(), shp_ext.end(), boost::algorithm::to_lower_copy(file_path.extension().string())) !=
      shp_ext.end()) {
    for (auto ext : shp_ext) {
      auto aux_file = file_path;
      if (!boost::filesystem::exists(aux_file.replace_extension(boost::algorithm::to_upper_copy(ext))) &&
          !boost::filesystem::exists(aux_file.replace_extension(ext))) {
        throw std::runtime_error("required file for shapefile does not exist: " + aux_file.filename().string());
      }
    }
  }
}

void MapDHandler::create_table(const TSessionId& session,
                               const std::string& table_name,
                               const TRowDescriptor& rd,
                               const TTableType::type table_type) {
  check_read_only("create_table");

  if (table_name != sanitize_name(table_name)) {
    THROW_MAPD_EXCEPTION("Invalid characters in table name: " + table_name);
  }

  auto rds = rd;

  if (table_type == TTableType::POLYGON) {
    rds.push_back(create_array_column(TDatumType::DOUBLE, MAPD_GEO_PREFIX + "coords"));
    rds.push_back(create_array_column(TDatumType::INT, MAPD_GEO_PREFIX + "indices"));
    rds.push_back(create_array_column(TDatumType::INT, MAPD_GEO_PREFIX + "linedrawinfo"));
    rds.push_back(create_array_column(TDatumType::INT, MAPD_GEO_PREFIX + "polydrawinfo"));
  }

  std::string stmt{"CREATE TABLE " + table_name};
  std::vector<std::string> col_stmts;

  for (auto col : rds) {
    if (col.col_name != sanitize_name(col.col_name)) {
      THROW_MAPD_EXCEPTION("Invalid characters in column name: " + col.col_name);
    }
    if (col.col_type.type == TDatumType::INTERVAL_DAY_TIME || col.col_type.type == TDatumType::INTERVAL_YEAR_MONTH) {
      THROW_MAPD_EXCEPTION("Unsupported type: " + thrift_to_name(col.col_type) + " for column: " + col.col_name);
    }
    // if no precision or scale passed in set to default 14,7
    if (col.col_type.precision == 0 && col.col_type.precision == 0) {
      col.col_type.precision = 14;
      col.col_type.scale = 7;
    }

    std::string col_stmt;
    col_stmt.append(col.col_name + " " + thrift_to_name(col.col_type) + " ");

    // As of 2016-06-27 the Immerse v1 frontend does not explicitly set the
    // `nullable` argument, leading this to default to false. Uncomment for v2.
    // if (!col.col_type.nullable) col_stmt.append("NOT NULL ");

    if (thrift_to_encoding(col.col_type.encoding) != kENCODING_NONE) {
      col_stmt.append("ENCODING " + thrift_to_encoding_name(col.col_type) + " ");
    }
    // deal with special case of non DICT encoded strings
    if (thrift_to_encoding(col.col_type.encoding) == kENCODING_NONE && col.col_type.type == TDatumType::STR) {
      col_stmt.append("ENCODING NONE");
    }
    col_stmts.push_back(col_stmt);
  }

  stmt.append(" (" + boost::algorithm::join(col_stmts, ", ") + ");");

  TQueryResult ret;
  sql_execute(ret, session, stmt, true, "", -1, -1);
}

void MapDHandler::import_table(const TSessionId& session,
                               const std::string& table_name,
                               const std::string& file_name,
                               const TCopyParams& cp) {
  check_read_only("import_table");
  LOG(INFO) << "import_table " << table_name << " from " << file_name;
  const auto session_info = get_session(session);
  auto& cat = session_info.get_catalog();

  const TableDescriptor* td = cat.getMetadataForTable(table_name);
  if (td == nullptr) {
    THROW_MAPD_EXCEPTION("Table " + table_name + " does not exist.");
  }
  check_table_load_privileges(session, table_name);

  auto file_path = import_path_ / session / boost::filesystem::path(file_name).filename();
  if (!boost::filesystem::exists(file_path)) {
    THROW_MAPD_EXCEPTION("File does not exist: " + file_path.filename().string());
  }

  Importer_NS::CopyParams copy_params = thrift_to_copyparams(cp);

  // TODO(andrew): add delimiter detection to Importer
  if (copy_params.delimiter == '\0') {
    copy_params.delimiter = ',';
    if (boost::filesystem::extension(file_path) == ".tsv") {
      copy_params.delimiter = '\t';
    }
  }

  try {
    std::unique_ptr<Importer_NS::Importer> importer;
    if (leaf_aggregator_.leafCount() > 0) {
      importer.reset(new Importer_NS::Importer(
          new DistributedLoader(session_info, td, &leaf_aggregator_), file_path.string(), copy_params));
    } else {
      importer.reset(new Importer_NS::Importer(cat, td, file_path.string(), copy_params));
    }
    auto ms = measure<>::execution([&]() { importer->import(); });
    std::cout << "Total Import Time: " << (double)ms / 1000.0 << " Seconds." << std::endl;
  } catch (const std::exception& e) {
    THROW_MAPD_EXCEPTION("Exception: " + std::string(e.what()));
  }
}

void MapDHandler::import_geo_table(const TSessionId& session,
                                   const std::string& table_name,
                                   const std::string& file_name_in,
                                   const TCopyParams& cp,
                                   const TRowDescriptor& row_desc) {
  check_read_only("import_table");
  const auto session_info = get_session(session);
  auto& cat = session_info.get_catalog();

  // Assume relative paths are relative to data_path / mapd_import / <session>
  std::string file_name{file_name_in};
  if (!boost::filesystem::path(file_name).is_absolute()) {
    auto file_path = import_path_ / session / boost::filesystem::path(file_name).filename();
    file_name = file_path.string();
  }

  LOG(INFO) << "import_geo_table " << table_name << " from " << file_name;

  auto file_path = boost::filesystem::path(file_name);
  if (!boost::filesystem::exists(file_path)) {
    THROW_MAPD_EXCEPTION("File does not exist: " + file_path.filename().string());
  }
  try {
    check_geospatial_files(file_path);
  } catch (const std::exception& e) {
    THROW_MAPD_EXCEPTION("import_geo_table error: " + std::string(e.what()));
  }

  TRowDescriptor rd;
  if (cat.getMetadataForTable(table_name) == nullptr) {
    TDetectResult cds;
    TCopyParams cp;
    cp.table_type = TTableType::POLYGON;
    detect_column_types(cds, session, file_name_in, cp);
    create_table(session, table_name, cds.row_set.row_desc, TTableType::POLYGON);
    rd = cds.row_set.row_desc;
  } else if (row_desc.size() > 0) {
    rd = row_desc;
  } else {
    THROW_MAPD_EXCEPTION("Could not append file " + file_path.filename().string() + " to " + table_name +
                         ": not currently supported.");
  }

  std::map<std::string, std::string> colname_to_src;
  for (auto r : rd) {
    colname_to_src[r.col_name] = r.src_name.length() > 0 ? r.src_name : sanitize_name(r.src_name);
  }

  const TableDescriptor* td = cat.getMetadataForTable(table_name);
  if (td == nullptr) {
    THROW_MAPD_EXCEPTION("Table " + table_name + " does not exist.");
  }
  check_table_load_privileges(session, table_name);

  Importer_NS::CopyParams copy_params = thrift_to_copyparams(cp);

  try {
    std::unique_ptr<Importer_NS::Importer> importer;
    if (leaf_aggregator_.leafCount() > 0) {
      importer.reset(new Importer_NS::Importer(
          new DistributedLoader(session_info, td, &leaf_aggregator_), file_path.string(), copy_params));
    } else {
      importer.reset(new Importer_NS::Importer(cat, td, file_path.string(), copy_params));
    }
    auto ms = measure<>::execution([&]() { importer->importGDAL(colname_to_src); });
    std::cout << "Total Import Time: " << (double)ms / 1000.0 << " Seconds." << std::endl;
  } catch (const std::exception& e) {
    THROW_MAPD_EXCEPTION(std::string("import_geo_table failed: ") + e.what());
  }
}

void MapDHandler::import_table_status(TImportStatus& _return, const TSessionId& session, const std::string& import_id) {
  LOG(INFO) << "import_table_status " << import_id;
  auto is = Importer_NS::Importer::get_import_status(import_id);
  _return.elapsed = is.elapsed.count();
  _return.rows_completed = is.rows_completed;
  _return.rows_estimated = is.rows_estimated;
  _return.rows_rejected = is.rows_rejected;
}

void MapDHandler::start_heap_profile(const TSessionId& session) {
  const auto session_info = get_session(session);
#ifdef HAVE_PROFILER
  if (IsHeapProfilerRunning()) {
    THROW_MAPD_EXCEPTION("Profiler already started");
  }
  HeapProfilerStart("mapd");
#else
  THROW_MAPD_EXCEPTION("Profiler not enabled");
#endif  // HAVE_PROFILER
}

void MapDHandler::stop_heap_profile(const TSessionId& session) {
  const auto session_info = get_session(session);
#ifdef HAVE_PROFILER
  if (!IsHeapProfilerRunning()) {
    THROW_MAPD_EXCEPTION("Profiler not running");
  }
  HeapProfilerStop();
#else
  THROW_MAPD_EXCEPTION("Profiler not enabled");
#endif  // HAVE_PROFILER
}

void MapDHandler::get_heap_profile(std::string& profile, const TSessionId& session) {
  const auto session_info = get_session(session);
#ifdef HAVE_PROFILER
  if (!IsHeapProfilerRunning()) {
    THROW_MAPD_EXCEPTION("Profiler not running");
  }
  auto profile_buff = GetHeapProfile();
  profile = profile_buff;
  free(profile_buff);
#else
  THROW_MAPD_EXCEPTION("Profiler not enabled");
#endif  // HAVE_PROFILER
}

SessionMap::iterator MapDHandler::get_session_it(const TSessionId& session) {
  auto session_it = sessions_.find(session);
  if (session_it == sessions_.end()) {
    THROW_MAPD_EXCEPTION("Session not valid.");
  }
  session_it->second->update_time();
  return session_it;
}

Catalog_Namespace::SessionInfo MapDHandler::get_session(const TSessionId& session) {
  mapd_shared_lock<mapd_shared_mutex> read_lock(sessions_mutex_);
  return *get_session_it(session)->second;
}

void MapDHandler::check_table_load_privileges(const TSessionId& session, const std::string& table_name) {
  const auto session_info = MapDHandler::get_session(session);
  auto user_metadata = session_info.get_currentUser();
  auto& cat = session_info.get_catalog();
  auto& sys_cat = static_cast<Catalog_Namespace::SysCatalog&>(cat);
  DBObject dbObject(table_name, TableDBObjectType);
  sys_cat.populateDBObjectKey(dbObject, cat);
  dbObject.setPrivileges(AccessPrivileges::INSERT);
  std::vector<DBObject> privObjects;
  privObjects.push_back(dbObject);
  if (cat.isAccessPrivCheckEnabled() && !sys_cat.checkPrivileges(user_metadata, privObjects)) {
    THROW_MAPD_EXCEPTION("Violation of access privileges: user " + user_metadata.userName +
                         " has no insert privileges for table " + table_name + ".");
  }
}

void MapDHandler::set_execution_mode_nolock(Catalog_Namespace::SessionInfo* session_ptr,
                                            const TExecuteMode::type mode) {
  const std::string& user_name = session_ptr->get_currentUser().userName;
  switch (mode) {
    case TExecuteMode::GPU:
      if (cpu_mode_only_) {
        TMapDException e;
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
    case TExecuteMode::HYBRID:
      if (cpu_mode_only_) {
        TMapDException e;
        e.error_msg = "Cannot switch to Hybrid mode in a server started in CPU-only mode.";
        throw e;
      }
      session_ptr->set_executor_device_type(ExecutorDeviceType::Hybrid);
      LOG(INFO) << "User " << user_name << " sets HYBRID mode.";
  }
}

void MapDHandler::execute_rel_alg(TQueryResult& _return,
                                  const std::string& query_ra,
                                  const bool column_format,
                                  const Catalog_Namespace::SessionInfo& session_info,
                                  const ExecutorDeviceType executor_device_type,
                                  const int32_t first_n,
                                  const int32_t at_most_n,
                                  const bool just_explain,
                                  const bool just_validate) const {
  const auto& cat = session_info.get_catalog();
  CompilationOptions co = {executor_device_type, true, ExecutorOptLevel::Default, g_enable_dynamic_watchdog};
  ExecutionOptions eo = {false,
                         allow_multifrag_,
                         just_explain,
                         allow_loop_joins_ || just_validate,
                         g_enable_watchdog,
                         jit_debug_,
                         just_validate,
                         g_enable_dynamic_watchdog,
                         g_dynamic_watchdog_time_limit};
  auto executor = Executor::getExecutor(
      cat.get_currentDB().dbId, jit_debug_ ? "/tmp" : "", jit_debug_ ? "mapdquery" : "", mapd_parameters_, nullptr);
  RelAlgExecutor ra_executor(executor.get(), cat);
  ExecutionResult result{
      std::make_shared<ResultSet>(
          std::vector<TargetInfo>{}, ExecutorDeviceType::CPU, QueryMemoryDescriptor{}, nullptr, nullptr),
      {}};
  _return.execution_time_ms +=
      measure<>::execution([&]() { result = ra_executor.executeRelAlgQuery(query_ra, co, eo, nullptr); });
  // reduce execution time by the time spent during queue waiting
  _return.execution_time_ms -= result.getRows()->getQueueTime();
  if (just_explain) {
    convert_explain(_return, *result.getRows(), column_format);
  } else {
    convert_rows(_return, result.getTargetsMeta(), *result.getRows(), column_format, first_n, at_most_n);
  }
}

void MapDHandler::execute_rel_alg_df(TDataFrame& _return,
                                     const std::string& query_ra,
                                     const Catalog_Namespace::SessionInfo& session_info,
                                     const ExecutorDeviceType device_type,
                                     const size_t device_id,
                                     const int32_t first_n) const {
  const auto& cat = session_info.get_catalog();
  CHECK(device_type == ExecutorDeviceType::CPU || session_info.get_executor_device_type() == ExecutorDeviceType::GPU);
  CompilationOptions co = {device_type, true, ExecutorOptLevel::Default, g_enable_dynamic_watchdog};
  ExecutionOptions eo = {false,
                         allow_multifrag_,
                         false,
                         allow_loop_joins_,
                         g_enable_watchdog,
                         jit_debug_,
                         false,
                         g_enable_dynamic_watchdog,
                         g_dynamic_watchdog_time_limit};
  auto executor = Executor::getExecutor(
      cat.get_currentDB().dbId, jit_debug_ ? "/tmp" : "", jit_debug_ ? "mapdquery" : "", mapd_parameters_, nullptr);
  RelAlgExecutor ra_executor(executor.get(), cat);
  const auto result = ra_executor.executeRelAlgQuery(query_ra, co, eo, nullptr);
  const auto rs = result.getRows();
  const auto copy = rs->getArrowCopy(data_mgr_.get(), device_type, device_id, getTargetNames(result.getTargetsMeta()));
  _return.sm_handle = std::string(copy.sm_handle.begin(), copy.sm_handle.end());
  _return.sm_size = copy.sm_size;
  _return.df_handle = std::string(copy.df_handle.begin(), copy.df_handle.end());
  if (device_type == ExecutorDeviceType::GPU) {
    std::lock_guard<std::mutex> map_lock(handle_to_dev_ptr_mutex_);
    CHECK(!ipc_handle_to_dev_ptr_.count(_return.df_handle));
    ipc_handle_to_dev_ptr_.insert(std::make_pair(_return.df_handle, copy.df_dev_ptr));
  }
  _return.df_size = copy.df_size;
}

void MapDHandler::execute_root_plan(TQueryResult& _return,
                                    const Planner::RootPlan* root_plan,
                                    const bool column_format,
                                    const Catalog_Namespace::SessionInfo& session_info,
                                    const ExecutorDeviceType executor_device_type,
                                    const int32_t first_n) const {
  auto executor = Executor::getExecutor(root_plan->get_catalog().get_currentDB().dbId,
                                        jit_debug_ ? "/tmp" : "",
                                        jit_debug_ ? "mapdquery" : "",
                                        mapd_parameters_,
                                        render_handler_ ? render_handler_->get_render_manager() : nullptr);
  std::shared_ptr<ResultSet> results;
  _return.execution_time_ms += measure<>::execution([&]() {
    results = executor->execute(root_plan,
                                session_info,
                                true,
                                executor_device_type,
                                ExecutorOptLevel::Default,
                                allow_multifrag_,
                                allow_loop_joins_);
  });
  // reduce execution time by the time spent during queue waiting
  _return.execution_time_ms -= results->getQueueTime();
  if (root_plan->get_plan_dest() == Planner::RootPlan::Dest::kEXPLAIN) {
    convert_explain(_return, *results, column_format);
    return;
  }
  const auto plan = root_plan->get_plan();
  CHECK(plan);
  const auto& targets = plan->get_targetlist();
  convert_rows(_return, getTargetMetaInfo(targets), *results, column_format, -1, -1);
}

std::vector<TargetMetaInfo> MapDHandler::getTargetMetaInfo(
    const std::vector<std::shared_ptr<Analyzer::TargetEntry>>& targets) const {
  std::vector<TargetMetaInfo> result;
  for (const auto target : targets) {
    CHECK(target);
    CHECK(target->get_expr());
    result.emplace_back(target->get_resname(), target->get_expr()->get_type_info());
  }
  return result;
}

std::vector<std::string> MapDHandler::getTargetNames(
    const std::vector<std::shared_ptr<Analyzer::TargetEntry>>& targets) const {
  std::vector<std::string> names;
  for (const auto target : targets) {
    CHECK(target);
    CHECK(target->get_expr());
    names.push_back(target->get_resname());
  }
  return names;
}

std::vector<std::string> MapDHandler::getTargetNames(const std::vector<TargetMetaInfo>& targets) const {
  std::vector<std::string> names;
  for (const auto target : targets) {
    names.push_back(target.get_resname());
  }
  return names;
}

TRowDescriptor MapDHandler::convert_target_metainfo(const std::vector<TargetMetaInfo>& targets) const {
  TRowDescriptor row_desc;
  TColumnType proj_info;
  size_t i = 0;
  for (const auto target : targets) {
    proj_info.col_name = target.get_resname();
    if (proj_info.col_name.empty()) {
      proj_info.col_name = "result_" + std::to_string(i + 1);
    }
    const auto& target_ti = target.get_type_info();
    proj_info.col_type.type = type_to_thrift(target_ti);
    proj_info.col_type.encoding = encoding_to_thrift(target_ti);
    proj_info.col_type.nullable = !target_ti.get_notnull();
    proj_info.col_type.is_array = target_ti.get_type() == kARRAY;
    proj_info.col_type.precision = target_ti.get_precision();
    proj_info.col_type.scale = target_ti.get_scale();
    proj_info.col_type.comp_param = target_ti.get_comp_param();
    row_desc.push_back(proj_info);
    ++i;
  }
  return row_desc;
}

template <class R>
void MapDHandler::convert_rows(TQueryResult& _return,
                               const std::vector<TargetMetaInfo>& targets,
                               const R& results,
                               const bool column_format,
                               const int32_t first_n,
                               const int32_t at_most_n) const {
  _return.row_set.row_desc = convert_target_metainfo(targets);
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
        THROW_MAPD_EXCEPTION("The result contains more rows than the specified cap of " + std::to_string(at_most_n));
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
        THROW_MAPD_EXCEPTION("The result contains more rows than the specified cap of " + std::to_string(at_most_n));
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

TRowDescriptor MapDHandler::fixup_row_descriptor(const TRowDescriptor& row_desc,
                                                 const Catalog_Namespace::Catalog& cat) {
  TRowDescriptor fixedup_row_desc;
  for (const TColumnType& col_desc : row_desc) {
    auto fixedup_col_desc = col_desc;
    if (col_desc.col_type.encoding == TEncodingType::DICT && col_desc.col_type.comp_param > 0) {
      const auto dd = cat.getMetadataForDict(col_desc.col_type.comp_param, false);
      fixedup_col_desc.col_type.comp_param = dd->dictNBits;
    }
    fixedup_row_desc.push_back(fixedup_col_desc);
  }
  return fixedup_row_desc;
}

// create simple result set to return a single column result
void MapDHandler::create_simple_result(TQueryResult& _return,
                                       const ResultSet& results,
                                       const bool column_format,
                                       const std::string label) const {
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

void MapDHandler::convert_explain(TQueryResult& _return, const ResultSet& results, const bool column_format) const {
  create_simple_result(_return, results, column_format, "Explanation");
}

void MapDHandler::convert_result(TQueryResult& _return, const ResultSet& results, const bool column_format) const {
  create_simple_result(_return, results, column_format, "Result");
}

namespace {

void check_table_not_sharded(const Catalog_Namespace::Catalog& cat, const int table_id) {
  const auto td = cat.getMetadataForTable(table_id);
  CHECK(td);
  if (td->nShards) {
    throw std::runtime_error("Cannot execute a cluster insert into a sharded table");
  }
}

}  // namespace

void MapDHandler::sql_execute_impl(TQueryResult& _return,
                                   const Catalog_Namespace::SessionInfo& session_info,
                                   const std::string& query_str,
                                   const bool column_format,
                                   const std::string& nonce,
                                   const ExecutorDeviceType executor_device_type,
                                   const int32_t first_n,
                                   const int32_t at_most_n) {
  if (leaf_handler_) {
    leaf_handler_->flush_queue();
  }

  _return.nonce = nonce;
  _return.execution_time_ms = 0;
  auto& cat = session_info.get_catalog();

  SQLParser parser;
  std::list<std::unique_ptr<Parser::Stmt>> parse_trees;
  std::string last_parsed;
  int num_parse_errors = 0;
  Planner::RootPlan* root_plan{nullptr};
  try {
    ParserWrapper pw{query_str};
    if (!pw.is_ddl && !pw.is_update_dml && !pw.is_other_explain) {
      std::string query_ra;
      _return.execution_time_ms += measure<>::execution([&]() { query_ra = parse_to_ra(query_str, session_info); });
      if (pw.is_select_calcite_explain) {
        // return the ra as the result
        convert_explain(_return, ResultSet(query_ra), true);
        return;
      }
      execute_rel_alg(_return,
                      query_ra,
                      column_format,
                      session_info,
                      executor_device_type,
                      first_n,
                      at_most_n,
                      pw.is_select_explain,
                      false);
      return;
    }
    LOG(INFO) << "passing query to legacy processor";
  } catch (std::exception& e) {
    THROW_MAPD_EXCEPTION(std::string("Exception: ") + e.what());
  }
  try {
    // check for COPY TO stmt replace as required parser expects #~# markers
    const auto result = apply_copy_to_shim(query_str);
    num_parse_errors = parser.parse(result, parse_trees, last_parsed);
  } catch (std::exception& e) {
    THROW_MAPD_EXCEPTION(std::string("Exception: ") + e.what());
  }
  if (num_parse_errors > 0) {
    THROW_MAPD_EXCEPTION("Syntax error at: " + last_parsed);
  }
  for (const auto& stmt : parse_trees) {
    try {
      auto select_stmt = dynamic_cast<Parser::SelectStmt*>(stmt.get());
      if (!select_stmt) {
        check_read_only("Non-SELECT statements");
      }
      Parser::DDLStmt* ddl = dynamic_cast<Parser::DDLStmt*>(stmt.get());
      Parser::ExplainStmt* explain_stmt = nullptr;
      if (ddl != nullptr)
        explain_stmt = dynamic_cast<Parser::ExplainStmt*>(ddl);
      if (ddl != nullptr && explain_stmt == nullptr) {
        const auto copy_stmt = dynamic_cast<Parser::CopyTableStmt*>(ddl);
        if (g_cluster && copy_stmt && !leaf_aggregator_.leafCount()) {
          // Sharded table rows need to be routed to the leaf by an aggregator.
          check_table_not_sharded(cat, copy_stmt->get_table());
        }
        if (copy_stmt && leaf_aggregator_.leafCount() > 0) {
          _return.execution_time_ms +=
              measure<>::execution([&]() { execute_distributed_copy_statement(copy_stmt, session_info); });
        } else {
          _return.execution_time_ms += measure<>::execution([&]() { ddl->execute(session_info); });
        }
        // check if it was a copy statement gather response message
        if (copy_stmt) {
          convert_result(_return, ResultSet(*copy_stmt->return_message.get()), true);
        }
      } else {
        const Parser::DMLStmt* dml;
        if (explain_stmt != nullptr)
          dml = explain_stmt->get_stmt();
        else
          dml = dynamic_cast<Parser::DMLStmt*>(stmt.get());
        Analyzer::Query query;
        dml->analyze(cat, query);
        Planner::Optimizer optimizer(query, cat);
        root_plan = optimizer.optimize();
        CHECK(root_plan);
        std::unique_ptr<Planner::RootPlan> plan_ptr(root_plan);  // make sure it's deleted
        if (g_cluster && plan_ptr->get_stmt_type() == kINSERT) {
          check_table_not_sharded(session_info.get_catalog(), plan_ptr->get_result_table_id());
        }
        if (explain_stmt != nullptr) {
          root_plan->set_plan_dest(Planner::RootPlan::Dest::kEXPLAIN);
        }
        execute_root_plan(_return, root_plan, column_format, session_info, executor_device_type, first_n);
      }
    } catch (std::exception& e) {
      THROW_MAPD_EXCEPTION(std::string("Exception: ") + e.what());
    }
  }
}

void MapDHandler::execute_distributed_copy_statement(Parser::CopyTableStmt* copy_stmt,
                                                     const Catalog_Namespace::SessionInfo& session_info) {
  auto importer_factory = [&session_info, this](const Catalog_Namespace::Catalog& catalog,
                                                const TableDescriptor* td,
                                                const std::string& file_path,
                                                const Importer_NS::CopyParams& copy_params) {
    return boost::make_unique<Importer_NS::Importer>(
        new DistributedLoader(session_info, td, &leaf_aggregator_), file_path, copy_params);
  };
  copy_stmt->execute(session_info, importer_factory);
}

Planner::RootPlan* MapDHandler::parse_to_plan(const std::string& query_str,
                                              const Catalog_Namespace::SessionInfo& session_info) {
  auto& cat = session_info.get_catalog();
  ParserWrapper pw{query_str};
  // if this is a calcite select or explain select run in calcite
  if (!pw.is_ddl && !pw.is_update_dml && !pw.is_other_explain) {
    const std::string actual_query{pw.is_select_explain || pw.is_select_calcite_explain ? pw.actual_query : query_str};
    const auto query_ra = calcite_->process(session_info,
                                            legacy_syntax_ ? pg_shim(actual_query) : actual_query,
                                            legacy_syntax_,
                                            pw.is_select_calcite_explain);
    auto root_plan = translate_query(query_ra, cat);
    CHECK(root_plan);
    if (pw.is_select_explain) {
      root_plan->set_plan_dest(Planner::RootPlan::Dest::kEXPLAIN);
    }
    return root_plan;
  }
  return nullptr;
}

std::string MapDHandler::parse_to_ra(const std::string& query_str, const Catalog_Namespace::SessionInfo& session_info) {
  ParserWrapper pw{query_str};
  const std::string actual_query{pw.is_select_explain || pw.is_select_calcite_explain ? pw.actual_query : query_str};
  if (!pw.is_ddl && !pw.is_update_dml && !pw.is_other_explain) {
    return calcite_->process(session_info,
                             legacy_syntax_ ? pg_shim(actual_query) : actual_query,
                             legacy_syntax_,
                             pw.is_select_calcite_explain);
  }
  return "";
}

void MapDHandler::execute_first_step(TStepResult& _return, const TPendingQuery& pending_query) {
  if (!leaf_handler_) {
    THROW_MAPD_EXCEPTION("Distributed support is disabled.");
  }
  LOG(INFO) << "execute_first_step :  id:" << pending_query.id;
  auto time_ms = measure<>::execution([&]() {
    try {
      leaf_handler_->execute_first_step(_return, pending_query);
    } catch (std::exception& e) {
      THROW_MAPD_EXCEPTION(std::string("Exception: ") + e.what());
    }
  });
  LOG(INFO) << "execute_first_step-COMPLETED " << time_ms << "ms";
}

void MapDHandler::start_query(TPendingQuery& _return,
                              const TSessionId& session,
                              const std::string& query_ra,
                              const bool just_explain) {
  if (!leaf_handler_) {
    THROW_MAPD_EXCEPTION("Distributed support is disabled.");
  }
  LOG(INFO) << "start_query :" << session << ":" << just_explain;
  auto time_ms = measure<>::execution([&]() {
    try {
      leaf_handler_->start_query(_return, session, query_ra, just_explain);
    } catch (std::exception& e) {
      THROW_MAPD_EXCEPTION(std::string("Exception: ") + e.what());
    }
  });
  LOG(INFO) << "start_query-COMPLETED " << time_ms << "ms "
            << "id is " << _return.id;
}

void MapDHandler::broadcast_serialized_rows(const std::string& serialized_rows,
                                            const TRowDescriptor& row_desc,
                                            const TQueryId query_id) {
  if (!leaf_handler_) {
    THROW_MAPD_EXCEPTION("Distributed support is disabled.");
  }
  LOG(INFO) << "BROADCAST-SERIALIZED-ROWS  id:" << query_id;
  auto time_ms = measure<>::execution([&]() {
    try {
      leaf_handler_->broadcast_serialized_rows(serialized_rows, row_desc, query_id);
    } catch (std::exception& e) {
      THROW_MAPD_EXCEPTION(std::string("Exception: ") + e.what());
    }
  });
  LOG(INFO) << "BROADCAST-SERIALIZED-ROWS COMPLETED " << time_ms << "ms";
}

void MapDHandler::insert_data(const TSessionId& session, const TInsertData& thrift_insert_data) {
  static std::mutex insert_mutex;  // TODO: split lock, make it per table
  CHECK_EQ(thrift_insert_data.column_ids.size(), thrift_insert_data.data.size());
  const auto session_info = get_session(session);
  auto& cat = session_info.get_catalog();
  Fragmenter_Namespace::InsertData insert_data;
  insert_data.databaseId = thrift_insert_data.db_id;
  insert_data.tableId = thrift_insert_data.table_id;
  insert_data.columnIds = thrift_insert_data.column_ids;
  std::vector<std::unique_ptr<std::vector<std::string>>> none_encoded_string_columns;
  std::vector<std::unique_ptr<std::vector<ArrayDatum>>> array_columns;
  for (size_t col_idx = 0; col_idx < insert_data.columnIds.size(); ++col_idx) {
    const int column_id = insert_data.columnIds[col_idx];
    DataBlockPtr p;
    const auto cd = cat.getMetadataForColumn(insert_data.tableId, column_id);
    CHECK(cd);
    const auto& ti = cd->columnType;
    if (ti.is_number() || ti.is_time() || ti.is_boolean()) {
      p.numbersPtr = (int8_t*)thrift_insert_data.data[col_idx].fixed_len_data.data();
    } else if (ti.is_string()) {
      if (ti.get_compression() == kENCODING_DICT) {
        p.numbersPtr = (int8_t*)thrift_insert_data.data[col_idx].fixed_len_data.data();
      } else {
        CHECK_EQ(kENCODING_NONE, ti.get_compression());
        none_encoded_string_columns.emplace_back(new std::vector<std::string>());
        auto& none_encoded_strings = none_encoded_string_columns.back();
        CHECK_EQ(static_cast<size_t>(thrift_insert_data.num_rows),
                 thrift_insert_data.data[col_idx].var_len_data.size());
        for (const auto& varlen_str : thrift_insert_data.data[col_idx].var_len_data) {
          none_encoded_strings->push_back(varlen_str.payload);
        }
        p.stringsPtr = none_encoded_strings.get();
      }
    } else {
      CHECK(ti.is_array());
      array_columns.emplace_back(new std::vector<ArrayDatum>());
      auto& array_column = array_columns.back();
      CHECK_EQ(static_cast<size_t>(thrift_insert_data.num_rows), thrift_insert_data.data[col_idx].var_len_data.size());
      for (const auto& t_arr_datum : thrift_insert_data.data[col_idx].var_len_data) {
        if (t_arr_datum.is_null) {
          array_column->emplace_back(0, nullptr, true);
        } else {
          ArrayDatum arr_datum;
          arr_datum.length = t_arr_datum.payload.size();
          arr_datum.pointer = (int8_t*)t_arr_datum.payload.data();
          arr_datum.is_null = false;
          array_column->push_back(arr_datum);
        }
      }
      p.arraysPtr = array_column.get();
    }
    insert_data.data.push_back(p);
  }
  insert_data.numRows = thrift_insert_data.num_rows;
  const auto td = cat.getMetadataForTable(insert_data.tableId);
  try {
    td->fragmenter->insertDataNoCheckpoint(insert_data);
  } catch (const std::exception& e) {
    THROW_MAPD_EXCEPTION(std::string("Exception: ") + e.what());
  }
}

void MapDHandler::render_vega_raw_pixels(TRawPixelDataResult& _return,
                                         const TSessionId& session,
                                         const int64_t widget_id,
                                         const int16_t node_idx,
                                         const std::string& vega_json,
                                         const std::string& nonce) {
  if (!render_handler_) {
    THROW_MAPD_EXCEPTION("Backend rendering is disabled.");
  }

  const auto session_info = MapDHandler::get_session(session);
  LOG(INFO) << "RENDER_VEGA_RAW_PIXELS :" << session << ":widget_id:" << widget_id << ":node_idx:" << node_idx
            << ":vega_json:" << vega_json << ":nonce:" << nonce;

  _return.total_time_ms = measure<>::execution([&]() {
    try {
      render_handler_->render_vega_raw_pixels(_return, session_info, widget_id, node_idx, vega_json);
    } catch (std::exception& e) {
      THROW_MAPD_EXCEPTION(std::string("Exception: ") + e.what());
    }
  });

  LOG(INFO) << "RENDER_VEGA_RAW_PIXELS COMPLETED nonce: " << nonce << " Total: " << _return.total_time_ms
            << " (ms), Total Execution: " << _return.execution_time_ms
            << " (ms), Total Render: " << _return.render_time_ms << " (ms)";
}

void MapDHandler::checkpoint(const TSessionId& session, const int32_t db_id, const int32_t table_id) {
  const auto session_info = get_session(session);
  auto& cat = session_info.get_catalog();
  cat.get_dataMgr().checkpoint(db_id, table_id);
}

// check and reset epoch if a request has been made
void MapDHandler::set_table_epoch(const TSessionId& session, const int db_id, const int table_id, const int new_epoch) {
  const auto session_info = get_session(session);
  if (!session_info.get_currentUser().isSuper) {
    throw std::runtime_error("Only superuser can set_table_epoch");
  }
  auto& cat = session_info.get_catalog();

  if (leaf_aggregator_.leafCount() > 0) {
    return leaf_aggregator_.set_table_epochLeaf(session_info, db_id, table_id, new_epoch);
  }
  cat.setTableEpoch(db_id, table_id, new_epoch);
}

// check and reset epoch if a request has been made
void MapDHandler::set_table_epoch_by_name(const TSessionId& session,
                                          const std::string& table_name,
                                          const int new_epoch) {
  const auto session_info = get_session(session);
  if (!session_info.get_currentUser().isSuper) {
    throw std::runtime_error("Only superuser can set_table_epoch");
  }
  auto& cat = session_info.get_catalog();
  auto td =
      cat.getMetadataForTable(table_name, false);  // don't populate fragmenter on this call since we only want metadata
  int32_t db_id = cat.get_currentDB().dbId;
  if (leaf_aggregator_.leafCount() > 0) {
    return leaf_aggregator_.set_table_epochLeaf(session_info, db_id, td->tableId, new_epoch);
  }
  cat.setTableEpoch(db_id, td->tableId, new_epoch);
}

int32_t MapDHandler::get_table_epoch(const TSessionId& session, const int32_t db_id, const int32_t table_id) {
  const auto session_info = get_session(session);
  auto& cat = session_info.get_catalog();

  if (leaf_aggregator_.leafCount() > 0) {
    return leaf_aggregator_.get_table_epochLeaf(session_info, db_id, table_id);
  }
  return cat.getTableEpoch(db_id, table_id);
}

int32_t MapDHandler::get_table_epoch_by_name(const TSessionId& session, const std::string& table_name) {
  const auto session_info = get_session(session);
  auto& cat = session_info.get_catalog();
  auto td =
      cat.getMetadataForTable(table_name, false);  // don't populate fragmenter on this call since we only want metadata
  int32_t db_id = cat.get_currentDB().dbId;
  if (leaf_aggregator_.leafCount() > 0) {
    return leaf_aggregator_.get_table_epochLeaf(session_info, db_id, td->tableId);
  }
  return cat.getTableEpoch(db_id, td->tableId);
}

void MapDHandler::set_license_key(TLicenseInfo& _return,
                                  const TSessionId& session,
                                  const std::string& key,
                                  const std::string& nonce) {
  check_read_only("set_license_key");
  const auto session_info = get_session(session);
  THROW_MAPD_EXCEPTION(std::string("Licensing not supported."));
}

void MapDHandler::get_license_claims(TLicenseInfo& _return, const TSessionId& session, const std::string& nonce) {
  const auto session_info = get_session(session);
  _return.claims.push_back("");
}
