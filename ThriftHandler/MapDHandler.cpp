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
#include "gen-cpp/CalciteServer.h"

#include "QueryEngine/RelAlgExecutor.h"

#include "Catalog/Catalog.h"
#include "Fragmenter/InsertOrderFragmenter.h"
#include "Import/Importer.h"
#include "MapDDistributedHandler.h"
#include "MapDRenderHandler.h"
#include "Parser/ParserWrapper.h"
#include "Parser/ReservedKeywords.h"
#include "Parser/parser.h"
#include "Planner/Planner.h"
#include "QueryEngine/CalciteAdapter.h"
#include "QueryEngine/Execute.h"
#include "QueryEngine/ExtensionFunctionsWhitelist.h"
#include "QueryEngine/GpuMemUtils.h"
#include "QueryEngine/JoinFilterPushDown.h"
#include "QueryEngine/JsonAccessors.h"
#include "Shared/MapDParameters.h"
#include "Shared/SQLTypeUtilities.h"
#include "Shared/StringTransform.h"
#include "Shared/geo_types.h"
#include "Shared/geosupport.h"
#include "Shared/import_helpers.h"
#include "Shared/mapd_shared_mutex.h"
#include "Shared/measure.h"
#include "Shared/scope.h"

#include <fcntl.h>
#include <glog/logging.h>
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
#include <csignal>
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

using Catalog_Namespace::Catalog;
using Catalog_Namespace::SysCatalog;
using namespace Lock_Namespace;

#define INVALID_SESSION_ID ""

#define THROW_MAPD_EXCEPTION(errstr) \
  TMapDException ex;                 \
  ex.error_msg = errstr;             \
  LOG(ERROR) << ex.error_msg;        \
  throw ex;

std::string generate_random_string(const size_t len);

MapDHandler::MapDHandler(const std::vector<LeafHostInfo>& db_leaves,
                         const std::vector<LeafHostInfo>& string_leaves,
                         const std::string& base_data_path,
                         const std::string& executor_device,
                         const bool allow_multifrag,
                         const bool jit_debug,
                         const bool read_only,
                         const bool allow_loop_joins,
                         const bool enable_rendering,
                         const size_t render_mem_bytes,
                         const int num_gpus,
                         const int start_gpu,
                         const size_t reserved_gpu_mem,
                         const size_t num_reader_threads,
                         const AuthMetadata authMetadata,
                         const MapDParameters& mapd_parameters,
                         const std::string& db_convert_dir,
                         const bool legacy_syntax,
                         const bool access_priv_check,
                         const int idle_session_duration,
                         const int max_session_duration)
    : leaf_aggregator_(db_leaves)
    , string_leaves_(string_leaves)
    , base_data_path_(base_data_path)
    , random_gen_(std::random_device{}())
    , session_id_dist_(0, INT32_MAX)
    , jit_debug_(jit_debug)
    , allow_multifrag_(allow_multifrag)
    , read_only_(read_only)
    , allow_loop_joins_(allow_loop_joins)
    , mapd_parameters_(mapd_parameters)
    , legacy_syntax_(legacy_syntax)
    , super_user_rights_(false)
    , access_priv_check_(access_priv_check)
    , idle_session_duration_(idle_session_duration * 60)
    , max_session_duration_(max_session_duration * 60)
    , _was_geo_copy_from(false) {
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
  } else {
    executor_device_type_ = ExecutorDeviceType::CPU;
    cpu_mode_only_ = true;
  }
  const auto data_path = boost::filesystem::path(base_data_path_) / "mapd_data";
  // calculate the total amount of memory we need to reserve from each gpu that the Buffer
  // manage cannot ask for
  size_t total_reserved = reserved_gpu_mem;
  if (enable_rendering) {
    total_reserved += render_mem_bytes;
  }
  data_mgr_.reset(new Data_Namespace::DataMgr(data_path.string(),
                                              mapd_parameters,
                                              !cpu_mode_only_,
                                              num_gpus,
                                              db_convert_dir,
                                              start_gpu,
                                              total_reserved,
                                              num_reader_threads));

  std::string calcite_session_prefix = "calcite-" + generate_random_string(64);

  calcite_ =
      std::make_shared<Calcite>(mapd_parameters, base_data_path_, calcite_session_prefix);

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
  }
  SysCatalog::instance().init(base_data_path_,
                              data_mgr_,
                              authMetadata,
                              calcite_,
                              false,
                              access_priv_check,
                              &string_leaves_);
  import_path_ = boost::filesystem::path(base_data_path_) / "mapd_import";
  start_time_ = std::time(nullptr);

  if (enable_rendering) {
    try {
      render_handler_.reset(
          new MapDRenderHandler(this, render_mem_bytes, num_gpus, start_gpu));
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
void MapDHandler::internal_connect(TSessionId& session,
                                   const std::string& user,
                                   const std::string& dbname) {
  mapd_lock_guard<mapd_shared_mutex> write_lock(sessions_mutex_);
  Catalog_Namespace::UserMetadata user_meta;
  if (!SysCatalog::instance().getMetadataForUser(user, user_meta)) {
    THROW_MAPD_EXCEPTION(std::string("User ") + user + " does not exist.");
  }
  std::shared_ptr<Catalog> cat = nullptr;
  try {
    cat = SysCatalog::instance().login(dbname, user, std::string(""), user_meta, false);
  } catch (std::exception& e) {
    THROW_MAPD_EXCEPTION(e.what());
  }

  if (SysCatalog::instance().arePrivilegesOn()) {
    DBObject dbObject(dbname, DatabaseDBObjectType);
    dbObject.loadKey(*cat);
    dbObject.setPrivileges(AccessPrivileges::ACCESS);
    std::vector<DBObject> dbObjects;
    dbObjects.push_back(dbObject);
    if (!SysCatalog::instance().checkPrivileges(user_meta, dbObjects)) {
      THROW_MAPD_EXCEPTION("Unauthorized Access: user " + user_meta.userName +
                           " is not allowed to access database " + dbname + ".");
    }
  }
  connectImpl(session, user, std::string(""), dbname, user_meta, cat);
}

void MapDHandler::connect(TSessionId& session,
                          const std::string& user,
                          const std::string& passwd,
                          const std::string& dbname) {
  mapd_lock_guard<mapd_shared_mutex> write_lock(sessions_mutex_);
  Catalog_Namespace::UserMetadata user_meta;
  std::shared_ptr<Catalog> cat = nullptr;
  try {
    cat = SysCatalog::instance().login(
        dbname, user, passwd, user_meta, !super_user_rights_);
  } catch (std::exception& e) {
    THROW_MAPD_EXCEPTION(e.what());
  }
  if (!SysCatalog::instance().getMetadataForUser(user, user_meta)) {
    THROW_MAPD_EXCEPTION(std::string("User ") + user + " does not exist.");
  }
  if (SysCatalog::instance().arePrivilegesOn()) {
    DBObject dbObject(dbname, DatabaseDBObjectType);
    dbObject.loadKey(*cat);
    dbObject.setPrivileges(AccessPrivileges::ACCESS);
    std::vector<DBObject> dbObjects;
    dbObjects.push_back(dbObject);
    if (!SysCatalog::instance().checkPrivileges(user_meta, dbObjects)) {
      THROW_MAPD_EXCEPTION("Unauthorized Access: user " + user_meta.userName +
                           " is not allowed to access database " + dbname + ".");
    }
  }
  connectImpl(session, user, passwd, dbname, user_meta, cat);
}

void MapDHandler::connectImpl(TSessionId& session,
                              const std::string& user,
                              const std::string& passwd,
                              const std::string& dbname,
                              Catalog_Namespace::UserMetadata& user_meta,
                              std::shared_ptr<Catalog> cat) {
  session = INVALID_SESSION_ID;
  while (true) {
    session = generate_random_string(32);
    auto session_it = sessions_.find(session);
    if (session_it == sessions_.end()) {
      break;
    }
  }
  sessions_[session].reset(
      new Catalog_Namespace::SessionInfo(cat, user_meta, executor_device_type_, session));
  if (!super_user_rights_) {  // no need to connect to leaf_aggregator_ at this time while
                              // doing warmup
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
  if (render_handler_) {
    render_handler_->disconnect(session);
  }
  auto session_it = MapDHandler::get_session_it(session);
  const auto dbname = session_it->second->get_catalog().get_currentDB().dbName;
  LOG(INFO) << "User " << session_it->second->get_currentUser().userName
            << " disconnected from database " << dbname << std::endl;
  sessions_.erase(session_it);
}

void MapDHandler::interrupt(const TSessionId& session) {
  if (g_enable_dynamic_watchdog) {
    // Shared lock to allow simultaneous interrupts of multiple sessions
    mapd_shared_lock<mapd_shared_mutex> read_lock(sessions_mutex_);
    if (leaf_aggregator_.leafCount() > 0) {
      leaf_aggregator_.interrupt(session);
    }
    auto session_it = get_session_it(session);
    const auto dbname = session_it->second->get_catalog().get_currentDB().dbName;
    auto session_info_ptr = session_it->second.get();
    auto& cat = session_info_ptr->get_catalog();
    auto executor = Executor::getExecutor(cat.get_currentDB().dbId,
                                          jit_debug_ ? "/tmp" : "",
                                          jit_debug_ ? "mapdquery" : "",
                                          mapd_parameters_,
                                          nullptr);
    CHECK(executor);

    VLOG(1) << "Received interrupt: "
            << "Session " << *session_it->second << ", Executor " << executor
            << ", leafCount " << leaf_aggregator_.leafCount() << ", User "
            << session_it->second->get_currentUser().userName << ", Database " << dbname
            << std::endl;

    executor->interrupt();

    LOG(INFO) << "User " << session_it->second->get_currentUser().userName
              << " interrupted session with database " << dbname << std::endl;
  }
}

void MapDHandler::get_server_status(TServerStatus& _return, const TSessionId& session) {
  const auto rendering_enabled = bool(render_handler_);
  _return.read_only = read_only_;
  _return.version = MAPD_RELEASE;
  _return.rendering_enabled = rendering_enabled;
  _return.poly_rendering_enabled =
      rendering_enabled && !(leaf_aggregator_.leafCount() > 0);
  _return.start_time = start_time_;
  _return.edition = MAPD_EDITION;
  _return.host_name = "aggregator";
}

void MapDHandler::get_status(std::vector<TServerStatus>& _return,
                             const TSessionId& session) {
  const auto rendering_enabled = bool(render_handler_);
  TServerStatus ret;
  ret.read_only = read_only_;
  ret.version = MAPD_RELEASE;
  ret.rendering_enabled = rendering_enabled;
  ret.poly_rendering_enabled = rendering_enabled && !(leaf_aggregator_.leafCount() > 0);
  ret.start_time = start_time_;
  ret.edition = MAPD_EDITION;
  ret.host_name = "aggregator";
  _return.push_back(ret);
  if (leaf_aggregator_.leafCount() > 0) {
    std::vector<TServerStatus> leaf_status = leaf_aggregator_.getLeafStatus(session);
    _return.insert(_return.end(), leaf_status.begin(), leaf_status.end());
  }
}

void MapDHandler::get_hardware_info(TClusterHardwareInfo& _return,
                                    const TSessionId& session) {
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
    _return.hardware_info.insert(_return.hardware_info.end(),
                                 leaf_hardware.hardware_info.begin(),
                                 leaf_hardware.hardware_info.end());
  }
}

void MapDHandler::value_to_thrift_column(const TargetValue& tv,
                                         const SQLTypeInfo& ti,
                                         TColumn& column) {
  if (ti.is_array()) {
    const auto list_tv = boost::get<std::vector<ScalarTargetValue>>(&tv);
    CHECK(list_tv);
    TColumn tColumn;
    for (const auto& elem_tv : *list_tv) {
      value_to_thrift_column(elem_tv, ti.get_elem_type(), tColumn);
    }
    column.data.arr_col.push_back(tColumn);
    column.nulls.push_back(list_tv->size() == 0 && !ti.get_notnull());
  } else if (ti.is_geometry()) {
    const auto list_tv = boost::get<std::vector<ScalarTargetValue>>(&tv);
    if (list_tv) {
      auto elem_type = SQLTypeInfo(kDOUBLE, false);
      TColumn tColumn;
      for (const auto& elem_tv : *list_tv) {
        value_to_thrift_column(elem_tv, elem_type, tColumn);
      }
      column.data.arr_col.push_back(tColumn);
      column.nulls.push_back(list_tv->size() == 0 && !ti.get_notnull());
    } else {
      const auto scalar_tv = boost::get<ScalarTargetValue>(&tv);
      CHECK(scalar_tv);
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
    }
  } else {
    const auto scalar_tv = boost::get<ScalarTargetValue>(&tv);
    CHECK(scalar_tv);
    if (boost::get<int64_t>(scalar_tv)) {
      int64_t data = *(boost::get<int64_t>(scalar_tv));

      if (is_member_of_typeset<kNUMERIC, kDECIMAL>(ti)) {
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
          column.nulls.push_back(data == NULL_TINYINT);
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
          if (sizeof(time_t) == 4) {
            column.nulls.push_back(data == NULL_INT && !ti.get_notnull());
          } else {
            column.nulls.push_back(data == NULL_BIGINT && !ti.get_notnull());
          }
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
    int64_t data = *(boost::get<int64_t>(scalar_tv));

    if (is_member_of_typeset<kNUMERIC, kDECIMAL>(ti)) {
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
        if (sizeof(time_t) == 4) {
          datum.is_null = (datum.val.int_val == NULL_INT);
        } else {
          datum.is_null = (datum.val.int_val == NULL_BIGINT);
        }
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

namespace {

std::string hide_sensitive_data(const std::string& query_str) {
  auto result = query_str;
  static const std::vector<std::string> patterns{
      R"(^(CREATE|ALTER)\s+?USER.+password\s*?=\s*?'(?<pwd>.+?)'.+)",
      R"(^COPY.+FROM.+WITH.+s3_access_key\s*?=\s*?'(?<pwd>.+?)'.+)",
      R"(^COPY.+FROM.+WITH.+s3_secret_key\s*?=\s*?'(?<pwd>.+?)'.+)",
  };
  for (const auto& pattern : patterns) {
    boost::regex passwd{pattern, boost::regex_constants::perl | boost::regex::icase};
    boost::smatch matches;
    if (boost::regex_search(result, matches, passwd)) {
      result.replace(
          matches["pwd"].first - result.begin(), matches["pwd"].length(), "XXXXXXXX");
    }
  }
  return result;
}

}  // namespace

void MapDHandler::sql_execute(TQueryResult& _return,
                              const TSessionId& session,
                              const std::string& query_str,
                              const bool column_format,
                              const std::string& nonce,
                              const int32_t first_n,
                              const int32_t at_most_n) {
  ScopeGuard reset_was_geo_copy_from = [&] { _was_geo_copy_from = false; };
  if (first_n >= 0 && at_most_n >= 0) {
    THROW_MAPD_EXCEPTION(std::string("At most one of first_n and at_most_n can be set"));
  }
  const auto session_info = MapDHandler::get_session(session);
  const auto session_it = get_session_it(session);
  LOG(INFO) << "sql_execute :" << *session_it->second
            << " :query_str:" << hide_sensitive_data(query_str);
  if (leaf_aggregator_.leafCount() > 0) {
    if (!agg_handler_) {
      THROW_MAPD_EXCEPTION("Distributed support is disabled.");
    }
    _return.total_time_ms = measure<>::execution([&]() {
      try {
        agg_handler_->cluster_execute(
            _return, session_info, query_str, column_format, nonce, first_n, at_most_n);
      } catch (std::exception& e) {
        const auto mapd_exception = dynamic_cast<const TMapDException*>(&e);
        THROW_MAPD_EXCEPTION(mapd_exception ? mapd_exception->error_msg
                                            : (std::string("Exception: ") + e.what()));
      }
      _return.nonce = nonce;
    });
    LOG(INFO) << "sql_execute-COMPLETED Distributed Execute Time: "
              << _return.total_time_ms << " (ms)";
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

  // if the SQL statement we just executed was a geo COPY FROM, the import
  // parameters were captured, and this flag set, so we do the actual import here
  if (_was_geo_copy_from) {
    // import_geo_table() calls create_table() which calls this function to
    // do the work, so reset the flag now to avoid executing this part a
    // second time at the end of that, which would fail as the table was
    // already created! Also reset the flag with a ScopeGuard on exiting
    // this function any other way, such as an exception from the code above!
    _was_geo_copy_from = false;

    // now do (and time) the import
    _return.total_time_ms = measure<>::execution([&]() {
      import_geo_table(session,
                       _geo_copy_from_table,
                       _geo_copy_from_file_name,
                       copyparams_to_thrift(_geo_copy_from_copy_params),
                       TRowDescriptor());
    });
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
      THROW_MAPD_EXCEPTION(
          std::string("Exception: GPU mode is not allowed in this session"));
    }
    if (!data_mgr_->gpusPresent()) {
      THROW_MAPD_EXCEPTION(std::string("Exception: no GPU is available in this server"));
    }
    if (device_id < 0 || device_id >= data_mgr_->cudaMgr_->getDeviceCount()) {
      THROW_MAPD_EXCEPTION(
          std::string("Exception: invalid device_id or unavailable GPU with this ID"));
    }
  }
  LOG(INFO) << hide_sensitive_data(query_str);
  int64_t total_time_ms = measure<>::execution([&]() {
    SQLParser parser;
    std::list<std::unique_ptr<Parser::Stmt>> parse_trees;
    std::string last_parsed;
    try {
      ParserWrapper pw{query_str};
      if (!pw.is_ddl && !pw.is_update_dml && !pw.is_other_explain) {
        std::string query_ra;
        std::map<std::string, bool> tableNames;
        execution_time_ms += measure<>::execution(
            [&]() { query_ra = parse_to_ra(query_str, {}, session_info, &tableNames); });

        // COPY_TO/SELECT: get read ExecutorOuterLock >> read UpdateDeleteLock locks
        mapd_shared_lock<mapd_shared_mutex> executeReadLock(
            *LockMgr<mapd_shared_mutex, bool>::getMutex(ExecutorOuterLock, true));
        std::vector<std::shared_ptr<VLock>> upddelLocks;
        getTableLocks<mapd_shared_mutex>(session_info.get_catalog(),
                                         tableNames,
                                         upddelLocks,
                                         LockType::UpdateDeleteLock);

        if (pw.is_select_calcite_explain) {
          throw std::runtime_error("explain is not unsupported by current thrift API");
        }
        execute_rel_alg_df(_return,
                           query_ra,
                           session_info,
                           device_type == TDeviceType::CPU ? ExecutorDeviceType::CPU
                                                           : ExecutorDeviceType::GPU,
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
    THROW_MAPD_EXCEPTION(
        "Exception: DDL or update DML are not unsupported by current thrift API");
  });
  LOG(INFO) << "Total: " << total_time_ms << " (ms), Execution: " << execution_time_ms
            << " (ms)";
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
      ex.error_msg = std::string(
          "Exception: current data frame handle is not bookkept or been inserted twice");
      LOG(ERROR) << ex.error_msg;
      throw ex;
    }
    dev_ptr = ipc_handle_to_dev_ptr_[df.df_handle];
    ipc_handle_to_dev_ptr_.erase(df.df_handle);
  }
  std::vector<char> sm_handle(df.sm_handle.begin(), df.sm_handle.end());
  std::vector<char> df_handle(df.df_handle.begin(), df.df_handle.end());
  ArrowResult result{sm_handle, df.sm_size, df_handle, df.df_size, dev_ptr};
  deallocate_arrow_result(
      result,
      device_type == TDeviceType::CPU ? ExecutorDeviceType::CPU : ExecutorDeviceType::GPU,
      device_id,
      data_mgr_.get());
}

std::string MapDHandler::apply_copy_to_shim(const std::string& query_str) {
  auto result = query_str;
  {
    // boost::regex copy_to{R"(COPY\s\((.*)\)\sTO\s(.*))", boost::regex::extended |
    // boost::regex::icase};
    boost::regex copy_to{R"(COPY\s*\(([^#])(.+)\)\s+TO\s)",
                         boost::regex::extended | boost::regex::icase};
    apply_shim(result, copy_to, [](std::string& result, const boost::smatch& what) {
      result.replace(
          what.position(), what.length(), "COPY (#~#" + what[1] + what[2] + "#~#) TO  ");
    });
  }
  return result;
}

void MapDHandler::sql_validate(TTableDescriptor& _return,
                               const TSessionId& session,
                               const std::string& query_str) {
  std::unique_ptr<const Planner::RootPlan> root_plan;
  const auto session_info = get_session(session);
  ParserWrapper pw{query_str};
  if (pw.is_select_explain || pw.is_other_explain || pw.is_ddl || pw.is_update_dml) {
    THROW_MAPD_EXCEPTION("Can only validate SELECT statements.");
  }
  MapDHandler::validate_rel_alg(_return, query_str, session_info);
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

void MapDHandler::get_completion_hints(std::vector<TCompletionHint>& hints,
                                       const TSessionId& session,
                                       const std::string& sql,
                                       const int cursor) {
  std::vector<std::string> visible_tables;  // Tables allowed for the given session.
  get_completion_hints_unsorted(hints, visible_tables, session, sql, cursor);
  const auto proj_tokens = extract_projection_tokens_for_completion(sql);
  auto compatible_table_names = get_uc_compatible_table_names_by_column(
      proj_tokens.uc_column_names, visible_tables, session);
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
          // Between two tables, one which is compatible with the specified projections
          // and one which isn't, pick the one which is compatible.
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

void MapDHandler::get_completion_hints_unsorted(std::vector<TCompletionHint>& hints,
                                                std::vector<std::string>& visible_tables,
                                                const TSessionId& session,
                                                const std::string& sql,
                                                const int cursor) {
  const auto session_info = get_session(session);
  try {
    get_tables(visible_tables, session);
    // Filter out keywords suggested by Calcite which we don't support.
    hints = just_whitelisted_keyword_hints(
        calcite_->getCompletionHints(session_info, visible_tables, sql, cursor));
  } catch (const std::exception& e) {
    TMapDException ex;
    ex.error_msg = "Exception: " + std::string(e.what());
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
  get_token_based_completions(hints, session, visible_tables, sql, cursor);
}

void MapDHandler::get_token_based_completions(
    std::vector<TCompletionHint>& hints,
    const TSessionId& session,
    const std::vector<std::string>& visible_tables,
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
    const auto column_names_by_table =
        fill_column_names_by_table(visible_tables, session);
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
MapDHandler::fill_column_names_by_table(const std::vector<std::string>& table_names,
                                        const TSessionId& session) {
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

std::unordered_set<std::string> MapDHandler::get_uc_compatible_table_names_by_column(
    const std::unordered_set<std::string>& uc_column_names,
    const std::vector<std::string>& table_names,
    const TSessionId& session) {
  std::unordered_set<std::string> compatible_table_names_by_column;
  for (const auto& table_name : table_names) {
    TTableDetails table_details;
    get_table_details(table_details, session, table_name);
    for (const auto& column_type : table_details.row_desc) {
      if (uc_column_names.find(to_upper(column_type.col_name)) != uc_column_names.end()) {
        compatible_table_names_by_column.emplace(to_upper(table_name));
        break;
      }
    }
  }
  return compatible_table_names_by_column;
}

void MapDHandler::validate_rel_alg(TTableDescriptor& _return,
                                   const std::string& query_str,
                                   const Catalog_Namespace::SessionInfo& session_info) {
  try {
    const auto query_ra = parse_to_ra(query_str, {}, session_info);
    TQueryResult result;
    MapDHandler::execute_rel_alg(result,
                                 query_ra,
                                 true,
                                 session_info,
                                 ExecutorDeviceType::CPU,
                                 -1,
                                 -1,
                                 false,
                                 true,
                                 false,
                                 false);
    const auto& row_desc =
        fixup_row_descriptor(result.row_set.row_desc, session_info.get_catalog());
    for (const auto& col_desc : row_desc) {
      const auto it_ok = _return.insert(std::make_pair(col_desc.col_name, col_desc));
      CHECK(it_ok.second);
    }
  } catch (std::exception& e) {
    THROW_MAPD_EXCEPTION(std::string("Exception: ") + e.what());
  }
}

static void check_privileges_on(const std::string& api_name) {
  if (!SysCatalog::instance().arePrivilegesOn()) {
    THROW_MAPD_EXCEPTION(api_name + " API requires database privileges turned on");
  }
}

void MapDHandler::get_roles(std::vector<std::string>& roles, const TSessionId& session) {
  check_privileges_on("get_roles");
  auto session_info = get_session(session);
  if (!session_info.get_currentUser().isSuper) {
    roles =
        SysCatalog::instance().getRoles(session_info.get_currentUser().userName,
                                        session_info.get_catalog().get_currentDB().dbId);
  } else {
    roles = SysCatalog::instance().getRoles(
        false, true, session_info.get_currentUser().userName);
  }
}

static TDBObject serialize_db_object(const std::string& roleName,
                                     const DBObject& inObject) {
  TDBObject outObject;
  outObject.objectName = inObject.getName();
  outObject.grantee = roleName;
  const auto ap = inObject.getPrivileges();
  switch (inObject.getObjectKey().permissionType) {
    case DatabaseDBObjectType:
      outObject.objectType = TDBObjectType::DatabaseDBObjectType;
      outObject.privs.push_back(ap.hasPermission(DatabasePrivileges::CREATE_DATABASE));
      outObject.privs.push_back(ap.hasPermission(DatabasePrivileges::DROP_DATABASE));
      outObject.privs.push_back(ap.hasPermission(DatabasePrivileges::VIEW_SQL_EDITOR));
      outObject.privs.push_back(ap.hasPermission(DatabasePrivileges::ACCESS));

      break;
    case TableDBObjectType:
      outObject.objectType = TDBObjectType::TableDBObjectType;
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
      outObject.objectType = TDBObjectType::DashboardDBObjectType;
      outObject.privs.push_back(ap.hasPermission(DashboardPrivileges::CREATE_DASHBOARD));
      outObject.privs.push_back(ap.hasPermission(DashboardPrivileges::DELETE_DASHBOARD));
      outObject.privs.push_back(ap.hasPermission(DashboardPrivileges::VIEW_DASHBOARD));
      outObject.privs.push_back(ap.hasPermission(DashboardPrivileges::EDIT_DASHBOARD));

      break;
    case ViewDBObjectType:
      outObject.objectType = TDBObjectType::ViewDBObjectType;
      outObject.privs.push_back(ap.hasPermission(ViewPrivileges::CREATE_VIEW));
      outObject.privs.push_back(ap.hasPermission(ViewPrivileges::DROP_VIEW));
      outObject.privs.push_back(ap.hasPermission(ViewPrivileges::SELECT_FROM_VIEW));
      outObject.privs.push_back(ap.hasPermission(ViewPrivileges::INSERT_INTO_VIEW));
      outObject.privs.push_back(ap.hasPermission(ViewPrivileges::UPDATE_IN_VIEW));
      outObject.privs.push_back(ap.hasPermission(ViewPrivileges::DELETE_FROM_VIEW));

      break;
    default:
      CHECK(false);
  }
  return outObject;
}

bool MapDHandler::has_database_permission(const AccessPrivileges& privs,
                                          const TDBObjectPermissions& permissions) {
  if (!permissions.__isset.database_permissions_) {
    THROW_MAPD_EXCEPTION("Database permissions not set for check.")
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

bool MapDHandler::has_table_permission(const AccessPrivileges& privs,
                                       const TDBObjectPermissions& permissions) {
  if (!permissions.__isset.table_permissions_) {
    THROW_MAPD_EXCEPTION("Table permissions not set for check.")
  }
  auto perms = permissions.table_permissions_;
  if ((perms.create_ && !privs.hasPermission(TablePrivileges::CREATE_TABLE)) ||
      (perms.drop_ && !privs.hasPermission(TablePrivileges::DROP_TABLE)) ||
      (perms.select_ && !privs.hasPermission(TablePrivileges::SELECT_FROM_TABLE)) ||
      (perms.insert_ && !privs.hasPermission(TablePrivileges::INSERT_INTO_TABLE)) ||
      (perms.update_ && !privs.hasPermission(TablePrivileges::UPDATE_IN_TABLE)) ||
      (perms.delete_ && !privs.hasPermission(TablePrivileges::DELETE_FROM_TABLE)) ||
      (perms.truncate_ && !privs.hasPermission(TablePrivileges::TRUNCATE_TABLE))) {
    return false;
  } else {
    return true;
  }
}

bool MapDHandler::has_dashboard_permission(const AccessPrivileges& privs,
                                           const TDBObjectPermissions& permissions) {
  if (!permissions.__isset.dashboard_permissions_) {
    THROW_MAPD_EXCEPTION("Dashboard permissions not set for check.")
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

bool MapDHandler::has_view_permission(const AccessPrivileges& privs,
                                      const TDBObjectPermissions& permissions) {
  if (!permissions.__isset.view_permissions_) {
    THROW_MAPD_EXCEPTION("View permissions not set for check.")
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

bool MapDHandler::has_object_privilege(const TSessionId& sessionId,
                                       const std::string& granteeName,
                                       const std::string& objectName,
                                       const TDBObjectType::type objectType,
                                       const TDBObjectPermissions& permissions) {
  check_privileges_on("has_object_privilege");
  auto session = get_session(sessionId);
  auto& cat = session.get_catalog();
  auto current_user = session.get_currentUser();
  if (!current_user.isSuper && !current_user.isReallySuper &&
      !SysCatalog::instance().isRoleGrantedToGrantee(
          current_user.userName, granteeName, false)) {
    THROW_MAPD_EXCEPTION(
        "Users except superusers can only check privileges for self or roles granted to "
        "them.")
  }
  Catalog_Namespace::UserMetadata user_meta;
  if (SysCatalog::instance().getMetadataForUser(granteeName, user_meta) &&
      (user_meta.isSuper || user_meta.isReallySuper)) {
    return true;
  }
  Grantee* grnt = SysCatalog::instance().getGrantee(granteeName);
  if (!grnt) {
    THROW_MAPD_EXCEPTION("User or Role " + granteeName + " does not exist.")
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
    default:
      THROW_MAPD_EXCEPTION("Invalid object type (" + std::to_string(objectType) + ").");
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

void MapDHandler::get_db_objects_for_grantee(std::vector<TDBObject>& TDBObjectsForRole,
                                             const TSessionId& sessionId,
                                             const std::string& roleName) {
  check_privileges_on("get_db_objects_for_grantee");
  auto session = get_session(sessionId);
  auto user = session.get_currentUser();
  if (!user.isSuper &&
      !SysCatalog::instance().isRoleGrantedToGrantee(user.userName, roleName, false)) {
    return;
  }
  auto* rl = SysCatalog::instance().getGrantee(roleName);
  if (rl) {
    auto dbId = session.get_catalog().get_currentDB().dbId;
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
    THROW_MAPD_EXCEPTION("User or role " + roleName + " does not exist.");
  }
}

void MapDHandler::get_db_object_privs(std::vector<TDBObject>& TDBObjects,
                                      const TSessionId& sessionId,
                                      const std::string& objectName,
                                      const TDBObjectType::type type) {
  check_privileges_on("get_db_object_privs");
  auto session = get_session(sessionId);
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
    default:
      THROW_MAPD_EXCEPTION("Failed to get object privileges for " + objectName +
                           ": unknown object type (" + std::to_string(type) + ").");
  }
  DBObject object_to_find(objectName, object_type);

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
      auto& cat = session.get_catalog();
      auto td = cat.getMetadataForTable(objectName, false);
      if (td) {
        object_type = td->isView ? ViewDBObjectType : TableDBObjectType;
        object_to_find = DBObject(objectName, object_type);
      }
    }
    object_to_find.loadKey(session.get_catalog());
  } catch (const std::exception&) {
    THROW_MAPD_EXCEPTION("Object with name " + objectName + " does not exist.");
  }

  // object type on database level
  DBObject object_to_find_dblevel("", object_type);
  object_to_find_dblevel.loadKey(session.get_catalog());
  // if user is superuser respond with a full priv
  if (session.get_currentUser().isSuper) {
    // using ALL_TABLE here to set max permissions
    DBObject dbObj{object_to_find.getObjectKey(),
                   AccessPrivileges::ALL_TABLE,
                   session.get_currentUser().userId};
    dbObj.setName("super");
    TDBObjects.push_back(serialize_db_object(session.get_currentUser().userName, dbObj));
  };

  std::vector<std::string> grantees = SysCatalog::instance().getRoles(
      true, session.get_currentUser().isSuper, session.get_currentUser().userName);
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

void MapDHandler::get_all_roles_for_user(std::vector<std::string>& roles,
                                         const TSessionId& sessionId,
                                         const std::string& granteeName) {
  check_privileges_on("get_all_roles_for_user");
  auto session = get_session(sessionId);
  auto* grantee = SysCatalog::instance().getGrantee(granteeName);
  if (grantee) {
    if (session.get_currentUser().isSuper) {
      roles = grantee->getRoles();
    } else if (grantee->isUser()) {
      if (session.get_currentUser().userName == granteeName) {
        roles = grantee->getRoles();
      } else {
        THROW_MAPD_EXCEPTION(
            "Only a superuser is authorized to request list of roles granted to another "
            "user.");
      }
    } else {
      CHECK(!grantee->isUser());
      // granteeName is actually a roleName here and we can check a role
      // only if it is granted to us
      if (SysCatalog::instance().isRoleGrantedToGrantee(
              session.get_currentUser().userName, granteeName, false)) {
        roles = grantee->getRoles();
      } else {
        THROW_MAPD_EXCEPTION("A user can check only roles granted to him.");
      }
    }
  } else {
    THROW_MAPD_EXCEPTION("Grantee " + granteeName + " does not exist.");
  }
}

std::string dump_table_col_names(
    const std::map<std::string, std::vector<std::string>>& table_col_names) {
  std::ostringstream oss;
  for (const auto table_col : table_col_names) {
    oss << ":" << table_col.first;
    for (const auto col : table_col.second) {
      oss << "," << col;
    }
  }
  return oss.str();
}

void MapDHandler::get_result_row_for_pixel(
    TPixelTableRowResult& _return,
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

  const auto session_it = get_session_it(session);
  LOG(INFO) << "get_result_row_for_pixel :" << *session_it->second
            << " :widget_id:" << widget_id << ":pixel.x:" << pixel.x
            << ":pixel.y:" << pixel.y << ":column_format:" << column_format
            << ":pixel_radius:" << pixel_radius << ":table_col_names"
            << dump_table_col_names(table_col_names) << ":nonce:" << nonce;

  auto time_ms = measure<>::execution([&]() {
    try {
      render_handler_->get_result_row_for_pixel(_return,
                                                *session_it->second,
                                                widget_id,
                                                pixel,
                                                table_col_names,
                                                column_format,
                                                pixel_radius,
                                                nonce);
    } catch (std::exception& e) {
      THROW_MAPD_EXCEPTION(std::string("Exception: ") + e.what());
    }
  });

  LOG(INFO) << "get_result_row_for_pixel-COMPLETED nonce: " << nonce
            << ", Execute Time: " << time_ms << " (ms)";
}

namespace {

inline void fixup_geo_column_descriptor(TColumnType& col_type,
                                        const SQLTypes subtype,
                                        const int output_srid) {
  col_type.col_type.precision = static_cast<int>(subtype);
  col_type.col_type.scale = output_srid;
}

}  // namespace

TColumnType MapDHandler::populateThriftColumnType(const Catalog* cat,
                                                  const ColumnDescriptor* cd) {
  TColumnType col_type;
  col_type.col_name = cd->columnName;
  col_type.src_name = cd->sourceName;
  col_type.col_type.type = type_to_thrift(cd->columnType);
  col_type.col_type.encoding = encoding_to_thrift(cd->columnType);
  col_type.col_type.nullable = !cd->columnType.get_notnull();
  col_type.col_type.is_array = cd->columnType.get_type() == kARRAY;
  if (col_type.col_type.is_array) {
    col_type.col_type.size = cd->columnType.get_size();  // only for arrays
  }
  if (IS_GEO(cd->columnType.get_type())) {
    fixup_geo_column_descriptor(
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
      THROW_MAPD_EXCEPTION("Dictionary doesn't exist");
    }
    col_type.col_type.comp_param = dd->dictNBits;
  } else {
    col_type.col_type.comp_param = cd->columnType.get_comp_param();
  }
  col_type.is_reserved_keyword = ImportHelpers::is_reserved_name(col_type.col_name);
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
  get_table_details_impl(_return, session, table_name, true, false);
}

void MapDHandler::get_table_details(TTableDetails& _return,
                                    const TSessionId& session,
                                    const std::string& table_name) {
  get_table_details_impl(_return, session, table_name, false, false);
}

void MapDHandler::get_table_details_impl(TTableDetails& _return,
                                         const TSessionId& session,
                                         const std::string& table_name,
                                         const bool get_system,
                                         const bool get_physical) {
  auto session_info = get_session(session);
  auto& cat = session_info.get_catalog();
  auto td = cat.getMetadataForTable(
      table_name,
      false);  // don't populate fragmenter on this call since we only want metadata
  if (!td) {
    THROW_MAPD_EXCEPTION("Table " + table_name + " doesn't exist");
  }
  if (td->isView) {
    try {
      if (!SysCatalog::instance().arePrivilegesOn() ||
          hasTableAccessPrivileges(td, session)) {
        session_info.make_superuser();
        const auto query_ra = parse_to_ra(td->viewSQL, {}, session_info);
        TQueryResult result;
        execute_rel_alg(result,
                        query_ra,
                        true,
                        session_info,
                        ExecutorDeviceType::CPU,
                        -1,
                        -1,
                        false,
                        true,
                        false,
                        false);
        _return.row_desc = fixup_row_descriptor(result.row_set.row_desc, cat);
      } else {
        THROW_MAPD_EXCEPTION("User has no access privileges to table " + table_name);
      }
    } catch (std::exception& e) {
      TColumnType tColumnType;
      tColumnType.col_name = "BROKEN_VIEW_PLEASE_FIX";
      _return.row_desc.push_back(tColumnType);
    }
  } else {
    try {
      if (!SysCatalog::instance().arePrivilegesOn() ||
          hasTableAccessPrivileges(td, session)) {
        const auto col_descriptors =
            cat.getAllColumnMetadataForTable(td->tableId, get_system, true, get_physical);
        const auto deleted_cd = cat.getDeletedColumn(td);
        for (const auto cd : col_descriptors) {
          if (cd == deleted_cd) {
            continue;
          }
          _return.row_desc.push_back(populateThriftColumnType(&cat, cd));
        }
      } else {
        THROW_MAPD_EXCEPTION("User has no access privileges to table " + table_name);
      }
    } catch (const std::runtime_error& e) {
      THROW_MAPD_EXCEPTION(e.what());
    }
  }
  _return.fragment_size = td->maxFragRows;
  _return.page_size = td->fragPageSize;
  _return.max_rows = td->maxRows;
  _return.view_sql = td->viewSQL;
  _return.shard_count = td->nShards;
  _return.key_metainfo = td->keyMetainfo;
  _return.is_temporary = td->persistenceLevel == Data_Namespace::MemoryLevel::CPU_LEVEL;
  _return.partition_detail =
      td->partitions.empty()
          ? TPartitionDetail::DEFAULT
          : (table_is_replicated(td)
                 ? TPartitionDetail::REPLICATED
                 : (td->partitions == "SHARDED" ? TPartitionDetail::SHARDED
                                                : TPartitionDetail::OTHER));
}

// DEPRECATED(2017-04-17) - use get_table_details()
void MapDHandler::get_row_descriptor(TRowDescriptor& _return,
                                     const TSessionId& session,
                                     const std::string& table_name) {
  LOG(ERROR) << "get_row_descriptor is deprecated, please fix application";
}

void MapDHandler::get_frontend_view(TFrontendView& _return,
                                    const TSessionId& session,
                                    const std::string& view_name) {
  const auto session_info = get_session(session);
  auto& cat = session_info.get_catalog();
  auto vd = cat.getMetadataForFrontendView(
      std::to_string(session_info.get_currentUser().userId), view_name);
  if (!vd) {
    THROW_MAPD_EXCEPTION("Dashboard " + view_name + " doesn't exist");
  }
  _return.view_name = view_name;
  _return.view_state = vd->viewState;
  _return.image_hash = vd->imageHash;
  _return.update_time = vd->updateTime;
  _return.view_metadata = vd->viewMetadata;
}

void MapDHandler::get_link_view(TFrontendView& _return,
                                const TSessionId& session,
                                const std::string& link) {
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

bool MapDHandler::hasTableAccessPrivileges(const TableDescriptor* td,
                                           const TSessionId& session) {
  const auto session_info = get_session(session);
  auto& cat = session_info.get_catalog();
  auto user_metadata = session_info.get_currentUser();

  if (user_metadata.isSuper) {
    return true;
  }

  DBObject dbObject(td->tableName, td->isView ? ViewDBObjectType : TableDBObjectType);
  dbObject.loadKey(cat);
  std::vector<DBObject> privObjects = {dbObject};

  return SysCatalog::instance().hasAnyPrivileges(user_metadata, privObjects);
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
    if (SysCatalog::instance().arePrivilegesOn() &&
        !hasTableAccessPrivileges(td, session)) {
      // skip table, as there are no privileges to access it
      continue;
    }
    table_names.push_back(td->tableName);
  }
}

void MapDHandler::get_tables(std::vector<std::string>& table_names,
                             const TSessionId& session) {
  get_tables_impl(table_names, session, GET_PHYSICAL_TABLES_AND_VIEWS);
}

void MapDHandler::get_physical_tables(std::vector<std::string>& table_names,
                                      const TSessionId& session) {
  get_tables_impl(table_names, session, GET_PHYSICAL_TABLES);
}

void MapDHandler::get_views(std::vector<std::string>& table_names,
                            const TSessionId& session) {
  get_tables_impl(table_names, session, GET_VIEWS);
}

void MapDHandler::get_tables_meta(std::vector<TTableMeta>& _return,
                                  const TSessionId& session) {
  const auto session_info = get_session(session);
  auto& cat = session_info.get_catalog();
  const auto tables = cat.getAllTableMetadata();
  _return.reserve(tables.size());

  for (const auto td : tables) {
    if (td->shard >= 0) {
      // skip shards, they're not standalone tables
      continue;
    }
    if (SysCatalog::instance().arePrivilegesOn() &&
        !hasTableAccessPrivileges(td, session)) {
      // skip table, as there are no privileges to access it
      continue;
    }

    TTableMeta ret;
    ret.table_name = td->tableName;
    ret.is_view = td->isView;
    ret.is_replicated = table_is_replicated(td);
    ret.shard_count = td->nShards;
    ret.max_rows = td->maxRows;

    std::set<TDatumType::type> col_datum_types;
    size_t num_cols = 0;
    if (td->isView) {
      try {
        const auto query_ra = parse_to_ra(td->viewSQL, {}, session_info);
        TQueryResult result;
        execute_rel_alg(result,
                        query_ra,
                        true,
                        session_info,
                        ExecutorDeviceType::CPU,
                        -1,
                        -1,
                        false,
                        true,
                        false,
                        false);
        num_cols = result.row_set.row_desc.size();
        for (const auto col : result.row_set.row_desc) {
          if (col.is_physical) {
            num_cols--;
            continue;
          }
          col_datum_types.insert(col.col_type.type);
        }
      } catch (std::exception& e) {
        LOG(WARNING) << "get_tables_meta: Ignoring broken view: " << td->tableName;
      }
    } else {
      try {
        if (!SysCatalog::instance().arePrivilegesOn() ||
            hasTableAccessPrivileges(td, session)) {
          const auto col_descriptors =
              cat.getAllColumnMetadataForTable(td->tableId, false, true, false);
          const auto deleted_cd = cat.getDeletedColumn(td);
          for (const auto cd : col_descriptors) {
            if (cd == deleted_cd) {
              continue;
            }
            col_datum_types.insert(type_to_thrift(cd->columnType));
          }
          num_cols = col_descriptors.size();
        } else {
          continue;
        }
      } catch (const std::runtime_error& e) {
        THROW_MAPD_EXCEPTION(e.what());
      }
    }

    ret.num_cols = num_cols;
    std::copy(col_datum_types.begin(),
              col_datum_types.end(),
              std::back_inserter(ret.col_datum_types));

    _return.push_back(ret);
  }
}

void MapDHandler::get_users(std::vector<std::string>& user_names,
                            const TSessionId& session) {
  std::list<Catalog_Namespace::UserMetadata> user_list;
  const auto session_info = get_session(session);

  if (SysCatalog::instance().arePrivilegesOn() &&
      !session_info.get_currentUser().isSuper) {
    user_list = SysCatalog::instance().getAllUserMetadata(
        session_info.get_catalog().get_currentDB().dbId);
  } else {
    user_list = SysCatalog::instance().getAllUserMetadata();
  }
  for (auto u : user_list) {
    user_names.push_back(u.userName);
  }
}

void MapDHandler::get_version(std::string& version) {
  version = MAPD_RELEASE;
}

void MapDHandler::clear_gpu_memory(const TSessionId& session) {
  const auto session_info = get_session(session);
  try {
    SysCatalog::instance().get_dataMgr().clearMemory(MemoryLevel::GPU_LEVEL);
  } catch (const std::exception& e) {
    THROW_MAPD_EXCEPTION(e.what());
  }
  if (render_handler_) {
    render_handler_->clear_gpu_memory();
  }

  if (leaf_aggregator_.leafCount() > 0) {
    leaf_aggregator_.clear_leaf_gpu_memory(session);
  }
}

void MapDHandler::clear_cpu_memory(const TSessionId& session) {
  const auto session_info = get_session(session);
  SysCatalog::instance().get_dataMgr().clearMemory(MemoryLevel::CPU_LEVEL);
  if (render_handler_) {
    render_handler_->clear_cpu_memory();
  }

  if (leaf_aggregator_.leafCount() > 0) {
    leaf_aggregator_.clear_leaf_cpu_memory(session);
  }
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
    internal_memory =
        SysCatalog::instance().get_dataMgr().getMemoryInfo(MemoryLevel::GPU_LEVEL);
  } else {
    mem_level = Data_Namespace::MemoryLevel::CPU_LEVEL;
    internal_memory =
        SysCatalog::instance().get_dataMgr().getMemoryInfo(MemoryLevel::CPU_LEVEL);
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
    std::vector<TNodeMemoryInfo> leafSummary =
        leaf_aggregator_.getLeafMemoryInfo(session, mem_level);
    _return.insert(_return.begin(), leafSummary.begin(), leafSummary.end());
  }
}

void MapDHandler::get_databases(std::vector<TDBInfo>& dbinfos,
                                const TSessionId& session) {
  const auto session_info = get_session(session);
  if (SysCatalog::instance().arePrivilegesOn() &&
      !session_info.get_currentUser().isSuper) {
    THROW_MAPD_EXCEPTION("Only a superuser is authorized to get list of databases.");
  }
  std::list<Catalog_Namespace::DBMetadata> db_list =
      SysCatalog::instance().getAllDBMetadata();
  std::list<Catalog_Namespace::UserMetadata> user_list =
      SysCatalog::instance().getAllUserMetadata();
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

void MapDHandler::get_frontend_views(std::vector<TFrontendView>& view_names,
                                     const TSessionId& session) {
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

void MapDHandler::set_execution_mode(const TSessionId& session,
                                     const TExecuteMode::type mode) {
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

void check_table_not_sharded(const Catalog& cat, const std::string& table_name) {
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
  check_table_load_privileges(session_info, table_name);
  if (rows.empty()) {
    return;
  }
  std::unique_ptr<Importer_NS::Loader> loader;
  if (leaf_aggregator_.leafCount() > 0) {
    loader.reset(new DistributedLoader(session_info, td, &leaf_aggregator_));
  } else {
    loader.reset(new Importer_NS::Loader(cat, td));
  }
  // TODO(andrew): nColumns should be number of non-virtual/non-system columns.
  //               Subtracting 1 (rowid) until TableDescriptor is updated.
  if (rows.front().cols.size() !=
      static_cast<size_t>(td->nColumns) - (td->hasDeletedCol ? 2 : 1)) {
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
        import_buffers[col_idx]->add_value(
            cd, row.cols[col_idx], row.cols[col_idx].is_null);
        col_idx++;
      }
    } catch (const std::exception& e) {
      for (size_t col_idx_to_pop = 0; col_idx_to_pop < col_idx; ++col_idx_to_pop) {
        import_buffers[col_idx_to_pop]->pop_value();
      }
      LOG(ERROR) << "Input exception thrown: " << e.what()
                 << ". Row discarded, issue at column : " << (col_idx + 1)
                 << " data :" << row;
    }
  }
  loader->load(import_buffers, rows.size());
}

void MapDHandler::prepare_columnar_loader(
    const Catalog_Namespace::SessionInfo& session_info,
    const std::string& table_name,
    size_t num_cols,
    std::unique_ptr<Importer_NS::Loader>* loader,
    std::vector<std::unique_ptr<Importer_NS::TypedImportBuffer>>* import_buffers) {
  auto& cat = session_info.get_catalog();
  if (g_cluster && !leaf_aggregator_.leafCount()) {
    // Sharded table rows need to be routed to the leaf by an aggregator.
    check_table_not_sharded(cat, table_name);
  }
  const TableDescriptor* td = cat.getMetadataForTable(table_name);
  if (td == nullptr) {
    THROW_MAPD_EXCEPTION("Table " + table_name + " does not exist.");
  }
  check_table_load_privileges(session_info, table_name);
  if (leaf_aggregator_.leafCount() > 0) {
    loader->reset(new DistributedLoader(session_info, td, &leaf_aggregator_));
  } else {
    loader->reset(new Importer_NS::Loader(cat, td));
  }
  // TODO(andrew): nColumns should be number of non-virtual/non-system columns.
  //               Subtracting 1 (rowid) until TableDescriptor is updated.
  if (num_cols != static_cast<size_t>(td->nColumns) - (td->hasDeletedCol ? 2 : 1) ||
      num_cols < 1) {
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
  const auto session_info = get_session(session);
  prepare_columnar_loader(
      session_info, table_name, cols.size(), &loader, &import_buffers);

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
          oss << "load_table_binary_columnar: Inconsistent number of rows in request,  "
                 "expecting "
              << numRows << " row, column " << col_idx << " has " << colRows << " rows";
          THROW_MAPD_EXCEPTION(oss.str());
        }
      }
      col_idx++;
    }
  } catch (const std::exception& e) {
    std::ostringstream oss;
    oss << "load_table_binary_columnar: Input exception thrown: " << e.what()
        << ". Issue at column : " << (col_idx + 1) << ". Import aborted";
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
    auto stream_buffer =
        std::make_shared<arrow::Buffer>(reinterpret_cast<const uint8_t*>(stream.c_str()),
                                        static_cast<int64_t>(stream.size()));

    arrow::io::BufferReader buf_reader(stream_buffer);
    std::shared_ptr<arrow::RecordBatchReader> batch_reader;
    ARROW_THRIFT_THROW_NOT_OK(
        arrow::ipc::RecordBatchStreamReader::Open(&buf_reader, &batch_reader));

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
  const auto session_info = get_session(session);
  prepare_columnar_loader(session_info,
                          table_name,
                          static_cast<size_t>(batch->num_columns()),
                          &loader,
                          &import_buffers);

  size_t numRows = 0;
  size_t col_idx = 0;
  try {
    for (auto cd : loader->get_column_descs()) {
      numRows = import_buffers[col_idx]->add_arrow_values(cd, *batch->column(col_idx));
      col_idx++;
    }
  } catch (const std::exception& e) {
    LOG(ERROR) << "Input exception thrown: " << e.what()
               << ". Issue at column : " << (col_idx + 1) << ". Import aborted";
    // TODO(tmostak): Go row-wise on binary columnar import to be consistent with our
    // other import paths
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
  check_table_load_privileges(session_info, table_name);
  if (rows.empty()) {
    return;
  }
  std::unique_ptr<Importer_NS::Loader> loader;
  if (leaf_aggregator_.leafCount() > 0) {
    loader.reset(new DistributedLoader(session_info, td, &leaf_aggregator_));
  } else {
    loader.reset(new Importer_NS::Loader(cat, td));
  }
  auto col_descs = loader->get_column_descs();
  auto geo_physical_cols = std::count_if(
      col_descs.begin(), col_descs.end(), [](auto cd) { return cd->isGeoPhyCol; });
  // TODO(andrew): nColumns should be number of non-virtual/non-system columns.
  //               Subtracting 1 (rowid) until TableDescriptor is updated.
  if (rows.front().cols.size() != static_cast<size_t>(td->nColumns) - geo_physical_cols -
                                      (td->hasDeletedCol ? 2 : 1)) {
    THROW_MAPD_EXCEPTION("Wrong number of columns to load into Table " + table_name +
                         " (" + std::to_string(rows.front().cols.size()) + " vs " +
                         std::to_string(td->nColumns - geo_physical_cols - 1) + ")");
  }
  std::vector<std::unique_ptr<Importer_NS::TypedImportBuffer>> import_buffers;
  for (auto cd : col_descs) {
    import_buffers.push_back(std::unique_ptr<Importer_NS::TypedImportBuffer>(
        new Importer_NS::TypedImportBuffer(cd, loader->get_string_dict(cd))));
  }
  Importer_NS::CopyParams copy_params;
  size_t rows_completed = 0;
  for (auto const& row : rows) {
    size_t import_idx = 0;  // index into the TStringRow being loaded
    size_t col_idx = 0;     // index into column description vector
    try {
      size_t skip_physical_cols = 0;
      for (auto cd : col_descs) {
        if (skip_physical_cols > 0) {
          if (!cd->isGeoPhyCol) {
            throw std::runtime_error("Unexpected physical column");
          }
          skip_physical_cols--;
          continue;
        }
        import_buffers[col_idx]->add_value(
            cd, row.cols[import_idx].str_val, row.cols[import_idx].is_null, copy_params);
        // Advance to the next column within the table
        col_idx++;

        if (cd->columnType.is_geometry()) {
          // Populate physical columns
          std::vector<double> coords, bounds;
          std::vector<int> ring_sizes, poly_rings;
          int render_group = 0;
          SQLTypeInfo ti;
          if (row.cols[import_idx].is_null ||
              !Geo_namespace::GeoTypesFactory::getGeoColumns(row.cols[import_idx].str_val,
                                                             ti,
                                                             coords,
                                                             bounds,
                                                             ring_sizes,
                                                             poly_rings,
                                                             false)) {
            throw std::runtime_error("Invalid geometry");
          }
          if (cd->columnType.get_type() != ti.get_type()) {
            throw std::runtime_error("Geometry type mismatch");
          }
          Importer_NS::Importer::set_geo_physical_import_buffer(cat,
                                                                cd,
                                                                import_buffers,
                                                                col_idx,
                                                                coords,
                                                                bounds,
                                                                ring_sizes,
                                                                poly_rings,
                                                                render_group);
          skip_physical_cols = cd->columnType.get_physical_cols();
        }
        // Advance to the next field within the row
        import_idx++;
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
  loader->load(import_buffers, rows_completed);
}

char MapDHandler::unescape_char(std::string str) {
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

Importer_NS::CopyParams MapDHandler::thrift_to_copyparams(const TCopyParams& cp) {
  Importer_NS::CopyParams copy_params;
  copy_params.has_header = cp.has_header;
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
  if (cp.s3_region.length() > 0) {
    copy_params.s3_region = cp.s3_region;
  }
  switch (cp.table_type) {
    case TTableType::POLYGON:
      copy_params.table_type = Importer_NS::TableType::POLYGON;
      break;
    case TTableType::DELIMITED:
      copy_params.table_type = Importer_NS::TableType::DELIMITED;
      break;
    default:
      THROW_MAPD_EXCEPTION("Invalid table_type in CopyParams: " +
                           std::to_string((int)cp.table_type));
      break;
  }
  switch (cp.geo_coords_encoding) {
    case TEncodingType::GEOINT:
      copy_params.geo_coords_encoding = kENCODING_GEOINT;
      break;
    case TEncodingType::NONE:
      copy_params.geo_coords_encoding = kENCODING_NONE;
      break;
    default:
      THROW_MAPD_EXCEPTION("Invalid geo_coords_encoding in CopyParams: " +
                           std::to_string((int)cp.geo_coords_encoding));
      break;
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
      THROW_MAPD_EXCEPTION("Invalid geo_coords_type in CopyParams: " +
                           std::to_string((int)cp.geo_coords_type));
      break;
  }
  switch (cp.geo_coords_srid) {
    case 4326:
    case 3857:
    case 900913:
      copy_params.geo_coords_srid = cp.geo_coords_srid;
      break;
    default:
      THROW_MAPD_EXCEPTION("Invalid geo_coords_srid in CopyParams (" +
                           std::to_string((int)cp.geo_coords_srid));
      break;
  }
  copy_params.sanitize_column_names = cp.sanitize_column_names;
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
  copy_params.s3_access_key = cp.s3_access_key;
  copy_params.s3_secret_key = cp.s3_secret_key;
  copy_params.s3_region = cp.s3_region;
  switch (cp.table_type) {
    case Importer_NS::TableType::POLYGON:
      copy_params.table_type = TTableType::POLYGON;
      break;
    default:
      copy_params.table_type = TTableType::DELIMITED;
      break;
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
      break;
  }
  copy_params.geo_coords_srid = cp.geo_coords_srid;
  copy_params.sanitize_column_names = cp.sanitize_column_names;
  return copy_params;
}

void add_vsi_network_prefix(std::string& path) {
  // do we support network file access?
  bool gdal_network = Importer_NS::Importer::gdalSupportsNetworkFileAccess();

  // modify head of filename based on source location
  if (boost::istarts_with(path, "http://") || boost::istarts_with(path, "https://")) {
    if (!gdal_network) {
      THROW_MAPD_EXCEPTION(
          "HTTP geo file import not supported! Update to GDAL 2.2 or later!");
    }
    // invoke GDAL CURL virtual file reader
    path = "/vsicurl/" + path;
  } else if (boost::istarts_with(path, "s3://")) {
    if (!gdal_network) {
      THROW_MAPD_EXCEPTION(
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

bool is_a_supported_geo_file(const std::string& path, bool include_gz) {
  if (include_gz) {
    if (boost::iends_with(path, ".geojson.gz") || boost::iends_with(path, ".json.gz")) {
      return true;
    }
  }
  if (boost::iends_with(path, ".shp") || boost::iends_with(path, ".geojson") ||
      boost::iends_with(path, ".json") || boost::iends_with(path, ".kml") ||
      boost::iends_with(path, ".kmz")) {
    return true;
  }
  return false;
}

bool is_a_supported_archive_file(const std::string& path) {
  if (boost::iends_with(path, ".zip")) {
    return true;
  } else if (boost::iends_with(path, ".tar") || boost::iends_with(path, ".tgz") ||
             boost::iends_with(path, ".tar.gz")) {
    return true;
  }
  return false;
}

std::string find_first_geo_file_in_archive(const std::string& archive_path,
                                           const Importer_NS::CopyParams& copy_params) {
  // get the recursive list of all files in the archive
  std::vector<std::string> files =
      Importer_NS::Importer::gdalGetAllFilesInArchive(archive_path, copy_params);

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
    if (is_a_supported_geo_file(file, false)) {
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

void MapDHandler::detect_column_types(TDetectResult& _return,
                                      const TSessionId& session,
                                      const std::string& file_name_in,
                                      const TCopyParams& cp) {
  check_read_only("detect_column_types");
  get_session(session);

  Importer_NS::CopyParams copy_params = thrift_to_copyparams(cp);

  std::string file_name{file_name_in};

  if (path_is_relative(file_name)) {
    // assume relative paths are relative to data_path / mapd_import / <session>
    auto file_path =
        import_path_ / session / boost::filesystem::path(file_name).filename();
    file_name = file_path.string();
  }

  // if it's a geo table, handle alternative paths (S3, HTTP, archive etc.)
  if (copy_params.table_type == Importer_NS::TableType::POLYGON) {
    if (is_a_supported_geo_file(file_name, true)) {
      // prepare to detect geo file directly
      add_vsi_network_prefix(file_name);
      add_vsi_geo_prefix(file_name);
    } else if (is_a_supported_archive_file(file_name)) {
      // find the archive file
      add_vsi_network_prefix(file_name);
      if (!Importer_NS::Importer::gdalFileExists(file_name, copy_params)) {
        THROW_MAPD_EXCEPTION("Archive does not exist: " + file_name_in);
      }
      // find geo file in archive
      add_vsi_archive_prefix(file_name);
      std::string geo_file = find_first_geo_file_in_archive(file_name, copy_params);
      // prepare to detect that geo file
      if (geo_file.size()) {
        file_name = file_name + std::string("/") + geo_file;
      }
    } else {
      THROW_MAPD_EXCEPTION("File is not a supported geo or geo archive format: " +
                           file_name_in);
    }
  }

  auto file_path = boost::filesystem::path(file_name);
  // can be a s3 url
  if (!boost::istarts_with(file_name, "s3://")) {
    if (!boost::filesystem::path(file_name).is_absolute()) {
      file_path = import_path_ / session / boost::filesystem::path(file_name).filename();
      file_name = file_path.string();
    }

    if (copy_params.table_type == Importer_NS::TableType::POLYGON) {
      // check for geo file
      if (!Importer_NS::Importer::gdalFileExists(file_name, copy_params)) {
        THROW_MAPD_EXCEPTION("File does not exist: " + file_path.string());
      }
    } else {
      // check for regular file
      if (!boost::filesystem::exists(file_path)) {
        THROW_MAPD_EXCEPTION("File does not exist: " + file_path.string());
      }
    }
  }
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
        if (IS_GEO(t)) {
          // set this so encoding_to_thrift does the right thing
          ti.set_compression(copy_params.geo_coords_encoding);
          // fill in these directly
          col.col_type.precision = static_cast<int>(copy_params.geo_coords_type);
          col.col_type.scale = copy_params.geo_coords_srid;
          col.col_type.comp_param = copy_params.geo_coords_comp_param;
        }
        col.col_type.type = type_to_thrift(ti);
        col.col_type.encoding = encoding_to_thrift(ti);
        if (copy_params.sanitize_column_names) {
          col.col_name = ImportHelpers::sanitize_name(headers[col_idx]);
        } else {
          col.col_name = headers[col_idx];
        }
        col.is_reserved_keyword = ImportHelpers::is_reserved_name(col.col_name);
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
      // @TODO simon.eves get this from somewhere!
      const std::string geoColumnName(OMNISCI_GEO_PREFIX);

      check_geospatial_files(file_path, copy_params);
      std::list<ColumnDescriptor> cds = Importer_NS::Importer::gdalToColumnDescriptors(
          file_path.string(), geoColumnName, copy_params);
      for (auto cd : cds) {
        if (copy_params.sanitize_column_names) {
          cd.columnName = ImportHelpers::sanitize_name(cd.columnName);
        }
        _return.row_set.row_desc.push_back(populateThriftColumnType(nullptr, &cd));
      }
      std::map<std::string, std::vector<std::string>> sample_data;
      Importer_NS::Importer::readMetadataSampleGDAL(
          file_path.string(), geoColumnName, sample_data, 100, copy_params);
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
      _return.copy_params = copyparams_to_thrift(copy_params);
    }
  } catch (const std::exception& e) {
    THROW_MAPD_EXCEPTION("detect_column_types error: " + std::string(e.what()));
  }
}

Planner::RootPlan* MapDHandler::parse_to_plan_legacy(
    const std::string& query_str,
    const Catalog_Namespace::SessionInfo& session_info,
    const std::string& action /* render or validate */) {
  auto& cat = session_info.get_catalog();
  LOG(INFO) << action << ": " << hide_sensitive_data(query_str);
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

  const auto session_it = get_session_it(session);
  LOG(INFO) << "render_vega :" << *session_it->second << " :widget_id:" << widget_id
            << ":compression_level:" << compression_level << ":vega_json:" << vega_json
            << ":nonce:" << nonce;

  _return.total_time_ms = measure<>::execution([&]() {
    try {
      render_handler_->render_vega(
          _return, *session_it->second, widget_id, vega_json, compression_level, nonce);
    } catch (std::exception& e) {
      THROW_MAPD_EXCEPTION(std::string("Exception: ") + e.what());
    }
  });
  LOG(INFO) << "render_vega-COMPLETED nonce: " << nonce
            << " Total: " << _return.total_time_ms
            << " (ms), Total Execution: " << _return.execution_time_ms
            << " (ms), Total Render: " << _return.render_time_ms << " (ms)";
}

static bool is_allowed_on_dashboard(const Catalog_Namespace::SessionInfo& session_info,
                                    int32_t dashboard_id,
                                    AccessPrivileges requestedPermissions) {
  DBObject object(dashboard_id, DashboardDBObjectType);
  auto& catalog = session_info.get_catalog();
  auto& user = session_info.get_currentUser();
  object.loadKey(catalog);
  object.setPrivileges(requestedPermissions);
  std::vector<DBObject> privs = {object};
  return (!SysCatalog::instance().arePrivilegesOn() ||
          SysCatalog::instance().checkPrivileges(user, privs));
}

// dashboards
void MapDHandler::get_dashboard(TDashboard& dashboard,
                                const TSessionId& session,
                                const int32_t dashboard_id) {
  const auto session_info = get_session(session);
  auto& cat = session_info.get_catalog();
  Catalog_Namespace::UserMetadata user_meta;
  auto dash = cat.getMetadataForDashboard(dashboard_id);
  if (!dash) {
    THROW_MAPD_EXCEPTION("Dashboard with dashboard id " + std::to_string(dashboard_id) +
                         " doesn't exist");
  }
  if (!is_allowed_on_dashboard(
          session_info, dash->viewId, AccessPrivileges::VIEW_DASHBOARD)) {
    THROW_MAPD_EXCEPTION("User has no view privileges for the dashboard with id " +
                         std::to_string(dashboard_id));
  }
  user_meta.userName = "";
  SysCatalog::instance().getMetadataForUserById(dash->userId, user_meta);
  auto objects_list = SysCatalog::instance().getMetadataForObject(
      cat.get_currentDB().dbId,
      static_cast<int>(DBObjectType::DashboardDBObjectType),
      dashboard_id);
  dashboard.dashboard_name = dash->viewName;
  dashboard.dashboard_state = dash->viewState;
  dashboard.image_hash = dash->imageHash;
  dashboard.update_time = dash->updateTime;
  dashboard.dashboard_metadata = dash->viewMetadata;
  dashboard.dashboard_owner = dash->user;
  dashboard.dashboard_id = dash->viewId;
  if (objects_list.empty() ||
      (objects_list.size() == 1 && objects_list[0]->roleName == user_meta.userName)) {
    dashboard.is_dash_shared = false;
  } else {
    dashboard.is_dash_shared = true;
  }
}

void MapDHandler::get_dashboards(std::vector<TDashboard>& dashboards,
                                 const TSessionId& session) {
  const auto session_info = get_session(session);
  auto& cat = session_info.get_catalog();
  Catalog_Namespace::UserMetadata user_meta;
  const auto dashes = cat.getAllFrontendViewMetadata();
  user_meta.userName = "";
  for (const auto d : dashes) {
    SysCatalog::instance().getMetadataForUserById(d->userId, user_meta);
    if (is_allowed_on_dashboard(
            session_info, d->viewId, AccessPrivileges::VIEW_DASHBOARD)) {
      auto objects_list = SysCatalog::instance().getMetadataForObject(
          cat.get_currentDB().dbId,
          static_cast<int>(DBObjectType::DashboardDBObjectType),
          d->viewId);
      TDashboard dash;
      dash.dashboard_name = d->viewName;
      dash.image_hash = d->imageHash;
      dash.update_time = d->updateTime;
      dash.dashboard_metadata = d->viewMetadata;
      dash.dashboard_id = d->viewId;
      dash.dashboard_owner = d->user;
      // dash.is_dash_shared = !objects_list.empty();
      if (objects_list.empty() ||
          (objects_list.size() == 1 && objects_list[0]->roleName == user_meta.userName)) {
        dash.is_dash_shared = false;
      } else {
        dash.is_dash_shared = true;
      }
      dashboards.push_back(dash);
    }
  }
}

int32_t MapDHandler::create_dashboard(const TSessionId& session,
                                      const std::string& dashboard_name,
                                      const std::string& dashboard_state,
                                      const std::string& image_hash,
                                      const std::string& dashboard_metadata) {
  check_read_only("create_dashboard");
  const auto session_info = get_session(session);
  auto& cat = session_info.get_catalog();

  if (SysCatalog::instance().arePrivilegesOn() &&
      !session_info.checkDBAccessPrivileges(DBObjectType::DashboardDBObjectType,
                                            AccessPrivileges::CREATE_DASHBOARD)) {
    THROW_MAPD_EXCEPTION("Not enough privileges to create a dashboard.");
  }

  auto dash = cat.getMetadataForFrontendView(
      std::to_string(session_info.get_currentUser().userId), dashboard_name);
  if (dash) {
    THROW_MAPD_EXCEPTION("Dashboard with name: " + dashboard_name + " already exists.");
  }

  FrontendViewDescriptor vd;
  vd.viewName = dashboard_name;
  vd.viewState = dashboard_state;
  vd.imageHash = image_hash;
  vd.viewMetadata = dashboard_metadata;
  vd.userId = session_info.get_currentUser().userId;
  vd.user = session_info.get_currentUser().userName;

  try {
    auto id = cat.createFrontendView(vd);
    // TODO: transactionally unsafe
    SysCatalog::instance().createDBObject(
        session_info.get_currentUser(), dashboard_name, DashboardDBObjectType, cat, id);
    return id;
  } catch (const std::exception& e) {
    THROW_MAPD_EXCEPTION(std::string("Exception: ") + e.what());
  }
}

void MapDHandler::replace_dashboard(const TSessionId& session,
                                    const int32_t dashboard_id,
                                    const std::string& dashboard_name,
                                    const std::string& dashboard_owner,
                                    const std::string& dashboard_state,
                                    const std::string& image_hash,
                                    const std::string& dashboard_metadata) {
  check_read_only("replace_dashboard");
  const auto session_info = get_session(session);
  auto& cat = session_info.get_catalog();

  if (!is_allowed_on_dashboard(
          session_info, dashboard_id, AccessPrivileges::EDIT_DASHBOARD)) {
    THROW_MAPD_EXCEPTION("Not enough privileges to replace a dashboard.");
  }

  FrontendViewDescriptor vd;
  vd.viewName = dashboard_name;
  vd.viewState = dashboard_state;
  vd.imageHash = image_hash;
  vd.viewMetadata = dashboard_metadata;
  Catalog_Namespace::UserMetadata user;
  if (!SysCatalog::instance().getMetadataForUser(dashboard_owner, user)) {
    THROW_MAPD_EXCEPTION(std::string("Dashboard owner ") + dashboard_owner +
                         " does not exist");
  }
  vd.userId = user.userId;
  vd.user = dashboard_owner;
  vd.viewId = dashboard_id;

  try {
    cat.replaceDashboard(vd);
  } catch (const std::exception& e) {
    THROW_MAPD_EXCEPTION(std::string("Exception: ") + e.what());
  }
}

void MapDHandler::delete_dashboard(const TSessionId& session,
                                   const int32_t dashboard_id) {
  check_read_only("delete_dashboard");
  const auto session_info = get_session(session);
  auto& cat = session_info.get_catalog();
  auto dash = cat.getMetadataForDashboard(dashboard_id);
  if (!dash) {
    THROW_MAPD_EXCEPTION("Dashboard with id" + std::to_string(dashboard_id) +
                         " doesn't exist, so cannot delete it");
  }
  if (SysCatalog::instance().arePrivilegesOn()) {
    if (!is_allowed_on_dashboard(
            session_info, dash->viewId, AccessPrivileges::DELETE_DASHBOARD)) {
      THROW_MAPD_EXCEPTION("Not enough privileges to delete a dashboard.");
    }
  }
  try {
    cat.deleteMetadataForDashboard(dashboard_id);
  } catch (const std::exception& e) {
    THROW_MAPD_EXCEPTION(std::string("Exception: ") + e.what());
  }
}

std::vector<std::string> MapDHandler::get_valid_groups(const TSessionId& session,
                                                       int32_t dashboard_id,
                                                       std::vector<std::string> groups) {
  const auto session_info = get_session(session);
  auto& cat = session_info.get_catalog();
  auto dash = cat.getMetadataForDashboard(dashboard_id);
  if (!dash) {
    THROW_MAPD_EXCEPTION("Exception: Dashboard id " + std::to_string(dashboard_id) +
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
      THROW_MAPD_EXCEPTION("Exception: User/Role " + group + " does not exist");
    } else if (!user_meta.isSuper) {
      valid_groups.push_back(group);
    }
  }
  return valid_groups;
}

// NOOP: Grants not available for objects as of now
void MapDHandler::share_dashboard(const TSessionId& session,
                                  const int32_t dashboard_id,
                                  const std::vector<std::string>& groups,
                                  const std::vector<std::string>& objects,
                                  const TDashboardPermissions& permissions) {
  check_read_only("share_dashboard");
  std::vector<std::string> valid_groups;
  valid_groups = get_valid_groups(session, dashboard_id, groups);
  const auto session_info = get_session(session);
  auto& cat = session_info.get_catalog();
  // By default object type can only be dashboard
  DBObjectType object_type = DBObjectType::DashboardDBObjectType;
  DBObject object(dashboard_id, object_type);
  if (!permissions.create_ && !permissions.delete_ && !permissions.edit_ &&
      !permissions.view_) {
    THROW_MAPD_EXCEPTION("Atleast one privilege should be assigned for grants");
  } else {
    AccessPrivileges privs;

    object.resetPrivileges();
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
  }
  SysCatalog::instance().grantDBObjectPrivilegesBatch(valid_groups, {object}, cat);
}

void MapDHandler::unshare_dashboard(const TSessionId& session,
                                    const int32_t dashboard_id,
                                    const std::vector<std::string>& groups,
                                    const std::vector<std::string>& objects,
                                    const TDashboardPermissions& permissions) {
  check_read_only("unshare_dashboard");
  std::vector<std::string> valid_groups;
  valid_groups = get_valid_groups(session, dashboard_id, groups);
  const auto session_info = get_session(session);
  auto& cat = session_info.get_catalog();
  // By default object type can only be dashboard
  DBObjectType object_type = DBObjectType::DashboardDBObjectType;
  DBObject object(dashboard_id, object_type);
  if (!permissions.create_ && !permissions.delete_ && !permissions.edit_ &&
      !permissions.view_) {
    THROW_MAPD_EXCEPTION("Atleast one privilege should be assigned for revokes");
  } else {
    AccessPrivileges privs;

    object.resetPrivileges();
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
  }
  SysCatalog::instance().revokeDBObjectPrivilegesBatch(valid_groups, {object}, cat);
}

void MapDHandler::get_dashboard_grantees(
    std::vector<TDashboardGrantees>& dashboard_grantees,
    const TSessionId& session,
    int32_t dashboard_id) {
  const auto session_info = get_session(session);
  auto& cat = session_info.get_catalog();
  Catalog_Namespace::UserMetadata user_meta;
  auto dash = cat.getMetadataForDashboard(dashboard_id);
  if (!dash) {
    THROW_MAPD_EXCEPTION("Exception: Dashboard id " + std::to_string(dashboard_id) +
                         " does not exist");
  } else if (session_info.get_currentUser().userId != dash->userId &&
             !session_info.get_currentUser().isSuper) {
    THROW_MAPD_EXCEPTION(
        "User should be either owner of dashboard or super user to access grantees");
  }
  std::vector<ObjectRoleDescriptor*> objectsList;
  objectsList = SysCatalog::instance().getMetadataForObject(
      cat.get_currentDB().dbId,
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
  vd.user = session_info.get_currentUser().userName;

  try {
    auto id = cat.createFrontendView(vd);

    if (SysCatalog::instance().arePrivilegesOn()) {
      SysCatalog::instance().createDBObject(
          session_info.get_currentUser(), view_name, DashboardDBObjectType, cat, id);
    }
  } catch (const std::exception& e) {
    THROW_MAPD_EXCEPTION(std::string("Exception: ") + e.what());
  }
}

void MapDHandler::delete_frontend_view(const TSessionId& session,
                                       const std::string& view_name) {
  check_read_only("delete_frontend_view");
  const auto session_info = get_session(session);
  auto& cat = session_info.get_catalog();
  auto vd = cat.getMetadataForFrontendView(
      std::to_string(session_info.get_currentUser().userId), view_name);
  if (!vd) {
    THROW_MAPD_EXCEPTION("View " + view_name + " doesn't exist");
  }
  try {
    cat.deleteMetadataForFrontendView(
        std::to_string(session_info.get_currentUser().userId), view_name);
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

TColumnType MapDHandler::create_geo_column(const TDatumType::type type,
                                           const std::string& name,
                                           const bool is_array) {
  TColumnType ct;
  ct.col_name = name;
  ct.col_type.type = type;
  ct.col_type.is_array = is_array;
  return ct;
}

void MapDHandler::check_geospatial_files(const boost::filesystem::path file_path,
                                         const Importer_NS::CopyParams& copy_params) {
  const std::list<std::string> shp_ext{".shp", ".shx", ".dbf"};
  if (std::find(shp_ext.begin(),
                shp_ext.end(),
                boost::algorithm::to_lower_copy(file_path.extension().string())) !=
      shp_ext.end()) {
    for (auto ext : shp_ext) {
      auto aux_file = file_path;
      if (!Importer_NS::Importer::gdalFileExists(
              aux_file.replace_extension(boost::algorithm::to_upper_copy(ext)).string(),
              copy_params) &&
          !Importer_NS::Importer::gdalFileExists(aux_file.replace_extension(ext).string(),
                                                 copy_params)) {
        throw std::runtime_error("required file for shapefile does not exist: " +
                                 aux_file.filename().string());
      }
    }
  }
}

void MapDHandler::create_table(const TSessionId& session,
                               const std::string& table_name,
                               const TRowDescriptor& rd,
                               const TTableType::type table_type,
                               const TCreateParams& create_params) {
  check_read_only("create_table");

  if (ImportHelpers::is_reserved_name(table_name)) {
    THROW_MAPD_EXCEPTION("Invalid table name (reserved keyword): " + table_name);
  } else if (table_name != ImportHelpers::sanitize_name(table_name)) {
    THROW_MAPD_EXCEPTION("Invalid characters in table name: " + table_name);
  }

  auto rds = rd;

  // no longer need to manually add the poly column for a TTableType::POLYGON table
  // a column of the correct geo type has already been added
  // @TODO simon.eves rename TTableType::POLYGON to TTableType::GEO or something!

  std::string stmt{"CREATE TABLE " + table_name};
  std::vector<std::string> col_stmts;

  for (auto col : rds) {
    if (ImportHelpers::is_reserved_name(col.col_name)) {
      THROW_MAPD_EXCEPTION("Invalid column name (reserved keyword): " + col.col_name);
    } else if (col.col_name != ImportHelpers::sanitize_name(col.col_name)) {
      THROW_MAPD_EXCEPTION("Invalid characters in column name: " + col.col_name);
    }
    if (col.col_type.type == TDatumType::INTERVAL_DAY_TIME ||
        col.col_type.type == TDatumType::INTERVAL_YEAR_MONTH) {
      THROW_MAPD_EXCEPTION("Unsupported type: " + thrift_to_name(col.col_type) +
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

    // As of 2016-06-27 the Immerse v1 frontend does not explicitly set the
    // `nullable` argument, leading this to default to false. Uncomment for v2.
    // if (!col.col_type.nullable) col_stmt.append(" NOT NULL");

    if (thrift_to_encoding(col.col_type.encoding) != kENCODING_NONE) {
      col_stmt.append(" ENCODING " + thrift_to_encoding_name(col.col_type));
      if (thrift_to_encoding(col.col_type.encoding) == kENCODING_DICT ||
          thrift_to_encoding(col.col_type.encoding) == kENCODING_FIXED ||
          thrift_to_encoding(col.col_type.encoding) == kENCODING_GEOINT) {
        col_stmt.append("(" + std::to_string(col.col_type.comp_param) + ")");
      }
    } else if (col.col_type.type == TDatumType::STR) {
      // non DICT encoded strings
      col_stmt.append(" ENCODING NONE");
    } else if (col.col_type.type == TDatumType::POINT ||
               col.col_type.type == TDatumType::LINESTRING ||
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
  sql_execute(ret, session, stmt, true, "", -1, -1);
}

void MapDHandler::import_table(const TSessionId& session,
                               const std::string& table_name,
                               const std::string& file_name_in,
                               const TCopyParams& cp) {
  check_read_only("import_table");
  LOG(INFO) << "import_table " << table_name << " from " << file_name_in;
  const auto session_info = get_session(session);
  auto& cat = session_info.get_catalog();

  const TableDescriptor* td = cat.getMetadataForTable(table_name);
  if (td == nullptr) {
    THROW_MAPD_EXCEPTION("Table " + table_name + " does not exist.");
  }
  check_table_load_privileges(session_info, table_name);

  std::string file_name{file_name_in};
  auto file_path = boost::filesystem::path(file_name);
  Importer_NS::CopyParams copy_params = thrift_to_copyparams(cp);
  if (!boost::istarts_with(file_name, "s3://")) {
    if (!boost::filesystem::path(file_name).is_absolute()) {
      file_path = import_path_ / session / boost::filesystem::path(file_name).filename();
      file_name = file_path.string();
    }
    if (!boost::filesystem::exists(file_path)) {
      THROW_MAPD_EXCEPTION("File does not exist: " + file_path.string());
    }
  }

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
          new DistributedLoader(session_info, td, &leaf_aggregator_),
          file_path.string(),
          copy_params));
    } else {
      importer.reset(new Importer_NS::Importer(cat, td, file_path.string(), copy_params));
    }
    auto ms = measure<>::execution([&]() { importer->import(); });
    std::cout << "Total Import Time: " << (double)ms / 1000.0 << " Seconds." << std::endl;
  } catch (const std::exception& e) {
    THROW_MAPD_EXCEPTION("Exception: " + std::string(e.what()));
  }
}

namespace {

// helper functions for error checking below
// these would usefully be added as methods of TDatumType
// but that's not possible as it's auto-generated by Thrift

bool TTypeInfo_IsGeo(const TDatumType::type& t) {
  return (t == TDatumType::POLYGON || t == TDatumType::MULTIPOLYGON ||
          t == TDatumType::LINESTRING || t == TDatumType::POINT);
}

std::string TTypeInfo_TypeToString(const TDatumType::type& t) {
  std::stringstream ss;
  ss << t;
  return ss.str();
}

std::string TTypeInfo_GeoSubTypeToString(const int32_t p) {
  std::string result;
  switch (p) {
    case SQLTypes::kGEOGRAPHY:
      result = "GEOGRAPHY";
      break;
    case SQLTypes::kGEOMETRY:
      result = "GEOMETRY";
      break;
    default:
      result = "INVALID";
      break;
  }
  return result;
}

std::string TTypeInfo_EncodingToString(const TEncodingType::type& t) {
  std::stringstream ss;
  ss << t;
  return ss.str();
}

}  // namespace

#define THROW_COLUMN_ATTR_MISMATCH_EXCEPTION(attr, got, expected)                      \
  THROW_MAPD_EXCEPTION("Could not append geo file '" + file_path.filename().string() + \
                       "' to table '" + table_name + "'. Column '" + cd->columnName +  \
                       "' " + attr + " mismatch (got '" + got + "', expected '" +      \
                       expected + "')");

void MapDHandler::import_geo_table(const TSessionId& session,
                                   const std::string& table_name,
                                   const std::string& file_name_in,
                                   const TCopyParams& cp,
                                   const TRowDescriptor& row_desc) {
  check_read_only("import_table");
  const auto session_info = get_session(session);
  auto& cat = session_info.get_catalog();

  Importer_NS::CopyParams copy_params = thrift_to_copyparams(cp);

  std::string file_name{file_name_in};

  if (path_is_relative(file_name)) {
    // assume relative paths are relative to data_path / mapd_import / <session>
    auto file_path =
        import_path_ / session / boost::filesystem::path(file_name).filename();
    file_name = file_path.string();
  }

  if (is_a_supported_geo_file(file_name, true)) {
    // prepare to load geo file directly
    add_vsi_network_prefix(file_name);
    add_vsi_geo_prefix(file_name);
  } else if (is_a_supported_archive_file(file_name)) {
    // find the archive file
    add_vsi_network_prefix(file_name);
    if (!Importer_NS::Importer::gdalFileExists(file_name, copy_params)) {
      THROW_MAPD_EXCEPTION("Archive does not exist: " + file_name_in);
    }
    // find geo file in archive
    add_vsi_archive_prefix(file_name);
    std::string geo_file = find_first_geo_file_in_archive(file_name, copy_params);
    // prepare to load that geo file
    if (geo_file.size()) {
      file_name = file_name + std::string("/") + geo_file;
    }
  } else {
    THROW_MAPD_EXCEPTION("File is not a supported geo or geo archive file: " +
                         file_name_in);
  }

  // log what we're about to try to do
  LOG(INFO) << "import_geo_table: Creating table: " << table_name;
  LOG(INFO) << "import_geo_table: Original filename: " << file_name_in;
  LOG(INFO) << "import_geo_table: Actual filename: " << file_name;

  // use GDAL to check the primary file exists (even if on S3 and/or in archive)
  auto file_path = boost::filesystem::path(file_name);
  if (!Importer_NS::Importer::gdalFileExists(file_name, copy_params)) {
    THROW_MAPD_EXCEPTION("File does not exist: " + file_path.filename().string());
  }

  // use GDAL to check any dependent files exist (ditto)
  try {
    check_geospatial_files(file_path, copy_params);
  } catch (const std::exception& e) {
    THROW_MAPD_EXCEPTION("import_geo_table error: " + std::string(e.what()));
  }

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
    TCopyParams cp_copy = cp;  // retain S3 auth tokens
    cp_copy.table_type = TTableType::POLYGON;
    detect_column_types(cds, session, file_name_in, cp_copy);
    rd = cds.row_set.row_desc;

    // then, if the table does NOT already exist, create it
    const TableDescriptor* td = cat.getMetadataForTable(table_name);
    if (!td) {
      TCreateParams create_params;
      create_params.is_replicated = false;
      create_table(session, table_name, rd, TTableType::POLYGON, create_params);
    }
  }

  // by this point, the table should exist, one way or another
  const TableDescriptor* td = cat.getMetadataForTable(table_name);
  if (!td) {
    THROW_MAPD_EXCEPTION("Could not import geo file '" + file_path.filename().string() +
                         "' to table '" + table_name +
                         "'; table does not exist or failed to create.");
  }

  // then, we have to verify that the structure matches
  // get column descriptors (non-system, non-deleted, logical columns only)
  const auto col_descriptors =
      cat.getAllColumnMetadataForTable(td->tableId, false, false, false);
  // compare the column number, names, and types
  if (col_descriptors.size() != rd.size()) {
    THROW_MAPD_EXCEPTION("Could not append geo file '" + file_path.filename().string() +
                         "' to table '" + table_name + "'. Column count mismatch (got " +
                         std::to_string(rd.size()) + ", expecting " +
                         std::to_string(col_descriptors.size()) + ")");
  } else {
    int rd_index = 0;
    for (auto cd : col_descriptors) {
      TColumnType cd_col_type = populateThriftColumnType(&cat, cd);
      std::string gname = rd[rd_index].col_name;  // got
      std::string ename = cd->columnName;         // expecting
      TTypeInfo gti = rd[rd_index].col_type;      // got
      TTypeInfo eti = cd_col_type.col_type;       // expecting
      // check for name match
      if (gname != ename) {
        if (TTypeInfo_IsGeo(eti.type) && ename == LEGACY_GEO_PREFIX &&
            gname == OMNISCI_GEO_PREFIX) {
          // rename incoming geo column to match existing legacy default geo column
          rd[rd_index].col_name = gname;
          LOG(INFO) << "import_geo_table: Renaming incoming geo column to match existing "
                       "legacy default geo column";
        } else {
          THROW_COLUMN_ATTR_MISMATCH_EXCEPTION("name", gname, ename);
        }
      }
      // check for type attributes match
      // these attrs must always match regardless of type
      if (gti.type != eti.type) {
        THROW_COLUMN_ATTR_MISMATCH_EXCEPTION(
            "type", TTypeInfo_TypeToString(gti.type), TTypeInfo_TypeToString(eti.type));
      }
      if (gti.is_array != eti.is_array) {
        THROW_COLUMN_ATTR_MISMATCH_EXCEPTION(
            "array-ness", std::to_string(gti.is_array), std::to_string(eti.is_array));
      }
      if (gti.nullable != eti.nullable) {
        THROW_COLUMN_ATTR_MISMATCH_EXCEPTION(
            "nullability", std::to_string(gti.nullable), std::to_string(eti.nullable));
      }
      if (TTypeInfo_IsGeo(eti.type)) {
        // for geo, only these other attrs must also match
        // encoding and comp_param are allowed to differ
        // this allows appending to existing geo table
        // without needing to know the existing encoding
        if (gti.precision != eti.precision) {
          THROW_COLUMN_ATTR_MISMATCH_EXCEPTION(
              "geo sub-type",
              TTypeInfo_GeoSubTypeToString(gti.precision),
              TTypeInfo_GeoSubTypeToString(eti.precision));
        }
        if (gti.scale != eti.scale) {
          THROW_COLUMN_ATTR_MISMATCH_EXCEPTION(
              "SRID", std::to_string(gti.scale), std::to_string(eti.scale));
        }
        if (gti.encoding != eti.encoding) {
          LOG(INFO) << "import_geo_table: Ignoring geo encoding mismatch";
        }
        if (gti.comp_param != eti.comp_param) {
          LOG(INFO) << "import_geo_table: Ignoring geo comp_param mismatch";
        }
      } else {
        // non-geo, all other attrs must also match
        // @TODO consider relaxing some of these dependent on type
        // e.g. DECIMAL precision/scale, TEXT dict or non-dict, INTEGER up-sizing
        if (gti.precision != eti.precision) {
          THROW_COLUMN_ATTR_MISMATCH_EXCEPTION(
              "precision", std::to_string(gti.precision), std::to_string(eti.precision));
        }
        if (gti.scale != eti.scale) {
          THROW_COLUMN_ATTR_MISMATCH_EXCEPTION(
              "scale", std::to_string(gti.scale), std::to_string(eti.scale));
        }
        if (gti.encoding != eti.encoding) {
          THROW_COLUMN_ATTR_MISMATCH_EXCEPTION("encoding",
                                               TTypeInfo_EncodingToString(gti.encoding),
                                               TTypeInfo_EncodingToString(eti.encoding));
        }
        if (gti.comp_param != eti.comp_param) {
          THROW_COLUMN_ATTR_MISMATCH_EXCEPTION("comp param",
                                               std::to_string(gti.comp_param),
                                               std::to_string(eti.comp_param));
        }
      }
      rd_index++;
    }
  }

  std::map<std::string, std::string> colname_to_src;
  for (auto r : rd) {
    colname_to_src[r.col_name] =
        r.src_name.length() > 0 ? r.src_name : ImportHelpers::sanitize_name(r.src_name);
  }

  check_table_load_privileges(session_info, table_name);

  // Final check to ensure that we actually have a geo column
  // of the expected name and type before doing the actual import,
  // in case the user naively overrode the name or type in Immerse
  // Preview (which as of 6/8/18 it still allows you to do).
  // This avoids a fatal assert later when it fails to find the
  // column. We should make Immerse more robust and disallow this.
  bool have_geo_column_with_correct_name = false;
  for (const auto& r : rd) {
    if (r.col_type.type == TDatumType::POINT ||
        r.col_type.type == TDatumType::LINESTRING ||
        r.col_type.type == TDatumType::POLYGON ||
        r.col_type.type == TDatumType::MULTIPOLYGON) {
      // TODO(team): allow user to override the geo column name
      if (r.col_name == OMNISCI_GEO_PREFIX) {
        have_geo_column_with_correct_name = true;
      } else if (r.col_name == LEGACY_GEO_PREFIX) {
        CHECK(colname_to_src.find(r.col_name) != colname_to_src.end());
        // Normalize column names for geo append with legacy column naming scheme
        colname_to_src[r.col_name] = r.col_name;
        have_geo_column_with_correct_name = true;
      }
    }
  }
  if (!have_geo_column_with_correct_name) {
    THROW_MAPD_EXCEPTION("Table " + table_name +
                         " does not have a geo column with name '" + OMNISCI_GEO_PREFIX +
                         "'. Import aborted!");
  }

  try {
    std::unique_ptr<Importer_NS::Importer> importer;
    if (leaf_aggregator_.leafCount() > 0) {
      importer.reset(new Importer_NS::Importer(
          new DistributedLoader(session_info, td, &leaf_aggregator_),
          file_path.string(),
          copy_params));
    } else {
      importer.reset(new Importer_NS::Importer(cat, td, file_path.string(), copy_params));
    }
    auto ms = measure<>::execution([&]() { importer->importGDAL(colname_to_src); });
    std::cout << "Total Import Time: " << (double)ms / 1000.0 << " Seconds." << std::endl;
  } catch (const std::exception& e) {
    THROW_MAPD_EXCEPTION(std::string("import_geo_table failed: ") + e.what());
  }
}

#undef THROW_COLUMN_ATTR_MISMATCH_EXCEPTION

void MapDHandler::import_table_status(TImportStatus& _return,
                                      const TSessionId& session,
                                      const std::string& import_id) {
  LOG(INFO) << "import_table_status " << import_id;
  auto is = Importer_NS::Importer::get_import_status(import_id);
  _return.elapsed = is.elapsed.count();
  _return.rows_completed = is.rows_completed;
  _return.rows_estimated = is.rows_estimated;
  _return.rows_rejected = is.rows_rejected;
}

void MapDHandler::get_first_geo_file_in_archive(std::string& _return,
                                                const TSessionId& session,
                                                const std::string& archive_path_in,
                                                const TCopyParams& copy_params) {
  LOG(INFO) << "get_first_geo_file_in_archive " << archive_path_in;
  std::string archive_path(archive_path_in);

  if (path_is_relative(archive_path)) {
    // assume relative paths are relative to data_path / mapd_import / <session>
    auto file_path =
        import_path_ / session / boost::filesystem::path(archive_path).filename();
    archive_path = file_path.string();
  }

  if (is_a_supported_archive_file(archive_path)) {
    // find the archive file
    add_vsi_network_prefix(archive_path);
    if (!Importer_NS::Importer::gdalFileExists(archive_path,
                                               thrift_to_copyparams(copy_params))) {
      THROW_MAPD_EXCEPTION("Archive does not exist: " + archive_path_in);
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

void MapDHandler::get_all_files_in_archive(std::vector<std::string>& _return,
                                           const TSessionId& session,
                                           const std::string& archive_path_in,
                                           const TCopyParams& copy_params) {
  LOG(INFO) << "get_all_files_in_archive " << archive_path_in;
  std::string archive_path(archive_path_in);

  if (path_is_relative(archive_path)) {
    // assume relative paths are relative to data_path / mapd_import / <session>
    auto file_path =
        import_path_ / session / boost::filesystem::path(archive_path).filename();
    archive_path = file_path.string();
  }

  if (is_a_supported_archive_file(archive_path)) {
    // find the archive file
    add_vsi_network_prefix(archive_path);
    if (!Importer_NS::Importer::gdalFileExists(archive_path,
                                               thrift_to_copyparams(copy_params))) {
      THROW_MAPD_EXCEPTION("Archive does not exist: " + archive_path_in);
    }
    // find all files in archive
    add_vsi_archive_prefix(archive_path);
    _return = Importer_NS::Importer::gdalGetAllFilesInArchive(
        archive_path, thrift_to_copyparams(copy_params));
    // prepend them all with original path
    for (auto& s : _return) {
      s = archive_path_in + std::string("/") + s;
    }
  }
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

void MapDHandler::check_session_exp(const SessionMap::iterator& session_it) {
  time_t last_used_time = session_it->second->get_last_used_time();
  time_t creation_time = session_it->second->get_creation_time();
  if ((time(0) - last_used_time) > idle_session_duration_) {
    sessions_.erase(session_it);  // Already checked session existance in get_session_it
    THROW_MAPD_EXCEPTION("Idle Session Timeout. User should re-authenticate.")
  } else if ((time(0) - creation_time) > max_session_duration_) {
    sessions_.erase(session_it);  // Already checked session existance in get_session_it
    THROW_MAPD_EXCEPTION("Maximum active Session Timeout. User should re-authenticate.")
  }
}

SessionMap::iterator MapDHandler::get_session_it(const TSessionId& session) {
  auto calcite_session_prefix = calcite_->get_session_prefix();
  auto prefix_length = calcite_session_prefix.size();
  if (prefix_length) {
    if (0 == session.compare(0, prefix_length, calcite_session_prefix)) {
      // call coming from calcite, elevate user to be superuser
      auto session_it = sessions_.find(session.substr(prefix_length + 1));
      if (session_it == sessions_.end()) {
        THROW_MAPD_EXCEPTION("Session not valid.");
      }
      check_session_exp(session_it);
      session_it->second->make_superuser();
      session_it->second->update_last_used_time();
      return session_it;
    }
  }

  auto session_it = sessions_.find(session);
  if (session_it == sessions_.end()) {
    THROW_MAPD_EXCEPTION("Session not valid.");
  }
  check_session_exp(session_it);
  session_it->second->reset_superuser();
  session_it->second->update_last_used_time();
  return session_it;
}

Catalog_Namespace::SessionInfo MapDHandler::get_session(const TSessionId& session) {
  mapd_shared_lock<mapd_shared_mutex> read_lock(sessions_mutex_);
  return *get_session_it(session)->second;
}

void MapDHandler::check_table_load_privileges(
    const Catalog_Namespace::SessionInfo& session_info,
    const std::string& table_name) {
  auto user_metadata = session_info.get_currentUser();
  auto& cat = session_info.get_catalog();
  if (SysCatalog::instance().arePrivilegesOn()) {
    DBObject dbObject(table_name, TableDBObjectType);
    dbObject.loadKey(cat);
    dbObject.setPrivileges(AccessPrivileges::INSERT_INTO_TABLE);
    std::vector<DBObject> privObjects;
    privObjects.push_back(dbObject);
    if (!SysCatalog::instance().checkPrivileges(user_metadata, privObjects)) {
      THROW_MAPD_EXCEPTION("Violation of access privileges: user " +
                           user_metadata.userName +
                           " has no insert privileges for table " + table_name + ".");
    }
  }
}

void MapDHandler::check_table_load_privileges(const TSessionId& session,
                                              const std::string& table_name) {
  const auto session_info = MapDHandler::get_session(session);
  check_table_load_privileges(session_info, table_name);
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
  }
}

std::vector<PushedDownFilterInfo> MapDHandler::execute_rel_alg(
    TQueryResult& _return,
    const std::string& query_ra,
    const bool column_format,
    const Catalog_Namespace::SessionInfo& session_info,
    const ExecutorDeviceType executor_device_type,
    const int32_t first_n,
    const int32_t at_most_n,
    const bool just_explain,
    const bool just_validate,
    const bool find_push_down_candidates,
    const bool just_calcite_explain) const {
  INJECT_TIMER(execute_rel_alg);
  const auto& cat = session_info.get_catalog();
  CompilationOptions co = {
      executor_device_type, true, ExecutorOptLevel::Default, g_enable_dynamic_watchdog};
  ExecutionOptions eo = {g_enable_columnar_output,
                         allow_multifrag_,
                         just_explain,
                         allow_loop_joins_ || just_validate,
                         g_enable_watchdog,
                         jit_debug_,
                         just_validate,
                         g_enable_dynamic_watchdog,
                         g_dynamic_watchdog_time_limit,
                         find_push_down_candidates,
                         just_calcite_explain};
  auto executor = Executor::getExecutor(cat.get_currentDB().dbId,
                                        jit_debug_ ? "/tmp" : "",
                                        jit_debug_ ? "mapdquery" : "",
                                        mapd_parameters_,
                                        nullptr);
  RelAlgExecutor ra_executor(executor.get(), cat);
  ExecutionResult result{std::make_shared<ResultSet>(std::vector<TargetInfo>{},
                                                     ExecutorDeviceType::CPU,
                                                     QueryMemoryDescriptor(),
                                                     nullptr,
                                                     nullptr),
                         {}};
  _return.execution_time_ms += measure<>::execution(
      [&]() { result = ra_executor.executeRelAlgQuery(query_ra, co, eo, nullptr); });
  // reduce execution time by the time spent during queue waiting
  _return.execution_time_ms -= result.getRows()->getQueueTime();
  const auto& filter_push_down_info = result.getPushedDownFilterInfo();
  if (!filter_push_down_info.empty()) {
    return filter_push_down_info;
  }
  if (just_explain) {
    convert_explain(_return, *result.getRows(), column_format);
  } else if (!just_calcite_explain) {
    convert_rows(_return,
                 result.getTargetsMeta(),
                 *result.getRows(),
                 column_format,
                 first_n,
                 at_most_n);
  }
  return {};
}

void MapDHandler::execute_rel_alg_df(TDataFrame& _return,
                                     const std::string& query_ra,
                                     const Catalog_Namespace::SessionInfo& session_info,
                                     const ExecutorDeviceType device_type,
                                     const size_t device_id,
                                     const int32_t first_n) const {
  const auto& cat = session_info.get_catalog();
  CHECK(device_type == ExecutorDeviceType::CPU ||
        session_info.get_executor_device_type() == ExecutorDeviceType::GPU);
  CompilationOptions co = {
      device_type, true, ExecutorOptLevel::Default, g_enable_dynamic_watchdog};
  ExecutionOptions eo = {false,
                         allow_multifrag_,
                         false,
                         allow_loop_joins_,
                         g_enable_watchdog,
                         jit_debug_,
                         false,
                         g_enable_dynamic_watchdog,
                         g_dynamic_watchdog_time_limit,
                         false};
  auto executor = Executor::getExecutor(cat.get_currentDB().dbId,
                                        jit_debug_ ? "/tmp" : "",
                                        jit_debug_ ? "mapdquery" : "",
                                        mapd_parameters_,
                                        nullptr);
  RelAlgExecutor ra_executor(executor.get(), cat);
  const auto result = ra_executor.executeRelAlgQuery(query_ra, co, eo, nullptr);
  const auto rs = result.getRows();
  const auto copy = rs->getArrowCopy(data_mgr_.get(),
                                     device_type,
                                     device_id,
                                     getTargetNames(result.getTargetsMeta()),
                                     first_n);
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
  auto executor = Executor::getExecutor(
      root_plan->get_catalog().get_currentDB().dbId,
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

std::vector<std::string> MapDHandler::getTargetNames(
    const std::vector<TargetMetaInfo>& targets) const {
  std::vector<std::string> names;
  for (const auto target : targets) {
    names.push_back(target.get_resname());
  }
  return names;
}

TColumnType MapDHandler::convert_target_metainfo(const TargetMetaInfo& target,
                                                 const size_t idx) const {
  TColumnType proj_info;
  proj_info.col_name = target.get_resname();
  if (proj_info.col_name.empty()) {
    proj_info.col_name = "result_" + std::to_string(idx + 1);
  }
  const auto& target_ti = target.get_type_info();
  proj_info.col_type.type = type_to_thrift(target_ti);
  proj_info.col_type.encoding = encoding_to_thrift(target_ti);
  proj_info.col_type.nullable = !target_ti.get_notnull();
  proj_info.col_type.is_array = target_ti.get_type() == kARRAY;
  if (IS_GEO(target_ti.get_type())) {
    fixup_geo_column_descriptor(
        proj_info, target_ti.get_subtype(), target_ti.get_output_srid());
  } else {
    proj_info.col_type.precision = target_ti.get_precision();
    proj_info.col_type.scale = target_ti.get_scale();
  }
  proj_info.col_type.comp_param = target_ti.get_comp_param();
  return proj_info;
}

TRowDescriptor MapDHandler::convert_target_metainfo(
    const std::vector<TargetMetaInfo>& targets) const {
  TRowDescriptor row_desc;
  size_t i = 0;
  for (const auto target : targets) {
    row_desc.push_back(convert_target_metainfo(target, i));
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
  INJECT_TIMER(convert_rows);
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
        THROW_MAPD_EXCEPTION("The result contains more rows than the specified cap of " +
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
        THROW_MAPD_EXCEPTION("The result contains more rows than the specified cap of " +
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

TRowDescriptor MapDHandler::fixup_row_descriptor(const TRowDescriptor& row_desc,
                                                 const Catalog& cat) {
  TRowDescriptor fixedup_row_desc;
  for (const TColumnType& col_desc : row_desc) {
    auto fixedup_col_desc = col_desc;
    if (col_desc.col_type.encoding == TEncodingType::DICT &&
        col_desc.col_type.comp_param > 0) {
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

void MapDHandler::convert_explain(TQueryResult& _return,
                                  const ResultSet& results,
                                  const bool column_format) const {
  create_simple_result(_return, results, column_format, "Explanation");
}

void MapDHandler::convert_result(TQueryResult& _return,
                                 const ResultSet& results,
                                 const bool column_format) const {
  create_simple_result(_return, results, column_format, "Result");
}

namespace {

void check_table_not_sharded(const Catalog& cat, const int table_id) {
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
  std::unique_ptr<Planner::RootPlan> root_plan;

  /*
     Use this seq to simplify locking:
                  INSERT_VALUES: CheckpointLock [ >> write UpdateDeleteLock ]
                  INSERT_SELECT: CheckpointLock >> read UpdateDeleteLock [ >> write
     UpdateDeleteLock ] COPY_TO/SELECT: read UpdateDeleteLock COPY_FROM:  CheckpointLock [
     >> write UpdateDeleteLock ] DROP/TRUNC: CheckpointLock >> write UpdateDeleteLock
                  DELETE/UPDATE: CheckpointLock >> write UpdateDeleteLock
  */
  mapd_unique_lock<mapd_shared_mutex> chkptlLock;
  mapd_unique_lock<mapd_shared_mutex> upddelLock;
  mapd_unique_lock<mapd_shared_mutex> executeWriteLock;
  mapd_shared_lock<mapd_shared_mutex> executeReadLock;
  std::vector<std::shared_ptr<VLock>> upddelLocks;

  try {
    ParserWrapper pw{query_str};
    std::map<std::string, bool> tableNames;
    if (is_calcite_path_permissable(pw, read_only_)) {
      std::string query_ra;
      _return.execution_time_ms += measure<>::execution([&]() {
        // query_ra = TIME_WRAP(parse_to_ra)(query_str, session_info);
        query_ra = parse_to_ra(query_str, {}, session_info, &tableNames);
      });

      std::string query_ra_calcite_explain;
      if (pw.is_select_calcite_explain && (!g_enable_filter_push_down || g_cluster)) {
        // return the ra as the result
        convert_explain(_return, ResultSet(query_ra), true);
        return;
      } else if (pw.is_select_calcite_explain) {
        // removing the "explain calcite " from the beginning of the "query_str":
        std::string temp_query_str =
            query_str.substr(std::string("explain calcite ").length());
        query_ra_calcite_explain = parse_to_ra(temp_query_str, {}, session_info);
      }

      // UPDATE/DELETE needs to get a checkpoint lock as the first lock
      for (const auto& table : tableNames) {
        if (table.second) {
          chkptlLock = getTableLock<mapd_shared_mutex, mapd_unique_lock>(
              session_info.get_catalog(), table.first, LockType::CheckpointLock);
        }
      }
      // COPY_TO/SELECT: read ExecutorOuterLock >> read UpdateDeleteLock locks
      executeReadLock = mapd_shared_lock<mapd_shared_mutex>(
          *LockMgr<mapd_shared_mutex, bool>::getMutex(ExecutorOuterLock, true));
      getTableLocks<mapd_shared_mutex>(session_info.get_catalog(),
                                       tableNames,
                                       upddelLocks,
                                       LockType::UpdateDeleteLock);
      const auto filter_push_down_requests = execute_rel_alg(
          _return,
          pw.is_select_calcite_explain ? query_ra_calcite_explain : query_ra,
          column_format,
          session_info,
          executor_device_type,
          first_n,
          at_most_n,
          pw.is_select_explain,
          false,
          g_enable_filter_push_down && !g_cluster,
          pw.is_select_calcite_explain);
      if (pw.is_select_calcite_explain && filter_push_down_requests.empty()) {
        // we only reach here if filter push down was enabled, but no filter
        // push down candidate was found
        convert_explain(_return, ResultSet(query_ra), true);
        return;
      }
      if (!filter_push_down_requests.empty()) {
        execute_rel_alg_with_filter_push_down(_return,
                                              query_ra,
                                              column_format,
                                              session_info,
                                              executor_device_type,
                                              first_n,
                                              at_most_n,
                                              pw.is_select_explain,
                                              pw.is_select_calcite_explain,
                                              query_str,
                                              filter_push_down_requests);
      } else if (pw.is_select_calcite_explain && filter_push_down_requests.empty()) {
        // return the ra as the result:
        // If we reach here, the 'filter_push_down_request' turned out to be empty, i.e.,
        // no filter push down so we continue with the initial (unchanged) query's calcite
        // explanation.
        query_ra = parse_to_ra(query_str, {}, session_info);
        convert_explain(_return, ResultSet(query_ra), true);
        return;
      }
      if (pw.is_select_calcite_explain) {
        // If we reach here, the filter push down candidates has been selected and
        // proper output result has been already created.
        return;
      }
      if (pw.is_select_calcite_explain) {
        // return the ra as the result:
        // If we reach here, the 'filter_push_down_request' turned out to be empty, i.e.,
        // no filter push down so we continue with the initial (unchanged) query's calcite
        // explanation.
        query_ra = parse_to_ra(query_str, {}, session_info);
        convert_explain(_return, ResultSet(query_ra), true);
        return;
      }
      return;
    }
    LOG(INFO) << "passing query to legacy processor";
  } catch (std::exception& e) {
    if (strstr(e.what(), "java.lang.NullPointerException")) {
      THROW_MAPD_EXCEPTION(std::string("Exception: ") +
                           "query failed from broken view or other schema related issue");
    } else {
      THROW_MAPD_EXCEPTION(std::string("Exception: ") + e.what());
    }
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

  auto get_legacy_plan =
      [&cat](const Parser::DMLStmt* dml,
             const bool is_explain) -> std::unique_ptr<Planner::RootPlan> {
    Analyzer::Query query;
    dml->analyze(cat, query);
    Planner::Optimizer optimizer(query, cat);
    auto root_plan_ptr = std::unique_ptr<Planner::RootPlan>(optimizer.optimize());
    CHECK(root_plan_ptr);
    if (is_explain) {
      root_plan_ptr->set_plan_dest(Planner::RootPlan::Dest::kEXPLAIN);
    }
    return std::move(root_plan_ptr);
  };

  auto handle_ddl = [&cat,
                     &root_plan,
                     &get_legacy_plan,
                     &session_info,
                     &_return,
                     &chkptlLock,
                     &upddelLock,
                     &upddelLocks,
                     this](Parser::DDLStmt* ddl) -> bool {
    if (!ddl) {
      return false;
    }
    const auto explain_stmt = dynamic_cast<Parser::ExplainStmt*>(ddl);
    if (explain_stmt) {
      root_plan = get_legacy_plan(explain_stmt->get_stmt(), true);
      return false;
    }

    const auto show_create_stmt = dynamic_cast<Parser::ShowCreateTableStmt*>(ddl);
    if (show_create_stmt) {
      // ParserNode ShowCreateTableStmt is currently unimplemented
      throw std::runtime_error(
          "SHOW CREATE TABLE is currently unsupported. Use `\\d` from mapdql for table "
          "DDL.");
    }

    const auto import_stmt = dynamic_cast<Parser::CopyTableStmt*>(ddl);
    if (import_stmt) {
      // get the lock if the table exists
      // thus allowing COPY FROM to execute to create a table
      // is it safe to do this check without a lock?
      // if the table doesn't exist, the getTableLock will throw an exception anyway
      const TableDescriptor* td =
          session_info.get_catalog().getMetadataForTable(import_stmt->get_table());
      if (td) {
        // COPY_FROM: CheckpointLock [ >> write UpdateDeleteLocks ]
        chkptlLock =
            getTableLock<mapd_shared_mutex, mapd_unique_lock>(session_info.get_catalog(),
                                                              import_stmt->get_table(),
                                                              LockType::CheckpointLock);
        // [ write UpdateDeleteLocks ] lock is deferred in
        // InsertOrderFragmenter::deleteFragments
      }

      if (g_cluster && !leaf_aggregator_.leafCount()) {
        // Sharded table rows need to be routed to the leaf by an aggregator.
        check_table_not_sharded(cat, import_stmt->get_table());
      } else if (leaf_aggregator_.leafCount() > 0) {
        _return.execution_time_ms += measure<>::execution(
            [&]() { execute_distributed_copy_statement(import_stmt, session_info); });
      } else {
        _return.execution_time_ms +=
            measure<>::execution([&]() { ddl->execute(session_info); });
      }

      // Read response message
      convert_result(_return, ResultSet(*import_stmt->return_message.get()), true);

      // get geo_copy_from info
      _was_geo_copy_from = import_stmt->was_geo_copy_from();
      import_stmt->get_geo_copy_from_payload(
          _geo_copy_from_table, _geo_copy_from_file_name, _geo_copy_from_copy_params);
      return true;
    }

    // Check for DDL statements requiring locking and get locks
    auto export_stmt = dynamic_cast<Parser::ExportQueryStmt*>(ddl);
    if (export_stmt) {
      std::map<std::string, bool> tableNames;
      const auto query_string = export_stmt->get_select_stmt();
      const auto query_ra = parse_to_ra(query_string, {}, session_info, &tableNames);
      getTableLocks<mapd_shared_mutex>(session_info.get_catalog(),
                                       tableNames,
                                       upddelLocks,
                                       LockType::UpdateDeleteLock);
    }
    auto truncate_stmt = dynamic_cast<Parser::TruncateTableStmt*>(ddl);
    if (truncate_stmt) {
      chkptlLock =
          getTableLock<mapd_shared_mutex, mapd_unique_lock>(session_info.get_catalog(),
                                                            *truncate_stmt->get_table(),
                                                            LockType::CheckpointLock);
      upddelLock =
          getTableLock<mapd_shared_mutex, mapd_unique_lock>(session_info.get_catalog(),
                                                            *truncate_stmt->get_table(),
                                                            LockType::UpdateDeleteLock);
    }
    auto add_col_stmt = dynamic_cast<Parser::AddColumnStmt*>(ddl);
    if (add_col_stmt) {
      chkptlLock =
          getTableLock<mapd_shared_mutex, mapd_unique_lock>(session_info.get_catalog(),
                                                            *add_col_stmt->get_table(),
                                                            LockType::CheckpointLock);
      upddelLock =
          getTableLock<mapd_shared_mutex, mapd_unique_lock>(session_info.get_catalog(),
                                                            *add_col_stmt->get_table(),
                                                            LockType::UpdateDeleteLock);
    }

    _return.execution_time_ms +=
        measure<>::execution([&]() { ddl->execute(session_info); });
    return true;
  };

  for (const auto& stmt : parse_trees) {
    try {
      auto select_stmt = dynamic_cast<Parser::SelectStmt*>(stmt.get());
      if (!select_stmt) {
        check_read_only("Non-SELECT statements");
      }
      auto ddl = dynamic_cast<Parser::DDLStmt*>(stmt.get());
      if (handle_ddl(ddl)) {
        if (render_handler_) {
          render_handler_->handle_ddl(ddl);
        }
        continue;
      }
      if (!root_plan) {
        // assume DML / non-explain plan (an insert or copy statement)
        const auto dml = dynamic_cast<Parser::DMLStmt*>(stmt.get());
        CHECK(dml);
        root_plan = get_legacy_plan(dml, false);
        CHECK(root_plan);
      }
      if (auto stmtp = dynamic_cast<Parser::InsertQueryStmt*>(stmt.get())) {
        // INSERT_SELECT: CheckpointLock >> read UpdateDeleteLocks [ >> write
        // UpdateDeleteLocks ]
        chkptlLock = getTableLock<mapd_shared_mutex, mapd_unique_lock>(
            session_info.get_catalog(), *stmtp->get_table(), LockType::CheckpointLock);
        // >> read UpdateDeleteLock locks
        std::map<std::string, bool> tableNames;
        const auto query_string = stmtp->get_query()->to_string();
        const auto query_ra = parse_to_ra(query_str, {}, session_info, &tableNames);
        getTableLocks<mapd_shared_mutex>(session_info.get_catalog(),
                                         tableNames,
                                         upddelLocks,
                                         LockType::UpdateDeleteLock);
        // [ write UpdateDeleteLocks ] lock is deferred in
        // InsertOrderFragmenter::deleteFragments
        // TODO: this statement is not supported. once supported, it must not go thru
        // InsertOrderFragmenter::insertData, or deadlock will occur w/o moving the
        // following lock back to here!!!
      } else if (auto stmtp = dynamic_cast<Parser::InsertValuesStmt*>(stmt.get())) {
        // INSERT_VALUES: CheckpointLock >> write ExecutorOuterLock [ >> write
        // UpdateDeleteLocks ]
        chkptlLock = getTableLock<mapd_shared_mutex, mapd_unique_lock>(
            session_info.get_catalog(), *stmtp->get_table(), LockType::CheckpointLock);
        executeWriteLock = mapd_unique_lock<mapd_shared_mutex>(
            *LockMgr<mapd_shared_mutex, bool>::getMutex(ExecutorOuterLock, true));
        // [ write UpdateDeleteLocks ] lock is deferred in
        // InsertOrderFragmenter::deleteFragments
      }

      if (g_cluster && root_plan->get_stmt_type() == kINSERT) {
        check_table_not_sharded(session_info.get_catalog(),
                                root_plan->get_result_table_id());
      }
      execute_root_plan(_return,
                        root_plan.get(),
                        column_format,
                        session_info,
                        executor_device_type,
                        first_n);
    } catch (std::exception& e) {
      THROW_MAPD_EXCEPTION(std::string("Exception: ") + e.what());
    }
  }
}

void MapDHandler::execute_rel_alg_with_filter_push_down(
    TQueryResult& _return,
    std::string& query_ra,
    const bool column_format,
    const Catalog_Namespace::SessionInfo& session_info,
    const ExecutorDeviceType executor_device_type,
    const int32_t first_n,
    const int32_t at_most_n,
    const bool just_explain,
    const bool just_calcite_explain,
    const std::string& query_str,
    const std::vector<PushedDownFilterInfo> filter_push_down_requests) {
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
  _return.execution_time_ms += measure<>::execution(
      [&]() { query_ra = parse_to_ra(query_str, filter_push_down_info, session_info); });

  if (just_calcite_explain) {
    // return the new ra as the result
    convert_explain(_return, ResultSet(query_ra), true);
    return;
  }

  // execute the new relational algebra plan:
  execute_rel_alg(_return,
                  query_ra,
                  column_format,
                  session_info,
                  executor_device_type,
                  first_n,
                  at_most_n,
                  just_explain,
                  /*just_validate = */ false,
                  /*find_push_down_candidates = */ false,
                  /*just_calcite_explain = */ false);
}

void MapDHandler::execute_distributed_copy_statement(
    Parser::CopyTableStmt* copy_stmt,
    const Catalog_Namespace::SessionInfo& session_info) {
  auto importer_factory = [&session_info, this](
                              const Catalog& catalog,
                              const TableDescriptor* td,
                              const std::string& file_path,
                              const Importer_NS::CopyParams& copy_params) {
    return boost::make_unique<Importer_NS::Importer>(
        new DistributedLoader(session_info, td, &leaf_aggregator_),
        file_path,
        copy_params);
  };
  copy_stmt->execute(session_info, importer_factory);
}

Planner::RootPlan* MapDHandler::parse_to_plan(
    const std::string& query_str,
    const Catalog_Namespace::SessionInfo& session_info) {
  auto& cat = session_info.get_catalog();
  ParserWrapper pw{query_str};
  // if this is a calcite select or explain select run in calcite
  if (!pw.is_ddl && !pw.is_update_dml && !pw.is_other_explain) {
    const std::string actual_query{pw.is_select_explain || pw.is_select_calcite_explain
                                       ? pw.actual_query
                                       : query_str};
    const auto query_ra =
        calcite_
            ->process(session_info,
                      legacy_syntax_ ? pg_shim(actual_query) : actual_query,
                      {},
                      legacy_syntax_,
                      pw.is_select_calcite_explain)
            .plan_result;
    auto root_plan = translate_query(query_ra, cat);
    CHECK(root_plan);
    if (pw.is_select_explain) {
      root_plan->set_plan_dest(Planner::RootPlan::Dest::kEXPLAIN);
    }
    return root_plan;
  }
  return nullptr;
}

std::string MapDHandler::parse_to_ra(
    const std::string& query_str,
    const std::vector<TFilterPushDownInfo>& filter_push_down_info,
    const Catalog_Namespace::SessionInfo& session_info,
    std::map<std::string, bool>* tableNames) {
  INJECT_TIMER(parse_to_ra);
  ParserWrapper pw{query_str};
  const std::string actual_query{
      pw.is_select_explain || pw.is_select_calcite_explain ? pw.actual_query : query_str};
  if (is_calcite_path_permissable(pw)) {
    auto result = calcite_->process(session_info,
                                    legacy_syntax_ ? pg_shim(actual_query) : actual_query,
                                    filter_push_down_info,
                                    legacy_syntax_,
                                    pw.is_select_calcite_explain);
    if (tableNames) {
      for (const auto& table : result.resolved_accessed_objects.tables_selected_from) {
        (*tableNames)[table] = false;
      }
      for (const auto& tables :
           std::vector<decltype(result.resolved_accessed_objects.tables_inserted_into)>{
               result.resolved_accessed_objects.tables_inserted_into,
               result.resolved_accessed_objects.tables_updated_in,
               result.resolved_accessed_objects.tables_deleted_from}) {
        for (const auto& table : tables) {
          (*tableNames)[table] = true;
        }
      }
    }
    return result.plan_result;
  }
  return "";
}

void MapDHandler::check_table_consistency(TTableMeta& _return,
                                          const TSessionId& session,
                                          const int32_t table_id) {
  if (!leaf_handler_) {
    THROW_MAPD_EXCEPTION("Distributed support is disabled.");
  }
  auto time_ms = measure<>::execution([&]() {
    try {
      leaf_handler_->check_table_consistency(_return, session, table_id);
    } catch (std::exception& e) {
      THROW_MAPD_EXCEPTION(std::string("Exception: ") + e.what());
    }
  });
  LOG(INFO) << "Table consistency check COMPLETED in " << time_ms << " ms ";
}

void MapDHandler::start_query(TPendingQuery& _return,
                              const TSessionId& session,
                              const std::string& query_ra,
                              const bool just_explain) {
  if (!leaf_handler_) {
    THROW_MAPD_EXCEPTION("Distributed support is disabled.");
  }
  const auto session_it = get_session_it(session);
  LOG(INFO) << "start_query :" << *session_it->second << " :" << just_explain;
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

void MapDHandler::execute_first_step(TStepResult& _return,
                                     const TPendingQuery& pending_query) {
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

void MapDHandler::insert_data(const TSessionId& session,
                              const TInsertData& thrift_insert_data) {
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
    } else if (ti.is_geometry()) {
      none_encoded_string_columns.emplace_back(new std::vector<std::string>());
      auto& none_encoded_strings = none_encoded_string_columns.back();
      CHECK_EQ(static_cast<size_t>(thrift_insert_data.num_rows),
               thrift_insert_data.data[col_idx].var_len_data.size());
      for (const auto& varlen_str : thrift_insert_data.data[col_idx].var_len_data) {
        none_encoded_strings->push_back(varlen_str.payload);
      }
      p.stringsPtr = none_encoded_strings.get();
    } else {
      CHECK(ti.is_array());
      array_columns.emplace_back(new std::vector<ArrayDatum>());
      auto& array_column = array_columns.back();
      CHECK_EQ(static_cast<size_t>(thrift_insert_data.num_rows),
               thrift_insert_data.data[col_idx].var_len_data.size());
      for (const auto& t_arr_datum : thrift_insert_data.data[col_idx].var_len_data) {
        if (t_arr_datum.is_null) {
          array_column->emplace_back(0, nullptr, true);
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
  insert_data.numRows = thrift_insert_data.num_rows;
  const auto td = cat.getMetadataForTable(insert_data.tableId);
  try {
    // this should have the same lock seq as COPY FROM
    ChunkKey chunkKey = {insert_data.databaseId, insert_data.tableId};
    mapd_unique_lock<mapd_shared_mutex> tableLevelWriteLock(
        *Lock_Namespace::LockMgr<mapd_shared_mutex, ChunkKey>::getMutex(
            Lock_Namespace::LockType::CheckpointLock, chunkKey));
    // [ write UpdateDeleteLocks ] lock is deferred in
    // InsertOrderFragmenter::deleteFragments
    td->fragmenter->insertDataNoCheckpoint(insert_data);
  } catch (const std::exception& e) {
    THROW_MAPD_EXCEPTION(std::string("Exception: ") + e.what());
  }
}

void MapDHandler::start_render_query(TPendingRenderQuery& _return,
                                     const TSessionId& session,
                                     const int64_t widget_id,
                                     const int16_t node_idx,
                                     const std::string& vega_json) {
  if (!render_handler_) {
    THROW_MAPD_EXCEPTION("Backend rendering is disabled.");
  }
  const auto session_it = get_session_it(session);
  LOG(INFO) << "start_render_query :" << *session_it->second
            << " :widget_id:" << widget_id << ":vega_json:" << vega_json;
  auto time_ms = measure<>::execution([&]() {
    try {
      render_handler_->start_render_query(
          _return, session, widget_id, node_idx, vega_json);
    } catch (std::exception& e) {
      THROW_MAPD_EXCEPTION(std::string("Exception: ") + e.what());
    }
  });
  LOG(INFO) << "start_render_query-COMPLETED " << time_ms << "ms "
            << "id is " << _return.id;
}

void MapDHandler::execute_next_render_step(TRenderStepResult& _return,
                                           const TPendingRenderQuery& pending_render,
                                           const TRenderAggDataMap& merged_data) {
  if (!render_handler_) {
    THROW_MAPD_EXCEPTION("Backend rendering is disabled.");
  }

  LOG(INFO) << "execute_next_render_step: id:" << pending_render.id;
  auto time_ms = measure<>::execution([&]() {
    try {
      render_handler_->execute_next_render_step(_return, pending_render, merged_data);
    } catch (std::exception& e) {
      THROW_MAPD_EXCEPTION(std::string("Exception: ") + e.what());
    }
  });
  LOG(INFO) << "execute_next_render_step-COMPLETED id: " << pending_render.id
            << ", time: " << time_ms << "ms ";
}

void MapDHandler::checkpoint(const TSessionId& session,
                             const int32_t db_id,
                             const int32_t table_id) {
  const auto session_info = get_session(session);
  auto& cat = session_info.get_catalog();
  cat.get_dataMgr().checkpoint(db_id, table_id);
}

// check and reset epoch if a request has been made
void MapDHandler::set_table_epoch(const TSessionId& session,
                                  const int db_id,
                                  const int table_id,
                                  const int new_epoch) {
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
  auto td = cat.getMetadataForTable(
      table_name,
      false);  // don't populate fragmenter on this call since we only want metadata
  int32_t db_id = cat.get_currentDB().dbId;
  if (leaf_aggregator_.leafCount() > 0) {
    return leaf_aggregator_.set_table_epochLeaf(
        session_info, db_id, td->tableId, new_epoch);
  }
  cat.setTableEpoch(db_id, td->tableId, new_epoch);
}

int32_t MapDHandler::get_table_epoch(const TSessionId& session,
                                     const int32_t db_id,
                                     const int32_t table_id) {
  const auto session_info = get_session(session);
  auto& cat = session_info.get_catalog();

  if (leaf_aggregator_.leafCount() > 0) {
    return leaf_aggregator_.get_table_epochLeaf(session_info, db_id, table_id);
  }
  return cat.getTableEpoch(db_id, table_id);
}

int32_t MapDHandler::get_table_epoch_by_name(const TSessionId& session,
                                             const std::string& table_name) {
  const auto session_info = get_session(session);
  auto& cat = session_info.get_catalog();
  auto td = cat.getMetadataForTable(
      table_name,
      false);  // don't populate fragmenter on this call since we only want metadata
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

void MapDHandler::get_license_claims(TLicenseInfo& _return,
                                     const TSessionId& session,
                                     const std::string& nonce) {
  const auto session_info = get_session(session);
  _return.claims.push_back("");
}

void MapDHandler::shutdown() {
  if (calcite_) {
    calcite_->close_calcite_server();
  }

  if (render_handler_) {
    render_handler_->shutdown();
  }
}
