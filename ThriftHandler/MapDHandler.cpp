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
#ifdef HAVE_PROFILER
#include <gperftools/heap-profiler.h>
#endif  // HAVE_PROFILER
#include <thrift/concurrency/ThreadManager.h>
#include <thrift/concurrency/PlatformThreadFactory.h>
#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/protocol/TJSONProtocol.h>
#include <thrift/server/TThreadPoolServer.h>
#include <thrift/server/TThreadedServer.h>
#include <thrift/transport/THttpServer.h>
#include <thrift/transport/TServerSocket.h>
#include <thrift/transport/TBufferTransports.h>

#include "MapDRelease.h"

#include "Calcite/Calcite.h"

#include "QueryEngine/RelAlgExecutor.h"

#include "Catalog/Catalog.h"
#include "Fragmenter/InsertOrderFragmenter.h"
#include "Import/Importer.h"
#include "Parser/parser.h"
#include "Parser/ParserWrapper.h"
#include "Parser/ReservedKeywords.h"
#include "Planner/Planner.h"
#include "QueryEngine/CalciteAdapter.h"
#include "QueryEngine/Execute.h"
#include "QueryEngine/GpuMemUtils.h"
#include "QueryEngine/ExtensionFunctionsWhitelist.h"
#include "QueryEngine/JsonAccessors.h"
#include "Shared/geosupport.h"
#include "Shared/mapd_shared_mutex.h"
#include "Shared/measure.h"
#include "Shared/scope.h"
#include "Shared/StringTransform.h"
#include "Shared/MapDParameters.h"

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/filesystem.hpp>
#include <boost/make_shared.hpp>
#include <boost/program_options.hpp>
#include <boost/regex.hpp>
#include <boost/tokenizer.hpp>
#include <future>
#include <memory>
#include <string>
#include <fstream>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <random>
#include <map>
#include <cmath>
#include <typeinfo>
#include <thread>
#include <glog/logging.h>
#include <signal.h>
#include <unistd.h>
#include <fcntl.h>
#include <regex>

#ifdef ENABLE_ARROW_CONVERTER
#include "arrow/buffer.h"
#endif  // ENABLE_ARROW_CONVERTER

#define INVALID_SESSION_ID ""

std::string image_from_rendered_rows(const ResultRows& rendered_results) {
  const auto img_row = rendered_results.getNextRow(false, false);
  CHECK_EQ(size_t(1), img_row.size());
  const auto& img_tv = img_row.front();
  const auto scalar_tv = boost::get<ScalarTargetValue>(&img_tv);
  const auto nullable_sptr = boost::get<NullableString>(scalar_tv);
  CHECK(nullable_sptr);
  auto sptr = boost::get<std::string>(nullable_sptr);
  CHECK(sptr);
  return *sptr;
}

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
                         const std::string& start_epoch_table_name,
                         const int start_epoch,
                         const bool is_decr_start_epoch,
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
      enable_rendering_(enable_rendering),
      legacy_syntax_(legacy_syntax),
      start_epoch_table_name_(start_epoch_table_name),
      start_epoch_(start_epoch),
      is_decr_start_epoch_(is_decr_start_epoch),
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
  if (enable_rendering_) {
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
}

MapDHandler::~MapDHandler() {
  LOG(INFO) << "mapd_server exits." << std::endl;
}

void MapDHandler::check_read_only(const std::string& str) {
  if (MapDHandler::read_only_) {
    TMapDException ex;
    ex.error_msg = str + " disabled: server running in read-only mode.";
    LOG(ERROR) << ex.error_msg;
    throw ex;
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
    TMapDException ex;
    ex.error_msg = std::string("User ") + user + " does not exist.";
    LOG(ERROR) << ex.error_msg;
    throw ex;
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
    TMapDException ex;
    ex.error_msg = std::string("User ") + user + " does not exist.";
    LOG(ERROR) << ex.error_msg;
    throw ex;
  }
  if (!super_user_rights_) {
    if (!sys_cat_->checkPasswordForUser(passwd, user_meta)) {
      TMapDException ex;
      ex.error_msg = std::string("Password for User ") + user + " is incorrect.";
      LOG(ERROR) << ex.error_msg;
      throw ex;
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
    TMapDException ex;
    ex.error_msg = std::string("Database ") + dbname + " does not exist.";
    LOG(ERROR) << ex.error_msg;
    throw ex;
  }
  // insert privilege is being treated as access allowed for now
  Privileges privs;
  privs.insert_ = true;
  privs.select_ = false;
  if (!access_priv_check_) {  // proceed with old style access priv check for DB only
    if (!sys_cat_->checkPrivileges(user_meta, db_meta, privs)) {
      TMapDException ex;
      ex.error_msg = std::string("User ") + user + " is not authorized to access database " + dbname;
      LOG(ERROR) << ex.error_msg;
      throw ex;
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
    const auto start_epoch_session_info_ptr = sessions_[session];
    set_table_start_epoch(*start_epoch_session_info_ptr);
  } else {
    sessions_[session].reset(
        new Catalog_Namespace::SessionInfo(cat_it->second, user_meta, executor_device_type_, session));
    const auto start_epoch_session_info_ptr = sessions_[session];
    set_table_start_epoch(*start_epoch_session_info_ptr);
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
  _return.rendering_enabled = enable_rendering_;
  _return.start_time = start_time_;
  _return.edition = MAPD_EDITION;
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
                              const int32_t first_n) {
  const auto session_info = MapDHandler::get_session(session);
  if (leaf_aggregator_.leafCount() > 0) {
  } else {
    MapDHandler::sql_execute_impl(
        _return, session_info, query_str, column_format, nonce, session_info.get_executor_device_type(), first_n);
  }
}

void MapDHandler::sql_execute_df(TDataFrame& _return,
                                 const TSessionId& session,
                                 const std::string& query_str,
                                 const TDeviceType::type device_type,
                                 const int32_t device_id,
                                 const int32_t first_n) {
#ifdef ENABLE_ARROW_CONVERTER
  const auto session_info = MapDHandler::get_session(session);
  int64_t execution_time_ms = 0;
  if (device_type == TDeviceType::GPU) {
    const auto executor_device_type = session_info.get_executor_device_type();
    if (executor_device_type != ExecutorDeviceType::GPU) {
      TMapDException ex;
      ex.error_msg = std::string("Exception: GPU mode is not allowed in this session");
      LOG(ERROR) << ex.error_msg;
      throw ex;
    }
    if (!data_mgr_->gpusPresent()) {
      TMapDException ex;
      ex.error_msg = std::string("Exception: no GPU is available in this server");
      LOG(ERROR) << ex.error_msg;
      throw ex;
    }
    if (device_id < 0 || device_id >= data_mgr_->cudaMgr_->getDeviceCount()) {
      TMapDException ex;
      ex.error_msg = std::string("Exception: invalid device_id or unavailable GPU with this ID");
      LOG(ERROR) << ex.error_msg;
      throw ex;
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
      TMapDException ex;
      ex.error_msg = std::string("Exception: ") + e.what();
      LOG(ERROR) << ex.error_msg;
      throw ex;
    }
    TMapDException ex;
    ex.error_msg = std::string("Exception: DDL or update DML are not unsupported by current thrift API");
    LOG(ERROR) << ex.error_msg;
    throw ex;
  });
  LOG(INFO) << "Total: " << total_time_ms << " (ms), Execution: " << execution_time_ms << " (ms)";
#endif  // ENABLE_ARROW_CONVERTER
}

void MapDHandler::sql_execute_gdf(TDataFrame& _return,
                                  const TSessionId& session,
                                  const std::string& query_str,
                                  const int32_t device_id,
                                  const int32_t first_n) {
#ifdef ENABLE_ARROW_CONVERTER
  sql_execute_df(_return, session, query_str, TDeviceType::GPU, device_id, first_n);
#endif
}

namespace {

std::string apply_copy_to_shim(const std::string& query_str) {
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

}  // namespace

void MapDHandler::sql_validate(TTableDescriptor& _return, const TSessionId& session, const std::string& query_str) {
  std::unique_ptr<const Planner::RootPlan> root_plan;
  const auto session_info = get_session(session);
  ParserWrapper pw{query_str};
  if (pw.is_select_explain || pw.is_other_explain || pw.is_ddl || pw.is_update_dml) {
    TMapDException ex;
    ex.error_msg = "Can only validate SELECT statements.";
    LOG(ERROR) << ex.error_msg;
    throw ex;
  }
  MapDHandler::validate_rel_alg(_return, query_str, session_info);
}

void MapDHandler::validate_rel_alg(TTableDescriptor& _return,
                                   const std::string& query_str,
                                   const Catalog_Namespace::SessionInfo& session_info) {
  try {
    const auto query_ra = parse_to_ra(query_str, session_info);
    TQueryResult result;
    MapDHandler::execute_rel_alg(result, query_ra, true, session_info, ExecutorDeviceType::CPU, -1, false, true);
    const auto& row_desc = fixup_row_descriptor(result.row_set.row_desc, session_info.get_catalog());
    for (const auto& col_desc : row_desc) {
      const auto it_ok = _return.insert(std::make_pair(col_desc.col_name, col_desc));
      CHECK(it_ok.second);
    }
  } catch (std::exception& e) {
    TMapDException ex;
    ex.error_msg = std::string("Exception: ") + e.what();
    LOG(ERROR) << ex.error_msg;
    throw ex;
  }
}

// DEPRECATED - use get_row_for_pixel()
void MapDHandler::get_rows_for_pixels(TPixelResult& _return,
                                      const TSessionId& session,
                                      const int64_t widget_id,
                                      const std::vector<TPixel>& pixels,
                                      const std::string& table_name,
                                      const std::vector<std::string>& col_names,
                                      const bool column_format,
                                      const std::string& nonce) {
  LOG(ERROR) << "get_rows_for_pixels is deprecated, please fix application";
}

// DEPRECATED - use get_result_row_for_pixel()
void MapDHandler::get_row_for_pixel(TPixelRowResult& _return,
                                    const TSessionId& session,
                                    const int64_t widget_id,
                                    const TPixel& pixel,
                                    const std::string& table_name,
                                    const std::vector<std::string>& col_names,
                                    const bool column_format,
                                    const int32_t pixelRadius,
                                    const std::string& nonce) {
  LOG(ERROR) << "get_rows_for_pixel is deprecated, please fix application";
}

void MapDHandler::get_result_row_for_pixel(TPixelTableRowResult& _return,
                                           const TSessionId& session,
                                           const int64_t widget_id,
                                           const TPixel& pixel,
                                           const std::map<std::string, std::vector<std::string>>& table_col_names,
                                           const bool column_format,
                                           const int32_t pixelRadius,
                                           const std::string& nonce) {
  _return.nonce = nonce;
  if (!enable_rendering_) {
    TMapDException ex;
    ex.error_msg = "Backend rendering is disabled.";
    LOG(ERROR) << ex.error_msg;
    throw ex;
  }
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
    if (!cat->getMetadataForDict(dict_id)) {
      col_type.col_type.comp_param = 0;
      return col_type;
    }
    auto dd = cat->getMetadataForDict(dict_id, false);
    if (!dd) {
      TMapDException ex;
      ex.error_msg = "Dictionary doesn't exist";
      LOG(ERROR) << ex.error_msg;
      throw ex;
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
  auto td = cat.getMetadataForTable(table_name);
  if (!td) {
    TMapDException ex;
    ex.error_msg = "Table " + table_name + " doesn't exist";
    LOG(ERROR) << ex.error_msg;
    throw ex;
  }
  if (td->isView) {
    try {
      const auto query_ra = parse_to_ra(td->viewSQL, session_info);
      TQueryResult result;
      execute_rel_alg(result, query_ra, true, session_info, ExecutorDeviceType::CPU, -1, false, true);
      _return.row_desc = fixup_row_descriptor(result.row_set.row_desc, cat);
    } catch (std::exception& e) {
      TMapDException ex;
      ex.error_msg = std::string("Exception: ") + e.what();
      LOG(ERROR) << ex.error_msg;
      throw ex;
    }
  } else {
    const auto col_descriptors = cat.getAllColumnMetadataForTable(td->tableId, get_system, true);
    for (const auto cd : col_descriptors) {
      _return.row_desc.push_back(populateThriftColumnType(&cat, cd));
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
    TMapDException ex;
    ex.error_msg = "View " + view_name + " doesn't exist";
    LOG(ERROR) << ex.error_msg;
    throw ex;
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
    TMapDException ex;
    ex.error_msg = "Link " + link + " is not valid.";
    LOG(ERROR) << ex.error_msg;
    throw ex;
  }
  _return.view_state = ld->viewState;
  _return.view_name = ld->link;
  _return.update_time = ld->updateTime;
  _return.view_metadata = ld->viewMetadata;
}

void MapDHandler::get_tables(std::vector<std::string>& table_names, const TSessionId& session) {
  const auto session_info = get_session(session);
  auto& cat = session_info.get_catalog();
  const auto tables = cat.getAllTableMetadata();
  for (const auto td : tables) {
    if (td->shard >= 0) {
      // skip shards, they're not standalone tables
      continue;
    }
    table_names.push_back(td->tableName);
  }
}

void MapDHandler::get_users(std::vector<std::string>& user_names, const TSessionId& session) {
  const auto session_info = get_session(session);
  std::list<Catalog_Namespace::UserMetadata> user_list = sys_cat_->getAllUserMetadata();
  for (auto u : user_list) {
    user_names.push_back(u.userName);
  }
}

void MapDHandler::get_version(std::string& version) {
  version = MAPD_RELEASE;
}

void MapDHandler::get_memory_gpu(std::string& memory, const TSessionId& session) {
  const auto session_info = get_session(session);
  memory = sys_cat_->get_dataMgr().dumpLevel(MemoryLevel::GPU_LEVEL);
}

void MapDHandler::get_memory_cpu(std::string& memory, const TSessionId& session) {
  const auto session_info = get_session(session);
  memory = sys_cat_->get_dataMgr().dumpLevel(MemoryLevel::CPU_LEVEL);
}

void MapDHandler::clear_gpu_memory(const TSessionId& session) {
  const auto session_info = get_session(session);
  sys_cat_->get_dataMgr().clearMemory(MemoryLevel::GPU_LEVEL);
}

void MapDHandler::clear_cpu_memory(const TSessionId& session) {
  const auto session_info = get_session(session);
  sys_cat_->get_dataMgr().clearMemory(MemoryLevel::CPU_LEVEL);
}

TSessionId MapDHandler::getInvalidSessionId() const {
  return INVALID_SESSION_ID;
}

// void get_memory_summary(std::string& memory) { memory = sys_cat_->get_dataMgr().getMemorySummary(); }

void MapDHandler::get_memory_summary(TMemorySummary& memory, const TSessionId& session) {
  const auto session_info = get_session(session);
  Data_Namespace::memorySummary internal_memory = sys_cat_->get_dataMgr().getMemorySummary();
  memory.cpu_memory_in_use = internal_memory.cpuMemoryInUse;
  for (auto gpu : internal_memory.gpuSummary) {
    TGpuMemorySummary gs;
    gs.in_use = gpu.inUse;
    gs.max = gpu.max;
    gs.allocated = gpu.allocated;
    gs.is_allocation_capped = gpu.isAllocationCapped;
    memory.gpu_summary.push_back(gs);
  }
}

void MapDHandler::get_databases(std::vector<TDBInfo>& dbinfos, const TSessionId& session) {
  const auto session_info = get_session(session);
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
    TMapDException ex;
    ex.error_msg = "Table " + table_name + " does not exist.";
    LOG(ERROR) << ex.error_msg;
    throw ex;
  }
  std::unique_ptr<Importer_NS::Loader> loader;
  if (leaf_aggregator_.leafCount() > 0) {
    loader.reset(new DistributedLoader(session_info, td, &leaf_aggregator_));
  } else {
    loader.reset(new Importer_NS::Loader(cat, td));
  }
  // TODO(andrew): nColumns should be number of non-virtual/non-system columns.
  //               Subtracting 1 (rowid) until TableDescriptor is updated.
  if (rows.front().cols.size() != static_cast<size_t>(td->nColumns) - 1) {
    TMapDException ex;
    ex.error_msg = "Wrong number of columns to load into Table " + table_name;
    LOG(ERROR) << ex.error_msg;
    throw ex;
  }
  auto col_descs = loader->get_column_descs();
  std::vector<std::unique_ptr<Importer_NS::TypedImportBuffer>> import_buffers;
  for (auto cd : col_descs) {
    import_buffers.push_back(std::unique_ptr<Importer_NS::TypedImportBuffer>(
        new Importer_NS::TypedImportBuffer(cd, loader->get_string_dict(cd))));
  }
  for (auto row : rows) {
    try {
      int col_idx = 0;
      for (auto cd : col_descs) {
        import_buffers[col_idx]->add_value(cd, row.cols[col_idx], row.cols[col_idx].is_null);
        col_idx++;
      }
    } catch (const std::exception& e) {
      LOG(WARNING) << "load_table exception thrown: " << e.what() << ". Row discarded.";
    }
  }
  loader->load(import_buffers, rows.size());
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
    TMapDException ex;
    ex.error_msg = "Table " + table_name + " does not exist.";
    LOG(ERROR) << ex.error_msg;
    throw ex;
  }
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
    TMapDException ex;
    ex.error_msg = "Wrong number of columns to load into Table " + table_name + " (" +
                   std::to_string(rows.front().cols.size()) + " vs " + std::to_string(td->nColumns - 1) + ")";
    LOG(ERROR) << ex.error_msg;
    throw ex;
  }
  auto col_descs = loader->get_column_descs();
  std::vector<std::unique_ptr<Importer_NS::TypedImportBuffer>> import_buffers;
  for (auto cd : col_descs) {
    import_buffers.push_back(std::unique_ptr<Importer_NS::TypedImportBuffer>(
        new Importer_NS::TypedImportBuffer(cd, loader->get_string_dict(cd))));
  }
  size_t rows_completed = 0;
  size_t col_idx = 0;
  for (auto row : rows) {
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
    TMapDException ex;
    ex.error_msg = "File does not exist: " + file_path.string();
    LOG(ERROR) << ex.error_msg;
    throw ex;
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
    TMapDException ex;
    ex.error_msg = "detect_column_types error: " + std::string(e.what());
    LOG(ERROR) << ex.error_msg;
    throw ex;
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
    TMapDException ex;
    ex.error_msg = std::string("Exception: ") + e.what();
    LOG(ERROR) << ex.error_msg;
    throw ex;
  }
  if (num_parse_errors > 0) {
    TMapDException ex;
    ex.error_msg = "Syntax error at: " + last_parsed;
    LOG(ERROR) << ex.error_msg;
    throw ex;
  }
  if (parse_trees.size() != 1) {
    TMapDException ex;
    ex.error_msg = "Can only " + action + " a single query at a time.";
    LOG(ERROR) << ex.error_msg;
    throw ex;
  }
  Parser::Stmt* stmt = parse_trees.front().get();
  Parser::DDLStmt* ddl = dynamic_cast<Parser::DDLStmt*>(stmt);
  if (ddl != nullptr) {
    TMapDException ex;
    ex.error_msg = "Can only " + action + " SELECT statements.";
    LOG(ERROR) << ex.error_msg;
    throw ex;
  }
  auto dml = static_cast<Parser::DMLStmt*>(stmt);
  Analyzer::Query query;
  dml->analyze(cat, query);
  Planner::Optimizer optimizer(query, cat);
  return optimizer.optimize();
}

void MapDHandler::render(TRenderResult& _return,
                         const TSessionId& session,
                         const std::string& query_str_in,
                         const std::string& render_type,
                         const std::string& nonce) {
  LOG(ERROR) << "render is deprecated, please fix application";
}

void MapDHandler::render_vega(TRenderResult& _return,
                              const TSessionId& session,
                              const int64_t widget_id,
                              const std::string& vega_json,
                              const int compressionLevel,
                              const std::string& nonce) {
  _return.total_time_ms = measure<>::execution([&]() {
    _return.execution_time_ms = 0;
    _return.render_time_ms = 0;
    _return.nonce = nonce;
    if (!enable_rendering_) {
      TMapDException ex;
      ex.error_msg = "Backend rendering is disabled.";
      LOG(ERROR) << ex.error_msg;
      throw ex;
    }

    std::lock_guard<std::mutex> render_lock(render_mutex_);
    mapd_shared_lock<mapd_shared_mutex> read_lock(sessions_mutex_);
    auto session_it = get_session_it(session);
    auto session_info_ptr = session_it->second.get();

    const auto& cat = session_info_ptr->get_catalog();
    auto executor = Executor::getExecutor(
        cat.get_currentDB().dbId, jit_debug_ ? "/tmp" : "", jit_debug_ ? "mapdquery" : "", mapd_parameters_, nullptr);

  });
  LOG(INFO) << "Total: " << _return.total_time_ms << " (ms), Total Execution: " << _return.execution_time_ms
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
    TMapDException ex;
    ex.error_msg = std::string("Exception: ") + e.what();
    LOG(ERROR) << ex.error_msg;
    throw ex;
  }
}

void MapDHandler::delete_frontend_view(const TSessionId& session, const std::string& view_name) {
  check_read_only("delete_frontend_view");
  const auto session_info = get_session(session);
  auto& cat = session_info.get_catalog();
  auto vd = cat.getMetadataForFrontendView(std::to_string(session_info.get_currentUser().userId), view_name);
  if (!vd) {
    TMapDException ex;
    ex.error_msg = "View " + view_name + " doesn't exist";
    LOG(ERROR) << ex.error_msg;
    throw ex;
  }
  try {
    cat.deleteMetadataForFrontendView(std::to_string(session_info.get_currentUser().userId), view_name);
  } catch (const std::exception& e) {
    TMapDException ex;
    ex.error_msg = std::string("Exception: ") + e.what();
    LOG(ERROR) << ex.error_msg;
    throw ex;
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
    TMapDException ex;
    ex.error_msg = std::string("Exception: ") + e.what();
    LOG(ERROR) << ex.error_msg;
    throw ex;
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
    TMapDException ex;
    ex.error_msg = "Invalid characters in table name: " + table_name;
    LOG(ERROR) << ex.error_msg;
    throw ex;
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
      TMapDException ex;
      ex.error_msg = "Invalid characters in column name: " + col.col_name;
      LOG(ERROR) << ex.error_msg;
      throw ex;
    }
    if (col.col_type.type == TDatumType::INTERVAL_DAY_TIME || col.col_type.type == TDatumType::INTERVAL_YEAR_MONTH) {
      TMapDException ex;
      ex.error_msg = "Unsupported type: " + thrift_to_name(col.col_type) + " for column: " + col.col_name;
      LOG(ERROR) << ex.error_msg;
      throw ex;
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
  sql_execute(ret, session, stmt, true, "", -1);
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
    TMapDException ex;
    ex.error_msg = "Table " + table_name + " does not exist.";
    LOG(ERROR) << ex.error_msg;
    throw ex;
  }

  auto file_path = import_path_ / session / boost::filesystem::path(file_name).filename();
  if (!boost::filesystem::exists(file_path)) {
    TMapDException ex;
    ex.error_msg = "File does not exist: " + file_path.filename().string();
    LOG(ERROR) << ex.error_msg << " at " << file_path.string();
    throw ex;
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
    TMapDException ex;
    ex.error_msg = "Exception: " + std::string(e.what());
    LOG(ERROR) << ex.error_msg;
    throw ex;
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
    TMapDException ex;
    ex.error_msg = "File does not exist: " + file_path.filename().string();
    LOG(ERROR) << ex.error_msg << " at " << file_path.string();
    throw ex;
  }
  try {
    check_geospatial_files(file_path);
  } catch (const std::exception& e) {
    TMapDException ex;
    ex.error_msg = "import_geo_table error: " + std::string(e.what());
    LOG(ERROR) << ex.error_msg;
    throw ex;
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
    TMapDException ex;
    ex.error_msg =
        "Could not append file " + file_path.filename().string() + " to " + table_name + ": not currently supported.";
    LOG(ERROR) << ex.error_msg;
    throw ex;
  }

  std::map<std::string, std::string> colname_to_src;
  for (auto r : rd) {
    colname_to_src[r.col_name] = r.src_name.length() > 0 ? r.src_name : sanitize_name(r.src_name);
  }

  const TableDescriptor* td = cat.getMetadataForTable(table_name);
  if (td == nullptr) {
    TMapDException ex;
    ex.error_msg = "Table " + table_name + " does not exist.";
    LOG(ERROR) << ex.error_msg;
    throw ex;
  }

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
    TMapDException ex;
    ex.error_msg = std::string("import_geo_table failed: ") + e.what();
    LOG(ERROR) << ex.error_msg;
    throw ex;
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
    throw_profile_exception("Profiler already started");
  }
  HeapProfilerStart("mapd");
#else
  throw_profile_exception("Profiler not enabled");
#endif  // HAVE_PROFILER
}

void MapDHandler::stop_heap_profile(const TSessionId& session) {
  const auto session_info = get_session(session);
#ifdef HAVE_PROFILER
  if (!IsHeapProfilerRunning()) {
    throw_profile_exception("Profiler not running");
  }
  HeapProfilerStop();
#else
  throw_profile_exception("Profiler not enabled");
#endif  // HAVE_PROFILER
}

void MapDHandler::get_heap_profile(std::string& profile, const TSessionId& session) {
  const auto session_info = get_session(session);
#ifdef HAVE_PROFILER
  if (!IsHeapProfilerRunning()) {
    throw_profile_exception("Profiler not running");
  }
  auto profile_buff = GetHeapProfile();
  profile = profile_buff;
  free(profile_buff);
#else
  throw_profile_exception("Profiler not enabled");
#endif  // HAVE_PROFILER
}

SessionMap::iterator MapDHandler::get_session_it(const TSessionId& session) {
  auto session_it = sessions_.find(session);
  if (session_it == sessions_.end()) {
    TMapDException ex;
    ex.error_msg = "Session not valid.";
    LOG(ERROR) << ex.error_msg;
    throw ex;
  }
  session_it->second->update_time();
  return session_it;
}

Catalog_Namespace::SessionInfo MapDHandler::get_session(const TSessionId& session) {
  mapd_shared_lock<mapd_shared_mutex> read_lock(sessions_mutex_);
  return *get_session_it(session)->second;
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
  ExecutionResult result{ResultRows({}, {}, nullptr, nullptr, {}, executor_device_type), {}};
  _return.execution_time_ms +=
      measure<>::execution([&]() { result = ra_executor.executeRelAlgQuery(query_ra, co, eo, nullptr); });
  // reduce execution time by the time spent during queue waiting
  _return.execution_time_ms -= result.getRows().getQueueTime();
  if (just_explain) {
    convert_explain(_return, result.getRows(), column_format);
  } else {
    convert_rows(_return, result.getTargetsMeta(), result.getRows(), column_format, first_n);
  }
}

#ifdef ENABLE_ARROW_CONVERTER
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
  ExecutionResult result{ResultRows({}, {}, nullptr, nullptr, {}, device_type), {}};
  result = ra_executor.executeRelAlgQuery(query_ra, co, eo, nullptr);
  if (auto rs = result.getRows().getResultSet()) {
    const auto copy =
        rs->getArrowCopy(data_mgr_.get(), device_type, device_id, getTargetNames(result.getTargetsMeta()));
    _return.sm_handle = std::string(copy.sm_handle.begin(), copy.sm_handle.end());
    _return.sm_size = copy.sm_size;
    _return.df_handle = std::string(copy.df_handle.begin(), copy.df_handle.end());
    _return.df_size = copy.df_size;
  } else {
    throw std::runtime_error("use-result-set might not be set");
  }
}
#endif  // ENABLE_ARROW_CONVERTER

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
                                        nullptr);
  ResultRows results({}, {}, nullptr, nullptr, {}, executor_device_type);
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
  _return.execution_time_ms -= results.getQueueTime();
  if (root_plan->get_plan_dest() == Planner::RootPlan::Dest::kEXPLAIN) {
    convert_explain(_return, results, column_format);
    return;
  }
  const auto plan = root_plan->get_plan();
  CHECK(plan);
  const auto& targets = plan->get_targetlist();
  convert_rows(_return, getTargetMetaInfo(targets), results, column_format, -1);
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

#ifdef ENABLE_ARROW_CONVERTER
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
#endif

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
                               const int32_t first_n) const {
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
                                       const ResultRows& results,
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

void MapDHandler::convert_explain(TQueryResult& _return, const ResultRows& results, const bool column_format) const {
  create_simple_result(_return, results, column_format, "Explanation");
}

void MapDHandler::convert_result(TQueryResult& _return, const ResultRows& results, const bool column_format) const {
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
                                   const int32_t first_n) {
  _return.nonce = nonce;
  _return.execution_time_ms = 0;
  auto& cat = session_info.get_catalog();
  LOG(INFO) << query_str;
  _return.total_time_ms = measure<>::execution([&]() {
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
          convert_explain(_return, ResultRows(query_ra, 0), true);
          return;
        }
        execute_rel_alg(
            _return, query_ra, column_format, session_info, executor_device_type, first_n, pw.is_select_explain, false);
        return;
      }
      LOG(INFO) << "passing query to legacy processor";
    } catch (std::exception& e) {
      TMapDException ex;
      ex.error_msg = std::string("Exception: ") + e.what();
      LOG(ERROR) << ex.error_msg;
      throw ex;
    }
    try {
      // check for COPY TO stmt replace as required parser expects #~# markers
      const auto result = apply_copy_to_shim(query_str);
      num_parse_errors = parser.parse(result, parse_trees, last_parsed);
    } catch (std::exception& e) {
      TMapDException ex;
      ex.error_msg = std::string("Exception: ") + e.what();
      LOG(ERROR) << ex.error_msg;
      throw ex;
    }
    if (num_parse_errors > 0) {
      TMapDException ex;
      ex.error_msg = "Syntax error at: " + last_parsed;
      LOG(ERROR) << ex.error_msg;
      throw ex;
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
            convert_result(_return, ResultRows(*copy_stmt->return_message.get(), 0), true);
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
        TMapDException ex;
        ex.error_msg = std::string("Exception: ") + e.what();
        LOG(ERROR) << ex.error_msg;
        throw ex;
      }
    }
  });
  LOG(INFO) << "Total: " << _return.total_time_ms << " (ms), Execution: " << _return.execution_time_ms << " (ms)";
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

std::vector<TColumnRange> MapDHandler::column_ranges_to_thrift(const AggregatedColRange& column_ranges) {
  std::vector<TColumnRange> thrift_column_ranges;
  const auto& column_ranges_map = column_ranges.asMap();
  for (const auto& kv : column_ranges_map) {
    TColumnRange thrift_column_range;
    thrift_column_range.col_id = kv.first.col_id;
    thrift_column_range.table_id = kv.first.table_id;
    const auto& expr_range = kv.second;
    switch (expr_range.getType()) {
      case ExpressionRangeType::Integer:
        thrift_column_range.type = TExpressionRangeType::INTEGER;
        thrift_column_range.int_min = expr_range.getIntMin();
        thrift_column_range.int_max = expr_range.getIntMax();
        thrift_column_range.bucket = expr_range.getBucket();
        thrift_column_range.has_nulls = expr_range.hasNulls();
        break;
      case ExpressionRangeType::Float:
      case ExpressionRangeType::Double:
        thrift_column_range.type = expr_range.getType() == ExpressionRangeType::Float ? TExpressionRangeType::FLOAT
                                                                                      : TExpressionRangeType::DOUBLE;
        thrift_column_range.fp_min = expr_range.getFpMin();
        thrift_column_range.fp_max = expr_range.getFpMax();
        thrift_column_range.has_nulls = expr_range.hasNulls();
        break;
      case ExpressionRangeType::Invalid:
        thrift_column_range.type = TExpressionRangeType::INVALID;
        break;
      default:
        CHECK(false);
    }
    thrift_column_ranges.push_back(thrift_column_range);
  }
  return thrift_column_ranges;
}

std::vector<TDictionaryGeneration> MapDHandler::string_dictionary_generations_to_thrift(
    const StringDictionaryGenerations& dictionary_generations) {
  std::vector<TDictionaryGeneration> thrift_dictionary_generations;
  for (const auto& kv : dictionary_generations.asMap()) {
    TDictionaryGeneration thrift_dictionary_generation;
    thrift_dictionary_generation.dict_id = kv.first;
    thrift_dictionary_generation.entry_count = kv.second;
    thrift_dictionary_generations.push_back(thrift_dictionary_generation);
  }
  return thrift_dictionary_generations;
}

std::vector<TTableGeneration> MapDHandler::table_generations_to_thrift(const TableGenerations& table_generations) {
  std::vector<TTableGeneration> thrift_table_generations;
  for (const auto& kv : table_generations.asMap()) {
    TTableGeneration table_generation;
    table_generation.table_id = kv.first;
    table_generation.start_rowid = kv.second.start_rowid;
    table_generation.tuple_count = kv.second.tuple_count;
    thrift_table_generations.push_back(table_generation);
  }
  return thrift_table_generations;
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
    td->fragmenter->insertData(insert_data);
  } catch (const std::exception& e) {
    TMapDException ex;
    ex.error_msg = std::string("Exception: ") + e.what();
    LOG(ERROR) << ex.error_msg;
    throw ex;
  }
}

/*
 * There's a lot of code duplication between this endpoint and render_vega,
 * we need to do something about it ASAP. A helper to create the QueryExecCB
 * lambda looks like an option, but there might be better ones.
 */
void MapDHandler::render_vega_raw_pixels(TRawPixelDataResult& _return,
                                         const TSessionId& session,
                                         const int64_t widget_id,
                                         const int16_t node_idx,
                                         const std::string& vega_json) {
  CHECK_EQ(size_t(0), leaf_aggregator_.leafCount());
  _return.total_time_ms = measure<>::execution([&]() {
    _return.execution_time_ms = 0;
    _return.render_time_ms = 0;
    if (!enable_rendering_) {
      TMapDException ex;
      ex.error_msg = "Backend rendering is disabled.";
      LOG(ERROR) << ex.error_msg;
      throw ex;
    }

    std::lock_guard<std::mutex> render_lock(render_mutex_);
    mapd_shared_lock<mapd_shared_mutex> read_lock(sessions_mutex_);
    auto session_it = get_session_it(session);
    auto session_info_ptr = session_it->second.get();

    const auto& cat = session_info_ptr->get_catalog();
    auto executor = Executor::getExecutor(
        cat.get_currentDB().dbId, jit_debug_ ? "/tmp" : "", jit_debug_ ? "mapdquery" : "", mapd_parameters_, nullptr);

  });
  LOG(INFO) << "Total: " << _return.total_time_ms << " (ms), Total Execution: " << _return.execution_time_ms
            << " (ms), Total Render: " << _return.render_time_ms << " (ms)";
}

void MapDHandler::throw_profile_exception(const std::string& error_msg) {
  TMapDException ex;
  ex.error_msg = error_msg;
  LOG(ERROR) << ex.error_msg;
  throw ex;
}

void MapDHandler::execute_first_step(TStepResult& _return, const TPendingQuery& pending_query) {
  CHECK(false);
}

void MapDHandler::start_query(TPendingQuery& _return,
                              const TSessionId& session,
                              const std::string& query_ra,
                              const bool just_explain) {
  CHECK(false);
}

void MapDHandler::broadcast_serialized_rows(const std::string& serialized_rows,
                                            const TRowDescriptor& row_desc,
                                            const TQueryId query_id) {
  CHECK(false);
}

void MapDHandler::set_table_start_epoch(const Catalog_Namespace::SessionInfo& session_info) {
  if (start_epoch_table_name_.length() > 0) {
    auto& cat = session_info.get_catalog();
    auto td = cat.getMetadataForTable(start_epoch_table_name_);
    if (!td) {
      TMapDException ex;
      if (!is_decr_start_epoch_) {
        ex.error_msg =
            "Unable to set epoch for table " + start_epoch_table_name_ + " because the table does not exist.";
      } else {
        ex.error_msg =
            "Unable to decrement epoch for table " + start_epoch_table_name_ + " because the table does not exist.";
      }
      LOG(ERROR) << ex.error_msg;
      throw ex;
    }
    int tb_id = static_cast<int>(td->tableId);
    int db_id = static_cast<int>(cat.get_currentDB().dbId);
    data_mgr_->updateTableEpoch(db_id, tb_id, start_epoch_, is_decr_start_epoch_);
  }
}
