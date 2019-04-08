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
 * File:   Calcite.cpp
 * Author: michael
 *
 * Created on November 23, 2015, 9:33 AM
 */

#include "Calcite.h"
#include "Shared/ConfigResolve.h"
#include "Shared/mapd_shared_ptr.h"
#include "Shared/mapdpath.h"
#include "Shared/measure.h"

#include <glog/logging.h>
#include <thread>
#include <utility>
#include "Catalog/Catalog.h"

#include "Shared/fixautotools.h"

#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/transport/TSocket.h>
#include <thrift/transport/TTransportUtils.h>

#include "Shared/fixautotools.h"

#include "gen-cpp/CalciteServer.h"

using namespace rapidjson;
using namespace apache::thrift;
using namespace apache::thrift::protocol;
using namespace apache::thrift::transport;

namespace {
template <typename XDEBUG_OPTION,
          typename REMOTE_DEBUG_OPTION,
          typename... REMAINING_ARGS>
int wrapped_execl(char const* path,
                  XDEBUG_OPTION&& x_debug,
                  REMOTE_DEBUG_OPTION&& remote_debug,
                  REMAINING_ARGS&&... standard_args) {
  if (std::is_same<JVMRemoteDebugSelector, PreprocessorTrue>::value) {
    return execl(
        path, x_debug, remote_debug, std::forward<REMAINING_ARGS>(standard_args)...);
  }
  return execl(path, std::forward<REMAINING_ARGS>(standard_args)...);
}
}  // namespace

static void start_calcite_server_as_daemon(const int mapd_port,
                                           const int port,
                                           const std::string& data_dir,
                                           const size_t calcite_max_mem,
                                           const std::string& ssl_trust_store,
                                           const std::string& ssl_trust_password) {
  // todo MAT all platforms seem to respect /usr/bin/java - this could be a gotcha on some
  // weird thing
  std::string const xDebug = "-Xdebug";
  std::string const remoteDebug =
      "-agentlib:jdwp=transport=dt_socket,server=y,suspend=n,address=5005";
  std::string xmxP = "-Xmx" + std::to_string(calcite_max_mem) + "m";
  std::string jarP = "-jar";
  std::string jarD =
      mapd_root_abs_path() + "/bin/calcite-1.0-SNAPSHOT-jar-with-dependencies.jar";
  std::string extensionsP = "-e";
  std::string extensionsD = mapd_root_abs_path() + "/QueryEngine/";
  std::string dataP = "-d";
  std::string dataD = data_dir;
  std::string localPortP = "-p";
  std::string localPortD = std::to_string(port);
  std::string mapdPortP = "-m";
  std::string mapdPortD = std::to_string(mapd_port);
  std::string mapdTrustStoreD = "-T";
  std::string mapdTrustPasswd = "-P";
  std::string mapdLogDirectory = "-DMAPD_LOG_DIR=" + data_dir;

  int pid = fork();
  if (pid == 0) {
    int i = wrapped_execl("/usr/bin/java",
                          xDebug.c_str(),
                          remoteDebug.c_str(),
                          xmxP.c_str(),
                          mapdLogDirectory.c_str(),
                          jarP.c_str(),
                          jarD.c_str(),
                          extensionsP.c_str(),
                          extensionsD.c_str(),
                          dataP.c_str(),
                          dataD.c_str(),
                          localPortP.c_str(),
                          localPortD.c_str(),
                          mapdPortP.c_str(),
                          mapdPortD.c_str(),
                          mapdTrustStoreD.c_str(),
                          ssl_trust_store.c_str(),
                          mapdTrustPasswd.c_str(),
                          ssl_trust_password.c_str(),
                          (char*)0);
    LOG(INFO) << " Calcite server running after exe, return " << i;
  }
}

std::pair<mapd::shared_ptr<CalciteServerClient>, mapd::shared_ptr<TTransport>> get_client(
    int port) {
  mapd::shared_ptr<TTransport> socket(new TSocket("localhost", port));
  mapd::shared_ptr<TTransport> transport(new TBufferedTransport(socket));
  try {
    transport->open();

  } catch (TException& tx) {
    throw tx;
  }
  mapd::shared_ptr<TProtocol> protocol(new TBinaryProtocol(transport));
  mapd::shared_ptr<CalciteServerClient> client;
  client.reset(new CalciteServerClient(protocol));
  std::pair<mapd::shared_ptr<CalciteServerClient>, mapd::shared_ptr<TTransport>> ret;
  return std::make_pair(client, transport);
}

void Calcite::runServer(const int mapd_port,
                        const int port,
                        const std::string& data_dir,
                        const size_t calcite_max_mem) {
  LOG(INFO) << "Running calcite server as a daemon";

  // ping server to see if for any reason there is an orphaned one
  int ping_time = ping();
  if (ping_time > -1) {
    // we have an orphaned server shut it down
    LOG(ERROR)
        << "Appears to be orphaned Calcite serve already running, shutting it down";
    LOG(ERROR) << "Please check that you are not trying to run two servers on same port";
    LOG(ERROR) << "Attempting to shutdown orphaned Calcite server";
    try {
      std::pair<mapd::shared_ptr<CalciteServerClient>, mapd::shared_ptr<TTransport>>
          clientP = get_client(remote_calcite_port_);
      clientP.first->shutdown();
      clientP.second->close();
      LOG(ERROR) << "orphaned Calcite server shutdown";

    } catch (TException& tx) {
      LOG(ERROR) << "Failed to shutdown orphaned Calcite server, reason: " << tx.what();
    }
  }

  // start the calcite server as a seperate process
  start_calcite_server_as_daemon(
      mapd_port, port, data_dir, calcite_max_mem, ssl_trust_store_, ssl_trust_password_);

  // check for new server for 5 seconds max
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  for (int i = 2; i < 50; i++) {
    int ping_time = ping();
    if (ping_time > -1) {
      LOG(INFO) << "Calcite server start took " << i * 100 << " ms ";
      LOG(INFO) << "ping took " << ping_time << " ms ";
      server_available_ = true;
      return;
    } else {
      // wait 100 ms
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }
  server_available_ = false;
  LOG(FATAL) << "No calcite remote server running on port " << port;
}

// ping existing server
// return -1 if no ping response
int Calcite::ping() {
  try {
    auto ms = measure<>::execution([&]() {
      std::pair<mapd::shared_ptr<CalciteServerClient>, mapd::shared_ptr<TTransport>>
          clientP = get_client(remote_calcite_port_);
      clientP.first->ping();
      clientP.second->close();
    });
    return ms;

  } catch (TException& tx) {
    return -1;
  }
}

Calcite::Calcite(const int mapd_port,
                 const int calcite_port,
                 const std::string& data_dir,
                 const size_t calcite_max_mem,
                 const std::string& session_prefix)
    : server_available_(false), session_prefix_(session_prefix) {
  init(mapd_port, calcite_port, data_dir, calcite_max_mem);
}

void Calcite::init(const int mapd_port,
                   const int calcite_port,
                   const std::string& data_dir,
                   const size_t calcite_max_mem) {
  LOG(INFO) << "Creating Calcite Handler,  Calcite Port is " << calcite_port
            << " base data dir is " << data_dir;
  if (calcite_port < 0) {
    CHECK(false) << "JNI mode no longer supported.";
  }
  if (calcite_port == 0) {
    // dummy process for initdb
    remote_calcite_port_ = calcite_port;
    server_available_ = false;
  } else {
    remote_calcite_port_ = calcite_port;
    runServer(mapd_port, calcite_port, data_dir, calcite_max_mem);
    server_available_ = true;
  }
}

Calcite::Calcite(const MapDParameters& mapd_parameter,
                 const std::string& data_dir,
                 const std::string& session_prefix)
    : ssl_trust_store_(mapd_parameter.ssl_trust_store)
    , ssl_trust_password_(mapd_parameter.ssl_trust_password)
    , session_prefix_(session_prefix) {
  init(mapd_parameter.omnisci_server_port,
       mapd_parameter.calcite_port,
       data_dir,
       mapd_parameter.calcite_max_mem);
}

void Calcite::updateMetadata(std::string catalog, std::string table) {
  if (server_available_) {
    auto ms = measure<>::execution([&]() {
      std::pair<mapd::shared_ptr<CalciteServerClient>, mapd::shared_ptr<TTransport>>
          clientP = get_client(remote_calcite_port_);
      clientP.first->updateMetadata(catalog, table);
      clientP.second->close();
    });
    LOG(INFO) << "Time to updateMetadata " << ms << " (ms)";
  } else {
    LOG(INFO) << "Not routing to Calcite, server is not up";
  }
}

void checkPermissionForTables(const Catalog_Namespace::SessionInfo& session_info,
                              std::vector<std::string> tableOrViewNames,
                              AccessPrivileges tablePrivs,
                              AccessPrivileges viewPrivs) {
  Catalog_Namespace::Catalog& catalog = session_info.getCatalog();

  for (auto tableOrViewName : tableOrViewNames) {
    const TableDescriptor* tableMeta =
        catalog.getMetadataForTable(tableOrViewName, false);

    if (!tableMeta) {
      throw std::runtime_error("unknown table of view: " + tableOrViewName);
    }

    DBObjectKey key;
    key.dbId = catalog.getCurrentDB().dbId;
    key.permissionType = tableMeta->isView ? DBObjectType::ViewDBObjectType
                                           : DBObjectType::TableDBObjectType;
    key.objectId = tableMeta->tableId;
    AccessPrivileges privs = tableMeta->isView ? viewPrivs : tablePrivs;
    DBObject dbobject(key, privs, tableMeta->userId);
    std::vector<DBObject> privObjects{dbobject};

    if (!privs.hasAny()) {
      throw std::runtime_error("Operation not supported for object " + tableOrViewName);
    }

    if (!Catalog_Namespace::SysCatalog::instance().checkPrivileges(
            session_info.get_currentUser(), privObjects)) {
      throw std::runtime_error("Violation of access privileges: user " +
                               session_info.get_currentUser().userName +
                               " has no proper privileges for object " + tableOrViewName);
    }
  }
}

TPlanResult Calcite::process(
    const Catalog_Namespace::SessionInfo& session_info,
    const std::string sql_string,
    const std::vector<TFilterPushDownInfo>& filter_push_down_info,
    const bool legacy_syntax,
    const bool is_explain) {
  TPlanResult result = processImpl(
      session_info, sql_string, filter_push_down_info, legacy_syntax, is_explain);

  AccessPrivileges NOOP;

  if (!is_explain && Catalog_Namespace::SysCatalog::instance().arePrivilegesOn()) {
    // check the individual tables
    checkPermissionForTables(session_info,
                             result.primary_accessed_objects.tables_selected_from,
                             AccessPrivileges::SELECT_FROM_TABLE,
                             AccessPrivileges::SELECT_FROM_VIEW);
    checkPermissionForTables(session_info,
                             result.primary_accessed_objects.tables_inserted_into,
                             AccessPrivileges::INSERT_INTO_TABLE,
                             NOOP);
    checkPermissionForTables(session_info,
                             result.primary_accessed_objects.tables_updated_in,
                             AccessPrivileges::UPDATE_IN_TABLE,
                             NOOP);
    checkPermissionForTables(session_info,
                             result.primary_accessed_objects.tables_deleted_from,
                             AccessPrivileges::DELETE_FROM_TABLE,
                             NOOP);
  }

  return result;
}

std::vector<TCompletionHint> Calcite::getCompletionHints(
    const Catalog_Namespace::SessionInfo& session_info,
    const std::vector<std::string>& visible_tables,
    const std::string sql_string,
    const int cursor) {
  std::vector<TCompletionHint> hints;
  auto& cat = session_info.getCatalog();
  const auto user = session_info.get_currentUser().userName;
  const auto session = session_info.get_session_id();
  const auto catalog = cat.getCurrentDB().dbName;
  auto client = get_client(remote_calcite_port_);
  client.first->getCompletionHints(
      hints, user, session, catalog, visible_tables, sql_string, cursor);
  return hints;
}

std::vector<std::string> Calcite::get_db_objects(const std::string ra) {
  std::vector<std::string> v_db_obj;
  Document document;
  document.Parse(ra.c_str());
  const Value& rels = document["rels"];
  CHECK(rels.IsArray());
  for (auto& v : rels.GetArray()) {
    std::string relOp(v["relOp"].GetString());
    if (!relOp.compare("EnumerableTableScan")) {
      std::string x;
      auto t = v["table"].GetArray();
      x = t[1].GetString();
      v_db_obj.push_back(x);
    }
  }

  return v_db_obj;
}

TPlanResult Calcite::processImpl(
    const Catalog_Namespace::SessionInfo& session_info,
    const std::string sql_string,
    const std::vector<TFilterPushDownInfo>& filter_push_down_info,
    const bool legacy_syntax,
    const bool is_explain) {
  auto& cat = session_info.getCatalog();
  std::string user = session_info.get_currentUser().userName;
  std::string session = session_info.get_session_id();
  if (!session_prefix_.empty()) {
    // preprend session prefix, if present
    session = session_prefix_ + "/" + session;
  }
  std::string catalog = cat.getCurrentDB().dbName;

  LOG(INFO) << "User " << user << " catalog " << catalog << " sql '" << sql_string << "'";
  if (server_available_) {
    TPlanResult ret;
    try {
      auto ms = measure<>::execution([&]() {
        std::pair<mapd::shared_ptr<CalciteServerClient>, mapd::shared_ptr<TTransport>>
            clientP = get_client(remote_calcite_port_);
        clientP.first->process(ret,
                               user,
                               session,
                               catalog,
                               sql_string,
                               filter_push_down_info,
                               legacy_syntax,
                               is_explain);
        clientP.second->close();
      });

      // LOG(INFO) << ret.plan_result;
      LOG(INFO) << "Time in Thrift "
                << (ms > ret.execution_time_ms ? ms - ret.execution_time_ms : 0)
                << " (ms), Time in Java Calcite server " << ret.execution_time_ms
                << " (ms)";
      return ret;
    } catch (InvalidParseRequest& e) {
      throw std::invalid_argument(e.whyUp);
    }
  } else {
    LOG(INFO) << "Not routing to Calcite, server is not up";
    TPlanResult ret;
    ret.plan_result = "";
    return ret;
  }
}

std::string Calcite::getExtensionFunctionWhitelist() {
  if (server_available_) {
    TPlanResult ret;
    std::string whitelist;

    std::pair<mapd::shared_ptr<CalciteServerClient>, mapd::shared_ptr<TTransport>>
        clientP = get_client(remote_calcite_port_);
    clientP.first->getExtensionFunctionWhitelist(whitelist);
    clientP.second->close();
    VLOG(1) << whitelist;
    return whitelist;
  } else {
    LOG(INFO) << "Not routing to Calcite, server is not up";
    return "";
  }
  CHECK(false);
  return "";
}

void Calcite::close_calcite_server() {
  if (server_available_) {
    LOG(INFO) << "Shutting down Calcite server";
    try {
      std::pair<mapd::shared_ptr<CalciteServerClient>, mapd::shared_ptr<TTransport>>
          clientP = get_client(remote_calcite_port_);
      clientP.first->shutdown();
      clientP.second->close();
    } catch (const std::exception& e) {
      LOG(ERROR) << "Error shutting down Calcite server: " << e.what();
    }
    LOG(INFO) << "shut down Calcite";
    server_available_ = false;
  }
}

Calcite::~Calcite() {
  LOG(INFO) << "Destroy Calcite Class";
  close_calcite_server();
  LOG(INFO) << "End of Calcite Destructor ";
}
