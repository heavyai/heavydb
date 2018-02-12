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
#include "../Shared/mapdpath.h"
#include "Shared/ConfigResolve.h"
#include "Shared/measure.h"

#include <glog/logging.h>
#include <thread>
#include <utility>
#include "Catalog/Catalog.h"

using namespace std;
using namespace rapidjson;
using namespace apache::thrift;
using namespace apache::thrift::protocol;
using namespace apache::thrift::transport;

namespace {
template <typename XDEBUG_OPTION, typename REMOTE_DEBUG_OPTION, typename... REMAINING_ARGS>
int wrapped_execl(char const* path,
                  XDEBUG_OPTION&& x_debug,
                  REMOTE_DEBUG_OPTION&& remote_debug,
                  REMAINING_ARGS&&... standard_args) {
  if (std::is_same<JVMRemoteDebugSelector, PreprocessorTrue>::value) {
    return execl(path, x_debug, remote_debug, std::forward<REMAINING_ARGS>(standard_args)...);
  }
  return execl(path, std::forward<REMAINING_ARGS>(standard_args)...);
}
}  // namespace

static void start_calcite_server_as_daemon(const int mapd_port,
                                           const int port,
                                           const std::string& data_dir,
                                           const size_t calcite_max_mem) {
  // todo MAT all platforms seem to respect /usr/bin/java - this could be a gotcha on some weird thing
  std::string const xDebug = "-Xdebug";
  std::string const remoteDebug = "-agentlib:jdwp=transport=dt_socket,server=y,suspend=n,address=5005";
  std::string xmxP = "-Xmx" + std::to_string(calcite_max_mem) + "m";
  std::string jarP = "-jar";
  std::string jarD = mapd_root_abs_path() + "/bin/calcite-1.0-SNAPSHOT-jar-with-dependencies.jar";
  std::string extensionsP = "-e";
  std::string extensionsD = mapd_root_abs_path() + "/QueryEngine/";
  std::string dataP = "-d";
  std::string dataD = data_dir;
  std::string localPortP = "-p";
  std::string localPortD = std::to_string(port);
  std::string mapdPortP = "-m";
  std::string mapdPortD = std::to_string(mapd_port);

  int pid = fork();
  if (pid == 0) {
    int i = wrapped_execl("/usr/bin/java",
                          xDebug.c_str(),
                          remoteDebug.c_str(),
                          xmxP.c_str(),
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
                          (char*)0);
    LOG(INFO) << " Calcite server running after exe, return " << i;
  }
}

std::pair<boost::shared_ptr<CalciteServerClient>, boost::shared_ptr<TTransport>> get_client(int port) {
  boost::shared_ptr<TTransport> socket(new TSocket("localhost", port));
  boost::shared_ptr<TTransport> transport(new TBufferedTransport(socket));
  try {
    transport->open();

  } catch (TException& tx) {
    throw tx;
  }
  boost::shared_ptr<TProtocol> protocol(new TBinaryProtocol(transport));
  boost::shared_ptr<CalciteServerClient> client;
  client.reset(new CalciteServerClient(protocol));
  std::pair<boost::shared_ptr<CalciteServerClient>, boost::shared_ptr<TTransport>> ret;
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
    LOG(ERROR) << "Appears to be orphaned Calcite serve already running, shutting it down";
    LOG(ERROR) << "Please check that you are not trying to run two servers on same port";
    LOG(ERROR) << "Attempting to shutdown orphaned Calcite server";
    try {
      std::pair<boost::shared_ptr<CalciteServerClient>, boost::shared_ptr<TTransport>> clientP =
          get_client(remote_calcite_port_);
      clientP.first->shutdown();
      clientP.second->close();
      LOG(ERROR) << "orphaned Calcite server shutdown";

    } catch (TException& tx) {
      LOG(ERROR) << "Failed to shutdown orphaned Calcite server, reason: " << tx.what();
    }
  }

  // start the calcite server as a seperate process
  start_calcite_server_as_daemon(mapd_port, port, data_dir, calcite_max_mem);

  // check for new server for 5 seconds max
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  for (int i = 2; i < 50; i++) {
    int ping_time = ping();
    if (ping_time > -1) {
      LOG(INFO) << "Calcite server start took " << i * 100 << " ms " << endl;
      LOG(INFO) << "ping took " << ping_time << " ms " << endl;
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
      std::pair<boost::shared_ptr<CalciteServerClient>, boost::shared_ptr<TTransport>> clientP =
          get_client(remote_calcite_port_);
      clientP.first->ping();
      clientP.second->close();
    });
    return ms;

  } catch (TException& tx) {
    return -1;
  }
}

Calcite::Calcite(const int mapd_port, const int port, const std::string& data_dir, const size_t calcite_max_mem)
    : server_available_(false) {
  LOG(INFO) << "Creating Calcite Handler,  Calcite Port is " << port << " base data dir is " << data_dir;
  if (port < 0) {
    CHECK(false) << "JNI mode no longer supported.";
  }
  if (port == 0) {
    // dummy process for initdb
    remote_calcite_port_ = port;
    server_available_ = false;
  } else {
    remote_calcite_port_ = port;
    runServer(mapd_port, port, data_dir, calcite_max_mem);
    server_available_ = true;
  }
}

void Calcite::updateMetadata(string catalog, string table) {
  if (server_available_) {
    auto ms = measure<>::execution([&]() {
      std::pair<boost::shared_ptr<CalciteServerClient>, boost::shared_ptr<TTransport>> clientP =
          get_client(remote_calcite_port_);
      clientP.first->updateMetadata(catalog, table);
      clientP.second->close();
    });
    LOG(INFO) << "Time to updateMetadata " << ms << " (ms)" << endl;
  } else {
    LOG(INFO) << "Not routing to Calcite, server is not up" << endl;
  }
}

string Calcite::process(const Catalog_Namespace::SessionInfo& session_info,
                        const string sql_string,
                        const bool legacy_syntax,
                        const bool is_explain) {
  std::string ra = processImpl(session_info, sql_string, legacy_syntax, is_explain);

  // gather tables used in this query
  if (!is_explain) {
    // TODO MAT we need to extend the output from calcite to include views
    // basically add a structure that returns all objects even if it is explain
    // security requires explains to be restricted in real life

    Catalog_Namespace::Catalog& catalog = session_info.get_catalog();
    if (Catalog_Namespace::SysCatalog::instance().arePrivilegesOn()) {
      std::vector<DBObject> privObjects;
      std::vector<std::string> v_db_obj = get_db_objects(ra);
      for (size_t i = 0; i < v_db_obj.size(); i++) {
        DBObject dbObject(v_db_obj[i], TableDBObjectType);
        dbObject.loadKey(catalog);
        dbObject.setPrivileges(AccessPrivileges::SELECT);
        privObjects.push_back(dbObject);
      }
      if (!Catalog_Namespace::SysCatalog::instance().checkPrivileges(session_info.get_currentUser(), privObjects)) {
        throw std::runtime_error("Violation of access privileges: user " + session_info.get_currentUser().userName +
                                 " has no proper select privileges.");
      }
    }
  }

  return ra;
}

std::vector<TCompletionHint> Calcite::getCompletionHints(const Catalog_Namespace::SessionInfo& session_info,
                                                         const std::vector<std::string>& visible_tables,
                                                         const std::string sql_string,
                                                         const int cursor) {
  std::vector<TCompletionHint> hints;
  auto& cat = session_info.get_catalog();
  const auto user = session_info.get_currentUser().userName;
  const auto session = session_info.get_session_id();
  const auto catalog = cat.get_currentDB().dbName;
  auto client = get_client(remote_calcite_port_);
  client.first->getCompletionHints(hints, user, session, catalog, visible_tables, sql_string, cursor);
  return hints;
}

std::vector<string> Calcite::get_db_objects(const std::string ra) {
  std::vector<string> v_db_obj;
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

string Calcite::processImpl(const Catalog_Namespace::SessionInfo& session_info,
                            const string sql_string,
                            const bool legacy_syntax,
                            const bool is_explain) {
  auto& cat = session_info.get_catalog();
  string user = session_info.get_currentUser().userName;
  string session = session_info.get_session_id();
  string catalog = cat.get_currentDB().dbName;

  LOG(INFO) << "User " << user << " catalog " << catalog << " sql '" << sql_string << "'";
  if (server_available_) {
    TPlanResult ret;
    try {
      auto ms = measure<>::execution([&]() {

        std::pair<boost::shared_ptr<CalciteServerClient>, boost::shared_ptr<TTransport>> clientP =
            get_client(remote_calcite_port_);
        clientP.first->process(ret, user, session, catalog, sql_string, legacy_syntax, is_explain);
        clientP.second->close();
      });

      // LOG(INFO) << ret.plan_result << endl;
      LOG(INFO) << "Time in Thrift " << (ms > ret.execution_time_ms ? ms - ret.execution_time_ms : 0)
                << " (ms), Time in Java Calcite server " << ret.execution_time_ms << " (ms)" << endl;
      return ret.plan_result;
    } catch (InvalidParseRequest& e) {
      throw std::invalid_argument(e.whyUp);
    }
  } else {
    LOG(INFO) << "Not routing to Calcite, server is not up" << endl;
    return "";
  }
}

string Calcite::getExtensionFunctionWhitelist() {
  if (server_available_) {
    TPlanResult ret;
    string whitelist;

    std::pair<boost::shared_ptr<CalciteServerClient>, boost::shared_ptr<TTransport>> clientP =
        get_client(remote_calcite_port_);
    clientP.first->getExtensionFunctionWhitelist(whitelist);
    clientP.second->close();
    LOG(INFO) << whitelist << endl;
    return whitelist;
  } else {
    LOG(INFO) << "Not routing to Calcite, server is not up" << endl;
    return "";
  }
  CHECK(false);
  return "";
}

Calcite::~Calcite() {
  LOG(INFO) << "Destroy Calcite Class" << std::endl;
  if (server_available_) {
    // running server
    std::pair<boost::shared_ptr<CalciteServerClient>, boost::shared_ptr<TTransport>> clientP =
        get_client(remote_calcite_port_);
    clientP.first->shutdown();
    clientP.second->close();
  }
  LOG(INFO) << "End of Calcite Destructor ";
}
