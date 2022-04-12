/*
 * Copyright 2020 OmniSci, Inc.
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
#include "Logger/Logger.h"
#include "OSDependent/omnisci_path.h"
#include "Shared/SystemParameters.h"
#include "Shared/ThriftClient.h"
#include "Shared/fixautotools.h"
#include "Shared/measure.h"
#include "ThriftHandler/QueryState.h"

#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/transport/TSocket.h>
#include <thrift/transport/TTransportUtils.h>
#include <type_traits>

#ifdef _MSC_VER
#include <process.h>
#endif

#include "gen-cpp/CalciteServer.h"

#include "rapidjson/document.h"

#include <iostream>
#include <utility>

using namespace rapidjson;
using namespace apache::thrift;
using namespace apache::thrift::protocol;
using namespace apache::thrift::transport;

extern bool g_enable_watchdog;
namespace {
template <typename XDEBUG_OPTION,
          typename REMOTE_DEBUG_OPTION,
          typename... REMAINING_ARGS>
int wrapped_execlp(char const* path,
                   XDEBUG_OPTION&& x_debug,
                   REMOTE_DEBUG_OPTION&& remote_debug,
                   REMAINING_ARGS&&... standard_args) {
#ifdef ENABLE_JAVA_REMOTE_DEBUG
  return execlp(
      path, x_debug, remote_debug, std::forward<REMAINING_ARGS>(standard_args)...);
#else
  return execlp(path, std::forward<REMAINING_ARGS>(standard_args)...);
#endif
}
}  // namespace

static void start_calcite_server_as_daemon(const int db_port,
                                           const int port,
                                           const std::string& data_dir,
                                           const size_t calcite_max_mem,
                                           const std::string& ssl_trust_store,
                                           const std::string& ssl_trust_password_X,
                                           const std::string& ssl_keystore,
                                           const std::string& ssl_keystore_password_X,
                                           const std::string& ssl_key_file,
                                           const std::string& db_config_file,
                                           const std::string& udf_filename) {
  auto root_abs_path = omnisci::get_root_abs_path();
  std::string const xDebug = "-Xdebug";
  std::string const remoteDebug =
      "-agentlib:jdwp=transport=dt_socket,server=y,suspend=n,address=5005";
  std::string xmxP = "-Xmx" + std::to_string(calcite_max_mem) + "m";
  std::string jarP = "-jar";
  std::string jarD =
      root_abs_path + "/bin/calcite-1.0-SNAPSHOT-jar-with-dependencies.jar";
  std::string extensionsP = "-e";
  std::string extensionsD = root_abs_path + "/QueryEngine/";
  std::string dataP = "-d";
  std::string dataD = data_dir;
  std::string localPortP = "-p";
  std::string localPortD = std::to_string(port);
  std::string dbPortP = "-m";
  std::string dbPortD = std::to_string(db_port);
  std::string TrustStoreP = "-T";
  std::string TrustPasswdP = "-P";
  std::string ConfigFileP = "-c";
  std::string KeyStoreP = "-Y";
  std::string KeyStorePasswdP = "-Z";
  // FIXME: this path should be getting pulled from logger rather than hardcoded
  std::string logDirectory = "-DMAPD_LOG_DIR=" + data_dir + "/mapd_log/";
  std::string userDefinedFunctionsP = "";
  std::string userDefinedFunctionsD = "";

  if (!udf_filename.empty()) {
    userDefinedFunctionsP += "-u";
    userDefinedFunctionsD += udf_filename;
  }

  // If a config file hasn't been supplied then put the password in the params
  // otherwise send an empty string and Calcite should get it from the config file.
  std::string key_store_password = (db_config_file == "") ? ssl_keystore_password_X : "";
  std::string trust_store_password = (db_config_file == "") ? ssl_trust_password_X : "";
#ifdef _MSC_VER
  // TODO: enable UDF support
  std::vector<std::string> args_vec;
  args_vec.push_back("java");
  args_vec.push_back(xDebug);
  args_vec.push_back(remoteDebug);
  args_vec.push_back(xmxP);
  args_vec.push_back(logDirectory);
  args_vec.push_back(jarP);
  args_vec.push_back(jarD);
  args_vec.push_back(extensionsP);
  args_vec.push_back(extensionsD);
  args_vec.push_back(dataP);
  args_vec.push_back(dataD);
  args_vec.push_back(localPortP);
  args_vec.push_back(localPortD);
  args_vec.push_back(dbPortP);
  args_vec.push_back(dbPortD);
  if (!ssl_trust_store.empty()) {
    args_vec.push_back(TrustStoreP);
    args_vec.push_back(ssl_trust_store);
  }
  if (!trust_store_password.empty()) {
    args_vec.push_back(TrustPasswdP);
    args_vec.push_back(trust_store_password);
  }
  if (!ssl_keystore.empty()) {
    args_vec.push_back(KeyStoreP);
    args_vec.push_back(ssl_keystore);
  }
  if (!key_store_password.empty()) {
    args_vec.push_back(KeyStorePasswdP);
    args_vec.push_back(key_store_password);
  }
  if (!db_config_file.empty()) {
    args_vec.push_back(ConfigFileP);
    args_vec.push_back(db_config_file);
  }
  std::string args{boost::algorithm::join(args_vec, " ")};
  STARTUPINFO startup_info;
  PROCESS_INFORMATION proc_info;
  ZeroMemory(&startup_info, sizeof(startup_info));
  startup_info.cb = sizeof(startup_info);
  ZeroMemory(&proc_info, sizeof(proc_info));
  LOG(INFO) << "Startup command: " << args;
  std::wstring wargs = std::wstring(args.begin(), args.end());
  const auto ret = CreateProcess(NULL,
                                 (LPWSTR)wargs.c_str(),
                                 NULL,
                                 NULL,
                                 false,
                                 0,
                                 NULL,
                                 NULL,
                                 &startup_info,
                                 &proc_info);
  if (ret == 0) {
    LOG(FATAL) << "Failed to start Calcite server " << GetLastError();
  }
#else
  int pid = fork();
  if (pid == 0) {
    int i;

    if (udf_filename.empty()) {
      i = wrapped_execlp("java",
                         xDebug.c_str(),
                         remoteDebug.c_str(),
                         xmxP.c_str(),
                         logDirectory.c_str(),
                         jarP.c_str(),
                         jarD.c_str(),
                         extensionsP.c_str(),
                         extensionsD.c_str(),
                         dataP.c_str(),
                         dataD.c_str(),
                         localPortP.c_str(),
                         localPortD.c_str(),
                         dbPortP.c_str(),
                         dbPortD.c_str(),
                         TrustStoreP.c_str(),
                         ssl_trust_store.c_str(),
                         TrustPasswdP.c_str(),
                         trust_store_password.c_str(),
                         KeyStoreP.c_str(),
                         ssl_keystore.c_str(),
                         KeyStorePasswdP.c_str(),
                         key_store_password.c_str(),
                         ConfigFileP.c_str(),
                         db_config_file.c_str(),
                         (char*)0);
    } else {
      i = wrapped_execlp("java",
                         xDebug.c_str(),
                         remoteDebug.c_str(),
                         xmxP.c_str(),
                         logDirectory.c_str(),
                         jarP.c_str(),
                         jarD.c_str(),
                         extensionsP.c_str(),
                         extensionsD.c_str(),
                         dataP.c_str(),
                         dataD.c_str(),
                         localPortP.c_str(),
                         localPortD.c_str(),
                         dbPortP.c_str(),
                         dbPortD.c_str(),
                         TrustStoreP.c_str(),
                         ssl_trust_store.c_str(),
                         TrustPasswdP.c_str(),
                         trust_store_password.c_str(),
                         KeyStoreP.c_str(),
                         ssl_keystore.c_str(),
                         KeyStorePasswdP.c_str(),
                         key_store_password.c_str(),
                         ConfigFileP.c_str(),
                         db_config_file.c_str(),
                         userDefinedFunctionsP.c_str(),
                         userDefinedFunctionsD.c_str(),
                         (char*)0);
    }

    if (i) {
      int errsv = errno;
      LOG(FATAL) << "Failed to start Calcite server [errno=" << errsv
                 << "]: " << strerror(errsv);
    } else {
      LOG(INFO) << "Successfully started Calcite server";
    }
  }
#endif
}

std::pair<std::shared_ptr<CalciteServerClient>, std::shared_ptr<TTransport>>
Calcite::getClient(int port) {
  const auto transport = connMgr_->open_buffered_client_transport("localhost",
                                                                  port,
                                                                  ssl_ca_file_,
                                                                  true,
                                                                  service_keepalive_,
                                                                  service_timeout_,
                                                                  service_timeout_,
                                                                  service_timeout_);
  try {
    transport->open();

  } catch (TException& tx) {
    throw tx;
  } catch (std::exception& ex) {
    throw ex;
  }
  std::shared_ptr<TProtocol> protocol(new TBinaryProtocol(transport));
  std::shared_ptr<CalciteServerClient> client;
  client.reset(new CalciteServerClient(protocol));
  std::pair<std::shared_ptr<CalciteServerClient>, std::shared_ptr<TTransport>> ret;
  return std::make_pair(client, transport);
}

void Calcite::runServer(const int db_port,
                        const int port,
                        const std::string& data_dir,
                        const size_t calcite_max_mem,
                        const std::string& udf_filename) {
  LOG(INFO) << "Running Calcite server as a daemon";

  // ping server to see if for any reason there is an orphaned one
  int ping_time = ping();
  if (ping_time > -1) {
    // we have an orphaned server shut it down
    LOG(ERROR)
        << "Appears to be orphaned Calcite server already running, shutting it down";
    LOG(ERROR) << "Please check that you are not trying to run two servers on same port";
    LOG(ERROR) << "Attempting to shutdown orphaned Calcite server";
    try {
      auto clientP = getClient(remote_calcite_port_);
      clientP.first->shutdown();
      clientP.second->close();
      LOG(ERROR) << "orphaned Calcite server shutdown";

    } catch (TException& tx) {
      LOG(ERROR) << "Failed to shutdown orphaned Calcite server, reason: " << tx.what();
    }
  }

  // start the calcite server as a seperate process
  start_calcite_server_as_daemon(db_port,
                                 port,
                                 data_dir,
                                 calcite_max_mem,
                                 ssl_trust_store_,
                                 ssl_trust_password_,
                                 ssl_keystore_,
                                 ssl_keystore_password_,
                                 ssl_key_file_,
                                 db_config_file_,
                                 udf_filename);

  // check for new server for 30 seconds max
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  int retry_max = 300;
  for (int i = 2; i <= retry_max; i++) {
    int ping_time = ping(i, retry_max);
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
  LOG(FATAL) << "Could not connect to Calcite remote server running on port [" << port
             << "]";
}

// ping existing server
// return -1 if no ping response
// params set to default values in header
int Calcite::ping(int retry_num, int max_retry) {
  try {
    auto ms = measure<>::execution([&]() {
      auto clientP = getClient(remote_calcite_port_);
      clientP.first->ping();
      clientP.second->close();
    });
    return ms;

  } catch (TException& tx) {
    if (retry_num >= max_retry) {
      LOG(ERROR) << "Problems connecting to Calcite. Thrift error - " << tx.what();
    }
    return -1;
  }
}

Calcite::Calcite(const int db_port,
                 const int calcite_port,
                 const std::string& data_dir,
                 const size_t calcite_max_mem,
                 const size_t service_timeout,
                 const bool service_keepalive,
                 const std::string& udf_filename)
    : server_available_(false)
    , service_timeout_(service_timeout)
    , service_keepalive_(service_keepalive) {
  init(db_port, calcite_port, data_dir, calcite_max_mem, udf_filename);
}

void Calcite::init(const int db_port,
                   const int calcite_port,
                   const std::string& data_dir,
                   const size_t calcite_max_mem,
                   const std::string& udf_filename) {
  LOG(INFO) << "Creating Calcite Handler,  Calcite Port is " << calcite_port
            << " base data dir is " << data_dir;
  connMgr_ = std::make_shared<ThriftClientConnection>();
  if (calcite_port < 0) {
    CHECK(false) << "JNI mode no longer supported.";
  }
  if (calcite_port == 0) {
    // dummy process for initdb
    remote_calcite_port_ = calcite_port;
    server_available_ = false;
  } else {
    remote_calcite_port_ = calcite_port;
    runServer(db_port, calcite_port, data_dir, calcite_max_mem, udf_filename);
    server_available_ = true;
  }
}

Calcite::Calcite(const SystemParameters& system_parameters,
                 const std::string& data_dir,
                 const std::string& udf_filename)
    : service_timeout_(system_parameters.calcite_timeout)
    , service_keepalive_(system_parameters.calcite_keepalive)
    , ssl_trust_store_(system_parameters.ssl_trust_store)
    , ssl_trust_password_(system_parameters.ssl_trust_password)
    , ssl_key_file_(system_parameters.ssl_key_file)
    , ssl_keystore_(system_parameters.ssl_keystore)
    , ssl_keystore_password_(system_parameters.ssl_keystore_password)
    , ssl_ca_file_(system_parameters.ssl_trust_ca_file)
    , db_config_file_(system_parameters.config_file) {
  init(system_parameters.omnisci_server_port,
       system_parameters.calcite_port,
       data_dir,
       system_parameters.calcite_max_mem,
       udf_filename);
}

void Calcite::updateMetadata(std::string catalog, std::string table) {
  if (server_available_) {
    auto ms = measure<>::execution([&]() {
      auto clientP = getClient(remote_calcite_port_);
      clientP.first->updateMetadata(catalog, table);
      clientP.second->close();
    });
    LOG(INFO) << "Time to updateMetadata " << ms << " (ms)";
  } else {
    LOG(INFO) << "Not routing to Calcite, server is not up";
  }
}

TPlanResult Calcite::process(
    const std::string& user,
    const std::string& db_name,
    const std::string& sql_string,
    const std::string& schema_json,
    const std::string& session_id,
    const std::vector<TFilterPushDownInfo>& filter_push_down_info,
    const bool legacy_syntax,
    const bool is_explain,
    const bool is_view_optimize) {
  TPlanResult ret;
  if (server_available_) {
    try {
      // calcite_session_id would be an empty string when accessed by internal resources
      // that would not access `process` through handler instance, like for eg: Unit
      // Tests. In these cases we would use the session_id from query state.
      auto ms = measure<>::execution([&]() {
        auto clientP = getClient(remote_calcite_port_);
        auto calcite_query_parsing_option = getCalciteQueryParsingOption(
            legacy_syntax, is_explain, /*check_privileges=*/true);
        auto calcite_optimization_option =
            getCalciteOptimizationOption(is_view_optimize, g_enable_watchdog, {});
        clientP.first->process(ret,
                               user,
                               session_id,
                               db_name,
                               sql_string,
                               calcite_query_parsing_option,
                               calcite_optimization_option,
                               {},
                               schema_json);
        clientP.second->close();
      });

      // LOG(INFO) << ret.plan_result;
      LOG(INFO) << "Time in Thrift "
                << (ms > ret.execution_time_ms ? ms - ret.execution_time_ms : 0)
                << " (ms), Time in Java Calcite server " << ret.execution_time_ms
                << " (ms)";
    } catch (InvalidParseRequest& e) {
      throw std::invalid_argument(e.whyUp);
    } catch (const std::exception& ex) {
      LOG(FATAL)
          << "Error occurred trying to communicate with Calcite server, the error was: '"
          << ex.what() << "', omnisci_server restart will be required";
      return ret;  // satisfy return-type warning
    }
  } else {
    LOG(FATAL) << "Not routing to Calcite, server is not up";
  }
  return ret;
}

void Calcite::checkAccessedObjectsPrivileges(
    query_state::QueryStateProxy query_state_proxy,
    TPlanResult plan) const {
  UNREACHABLE();
}

std::vector<TCompletionHint> Calcite::getCompletionHints(
    const SessionInfo& session_info,
    const std::vector<std::string>& visible_tables,
    const std::string sql_string,
    const int cursor) {
  UNREACHABLE();
  std::vector<TCompletionHint> hints;
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

std::string Calcite::getExtensionFunctionWhitelist() {
  if (server_available_) {
    TPlanResult ret;
    std::string whitelist;

    auto clientP = getClient(remote_calcite_port_);
    clientP.first->getExtensionFunctionWhitelist(whitelist);
    clientP.second->close();
    VLOG(1) << whitelist;
    return whitelist;
  } else {
    LOG(FATAL) << "Not routing to Calcite, server is not up";
    return "";
  }
  CHECK(false);
  return "";
}

std::string Calcite::getUserDefinedFunctionWhitelist() {
  if (server_available_) {
    TPlanResult ret;
    std::string whitelist;

    auto clientP = getClient(remote_calcite_port_);
    clientP.first->getUserDefinedFunctionWhitelist(whitelist);
    clientP.second->close();
    VLOG(1) << "User defined functions whitelist loaded from Calcite: " << whitelist;
    return whitelist;
  } else {
    LOG(FATAL) << "Not routing to Calcite, server is not up";
    return "";
  }
  UNREACHABLE();
  return "";
}

void Calcite::close_calcite_server(bool log) {
  std::call_once(shutdown_once_flag_,
                 [this, log]() { this->inner_close_calcite_server(log); });
}

void Calcite::inner_close_calcite_server(bool log) {
  if (server_available_) {
    LOG_IF(INFO, log) << "Shutting down Calcite server";
    try {
      auto clientP = getClient(remote_calcite_port_);
      clientP.first->shutdown();
      clientP.second->close();
    } catch (const std::exception& e) {
      if (std::string(e.what()) != "connect() failed: Connection refused" &&
          std::string(e.what()) != "socket open() error: Connection refused" &&
          std::string(e.what()) != "No more data to read.") {
        std::cerr << "Error shutting down Calcite server: " << e.what() << std::endl;
      }  // else Calcite already shut down
    }
    LOG_IF(INFO, log) << "shut down Calcite";
    server_available_ = false;
  }
}

Calcite::~Calcite() {
  close_calcite_server(false);
}

std::string Calcite::getRuntimeExtensionFunctionWhitelist() {
  if (server_available_) {
    TPlanResult ret;
    std::string whitelist;
    auto clientP = getClient(remote_calcite_port_);
    clientP.first->getRuntimeExtensionFunctionWhitelist(whitelist);
    clientP.second->close();
    VLOG(1) << "Runtime extension functions whitelist loaded from Calcite: " << whitelist;
    return whitelist;
  } else {
    LOG(FATAL) << "Not routing to Calcite, server is not up";
    return "";
  }
  UNREACHABLE();
  return "";
}

void Calcite::setRuntimeExtensionFunctions(
    const std::vector<TUserDefinedFunction>& udfs,
    const std::vector<TUserDefinedTableFunction>& udtfs,
    bool isruntime) {
  if (server_available_) {
    auto clientP = getClient(remote_calcite_port_);
    clientP.first->setRuntimeExtensionFunctions(udfs, udtfs, isruntime);
    clientP.second->close();
  } else {
    LOG(FATAL) << "Not routing to Calcite, server is not up";
  }
}

TQueryParsingOption Calcite::getCalciteQueryParsingOption(bool legacy_syntax,
                                                          bool is_explain,
                                                          bool check_privileges) {
  TQueryParsingOption query_parsing_info;
  query_parsing_info.legacy_syntax = legacy_syntax;
  query_parsing_info.is_explain = is_explain;
  query_parsing_info.check_privileges = check_privileges;
  return query_parsing_info;
}

TOptimizationOption Calcite::getCalciteOptimizationOption(
    bool is_view_optimize,
    bool enable_watchdog,
    const std::vector<TFilterPushDownInfo>& filter_push_down_info) {
  TOptimizationOption optimization_option;
  optimization_option.filter_push_down_info = filter_push_down_info;
  optimization_option.is_view_optimize = is_view_optimize;
  optimization_option.enable_watchdog = enable_watchdog;
  return optimization_option;
}
