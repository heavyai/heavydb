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
 * @file   Calcite.h
 * @brief
 *
 */

#pragma once

#include "gen-cpp/calciteserver_types.h"
#include "gen-cpp/extension_functions_types.h"

// gen-cpp/calciteserver_types.h > thrift/Thrift.h >
// thrift/transport/PlatformSocket.h > winsock2.h > windows.h
#include "Shared/cleanup_global_namespace.h"

#include <mutex>
#include <string>
#include <vector>

using namespace apache::thrift::transport;

namespace {
constexpr char const* kCalciteUserName = "calcite";
constexpr char const* kCalciteUserPassword = "HyperInteractive";
}  // namespace

class CalciteServerClient;

namespace Catalog_Namespace {
class SessionInfo;
}  // namespace Catalog_Namespace

namespace query_state {
class QueryStateProxy;
}

struct SystemParameters;

class ThriftClientConnection;

// Forward declares for Thrift-generated classes
class TFilterPushDownInfo;
class TPlanResult;
class TCompletionHint;

class Calcite {
 public:
  Calcite(const int db_port,
          const int port,
          const std::string& data_dir,
          const size_t calcite_max_mem,
          const size_t service_timeout,
          const bool service_keepalive,
          const std::string& udf_filename = "");
  Calcite(const SystemParameters& db_parameters,
          const std::string& data_dir,
          const std::string& udf_filename = "");
  Calcite() {}
  // sql_string may differ from what is in query_state due to legacy_syntax option.
  TPlanResult process(query_state::QueryStateProxy,
                      std::string sql_string,
                      const TQueryParsingOption& query_parsing_option,
                      const TOptimizationOption& optimization_option,
                      const std::string& calcite_session_id = "");
  void checkAccessedObjectsPrivileges(query_state::QueryStateProxy query_state_prox,
                                      TPlanResult plan) const;
  std::vector<TCompletionHint> getCompletionHints(
      const Catalog_Namespace::SessionInfo& session_info,
      const std::vector<std::string>& visible_tables,
      const std::string sql_string,
      const int cursor);
  std::string getExtensionFunctionWhitelist();
  std::string getUserDefinedFunctionWhitelist();
  virtual void updateMetadata(std::string catalog, std::string table);
  void close_calcite_server(bool log = true);
  virtual ~Calcite();
  std::string getRuntimeExtensionFunctionWhitelist();
  void setRuntimeExtensionFunctions(const std::vector<TUserDefinedFunction>& udfs,
                                    const std::vector<TUserDefinedTableFunction>& udtfs,
                                    bool isruntime = true);
  static std::string const getInternalSessionProxyUserName() { return kCalciteUserName; }
  static std::string const getInternalSessionProxyPassword() {
    return kCalciteUserPassword;
  }
  TQueryParsingOption getCalciteQueryParsingOption(bool legacy_syntax,
                                                   bool is_explain,
                                                   bool check_privileges,
                                                   bool is_explain_detail);
  TOptimizationOption getCalciteOptimizationOption(
      bool is_view_optimize,
      bool enable_watchdog,
      const std::vector<TFilterPushDownInfo>& filter_push_down_info,
      bool distributed_mode);

 private:
  void init(const int db_port,
            const int port,
            const std::string& data_dir,
            const size_t calcite_max_mem,
            const std::string& udf_filename);
  void runServer(const int db_port,
                 const int port,
                 const std::string& data_dir,
                 const size_t calcite_max_mem,
                 const std::string& udf_filename);
  TPlanResult processImpl(query_state::QueryStateProxy,
                          std::string sql_string,
                          const TQueryParsingOption& query_parsing_option,
                          const TOptimizationOption& optimization_option,
                          const std::string& calcite_session_id);
  std::vector<std::string> get_db_objects(const std::string ra);
  void inner_close_calcite_server(bool log);
  std::pair<std::shared_ptr<CalciteServerClient>, std::shared_ptr<TTransport>> getClient(
      int port);

  int ping(int retry_num = 0, int max_retry = 50);

  std::shared_ptr<ThriftClientConnection> connMgr_;
  bool server_available_;
  size_t service_timeout_;
  bool service_keepalive_ = true;
  int remote_calcite_port_ = -1;
  std::string ssl_trust_store_;
  std::string ssl_trust_password_;
  std::string ssl_key_file_;
  std::string ssl_keystore_;
  std::string ssl_keystore_password_;
  std::string ssl_ca_file_;
  std::string db_config_file_;
  std::once_flag shutdown_once_flag_;
};
