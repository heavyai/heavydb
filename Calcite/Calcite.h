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
 * File:   Calcite.h
 * Author: michael
 *
 * Created on November 23, 2015, 9:33 AM
 */

#ifndef CALCITE_H
#define CALCITE_H

#include "Shared/mapd_shared_ptr.h"

#include <thrift/transport/TTransport.h>

#include <mutex>
#include <string>
#include <vector>

using namespace apache::thrift::transport;

class CalciteServerClient;

namespace Catalog_Namespace {
class SessionInfo;
}

namespace query_state {
class QueryStateProxy;
}

struct MapDParameters;

class ThriftClientConnection;

// Forward declares for Thrift-generated classes
class TFilterPushDownInfo;
class TPlanResult;
class TCompletionHint;

class Calcite final {
 public:
  Calcite(const int mapd_port,
          const int port,
          const std::string& data_dir,
          const size_t calcite_max_mem,
          const std::string& udf_filename = "")
      : Calcite(mapd_port, port, data_dir, calcite_max_mem, "", udf_filename){};
  Calcite(const int mapd_port,
          const int port,
          const std::string& data_dir,
          const size_t calcite_max_mem,
          const std::string& session_prefix,
          const std::string& udf_filename = "");
  Calcite(const MapDParameters& mapd_parameter,
          const std::string& data_dir,
          const std::string& session_prefix,
          const std::string& udf_filename = "");
  // sql_string may differ from what is in query_state due to legacy_syntax option.
  TPlanResult process(query_state::QueryStateProxy,
                      std::string sql_string,
                      const std::vector<TFilterPushDownInfo>& filter_push_down_info,
                      const bool legacy_syntax,
                      const bool is_explain,
                      const bool is_view_optimize);
  std::vector<TCompletionHint> getCompletionHints(
      const Catalog_Namespace::SessionInfo& session_info,
      const std::vector<std::string>& visible_tables,
      const std::string sql_string,
      const int cursor);
  std::string getExtensionFunctionWhitelist();
  std::string getUserDefinedFunctionWhitelist();
  void updateMetadata(std::string catalog, std::string table);
  void close_calcite_server(bool log = true);
  ~Calcite();
  std::string getRuntimeUserDefinedFunctionWhitelist();
  void setRuntimeUserDefinedFunction(std::string udf_string);
  std::string const& get_session_prefix() const { return session_prefix_; }

 private:
  void init(const int mapd_port,
            const int port,
            const std::string& data_dir,
            const size_t calcite_max_mem,
            const std::string& udf_filename);
  void runServer(const int mapd_port,
                 const int port,
                 const std::string& data_dir,
                 const size_t calcite_max_mem,
                 const std::string& udf_filename);
  TPlanResult processImpl(query_state::QueryStateProxy,
                          std::string sql_string,
                          const std::vector<TFilterPushDownInfo>& filter_push_down_info,
                          const bool legacy_syntax,
                          const bool is_explain,
                          const bool is_view_optimize);
  std::vector<std::string> get_db_objects(const std::string ra);
  void inner_close_calcite_server(bool log);
  std::pair<mapd::shared_ptr<CalciteServerClient>, mapd::shared_ptr<TTransport>>
  getClient(int port);

  int ping();

  mapd::shared_ptr<ThriftClientConnection> connMgr_;
  bool server_available_;
  int remote_calcite_port_ = -1;
  std::string ssl_trust_store_;
  std::string ssl_trust_password_;
  std::string ssl_key_file_;
  std::string ssl_keystore_;
  std::string ssl_keystore_password_;
  std::string ssl_cert_file_;
  std::string const session_prefix_;
  std::once_flag shutdown_once_flag_;
};

#endif /* CALCITE_H */
