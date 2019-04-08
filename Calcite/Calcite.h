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

#include <string>
#include <thread>
#include <vector>
#include "Shared/MapDParameters.h"
#include "rapidjson/document.h"

namespace Catalog_Namespace {
class SessionInfo;
}

// Forward declares for Thrift-generated classes
class TFilterPushDownInfo;
class TPlanResult;
class TCompletionHint;

class Calcite {
 public:
  Calcite(const int mapd_port,
          const int port,
          const std::string& data_dir,
          const size_t calcite_max_mem)
      : Calcite(mapd_port, port, data_dir, calcite_max_mem, ""){};
  Calcite(const int mapd_port,
          const int port,
          const std::string& data_dir,
          const size_t calcite_max_mem,
          const std::string& session_prefix);
  Calcite(const MapDParameters& mapd_parameter,
          const std::string& data_dir,
          const std::string& session_prefix);
  TPlanResult process(const Catalog_Namespace::SessionInfo& session_info,
                      const std::string sql_string,
                      const std::vector<TFilterPushDownInfo>& filter_push_down_info,
                      const bool legacy_syntax,
                      const bool is_explain);
  std::vector<TCompletionHint> getCompletionHints(
      const Catalog_Namespace::SessionInfo& session_info,
      const std::vector<std::string>& visible_tables,
      const std::string sql_string,
      const int cursor);
  std::string getExtensionFunctionWhitelist();
  void updateMetadata(std::string catalog, std::string table);
  void close_calcite_server();
  virtual ~Calcite();

  std::string& get_session_prefix() { return session_prefix_; }

 private:
  void init(const int mapd_port,
            const int port,
            const std::string& data_dir,
            const size_t calcite_max_mem);
  void runServer(const int mapd_port,
                 const int port,
                 const std::string& data_dir,
                 const size_t calcite_max_mem);
  TPlanResult processImpl(const Catalog_Namespace::SessionInfo& session_info,
                          const std::string sql_string,
                          const std::vector<TFilterPushDownInfo>& filter_push_down_info,
                          const bool legacy_syntax,
                          const bool is_explain);
  std::vector<std::string> get_db_objects(const std::string ra);

  std::thread calcite_server_thread_;
  int ping();

  bool server_available_;
  int remote_calcite_port_ = -1;
  std::string ssl_trust_store_;
  std::string ssl_trust_password_;
  std::string session_prefix_;
};

#endif /* CALCITE_H */
