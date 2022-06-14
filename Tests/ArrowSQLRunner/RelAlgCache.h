/*
 * Copyright 2022 Intel Corporation.
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

#pragma once

#include "Calcite/CalciteJNI.h"
#include "Shared/Config.h"
#include "Shared/hash.h"

#include <boost/functional/hash.hpp>

#include <unordered_map>

class RelAlgCache {
 public:
  RelAlgCache(std::shared_ptr<CalciteJNI> calcite,
              SchemaProviderPtr schema_provider,
              const Config& config);
  ~RelAlgCache();

  std::string process(const std::string& user,
                      const std::string& db_name,
                      const std::string& sql_string,
                      const std::vector<FilterPushDownInfo>& filter_push_down_info = {},
                      const bool legacy_syntax = false,
                      const bool is_explain = false,
                      const bool is_view_optimize = false);

 private:
  void put(const std::string& user,
           const std::string& db_name,
           const std::string& sql,
           bool legacy_syntax,
           bool is_explain,
           const std::string& schema_json,
           const std::string& ra);

  void load();
  void store() const;

  struct CacheKey {
    std::string sql;
    int schema_id;
    std::string user;
    std::string db_name;
    bool legacy_syntax;
    bool is_explain;

    bool operator==(const CacheKey& other) const {
      return sql == other.sql && schema_id == other.schema_id && user == other.user &&
             db_name == other.db_name && legacy_syntax == other.legacy_syntax &&
             is_explain == other.is_explain;
    }
  };

  struct CacheKeyHash {
    size_t operator()(const CacheKey& key) const {
      return boost::hash_value(std::tie(key.sql,
                                        key.schema_id,
                                        key.user,
                                        key.db_name,
                                        key.legacy_syntax,
                                        key.is_explain));
    }
  };

  std::shared_ptr<CalciteJNI> calcite_;
  SchemaProviderPtr schema_provider_;
  std::string build_cache_;
  std::string use_cache_;
  std::unordered_map<std::string, int> schema_ids_;
  std::unordered_map<CacheKey, std::string, CacheKeyHash> rel_alg_cache_;
};