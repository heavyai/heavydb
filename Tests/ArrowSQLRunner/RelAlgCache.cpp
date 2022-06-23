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

#include "RelAlgCache.h"

#include "Calcite/SchemaJson.h"
#include "Logger/Logger.h"
#include "QueryEngine/ExtensionFunctionsWhitelist.h"

#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/writer.h>

#include <fstream>

RelAlgCache::RelAlgCache(std::shared_ptr<CalciteJNI> calcite,
                         SchemaProviderPtr schema_provider,
                         const Config& config)
    : calcite_(calcite), schema_provider_(schema_provider) {
  build_cache_ = config.debug.build_ra_cache;
  use_cache_ = config.debug.use_ra_cache;

  if (!use_cache_.empty()) {
    load();
  }
}

RelAlgCache::~RelAlgCache() {
  if (!build_cache_.empty()) {
    try {
      store();
    } catch (std::exception& e) {
      std::cerr << "Cannot write RelAlg Cache: " << e.what();
    }
  }
}

std::string RelAlgCache::process(
    const std::string& db_name,
    const std::string& sql_string,
    const std::vector<FilterPushDownInfo>& filter_push_down_info,
    const bool legacy_syntax,
    const bool is_explain,
    const bool is_view_optimize) {
  CHECK(filter_push_down_info.empty());
  CHECK(!is_view_optimize);

  std::string schema_json;
  if (!use_cache_.empty() || !build_cache_.empty()) {
    schema_json = schema_to_json(schema_provider_);
  }

  if (!use_cache_.empty()) {
    auto schema_it = schema_ids_.find(schema_json);
    if (schema_it != schema_ids_.end()) {
      CacheKey key;
      key.sql = sql_string;
      key.schema_id = schema_it->second;
      key.db_name = db_name;
      key.legacy_syntax = legacy_syntax;
      key.is_explain = is_explain;

      if (rel_alg_cache_.count(key)) {
        return rel_alg_cache_.at(key);
      }
    }
  }

  // Report failure if Calcite is not available on cache miss.
  if (!calcite_) {
    throw std::runtime_error("Missing entry in RelAlgCache.\nQuery: " + sql_string +
                             "\nSchema: " + schema_json);
  }

  auto ra = calcite_->process(db_name,
                              sql_string,
                              filter_push_down_info,
                              legacy_syntax,
                              is_explain,
                              is_view_optimize);

  if (!build_cache_.empty()) {
    put(db_name, sql_string, legacy_syntax, is_explain, schema_json, ra);
  }

  return ra;
}

void RelAlgCache::put(const std::string& db_name,
                      const std::string& sql,
                      bool legacy_syntax,
                      bool is_explain,
                      const std::string& schema_json,
                      const std::string& ra) {
  auto schema_it = schema_ids_.find(schema_json);
  if (schema_it == schema_ids_.end()) {
    auto val = std::make_pair(schema_json, static_cast<int>(schema_ids_.size()));
    schema_it = schema_ids_.insert(val).first;
  }

  CacheKey key;
  key.sql = sql;
  key.schema_id = schema_it->second;
  key.db_name = db_name;
  key.legacy_syntax = legacy_syntax;
  key.is_explain = is_explain;
  if (rel_alg_cache_.count(key)) {
    if (rel_alg_cache_.at(key) != ra) {
      throw std::runtime_error("RelAlg cache entry mismatch for query: " + sql);
    }
  } else {
    rel_alg_cache_[key] = ra;
  }
}

void RelAlgCache::load() {
  std::ifstream fs(use_cache_);
  if (!fs.is_open()) {
    throw std::runtime_error("Cannot open file to read ral alg cache: " + use_cache_);
  }

  rapidjson::IStreamWrapper sw(fs);
  rapidjson::Document doc;
  doc.ParseStream(sw);

  if (!doc.HasMember("ext_fns") || !doc["ext_fns"].IsString() ||
      !doc.HasMember("udf_fns") || !doc["udf_fns"].IsString()) {
    throw std::runtime_error("Malformed RelAlg cache.");
  }

  ExtensionFunctionsWhitelist::add(doc["ext_fns"].GetString());
  ExtensionFunctionsWhitelist::addUdfs(doc["udf_fns"].GetString());

  if (!doc.HasMember("schema_ids") || !doc["schema_ids"].IsArray()) {
    throw std::runtime_error("Malformed RelAlg cache.");
  }

  for (auto& entry : doc["schema_ids"].GetArray()) {
    if (!entry.HasMember("id") || !entry["id"].IsNumber() || !entry.HasMember("value") ||
        !entry["value"].IsString()) {
      throw std::runtime_error("Malformed RelAlg cache.");
    }
    auto insert_res = schema_ids_.insert(
        std::make_pair(entry["value"].GetString(), entry["id"].GetInt()));
    if (!insert_res.second) {
      throw std::runtime_error("Malformed RelAlg cache.");
    }
  }

  if (!doc.HasMember("rel_alg") || !doc["rel_alg"].IsArray()) {
    throw std::runtime_error("Malformed RelAlg cache.");
  }

  for (auto& entry : doc["rel_alg"].GetArray()) {
    if (!entry.HasMember("sql") || !entry["sql"].IsString() ||
        !entry.HasMember("schema_id") || !entry["schema_id"].IsNumber() ||
        !entry.HasMember("db_name") || !entry["db_name"].IsString() ||
        !entry.HasMember("legacy_syntax") || !entry["legacy_syntax"].IsBool() ||
        !entry.HasMember("is_explain") || !entry["is_explain"].IsBool() ||
        !entry.HasMember("value") || !entry["value"].IsString()) {
      throw std::runtime_error("Malformed RelAlg cache.");
    }
    CacheKey key;
    key.sql = entry["sql"].GetString();
    key.schema_id = entry["schema_id"].GetInt();
    key.db_name = entry["db_name"].GetString();
    key.legacy_syntax = entry["legacy_syntax"].GetBool();
    key.is_explain = entry["is_explain"].GetBool();
    auto insert_res =
        rel_alg_cache_.insert(std::make_pair(key, entry["value"].GetString()));
    if (!insert_res.second) {
      throw std::runtime_error("Malformed RelAlg cache.");
    }
  }
}

void RelAlgCache::store() const {
  rapidjson::Document doc(rapidjson::kObjectType);

  doc.AddMember(rapidjson::StringRef("schema_ids"),
                rapidjson::Value(rapidjson::kArrayType),
                doc.GetAllocator());
  for (auto& pr : schema_ids_) {
    rapidjson::Value val(rapidjson::kObjectType);
    val.AddMember("id", rapidjson::Value().SetInt(pr.second), doc.GetAllocator());
    val.AddMember("value",
                  rapidjson::Value().SetString(rapidjson::StringRef(pr.first)),
                  doc.GetAllocator());
    doc["schema_ids"].PushBack(val, doc.GetAllocator());
  }

  doc.AddMember("rel_alg", rapidjson::Value(rapidjson::kArrayType), doc.GetAllocator());
  for (auto& pr : rel_alg_cache_) {
    rapidjson::Value rel_alg_entry(rapidjson::kObjectType);
    rel_alg_entry.AddMember(
        "sql",
        rapidjson::Value().SetString(rapidjson::StringRef(pr.first.sql)),
        doc.GetAllocator());
    rel_alg_entry.AddMember(
        "schema_id", rapidjson::Value().SetInt(pr.first.schema_id), doc.GetAllocator());
    rel_alg_entry.AddMember(
        "db_name",
        rapidjson::Value().SetString(rapidjson::StringRef(pr.first.db_name)),
        doc.GetAllocator());
    rel_alg_entry.AddMember("legacy_syntax",
                            rapidjson::Value().SetBool(pr.first.legacy_syntax),
                            doc.GetAllocator());
    rel_alg_entry.AddMember("is_explain",
                            rapidjson::Value().SetBool(pr.first.is_explain),
                            doc.GetAllocator());
    rel_alg_entry.AddMember("value",
                            rapidjson::Value().SetString(rapidjson::StringRef(pr.second)),
                            doc.GetAllocator());
    doc["rel_alg"].PushBack(rel_alg_entry, doc.GetAllocator());
  }

  std::string ext_fns = calcite_->getExtensionFunctionWhitelist();
  doc.AddMember("ext_fns",
                rapidjson::Value().SetString(rapidjson::StringRef(ext_fns)),
                doc.GetAllocator());
  std::string udf_fns = calcite_->getUserDefinedFunctionWhitelist();
  doc.AddMember("udf_fns",
                rapidjson::Value().SetString(rapidjson::StringRef(udf_fns)),
                doc.GetAllocator());

  std::ofstream fs(build_cache_);
  if (!fs.is_open()) {
    throw std::runtime_error("Cannot create file to write rel alg cache: " +
                             build_cache_);
  }

  rapidjson::OStreamWrapper sw(fs);
  rapidjson::Writer<rapidjson::OStreamWrapper> writer(sw);
  doc.Accept(writer);
}
