/*
 * Copyright 2019 OmniSci, Inc.
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

#include <string>
#include <vector>

#include "QueryEngine/ColumnFetcher.h"
#include "QueryEngine/PlanState.h"
#include "QueryEngine/SerializeToSql.h"
#include "QueryEngine/TargetMetaInfo.h"
#include "ThirdParty/sqlite3/sqlite3.h"

class Executor;

struct ExternalQueryTable {
  FetchResult fetch_result;
  std::vector<TargetMetaInfo> schema;
  std::string from_table;
  const Executor* executor;
};

struct ExternalQueryOutputSpec {
  QueryMemoryDescriptor query_mem_desc;
  std::vector<TargetInfo> target_infos;
  const Executor* executor;
};

class NativeExecutionError : public std::runtime_error {
 public:
  NativeExecutionError(const std::string& message) : std::runtime_error(message) {}
};

class SqliteMemDatabase {
 public:
  SqliteMemDatabase(const ExternalQueryTable& external_query_table);

  ~SqliteMemDatabase();

  void run(const std::string& sql);
  std::unique_ptr<ResultSet> runSelect(const std::string& sql,
                                       const ExternalQueryOutputSpec& output_spec);

 private:
  sqlite3* db_;
  ExternalQueryTable external_query_table_;
  static std::mutex session_mutex_;
};

std::unique_ptr<ResultSet> run_query_external(const ExecutionUnitSql& sql,
                                              const FetchResult& fetch_result,
                                              const PlanState* plan_state,
                                              const ExternalQueryOutputSpec& output_spec);

bool is_supported_type_for_extern_execution(const SQLTypeInfo& ti);
