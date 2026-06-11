/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
