/*
 * Copyright 2021 OmniSci, Inc.
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

#include "ArrowStorage/ArrowStorage.h"
#include "QueryEngine/CompilationOptions.h"
#include "QueryEngine/Descriptors/RelAlgExecutionDescriptor.h"

#include "BufferPoolStats.h"

namespace TestHelpers::ArrowSQLRunner {

extern bool g_hoist_literals;

constexpr int TEST_SCHEMA_ID = 1;
constexpr int TEST_DB_ID = (TEST_SCHEMA_ID << 24) + 1;

constexpr int CALCITE_PORT = 3278;

void init(size_t max_gpu_mem = 0);

void reset();

bool gpusPresent();

void printStats();

void createTable(
    const std::string& table_name,
    const std::vector<ArrowStorage::ColumnDescription>& columns,
    const ArrowStorage::TableOptions& options = ArrowStorage::TableOptions());

void dropTable(const std::string& table_name);

void insertCsvValues(const std::string& table_name, const std::string& values);

void insertJsonValues(const std::string& table_name, const std::string& values);

ExecutionResult runSqlQuery(const std::string& sql,
                            const CompilationOptions& co,
                            const ExecutionOptions& eo);

ExecutionResult runSqlQuery(const std::string& sql,
                            ExecutorDeviceType device_type,
                            const ExecutionOptions& eo);

ExecutionResult runSqlQuery(const std::string& sql,
                            ExecutorDeviceType device_type,
                            bool allow_loop_joins);

ExecutionOptions getExecutionOptions(bool allow_loop_joins, bool just_explain = false);

CompilationOptions getCompilationOptions(ExecutorDeviceType device_type);

std::shared_ptr<ResultSet> run_multiple_agg(const std::string& query_str,
                                            const ExecutorDeviceType device_type,
                                            const bool allow_loop_joins = true);

TargetValue run_simple_agg(const std::string& query_str,
                           const ExecutorDeviceType device_type,
                           const bool allow_loop_joins = true);

void run_sqlite_query(const std::string& query_string);

void c(const std::string& query_string, const ExecutorDeviceType device_type);

void c(const std::string& query_string,
       const std::string& sqlite_query_string,
       const ExecutorDeviceType device_type);

/* timestamp approximate checking for NOW() */
void cta(const std::string& query_string, const ExecutorDeviceType device_type);

void c_arrow(const std::string& query_string, const ExecutorDeviceType device_type);

void clearCpuMemory();

BufferPoolStats getBufferPoolStats(const Data_Namespace::MemoryLevel memory_level =
                                       Data_Namespace::MemoryLevel::CPU_LEVEL);

std::shared_ptr<ArrowStorage> getStorage();

DataMgr* getDataMgr();

}  // namespace TestHelpers::ArrowSQLRunner
