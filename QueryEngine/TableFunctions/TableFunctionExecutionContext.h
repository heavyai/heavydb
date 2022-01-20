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

#include "QueryEngine/CompilationOptions.h"
#include "QueryEngine/Descriptors/RowSetMemoryOwner.h"
#include "QueryEngine/RelAlgExecutionUnit.h"

struct InputTableInfo;
class TableFunctionCompilationContext;
class ColumnFetcher;
class Executor;

class TableFunctionExecutionContext {
 public:
  TableFunctionExecutionContext(std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner)
      : row_set_mem_owner_(row_set_mem_owner) {}

  // non-copyable
  TableFunctionExecutionContext(const TableFunctionExecutionContext&) = delete;
  TableFunctionExecutionContext& operator=(const TableFunctionExecutionContext&) = delete;

  ResultSetPtr execute(const TableFunctionExecutionUnit& exe_unit,
                       const std::vector<InputTableInfo>& table_infos,
                       const std::shared_ptr<CompilationContext>& compilation_context,
                       DataProvider* data_provider,
                       const ColumnFetcher& column_fetcher,
                       const ExecutorDeviceType device_type,
                       Executor* executor);

 private:
  ResultSetPtr launchCpuCode(
      const TableFunctionExecutionUnit& exe_unit,
      const std::shared_ptr<CpuCompilationContext>& compilation_context,
      std::vector<const int8_t*>& col_buf_ptrs,
      std::vector<int64_t>& col_sizes,
      const size_t elem_count,
      Executor* executor);
  ResultSetPtr launchGpuCode(
      const TableFunctionExecutionUnit& exe_unit,
      const std::shared_ptr<GpuCompilationContext>& compilation_context,
      std::vector<const int8_t*>& col_buf_ptrs,
      std::vector<int64_t>& col_sizes,
      const size_t elem_count,
      const int device_id,
      Executor* executor);

  std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner_;
};
