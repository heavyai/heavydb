/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "QueryEngine/CompilationOptions.h"
#include "QueryEngine/Descriptors/RowSetMemoryOwner.h"
#include "QueryEngine/RelAlgExecutionUnit.h"

struct InputTableInfo;
class TableFunctionCompilationContext;
class ColumnFetcher;
class Executor;

namespace gfx {
class GfxContext;
}

class TableFunctionExecutionContext {
 public:
  TableFunctionExecutionContext(std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                                gfx::GfxContext* gfx_context)
      : row_set_mem_owner_{row_set_mem_owner}, gfx_context_{gfx_context} {}

  // non-copyable
  TableFunctionExecutionContext(const TableFunctionExecutionContext&) = delete;
  TableFunctionExecutionContext& operator=(const TableFunctionExecutionContext&) = delete;

  ResultSetPtr execute(const TableFunctionExecutionUnit& exe_unit,
                       const std::vector<InputTableInfo>& table_infos,
                       const std::shared_ptr<CompilationContext>& compilation_context,
                       const ColumnFetcher& column_fetcher,
                       const ExecutorDeviceType device_type,
                       Executor* executor,
                       bool is_pre_launch_udtf);

 private:
  void launchPreCodeOnCpu(
      const TableFunctionExecutionUnit& exe_unit,
      const std::shared_ptr<CpuCompilationContext>& compilation_context,
      std::vector<const int8_t*>& col_buf_ptrs,
      std::vector<int64_t>& col_sizes,
      std::vector<const int8_t*>& input_str_dict_proxy_ptrs,
      const size_t elem_count,
      Executor* executor);

  ResultSetPtr launchCpuCode(
      const TableFunctionExecutionUnit& exe_unit,
      const std::shared_ptr<CpuCompilationContext>& compilation_context,
      std::vector<const int8_t*>& col_buf_ptrs,
      std::vector<int64_t>& col_sizes,
      std::vector<const int8_t*>& input_str_dict_proxy_ptrs,
      const size_t elem_count,
      std::vector<int8_t*>& output_str_dict_proxy_ptrs,
      Executor* executor);

  ResultSetPtr launchGpuCode(
      const TableFunctionExecutionUnit& exe_unit,
      const std::shared_ptr<GpuCompilationContext>& compilation_context,
      std::vector<const int8_t*>& col_buf_ptrs,
      std::vector<int64_t>& col_sizes,
      std::vector<const int8_t*>& input_str_dict_proxy_ptrs,
      const size_t elem_count,
      std::vector<int8_t*>& output_str_dict_proxy_ptrs,
      const int device_id,
      Executor* executor);

  std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner_;
  gfx::GfxContext* gfx_context_;
};
