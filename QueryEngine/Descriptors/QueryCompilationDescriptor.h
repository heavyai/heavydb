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
 * @file    QueryCompilationDescriptor.h
 * @brief   Container for compilation results and assorted options for a single execution
 * unit.
 *
 */

#pragma once

#include "QueryEngine/CgenState.h"
#include "QueryEngine/ColumnFetcher.h"
#include "QueryEngine/GpuSharedMemoryContext.h"
#include "QueryEngine/PlanState.h"

class CompilationContext;

struct CompilationResult {
  std::shared_ptr<CompilationContext> generated_code;
  std::unordered_map<int, CgenState::LiteralValues> literal_values;
  bool output_columnar;
  std::string llvm_ir;
  GpuSharedMemoryContext gpu_smem_context;

 public:
  std::string toString() const {
    auto result = ::typeName(this) + "{";
    result += ::toString(generated_code);
    result += ", literal_values=" + ::toString(literal_values);
    result += ", toString(output_columnar=" + ::toString(output_columnar);
    result += ", llvm_ir='''\n" + ::toString(llvm_ir) + "\n'''";
    result += ", " + ::toString(gpu_smem_context);
    result += "}";
    return result;
  };
};

class QueryCompilationDescriptor {
 public:
  QueryCompilationDescriptor()
      : compilation_device_type_(ExecutorDeviceType::CPU)
      , hoist_literals_(false)
      , actual_min_byte_width_(MAX_BYTE_WIDTH_SUPPORTED) {}

  std::unique_ptr<QueryMemoryDescriptor> compile(
      const size_t max_groups_buffer_entry_guess,
      const int8_t crt_min_byte_width,
      const bool has_cardinality_estimation,
      const RelAlgExecutionUnit& ra_exe_unit,
      const std::vector<InputTableInfo>& table_infos,
      const PlanState::DeletedColumnsMap& deleted_cols_map,
      const ColumnFetcher& column_fetcher,
      const CompilationOptions& co,
      const ExecutionOptions& eo,
      RenderInfo* render_info,
      Executor* executor);

  auto getCompilationResult() const { return compilation_result_; }

  std::string getIR() const {
    switch (compilation_device_type_) {
      case ExecutorDeviceType::CPU: {
        return std::string{"IR for the CPU:\n===============\n" +
                           compilation_result_.llvm_ir};
      }
      case ExecutorDeviceType::GPU: {
        return std::string{"IR for the GPU:\n===============\n" +
                           compilation_result_.llvm_ir};
      }
    }
    UNREACHABLE();
    return "";
  }

  ExecutorDeviceType getDeviceType() const { return compilation_device_type_; }
  bool hoistLiterals() const { return hoist_literals_; }
  int8_t getMinByteWidth() const { return actual_min_byte_width_; }

  std::string toString() const {
    auto result = ::typeName(this) + "{";
    result += ::toString(compilation_result_);
    result += ", " + ::toString(compilation_device_type_);
    result += ", " + ::toString(hoist_literals_);
    result += ", " + ::toString(actual_min_byte_width_);
    result += "}";
    return result;
  };

 private:
  CompilationResult compilation_result_;
  ExecutorDeviceType compilation_device_type_;
  bool hoist_literals_;
  int8_t actual_min_byte_width_;
};
