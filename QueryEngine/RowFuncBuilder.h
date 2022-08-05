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

#pragma once

#include <stack>
#include <vector>

#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Value.h>
#include <boost/algorithm/string/join.hpp>

#include "Logger/Logger.h"
#include "QueryEngine/BufferCompaction.h"
#include "QueryEngine/ColumnarResults.h"
#include "QueryEngine/CompilationOptions.h"
#include "QueryEngine/GpuMemUtils.h"
#include "QueryEngine/GpuSharedMemoryContext.h"
#include "QueryEngine/InputMetadata.h"
#include "QueryEngine/QueryExecutionContext.h"
#include "QueryEngine/RuntimeFunctions.h"
#include "QueryEngine/Utils/DiamondCodegen.h"
#include "SchemaMgr/SchemaProvider.h"
#include "Shared/sqltypes.h"

class RowFuncBuilder {
 public:
  RowFuncBuilder(Executor* executor,
                 const ExecutorDeviceType device_type,
                 const RelAlgExecutionUnit& ra_exe_unit,
                 const std::vector<InputTableInfo>& query_infos,
                 std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                 const std::optional<int64_t>& group_cardinality_estimation);

  // returns true iff checking the error code after every row
  // is required -- slow path group by queries for now
  bool codegen(llvm::Value* filter_result,
               llvm::BasicBlock* sc_false,
               QueryMemoryDescriptor& query_mem_desc,
               const CompilationOptions& co,
               const GpuSharedMemoryContext& gpu_smem_context);

 private:
  llvm::Value* codegenOutputSlot(llvm::Value* groups_buffer,
                                 const QueryMemoryDescriptor& query_mem_desc,
                                 const CompilationOptions& co,
                                 DiamondCodegen& diamond_codegen);

  std::tuple<llvm::Value*, llvm::Value*> codegenGroupBy(
      const QueryMemoryDescriptor& query_mem_desc,
      const CompilationOptions& co,
      DiamondCodegen& codegen);

  llvm::Value* codegenVarlenOutputBuffer(const QueryMemoryDescriptor& query_mem_desc);

  std::tuple<llvm::Value*, llvm::Value*> codegenSingleColumnPerfectHash(
      const QueryMemoryDescriptor& query_mem_desc,
      const CompilationOptions& co,
      llvm::Value* groups_buffer,
      llvm::Value* group_expr_lv_translated,
      llvm::Value* group_expr_lv_original,
      const int32_t row_size_quad);

  std::tuple<llvm::Value*, llvm::Value*> codegenMultiColumnPerfectHash(
      llvm::Value* groups_buffer,
      llvm::Value* group_key,
      llvm::Value* key_size_lv,
      const QueryMemoryDescriptor& query_mem_desc,
      const int32_t row_size_quad);
  llvm::Function* codegenPerfectHashFunction();

  std::tuple<llvm::Value*, llvm::Value*> codegenMultiColumnBaselineHash(
      const CompilationOptions& co,
      llvm::Value* groups_buffer,
      llvm::Value* group_key,
      llvm::Value* key_size_lv,
      const QueryMemoryDescriptor& query_mem_desc,
      const size_t key_width,
      const int32_t row_size_quad);

  llvm::Value* convertNullIfAny(const SQLTypeInfo& arg_type,
                                const TargetInfo& agg_info,
                                llvm::Value* target);

  bool codegenAggCalls(const std::tuple<llvm::Value*, llvm::Value*>& agg_out_ptr_w_idx,
                       llvm::Value* varlen_output_buffer,
                       const std::vector<llvm::Value*>& agg_out_vec,
                       QueryMemoryDescriptor& query_mem_desc,
                       const CompilationOptions& co,
                       const GpuSharedMemoryContext& gpu_smem_context,
                       DiamondCodegen& diamond_codegen);

  llvm::Value* codegenWindowRowPointer(const Analyzer::WindowFunction* window_func,
                                       const QueryMemoryDescriptor& query_mem_desc,
                                       const CompilationOptions& co,
                                       DiamondCodegen& diamond_codegen);

  llvm::Value* codegenAggColumnPtr(
      llvm::Value* output_buffer_byte_stream,
      llvm::Value* out_row_idx,
      const std::tuple<llvm::Value*, llvm::Value*>& agg_out_ptr_w_idx,
      const QueryMemoryDescriptor& query_mem_desc,
      const size_t chosen_bytes,
      const size_t agg_out_off,
      const size_t target_idx);

  void codegenEstimator(std::stack<llvm::BasicBlock*>& array_loops,
                        DiamondCodegen& diamond_codegen,
                        const QueryMemoryDescriptor& query_mem_desc,
                        const CompilationOptions&);

  void codegenCountDistinct(const size_t target_idx,
                            const Analyzer::Expr* target_expr,
                            std::vector<llvm::Value*>& agg_args,
                            const QueryMemoryDescriptor&,
                            const ExecutorDeviceType);

  void codegenApproxQuantile(const size_t target_idx,
                             const Analyzer::Expr* target_expr,
                             std::vector<llvm::Value*>& agg_args,
                             const QueryMemoryDescriptor& query_mem_desc,
                             const ExecutorDeviceType device_type);

  llvm::Value* getAdditionalLiteral(const int32_t off);

  std::vector<llvm::Value*> codegenAggArg(const Analyzer::Expr* target_expr,
                                          const CompilationOptions& co);

  llvm::Value* emitCall(const std::string& fname, const std::vector<llvm::Value*>& args);

  void checkErrorCode(llvm::Value* retCode);

  std::tuple<llvm::Value*, llvm::Value*> genLoadHashDesc(llvm::Value* groups_buffer);

  Executor* executor_;
  const Config& config_;
  const RelAlgExecutionUnit& ra_exe_unit_;
  const std::vector<InputTableInfo>& query_infos_;

  friend class Executor;
  friend class CodeGenerator;
  friend class ExecutionKernel;
  friend struct TargetExprCodegen;
  friend struct TargetExprCodegenBuilder;
};
