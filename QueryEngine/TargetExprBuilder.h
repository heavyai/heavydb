/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file    TargetExprBuilder.h
 * @brief   Helpers for codegen of target expressions
 *
 */

#pragma once

#include <Analyzer/Analyzer.h>
#include <Shared/TargetInfo.h>

#include "Descriptors/QueryMemoryDescriptor.h"
#include "GroupByAndAggregate.h"

#include <vector>

struct TargetExprCodegen {
  TargetExprCodegen(const Analyzer::Expr* target_expr,
                    TargetInfo& target_info,
                    const int32_t base_slot_index,
                    const size_t target_idx,
                    const bool is_group_by)
      : target_expr(target_expr)
      , target_info(target_info)
      , base_slot_index(base_slot_index)
      , target_idx(target_idx)
      , is_group_by(is_group_by) {}

  void codegen(GroupByAndAggregate* group_by_and_agg,
               Executor* executor,
               const QueryMemoryDescriptor& query_mem_desc,
               const CompilationOptions& co,
               const GpuSharedMemoryContext& gpu_smem_context,
               const std::tuple<llvm::Value*, llvm::Value*>& agg_out_ptr_w_idx,
               const std::vector<llvm::Value*>& agg_out_vec,
               llvm::Value* output_buffer_byte_stream,
               llvm::Value* out_row_idx,
               llvm::Value* varlen_output_buffer,
               DiamondCodegen& diamond_codegen,
               DiamondCodegen* sample_cfg = nullptr) const;

  void codegenAggregate(GroupByAndAggregate* group_by_and_agg,
                        Executor* executor,
                        const QueryMemoryDescriptor& query_mem_desc,
                        const CompilationOptions& co,
                        const std::vector<llvm::Value*>& target_lvs,
                        const std::tuple<llvm::Value*, llvm::Value*>& agg_out_ptr_w_idx,
                        const std::vector<llvm::Value*>& agg_out_vec,
                        llvm::Value* output_buffer_byte_stream,
                        llvm::Value* out_row_idx,
                        llvm::Value* varlen_output_buffer,
                        int32_t slot_index) const;

  friend std::ostream& operator<<(std::ostream& os,
                                  const TargetExprCodegen& target_expr_codegen);

  const Analyzer::Expr* target_expr;
  TargetInfo target_info;

  int32_t base_slot_index;
  size_t target_idx;
  bool is_group_by;
};

struct TargetExprCodegenBuilder {
  TargetExprCodegenBuilder(const RelAlgExecutionUnit& ra_exe_unit, const bool is_group_by)
      : ra_exe_unit(ra_exe_unit), is_group_by(is_group_by) {}

  void operator()(const Analyzer::Expr* target_expr,
                  const Executor* executor,
                  QueryMemoryDescriptor& query_mem_desc,
                  const CompilationOptions& co);

  void codegen(GroupByAndAggregate* group_by_and_agg,
               Executor* executor,
               const QueryMemoryDescriptor& query_mem_desc,
               const CompilationOptions& co,
               const GpuSharedMemoryContext& gpu_smem_context,
               const std::tuple<llvm::Value*, llvm::Value*>& agg_out_ptr_w_idx,
               const std::vector<llvm::Value*>& agg_out_vec,
               llvm::Value* output_buffer_byte_stream,
               llvm::Value* out_row_idx,
               llvm::Value* varlen_output_buffer,
               DiamondCodegen& diamond_codegen) const;

  void codegenSampleExpressions(
      GroupByAndAggregate* group_by_and_agg,
      Executor* executor,
      const QueryMemoryDescriptor& query_mem_desc,
      const CompilationOptions& co,
      const std::tuple<llvm::Value*, llvm::Value*>& agg_out_ptr_w_idx,
      const std::vector<llvm::Value*>& agg_out_vec,
      llvm::Value* output_buffer_byte_stream,
      llvm::Value* out_row_idx,
      DiamondCodegen& diamond_codegen) const;

  void codegenSingleSlotSampleExpression(
      GroupByAndAggregate* group_by_and_agg,
      Executor* executor,
      const QueryMemoryDescriptor& query_mem_desc,
      const CompilationOptions& co,
      const std::tuple<llvm::Value*, llvm::Value*>& agg_out_ptr_w_idx,
      const std::vector<llvm::Value*>& agg_out_vec,
      llvm::Value* output_buffer_byte_stream,
      llvm::Value* out_row_idx,
      DiamondCodegen& diamond_codegen) const;

  void codegenMultiSlotSampleExpressions(
      GroupByAndAggregate* group_by_and_agg,
      Executor* executor,
      const QueryMemoryDescriptor& query_mem_desc,
      const CompilationOptions& co,
      const std::tuple<llvm::Value*, llvm::Value*>& agg_out_ptr_w_idx,
      const std::vector<llvm::Value*>& agg_out_vec,
      llvm::Value* output_buffer_byte_stream,
      llvm::Value* out_row_idx,
      DiamondCodegen& diamond_codegen) const;

  llvm::Value* codegenSlotEmptyKey(llvm::Value* agg_col_ptr,
                                   std::vector<llvm::Value*>& target_lvs,
                                   Executor* executor,
                                   const QueryMemoryDescriptor& query_mem_desc,
                                   const int64_t init_val) const;

  size_t target_index_counter{0};
  size_t slot_index_counter{0};

  const RelAlgExecutionUnit& ra_exe_unit;

  std::vector<TargetExprCodegen> target_exprs_to_codegen;
  std::vector<TargetExprCodegen> sample_exprs_to_codegen;

  bool is_group_by;
};
