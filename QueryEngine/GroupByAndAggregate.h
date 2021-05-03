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

#ifndef QUERYENGINE_GROUPBYANDAGGREGATE_H
#define QUERYENGINE_GROUPBYANDAGGREGATE_H

#include "BufferCompaction.h"
#include "ColumnarResults.h"
#include "CompilationOptions.h"
#include "GpuMemUtils.h"
#include "GpuSharedMemoryContext.h"
#include "InputMetadata.h"
#include "QueryExecutionContext.h"
#include "Rendering/RenderInfo.h"
#include "RuntimeFunctions.h"

#include "QueryEngine/Utils/DiamondCodegen.h"

#include "../Shared/sqltypes.h"
#include "Logger/Logger.h"

#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Value.h>
#include <boost/algorithm/string/join.hpp>
#include <boost/make_unique.hpp>

#include <stack>
#include <vector>

extern bool g_enable_smem_group_by;
extern bool g_bigint_count;

struct ColRangeInfo {
  QueryDescriptionType hash_type_;
  int64_t min;
  int64_t max;
  int64_t bucket;
  bool has_nulls;
  bool isEmpty() { return min == 0 && max == -1; }
};

struct KeylessInfo {
  const bool keyless;
  const int32_t target_index;
};

class GroupByAndAggregate {
 public:
  GroupByAndAggregate(Executor* executor,
                      const ExecutorDeviceType device_type,
                      const RelAlgExecutionUnit& ra_exe_unit,
                      const std::vector<InputTableInfo>& query_infos,
                      std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                      const std::optional<int64_t>& group_cardinality_estimation);

  // returns true iff checking the error code after every row
  // is required -- slow path group by queries for now
  bool codegen(llvm::Value* filter_result,
               llvm::BasicBlock* sc_false,
               const QueryMemoryDescriptor& query_mem_desc,
               const CompilationOptions& co,
               const GpuSharedMemoryContext& gpu_smem_context);

  static size_t shard_count_for_top_groups(const RelAlgExecutionUnit& ra_exe_unit,
                                           const Catalog_Namespace::Catalog& catalog);

 private:
  bool gpuCanHandleOrderEntries(const std::list<Analyzer::OrderEntry>& order_entries);

  std::unique_ptr<QueryMemoryDescriptor> initQueryMemoryDescriptor(
      const bool allow_multifrag,
      const size_t max_groups_buffer_entry_count,
      const int8_t crt_min_byte_width,
      RenderInfo* render_info,
      const bool output_columnar_hint);

  std::unique_ptr<QueryMemoryDescriptor> initQueryMemoryDescriptorImpl(
      const bool allow_multifrag,
      const size_t max_groups_buffer_entry_count,
      const int8_t crt_min_byte_width,
      const bool sort_on_gpu_hint,
      RenderInfo* render_info,
      const bool must_use_baseline_sort,
      const bool output_columnar_hint);

  int64_t getShardedTopBucket(const ColRangeInfo& col_range_info,
                              const size_t shard_count) const;

  llvm::Value* codegenOutputSlot(llvm::Value* groups_buffer,
                                 const QueryMemoryDescriptor& query_mem_desc,
                                 const CompilationOptions& co,
                                 DiamondCodegen& diamond_codegen);

  std::tuple<llvm::Value*, llvm::Value*> codegenGroupBy(
      const QueryMemoryDescriptor& query_mem_desc,
      const CompilationOptions& co,
      DiamondCodegen& codegen);

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

  ColRangeInfo getColRangeInfo();

  static int64_t getBucketedCardinality(const ColRangeInfo& col_range_info);

  llvm::Value* convertNullIfAny(const SQLTypeInfo& arg_type,
                                const TargetInfo& agg_info,
                                llvm::Value* target);

  bool codegenAggCalls(const std::tuple<llvm::Value*, llvm::Value*>& agg_out_ptr_w_idx,
                       const std::vector<llvm::Value*>& agg_out_vec,
                       const QueryMemoryDescriptor& query_mem_desc,
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

  void codegenApproxMedian(const size_t target_idx,
                           const Analyzer::Expr* target_expr,
                           std::vector<llvm::Value*>& agg_args,
                           const QueryMemoryDescriptor& query_mem_desc,
                           const ExecutorDeviceType device_type);

  llvm::Value* getAdditionalLiteral(const int32_t off);

  std::vector<llvm::Value*> codegenAggArg(const Analyzer::Expr* target_expr,
                                          const CompilationOptions& co);

  llvm::Value* emitCall(const std::string& fname, const std::vector<llvm::Value*>& args);

  void checkErrorCode(llvm::Value* retCode);

  bool needsUnnestDoublePatch(llvm::Value const* val_ptr,
                              const std::string& agg_base_name,
                              const bool threads_share_memory,
                              const CompilationOptions& co) const;

  void prependForceSync();

  Executor* executor_;
  const RelAlgExecutionUnit& ra_exe_unit_;
  const std::vector<InputTableInfo>& query_infos_;
  std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner_;
  bool output_columnar_;
  const ExecutorDeviceType device_type_;

  const std::optional<int64_t> group_cardinality_estimation_;

  friend class Executor;
  friend class QueryMemoryDescriptor;
  friend class CodeGenerator;
  friend class ExecutionKernel;
  friend struct TargetExprCodegen;
  friend struct TargetExprCodegenBuilder;
};

inline int64_t extract_from_datum(const Datum datum, const SQLTypeInfo& ti) {
  const auto type = ti.is_decimal() ? decimal_to_int_type(ti) : ti.get_type();
  switch (type) {
    case kBOOLEAN:
      return datum.tinyintval;
    case kTINYINT:
      return datum.tinyintval;
    case kSMALLINT:
      return datum.smallintval;
    case kCHAR:
    case kVARCHAR:
    case kTEXT:
      CHECK_EQ(kENCODING_DICT, ti.get_compression());
    case kINT:
      return datum.intval;
    case kBIGINT:
      return datum.bigintval;
    case kTIME:
    case kTIMESTAMP:
    case kDATE:
      return datum.bigintval;
    default:
      abort();
  }
}

inline int64_t extract_min_stat(const ChunkStats& stats, const SQLTypeInfo& ti) {
  return extract_from_datum(stats.min, ti);
}

inline int64_t extract_max_stat(const ChunkStats& stats, const SQLTypeInfo& ti) {
  return extract_from_datum(stats.max, ti);
}

inline size_t get_count_distinct_sub_bitmap_count(const size_t bitmap_sz_bits,
                                                  const RelAlgExecutionUnit& ra_exe_unit,
                                                  const ExecutorDeviceType device_type) {
  // For count distinct on a column with a very small number of distinct values
  // contention can be very high, especially for non-grouped queries. We'll split
  // the bitmap into multiple sub-bitmaps which are unified to get the full result.
  // The threshold value for bitmap_sz_bits works well on Kepler.
  return bitmap_sz_bits < 50000 && ra_exe_unit.groupby_exprs.empty() &&
                 (device_type == ExecutorDeviceType::GPU || g_cluster)
             ? 64  // NB: must be a power of 2 to keep runtime offset computations cheap
             : 1;
}

#endif  // QUERYENGINE_GROUPBYANDAGGREGATE_H
