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
#include "InputMetadata.h"
#include "QueryExecutionContext.h"
#include "Rendering/RenderInfo.h"
#include "RuntimeFunctions.h"

#include "../Planner/Planner.h"
#include "../Shared/sqltypes.h"

#include <glog/logging.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Value.h>
#include <boost/algorithm/string/join.hpp>
#include <boost/make_unique.hpp>

#include <stack>
#include <vector>

extern bool g_enable_smem_group_by;

class ReductionRanOutOfSlots : public std::runtime_error {
 public:
  ReductionRanOutOfSlots() : std::runtime_error("ReductionRanOutOfSlots") {}
};

inline std::string nullable_str_to_string(const NullableString& str) {
  auto nptr = boost::get<void*>(&str);
  if (nptr) {
    CHECK(!*nptr);
    return "NULL";
  }
  auto sptr = boost::get<std::string>(&str);
  CHECK(sptr);
  return *sptr;
}

inline std::string datum_to_string(const TargetValue& tv,
                                   const SQLTypeInfo& ti,
                                   const std::string& delim) {
  if (ti.is_array()) {
    const auto list_tv = boost::get<std::vector<ScalarTargetValue>>(&tv);
    CHECK(list_tv);
    std::vector<std::string> elem_strs;
    elem_strs.reserve(list_tv->size());
    const auto& elem_ti = ti.get_elem_type();
    for (const auto& elem_tv : *list_tv) {
      elem_strs.push_back(datum_to_string(elem_tv, elem_ti, delim));
    }
    return "{" + boost::algorithm::join(elem_strs, delim) + "}";
  }
  const auto scalar_tv = boost::get<ScalarTargetValue>(&tv);
  if (ti.is_time()) {
    Datum datum;
    datum.timeval = *boost::get<int64_t>(scalar_tv);
    if (datum.timeval == NULL_BIGINT) {
      return "NULL";
    }
    return DatumToString(datum, ti);
  }
  if (ti.is_boolean()) {
    const auto bool_val = *boost::get<int64_t>(scalar_tv);
    return bool_val == NULL_BOOLEAN ? "NULL" : (bool_val ? "true" : "false");
  }
  auto iptr = boost::get<int64_t>(scalar_tv);
  if (iptr) {
    return *iptr == inline_int_null_val(ti) ? "NULL" : std::to_string(*iptr);
  }
  auto fptr = boost::get<float>(scalar_tv);
  if (fptr) {
    return *fptr == inline_fp_null_val(ti) ? "NULL" : std::to_string(*fptr);
  }
  auto dptr = boost::get<double>(scalar_tv);
  if (dptr) {
    return *dptr == inline_fp_null_val(ti.is_decimal() ? SQLTypeInfo(kDOUBLE, false) : ti)
               ? "NULL"
               : std::to_string(*dptr);
  }
  auto sptr = boost::get<NullableString>(scalar_tv);
  CHECK(sptr);
  return nullable_str_to_string(*sptr);
}

struct ColRangeInfo {
  const QueryDescriptionType hash_type_;
  const int64_t min;
  const int64_t max;
  const int64_t bucket;
  const bool has_nulls;
  bool isEmpty() { return min == 0 && max == -1; }
};

class GroupByAndAggregate {
 public:
  GroupByAndAggregate(Executor* executor,
                      const ExecutorDeviceType device_type,
                      const RelAlgExecutionUnit& ra_exe_unit,
                      RenderInfo* render_info,
                      const std::vector<InputTableInfo>& query_infos,
                      std::shared_ptr<RowSetMemoryOwner>,
                      const size_t max_groups_buffer_entry_count,
                      const size_t small_groups_buffer_entry_count,
                      const int8_t crt_min_byte_width,
                      const bool allow_multifrag,
                      const bool output_columnar_hint);

  const QueryMemoryDescriptor& getQueryMemoryDescriptor() const;

  bool outputColumnar() const;

  // returns true iff checking the error code after every row
  // is required -- slow path group by queries for now
  bool codegen(llvm::Value* filter_result,
               llvm::BasicBlock* sc_false,
               const CompilationOptions& co);

  static void addTransientStringLiterals(
      const RelAlgExecutionUnit& ra_exe_unit,
      Executor* executor,
      std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner);

  static size_t shard_count_for_top_groups(const RelAlgExecutionUnit& ra_exe_unit,
                                           const Catalog_Namespace::Catalog& catalog);

 private:
  struct DiamondCodegen {
    DiamondCodegen(llvm::Value* cond,
                   Executor* executor,
                   const bool chain_to_next,
                   const std::string& label_prefix,
                   DiamondCodegen* parent,
                   const bool share_false_edge_with_parent);
    void setChainToNext();
    void setFalseTarget(llvm::BasicBlock* cond_false);
    ~DiamondCodegen();

    Executor* executor_;
    llvm::BasicBlock* cond_true_;
    llvm::BasicBlock* cond_false_;
    llvm::BasicBlock* orig_cond_false_;
    bool chain_to_next_;
    DiamondCodegen* parent_;
  };

  bool supportedTypeForGpuSharedMemUsage(const SQLTypeInfo& target_type_info) const;

  bool supportedExprForGpuSharedMemUsage(Analyzer::Expr* expr) const;

  bool gpuCanHandleOrderEntries(const std::list<Analyzer::OrderEntry>& order_entries);

  void initQueryMemoryDescriptor(const bool allow_multifrag,
                                 const size_t max_groups_buffer_entry_count,
                                 const size_t small_groups_buffer_entry_count,
                                 const int8_t crt_min_byte_width,
                                 const bool sort_on_gpu_hint,
                                 RenderInfo* render_info,
                                 const bool must_use_baseline_sort,
                                 const bool output_columnar_hint);

  int64_t getShardedTopBucket(const ColRangeInfo& col_range_info,
                              const size_t shard_count) const;

  void addTransientStringLiterals();

  CountDistinctDescriptors initCountDistinctDescriptors();

  llvm::Value* codegenOutputSlot(llvm::Value* groups_buffer,
                                 const CompilationOptions& co,
                                 DiamondCodegen& diamond_codegen);

  std::tuple<llvm::Value*, llvm::Value*> codegenGroupBy(const CompilationOptions& co,
                                                        DiamondCodegen& codegen);

  std::tuple<llvm::Value*, llvm::Value*> codegenSingleColumnPerfectHash(
      const CompilationOptions& co,
      llvm::Value* groups_buffer,
      llvm::Value* group_expr_lv_translated,
      llvm::Value* group_expr_lv_original,
      const int32_t row_size_quad);

  std::tuple<llvm::Value*, llvm::Value*> codegenMultiColumnPerfectHash(
      llvm::Value* groups_buffer,
      llvm::Value* group_key,
      llvm::Value* key_size_lv,
      const int32_t row_size_quad);
  llvm::Function* codegenPerfectHashFunction();

  std::tuple<llvm::Value*, llvm::Value*> codegenMultiColumnBaselineHash(
      const CompilationOptions& co,
      llvm::Value* groups_buffer,
      llvm::Value* group_key,
      llvm::Value* key_size_lv,
      const size_t key_width,
      const int32_t row_size_quad);

  ColRangeInfo getColRangeInfo();

  ColRangeInfo getExprRangeInfo(const Analyzer::Expr* expr) const;

  static int64_t getBucketedCardinality(const ColRangeInfo& col_range_info);

  struct KeylessInfo {
    const bool keyless;
    const int32_t target_index;
    const int64_t init_val;
    const bool shared_mem_support;  // TODO(Saman) remove, all aggregate operations should
                                    // eventually be potentially done with shared memory.
                                    // The decision will be made when the query memory
                                    // descriptor is created, not here. This member just
                                    // indicates the possibility.
  };

  KeylessInfo getKeylessInfo(const std::vector<Analyzer::Expr*>& target_expr_list,
                             const bool is_group_by) const;

  llvm::Value* convertNullIfAny(const SQLTypeInfo& arg_type,
                                const SQLTypeInfo& agg_type,
                                const size_t chosen_bytes,
                                llvm::Value* target);
  bool codegenAggCalls(const std::tuple<llvm::Value*, llvm::Value*>& agg_out_ptr_w_idx,
                       const std::vector<llvm::Value*>& agg_out_vec,
                       const CompilationOptions&);

  llvm::Value* codegenAggColumnPtr(
      llvm::Value* output_buffer_byte_stream,
      llvm::Value* out_row_idx,
      const std::tuple<llvm::Value*, llvm::Value*>& agg_out_ptr_w_idx,
      const size_t chosen_bytes,
      const size_t agg_out_off,
      const size_t target_idx);

  void codegenEstimator(std::stack<llvm::BasicBlock*>& array_loops,
                        GroupByAndAggregate::DiamondCodegen& diamond_codegen,
                        const CompilationOptions&);

  void codegenCountDistinct(const size_t target_idx,
                            const Analyzer::Expr* target_expr,
                            std::vector<llvm::Value*>& agg_args,
                            const QueryMemoryDescriptor&,
                            const ExecutorDeviceType);

  llvm::Value* getAdditionalLiteral(const int32_t off);

  std::vector<llvm::Value*> codegenAggArg(const Analyzer::Expr* target_expr,
                                          const CompilationOptions& co);

  llvm::Value* emitCall(const std::string& fname, const std::vector<llvm::Value*>& args);

  bool needsUnnestDoublePatch(llvm::Value* val_ptr,
                              const std::string& agg_base_name,
                              const CompilationOptions& co) const;

  void prependForceSync();

  QueryMemoryDescriptor query_mem_desc_;
  Executor* executor_;
  const RelAlgExecutionUnit& ra_exe_unit_;
  const std::vector<InputTableInfo>& query_infos_;
  std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner_;
  bool output_columnar_;
  const ExecutorDeviceType device_type_;

  friend class Executor;
  friend class QueryMemoryDescriptor;
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
      return datum.timeval;
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

template <class T>
inline std::vector<int8_t> get_col_byte_widths(
    const T& col_expr_list,
    const std::vector<ssize_t>& target_group_by_indices) {
  if (!target_group_by_indices.empty()) {
    CHECK_EQ(col_expr_list.size(), target_group_by_indices.size());
  }
  std::vector<int8_t> col_widths;
  size_t col_expr_idx = 0;
  for (const auto col_expr : col_expr_list) {
    if (!target_group_by_indices.empty() && target_group_by_indices[col_expr_idx] != -1) {
      col_widths.push_back(0);
      ++col_expr_idx;
      continue;
    }
    if (!col_expr) {
      // row index
      col_widths.push_back(sizeof(int64_t));
    } else {
      const auto agg_info = target_info(col_expr);
      const auto chosen_type = get_compact_type(agg_info);
      if ((chosen_type.is_string() && chosen_type.get_compression() == kENCODING_NONE) ||
          chosen_type.is_array()) {
        col_widths.push_back(sizeof(int64_t));
        col_widths.push_back(sizeof(int64_t));
        ++col_expr_idx;
        continue;
      }
      if (chosen_type.is_geometry()) {
        for (auto i = 0; i < chosen_type.get_physical_coord_cols(); ++i) {
          col_widths.push_back(sizeof(int64_t));
          col_widths.push_back(sizeof(int64_t));
        }
        ++col_expr_idx;
        continue;
      }
      const auto col_expr_bitwidth = get_bit_width(chosen_type);
      CHECK_EQ(size_t(0), col_expr_bitwidth % 8);
      col_widths.push_back(static_cast<int8_t>(col_expr_bitwidth >> 3));
      // for average, we'll need to keep the count as well
      if (agg_info.agg_kind == kAVG) {
        CHECK(agg_info.is_agg);
        col_widths.push_back(sizeof(int64_t));
      }
    }
    ++col_expr_idx;
  }
  return col_widths;
}

inline int8_t get_min_byte_width() {
  return MAX_BYTE_WIDTH_SUPPORTED;
}

struct RelAlgExecutionUnit;

#endif  // QUERYENGINE_GROUPBYANDAGGREGATE_H
