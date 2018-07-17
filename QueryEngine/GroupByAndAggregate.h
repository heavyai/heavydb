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
#include "IteratorTable.h"
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

class QueryExecutionContext : boost::noncopyable {
 public:
  // TODO(alex): move init_agg_vals to GroupByBufferDescriptor, remove device_type
  QueryExecutionContext(const RelAlgExecutionUnit& ra_exe_unit,
                        const QueryMemoryDescriptor&,
                        const std::vector<int64_t>& init_agg_vals,
                        const Executor* executor,
                        const ExecutorDeviceType device_type,
                        const int device_id,
                        const std::vector<std::vector<const int8_t*>>& col_buffers,
                        const std::vector<std::vector<const int8_t*>>& iter_buffers,
                        const std::vector<std::vector<uint64_t>>& frag_offsets,
                        std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                        const bool output_columnar,
                        const bool sort_on_gpu,
                        RenderInfo*);

  ResultPtr getResult(const RelAlgExecutionUnit& ra_exe_unit,
                      const std::vector<size_t>& outer_tab_frag_ids,
                      const bool was_auto_device) const;

  // TOOD(alex): get rid of targets parameter
  RowSetPtr getRowSet(const RelAlgExecutionUnit& ra_exe_unit,
                      const QueryMemoryDescriptor& query_mem_desc,
                      const bool was_auto_device) const;
  RowSetPtr groupBufferToResults(const size_t i,
                                 const std::vector<Analyzer::Expr*>& targets,
                                 const bool was_auto_device) const;

  IterTabPtr getIterTab(const std::vector<Analyzer::Expr*>& targets,
                        const ssize_t frag_idx) const;

  std::vector<int64_t*> launchGpuCode(
      const RelAlgExecutionUnit& ra_exe_unit,
      const std::vector<std::pair<void*, void*>>& cu_functions,
      const bool hoist_literals,
      const std::vector<int8_t>& literal_buff,
      std::vector<std::vector<const int8_t*>> col_buffers,
      const std::vector<std::vector<int64_t>>& num_rows,
      const std::vector<std::vector<uint64_t>>& frag_row_offsets,
      const uint32_t frag_stride,
      const int32_t scan_limit,
      const std::vector<int64_t>& init_agg_vals,
      Data_Namespace::DataMgr* data_mgr,
      const unsigned block_size_x,
      const unsigned grid_size_x,
      const int device_id,
      int32_t* error_code,
      const uint32_t num_tables,
      const std::vector<int64_t>& join_hash_tables,
      RenderAllocatorMap* render_allocator_map);

  std::vector<int64_t*> launchCpuCode(
      const RelAlgExecutionUnit& ra_exe_unit,
      const std::vector<std::pair<void*, void*>>& fn_ptrs,
      const bool hoist_literals,
      const std::vector<int8_t>& literal_buff,
      std::vector<std::vector<const int8_t*>> col_buffers,
      const std::vector<std::vector<int64_t>>& num_rows,
      const std::vector<std::vector<uint64_t>>& frag_row_offsets,
      const uint32_t frag_stride,
      const int32_t scan_limit,
      const std::vector<int64_t>& init_agg_vals,
      int32_t* error_code,
      const uint32_t num_tables,
      const std::vector<int64_t>& join_hash_tables);

  bool hasNoFragments() const { return consistent_frag_sizes_.empty(); }

 private:
  const std::vector<const int8_t*>& getColumnFrag(const size_t table_idx,
                                                  int64_t& global_idx) const;
  bool isEmptyBin(const int64_t* group_by_buffer,
                  const size_t bin,
                  const size_t key_idx) const;

  void initColumnPerRow(const QueryMemoryDescriptor& query_mem_desc,
                        int8_t* row_ptr,
                        const size_t bin,
                        const int64_t* init_vals,
                        const std::vector<ssize_t>& bitmap_sizes);

  void initGroups(int64_t* groups_buffer,
                  const int64_t* init_vals,
                  const int32_t groups_buffer_entry_count,
                  const bool keyless,
                  const size_t warp_size);

  template <typename T>
  int8_t* initColumnarBuffer(T* buffer_ptr, const T init_val, const uint32_t entry_count);

  void initColumnarGroups(int64_t* groups_buffer,
                          const int64_t* init_vals,
                          const int32_t groups_buffer_entry_count,
                          const bool keyless);

  IterTabPtr groupBufferToTab(const size_t buf_idx,
                              const ssize_t frag_idx,
                              const std::vector<Analyzer::Expr*>& targets) const;

  uint32_t getFragmentStride(
      const RelAlgExecutionUnit& ra_exe_unit,
      const std::vector<std::pair<int, std::vector<size_t>>>& frag_ids) const;

#ifdef HAVE_CUDA
  enum {
      COL_BUFFERS,
      NUM_FRAGMENTS,
      FRAG_STRIDE,
      LITERALS,
      NUM_ROWS,
      FRAG_ROW_OFFSETS,
      MAX_MATCHED,
      TOTAL_MATCHED,
      INIT_AGG_VALS,
      GROUPBY_BUF,
      SMALL_BUF,
      ERROR_CODE,
      NUM_TABLES,
      JOIN_HASH_TABLES,
      KERN_PARAM_COUNT,
  };

  void initializeDynamicWatchdog(void* native_module, const int device_id) const;

  std::vector<CUdeviceptr> prepareKernelParams(
      const std::vector<std::vector<const int8_t*>>& col_buffers,
      const std::vector<int8_t>& literal_buff,
      const std::vector<std::vector<int64_t>>& num_rows,
      const std::vector<std::vector<uint64_t>>& frag_offsets,
      const uint32_t frag_stride,
      const int32_t scan_limit,
      const std::vector<int64_t>& init_agg_vals,
      const std::vector<int32_t>& error_codes,
      const uint32_t num_tables,
      const std::vector<int64_t>& join_hash_tables,
      Data_Namespace::DataMgr* data_mgr,
      const int device_id,
      const bool hoist_literals,
      const bool is_group_by) const;

  std::pair<CUdeviceptr, CUdeviceptr> prepareTopNHeapsDevBuffer(
      Data_Namespace::DataMgr* data_mgr,
      const CUdeviceptr init_agg_vals_dev_ptr,
      const size_t n,
      const int device_id,
      const unsigned block_size_x,
      const unsigned grid_size_x) const;

  GpuQueryMemory prepareGroupByDevBuffer(Data_Namespace::DataMgr* data_mgr,
                                         RenderAllocator* render_allocator,
                                         const RelAlgExecutionUnit& ra_exe_unit,
                                         const CUdeviceptr init_agg_vals_dev_ptr,
                                         const int device_id,
                                         const unsigned block_size_x,
                                         const unsigned grid_size_x,
                                         const bool can_sort_on_gpu) const;
#endif

  std::vector<ssize_t> allocateCountDistinctBuffers(const bool deferred);
  int64_t allocateCountDistinctBitmap(const size_t bitmap_byte_sz);
  int64_t allocateCountDistinctSet();

  std::vector<ColumnLazyFetchInfo> getColLazyFetchInfo(
      const std::vector<Analyzer::Expr*>& target_exprs) const;

  void allocateCountDistinctGpuMem();

  RowSetPtr groupBufferToDeinterleavedResults(const size_t i) const;

  const QueryMemoryDescriptor& query_mem_desc_;
  std::vector<int64_t> init_agg_vals_;
  const Executor* executor_;
  const ExecutorDeviceType device_type_;
  const int device_id_;
  const std::vector<std::vector<const int8_t*>>& col_buffers_;
  const std::vector<std::vector<const int8_t*>>& iter_buffers_;
  const std::vector<std::vector<uint64_t>>& frag_offsets_;
  const std::vector<int64_t> consistent_frag_sizes_;
  const size_t num_buffers_;

  std::vector<int64_t*> group_by_buffers_;
  std::vector<int64_t*> small_group_by_buffers_;
  std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner_;
  const bool output_columnar_;
  const bool sort_on_gpu_;

  mutable std::vector<std::unique_ptr<ResultSet>> result_sets_;
  mutable std::unique_ptr<ResultSet> estimator_result_set_;
  CUdeviceptr count_distinct_bitmap_mem_;
  int8_t* count_distinct_bitmap_host_mem_;
  int8_t* count_distinct_bitmap_crt_ptr_;
  size_t count_distinct_bitmap_mem_bytes_;

  friend class Executor;
  friend void copy_group_by_buffers_from_gpu(
      Data_Namespace::DataMgr* data_mgr,
      const QueryExecutionContext* query_exe_context,
      const GpuQueryMemory& gpu_query_mem,
      const RelAlgExecutionUnit& ra_exe_unit,
      const unsigned block_size_x,
      const unsigned grid_size_x,
      const int device_id,
      const bool prepend_index_buffer);
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

  void patchGroupbyCall(llvm::CallInst* call_site);

  // returns true iff checking the error code after every row
  // is required -- slow path group by queries for now
  bool codegen(llvm::Value* filter_result,
               llvm::Value* nonjoin_filter_result,
               llvm::BasicBlock* sc_false,
               const CompilationOptions& co);

  static void addTransientStringLiterals(
      const RelAlgExecutionUnit& ra_exe_unit,
      Executor* executor,
      std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner);

 private:
  struct ColRangeInfo {
    const GroupByColRangeType hash_type_;
    const int64_t min;
    const int64_t max;
    const int64_t bucket;
    const bool has_nulls;
    bool isEmpty() { return min == 0 && max == -1; }
  };

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
                                 const bool must_use_baseline_sort);

  int64_t getShardedTopBucket(const ColRangeInfo& col_range_info,
                              const size_t shard_count) const;

  void addTransientStringLiterals();

  CountDistinctDescriptors initCountDistinctDescriptors();

  llvm::Value* codegenOutputSlot(llvm::Value* groups_buffer,
                                 const CompilationOptions& co,
                                 DiamondCodegen& diamond_codegen);

  std::tuple<llvm::Value*, llvm::Value*> codegenGroupBy(const CompilationOptions& co,
                                                        DiamondCodegen& codegen);

  llvm::Function* codegenPerfectHashFunction();

  GroupByAndAggregate::ColRangeInfo getColRangeInfo();

  GroupByAndAggregate::ColRangeInfo getExprRangeInfo(const Analyzer::Expr* expr) const;

  static int64_t getBucketedCardinality(
      const GroupByAndAggregate::ColRangeInfo& col_range_info);

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
#ifdef ENABLE_COMPACTION
  bool detectOverflowAndUnderflow(llvm::Value* agg_addr,
                                  llvm::Value* val,
                                  const TargetInfo& agg_info,
                                  const size_t chosen_bytes,
                                  const bool need_skip_null,
                                  const std::string& agg_base_name);
#endif
  bool codegenAggCalls(const std::tuple<llvm::Value*, llvm::Value*>& agg_out_ptr_w_idx,
                       const std::vector<llvm::Value*>& agg_out_vec,
                       const CompilationOptions&);

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
};

inline std::vector<Analyzer::Expr*> get_agg_target_exprs(
    const std::vector<std::shared_ptr<Analyzer::TargetEntry>>& target_list) {
  std::vector<Analyzer::Expr*> result;
  for (auto target : target_list) {
    auto target_expr = target->get_expr();
    CHECK(target_expr);
    result.push_back(target_expr);
  }
  return result;
}

inline std::vector<Analyzer::Expr*> get_agg_target_exprs(const Planner::Plan* plan) {
  const auto& target_list = plan->get_targetlist();
  return get_agg_target_exprs(target_list);
}

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
#ifdef ENABLE_COMPACTION
  return 4;
#else
  return MAX_BYTE_WIDTH_SUPPORTED;
#endif
}

struct RelAlgExecutionUnit;

size_t shard_count_for_top_groups(const RelAlgExecutionUnit& ra_exe_unit,
                                  const Catalog_Namespace::Catalog& catalog);

#endif  // QUERYENGINE_GROUPBYANDAGGREGATE_H
