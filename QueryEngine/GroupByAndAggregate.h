#ifndef QUERYENGINE_GROUPBYANDAGGREGATE_H
#define QUERYENGINE_GROUPBYANDAGGREGATE_H

#include "BufferCompaction.h"
#include "CompilationOptions.h"
#include "GpuMemUtils.h"
#include "ResultRows.h"
#include "RuntimeFunctions.h"

#include "../Fragmenter/Fragmenter.h"
#include "../Planner/Planner.h"
#include "../Shared/checked_alloc.h"
#include "../Shared/sqltypes.h"

#include "SqlTypesLayout.h"

#include <boost/algorithm/string/join.hpp>
#include <glog/logging.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Value.h>

#include <vector>

class ReductionRanOutOfSlots : public std::runtime_error {
 public:
  ReductionRanOutOfSlots() : std::runtime_error("ReductionRanOutOfSlots") {}
};

class ColumnarResults {
 public:
  ColumnarResults(const ResultRows& rows, const size_t num_columns, const std::vector<SQLTypeInfo>& target_types)
      : column_buffers_(num_columns), num_rows_(rows.rowCount()), target_types_(target_types) {
    column_buffers_.resize(num_columns);
    for (size_t i = 0; i < num_columns; ++i) {
      CHECK(!target_types[i].is_array());
      CHECK(!target_types[i].is_string() ||
            (target_types[i].get_compression() == kENCODING_DICT && target_types[i].get_size() == 4));
      column_buffers_[i] = static_cast<const int8_t*>(checked_malloc(num_rows_ * (get_bit_width(target_types[i]) / 8)));
    }
    size_t row_idx{0};
    while (true) {
      const auto crt_row = rows.getNextRow(false, false);
      if (crt_row.empty()) {
        break;
      }
      for (size_t i = 0; i < num_columns; ++i) {
        const auto col_val = crt_row[i];
        const auto scalar_col_val = boost::get<ScalarTargetValue>(&col_val);
        CHECK(scalar_col_val);
        auto i64_p = boost::get<int64_t>(scalar_col_val);
        if (i64_p) {
          switch (get_bit_width(target_types[i])) {
            case 8:
              ((int8_t*)column_buffers_[i])[row_idx] = static_cast<int8_t>(*i64_p);
              break;
            case 16:
              ((int16_t*)column_buffers_[i])[row_idx] = static_cast<int16_t>(*i64_p);
              break;
            case 32:
              ((int32_t*)column_buffers_[i])[row_idx] = static_cast<int32_t>(*i64_p);
              break;
            case 64:
              ((int64_t*)column_buffers_[i])[row_idx] = *i64_p;
              break;
            default:
              CHECK(false);
          }
        } else {
          CHECK(target_types[i].is_fp());
          switch (target_types[i].get_type()) {
            case kFLOAT: {
              auto float_p = boost::get<float>(scalar_col_val);
              ((float*)column_buffers_[i])[row_idx] = static_cast<float>(*float_p);
              break;
            }
            case kDOUBLE: {
              auto double_p = boost::get<double>(scalar_col_val);
              ((double*)column_buffers_[i])[row_idx] = static_cast<double>(*double_p);
              break;
            }
            default:
              CHECK(false);
          }
        }
      }
      ++row_idx;
    }
  }

  ~ColumnarResults() {
    for (const auto column_buffer : column_buffers_) {
      free((void*)column_buffer);
    }
  }

  const std::vector<const int8_t*>& getColumnBuffers() const { return column_buffers_; }

  const size_t size() const { return num_rows_; }

  const SQLTypeInfo& getColumnType(const int col_id) const {
    CHECK_GE(col_id, 0);
    CHECK_LT(static_cast<size_t>(col_id), target_types_.size());
    return target_types_[col_id];
  }

 private:
  std::vector<const int8_t*> column_buffers_;
  const size_t num_rows_;
  const std::vector<SQLTypeInfo> target_types_;
};

namespace {

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

inline std::string datum_to_string(const TargetValue& tv, const SQLTypeInfo& ti, const std::string& delim) {
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
    return *dptr == inline_fp_null_val(ti.is_decimal() ? SQLTypeInfo(kDOUBLE, false) : ti) ? "NULL"
                                                                                           : std::to_string(*dptr);
  }
  auto sptr = boost::get<NullableString>(scalar_tv);
  CHECK(sptr);
  return nullable_str_to_string(*sptr);
}

class ScopedScratchBuffer {
 public:
  ScopedScratchBuffer(const size_t num_bytes, Data_Namespace::DataMgr* data_mgr, const int device_id)
      : data_mgr_(data_mgr), ab_(alloc_gpu_abstract_buffer(data_mgr_, num_bytes, device_id)) {}
  ~ScopedScratchBuffer() { data_mgr_->freeAllBuffers(); }
  CUdeviceptr getPtr() const { return reinterpret_cast<CUdeviceptr>(ab_->getMemoryPtr()); }

 private:
  Data_Namespace::DataMgr* data_mgr_;
  Data_Namespace::AbstractBuffer* ab_;
};

}  // namespace

inline std::string row_col_to_string(const ResultRows& rows,
                                     const size_t row_idx,
                                     const size_t i,
                                     const std::string& delim = ", ") {
  const auto tv = rows.getRowAt(row_idx, i, true);
  const auto ti = rows.getColType(i);
  return datum_to_string(tv, ti, delim);
}

inline std::string row_col_to_string(const std::vector<TargetValue>& row,
                                     const size_t i,
                                     const SQLTypeInfo& ti,
                                     const std::string& delim = ", ") {
  return datum_to_string(row[i], ti, delim);
}

class QueryExecutionContext : boost::noncopyable {
 public:
  // TODO(alex): move init_agg_vals to GroupByBufferDescriptor, remove device_type
  QueryExecutionContext(const QueryMemoryDescriptor&,
                        const std::vector<int64_t>& init_agg_vals,
                        const Executor* executor,
                        const ExecutorDeviceType device_type,
                        const int device_id,
                        const std::vector<std::vector<const int8_t*>>& col_buffers,
                        std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                        const bool output_columnar,
                        const bool sort_on_gpu,
                        RenderAllocatorMap*);

  // TOOD(alex): get rid of targets parameter
  ResultRows getRowSet(const std::vector<Analyzer::Expr*>& targets,
                       const QueryMemoryDescriptor& query_mem_desc,
                       const bool was_auto_device) const noexcept;
  ResultRows groupBufferToResults(const size_t i,
                                  const std::vector<Analyzer::Expr*>& targets,
                                  const bool was_auto_device) const;

  std::vector<int64_t*> launchGpuCode(const std::vector<void*>& cu_functions,
                                      const bool hoist_literals,
                                      const std::vector<int8_t>& literal_buff,
                                      std::vector<std::vector<const int8_t*>> col_buffers,
                                      const std::vector<int64_t>& num_rows,
                                      const std::vector<uint64_t>& frag_row_offsets,
                                      const int32_t scan_limit,
                                      const std::vector<int64_t>& init_agg_vals,
                                      Data_Namespace::DataMgr* data_mgr,
                                      const unsigned block_size_x,
                                      const unsigned grid_size_x,
                                      const int device_id,
                                      int32_t* error_code,
                                      const uint32_t num_tables,
                                      const int64_t join_hash_table,
                                      RenderAllocatorMap* render_allocator_map) const;

 private:
  bool isEmptyBin(const int64_t* group_by_buffer, const size_t bin, const size_t key_idx) const;
  void outputBin(ResultRows& results,
                 const std::vector<Analyzer::Expr*>& targets,
                 int64_t* group_by_buffer,
                 const size_t bin) const;

  void initColumnPerRow(int8_t* row_ptr,
                        const size_t bin,
                        const int64_t* init_vals,
                        const std::vector<ssize_t>& bitmap_sizes);

  void initGroups(int64_t* groups_buffer,
                  const int64_t* init_vals,
                  const int32_t groups_buffer_entry_count,
                  const bool keyless,
                  const size_t warp_size);

  template <typename T>
  int8_t* initColumnarBuffer(T* buffer_ptr,
                             const T init_val,
                             const uint32_t entry_count,
                             const ssize_t bitmap_sz = -1,
                             const bool key_or_col = true);

  void initColumnarGroups(int64_t* groups_buffer,
                          const int64_t* init_vals,
                          const int32_t groups_buffer_entry_count,
                          const bool keyless);
#ifdef HAVE_CUDA
  enum {
    COL_BUFFERS,
    NUM_FRAGMENTS,
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
    JOIN_HASH_TABLE,
    KERN_PARAM_COUNT,
  };

  std::vector<CUdeviceptr> prepareKernelParams(const std::vector<std::vector<const int8_t*>>& col_buffers,
                                               const std::vector<int8_t>& literal_buff,
                                               const std::vector<int64_t>& num_rows,
                                               const std::vector<uint64_t>& frag_row_offsets,
                                               const int32_t scan_limit,
                                               const std::vector<int64_t>& init_agg_vals,
                                               const std::vector<int32_t>& error_codes,
                                               const uint32_t num_tables,
                                               const int64_t join_hash_table,
                                               Data_Namespace::DataMgr* data_mgr,
                                               const int device_id,
                                               const bool hoist_literals,
                                               const bool is_group_by) const;

  GpuQueryMemory prepareGroupByDevBuffer(Data_Namespace::DataMgr* data_mgr,
                                         RenderAllocator* render_allocator,
                                         const CUdeviceptr init_agg_vals_dev_ptr,
                                         const int device_id,
                                         const unsigned block_size_x,
                                         const unsigned grid_size_x,
                                         const bool can_sort_on_gpu) const;
#endif

  std::vector<ssize_t> allocateCountDistinctBuffers(const bool deferred);
  int64_t allocateCountDistinctBitmap(const size_t bitmap_sz);
  int64_t allocateCountDistinctSet();

  const QueryMemoryDescriptor& query_mem_desc_;
  std::vector<int64_t> init_agg_vals_;
  const Executor* executor_;
  const ExecutorDeviceType device_type_;
  const int device_id_;
  const std::vector<std::vector<const int8_t*>>& col_buffers_;
  const size_t num_buffers_;

  std::vector<int64_t*> group_by_buffers_;
  std::vector<int64_t*> small_group_by_buffers_;
  std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner_;
  const bool output_columnar_;
  const bool sort_on_gpu_;

  friend class Executor;
  friend void copy_group_by_buffers_from_gpu(Data_Namespace::DataMgr* data_mgr,
                                             const QueryExecutionContext* query_exe_context,
                                             const GpuQueryMemory& gpu_query_mem,
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
                      const bool render_output,
                      const std::vector<Fragmenter_Namespace::TableInfo>& query_infos,
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
  bool codegen(llvm::Value* filter_result, const CompilationOptions& co);

 private:
  struct ColRangeInfo {
    const GroupByColRangeType hash_type_;
    const int64_t min;
    const int64_t max;
    const int64_t bucket;
    const bool has_nulls;
  };

  struct DiamondCodegen {
    DiamondCodegen(llvm::Value* cond,
                   Executor* executor,
                   const bool chain_to_next,
                   const std::string& label_prefix,
                   DiamondCodegen* parent = nullptr);
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

  bool gpuCanHandleOrderEntries(const std::list<Analyzer::OrderEntry>& order_entries);

  void initQueryMemoryDescriptor(const bool allow_multifrag,
                                 const size_t max_groups_buffer_entry_count,
                                 const size_t small_groups_buffer_entry_count,
                                 const int8_t crt_min_byte_width,
                                 const bool sort_on_gpu_hint,
                                 const bool render_output);
  void addTransientStringLiterals();

  CountDistinctDescriptors initCountDistinctDescriptors();

  std::tuple<llvm::Value*, llvm::Value*> codegenGroupBy(const CompilationOptions& co, DiamondCodegen& codegen);

  llvm::Function* codegenPerfectHashFunction();

  GroupByAndAggregate::ColRangeInfo getColRangeInfo();

  GroupByAndAggregate::ColRangeInfo getExprRangeInfo(const Analyzer::Expr* expr) const;

  static int64_t getBucketedCardinality(const GroupByAndAggregate::ColRangeInfo& col_range_info);

  struct KeylessInfo {
    const bool keyless;
    const int32_t target_index;
    const int64_t init_val;
  };

  KeylessInfo getKeylessInfo(const std::vector<Analyzer::Expr*>& target_expr_list, const bool is_group_by) const;

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

  void codegenCountDistinct(const size_t target_idx,
                            const Analyzer::Expr* target_expr,
                            std::vector<llvm::Value*>& agg_args,
                            const QueryMemoryDescriptor&,
                            const ExecutorDeviceType);

  std::vector<llvm::Value*> codegenAggArg(const Analyzer::Expr* target_expr, const CompilationOptions& co);

  llvm::Value* emitCall(const std::string& fname, const std::vector<llvm::Value*>& args);

  bool isUnnestDouble(llvm::Value* val_ptr, const std::string& agg_base_name, const CompilationOptions& co) const;

  void prependForceSync();

  QueryMemoryDescriptor query_mem_desc_;
  Executor* executor_;
  const RelAlgExecutionUnit& ra_exe_unit_;
  const std::vector<Fragmenter_Namespace::TableInfo>& query_infos_;
  std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner_;
  bool output_columnar_;

  friend class Executor;
};

namespace {

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
      CHECK(false);
  }
}

inline int64_t extract_min_stat(const ChunkStats& stats, const SQLTypeInfo& ti) {
  return extract_from_datum(stats.min, ti);
}

inline int64_t extract_max_stat(const ChunkStats& stats, const SQLTypeInfo& ti) {
  return extract_from_datum(stats.max, ti);
}

}  // namespace

template <class T>
inline std::vector<int8_t> get_col_byte_widths(const T& col_expr_list) {
  std::vector<int8_t> col_widths;
  for (const auto col_expr : col_expr_list) {
    if (!col_expr) {
      // row index
      col_widths.push_back(sizeof(int64_t));
    } else {
      const auto agg_info = target_info(col_expr);
      const auto chosen_type = get_compact_type(agg_info);
      if ((chosen_type.is_string() && chosen_type.get_compression() == kENCODING_NONE) || chosen_type.is_array()) {
        col_widths.push_back(sizeof(int64_t));
        col_widths.push_back(sizeof(int64_t));
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

int8_t pick_target_compact_width(const RelAlgExecutionUnit& ra_exe_unit,
                                 const std::vector<Fragmenter_Namespace::TableInfo>& query_infos,
                                 const int8_t crt_min_byte_width);

#endif  // QUERYENGINE_GROUPBYANDAGGREGATE_H
