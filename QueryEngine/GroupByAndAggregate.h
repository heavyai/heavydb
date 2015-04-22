#ifndef QUERYENGINE_GROUPBYANDAGGREGATE_H
#define QUERYENGINE_GROUPBYANDAGGREGATE_H

#include "GpuMemUtils.h"
#include "../Fragmenter/Fragmenter.h"
#include "../Planner/Planner.h"
#include "../Shared/sqltypes.h"

#include <boost/noncopyable.hpp>
#include <boost/range/adaptor/reversed.hpp>
#include <boost/variant.hpp>
#include <glog/logging.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Value.h>

#include <mutex>
#include <unordered_map>
#include <set>
#include <vector>


class Executor;

enum class GroupByColRangeType {
  OneColKnownRange,       // statically known range, only possible for column expressions
  OneColGuessedRange,     // best guess: small hash for the guess plus overflow for outliers
  MultiCol,
  MultiColKnownCardinality,
  Scan,                   // the plan is not a group by plan
};

// Private: each thread has its own memory, no atomic operations required
// Shared: threads in the same block share memory, atomic operations required
enum class GroupByMemSharing {
  Private,
  Shared
};

struct QueryMemoryDescriptor;

typedef boost::variant<int64_t, double, std::string> TargetValue;

struct TargetInfo {
  bool is_agg;
  SQLAgg agg_kind;
  SQLTypeInfo sql_type;
  bool skip_null_val;
  bool is_distinct;
};

enum class CountDistinctImplType {
  Bitmap,
  StdSet
};

inline size_t bitmap_size_bytes(const size_t bitmap_sz) {
  size_t bitmap_byte_sz = bitmap_sz / 8;
  if (bitmap_sz % 8) {
    ++bitmap_byte_sz;
  }
  return bitmap_byte_sz;
}

struct CountDistinctDescriptor {
  const Executor* executor_;
  CountDistinctImplType impl_type_;
  int64_t min_val;
  int64_t bitmap_sz_bits;

  size_t bitmapSizeBytes() const {
    CHECK(impl_type_ == CountDistinctImplType::Bitmap);
    return bitmap_size_bytes(bitmap_sz_bits);
  }
};

typedef std::unordered_map<size_t, CountDistinctDescriptor> CountDistinctDescriptors;

class RowSetMemoryOwner;

struct QueryMemoryDescriptor {
  const Executor* executor_;
  GroupByColRangeType hash_type;
  bool keyless_hash;
  bool interleaved_bins_on_gpu;
  std::vector<int8_t> group_col_widths;
  std::vector<int8_t> agg_col_widths;
  size_t entry_count;                    // the number of entries in the main buffer
  size_t entry_count_small;              // the number of entries in the small buffer
  int64_t min_val;                       // meaningful for OneCol{KnownRange, ConsecutiveKeys} only
  GroupByMemSharing sharing;             // meaningful for GPU only
  CountDistinctDescriptors count_distinct_descriptors_;

  std::unique_ptr<QueryExecutionContext> getQueryExecutionContext(
    const std::vector<int64_t>& init_agg_vals,
    const Executor* executor,
    const ExecutorDeviceType device_type,
    const int device_id,
    const std::vector<const int8_t*>& col_buffers,
    std::shared_ptr<RowSetMemoryOwner>) const;

  size_t getBufferSizeQuad(const ExecutorDeviceType device_type) const;
  size_t getSmallBufferSizeQuad() const;

  size_t getBufferSizeBytes(const ExecutorDeviceType device_type) const;
  size_t getSmallBufferSizeBytes() const;

  // TODO(alex): remove
  bool usesGetGroupValueFast() const;

  bool threadsShareMemory() const;

  bool lazyInitGroups(const ExecutorDeviceType) const;

  bool interleavedBins(const ExecutorDeviceType) const;

  size_t sharedMemBytes(const ExecutorDeviceType) const;
};

inline int64_t bitmap_set_size(const int64_t bitmap_ptr,
                               const int target_idx,
                               const CountDistinctDescriptors& count_distinct_descriptors) {
  const auto count_distinct_desc_it = count_distinct_descriptors.find(target_idx);
  CHECK(count_distinct_desc_it != count_distinct_descriptors.end());
  if (count_distinct_desc_it->second.impl_type_ != CountDistinctImplType::Bitmap) {
    CHECK(count_distinct_desc_it->second.impl_type_ == CountDistinctImplType::StdSet);
    return reinterpret_cast<std::set<int64_t>*>(bitmap_ptr)->size();
  }
  int64_t set_size { 0 };
  auto set_vals = reinterpret_cast<const int8_t*>(bitmap_ptr);
  for (size_t i = 0; i < count_distinct_desc_it->second.bitmapSizeBytes(); ++i) {
    for (auto bit_idx = 0; bit_idx < 8; ++bit_idx) {
      if (set_vals[i] & (1 << bit_idx)) {
        ++set_size;
      }
    }
  }
  return set_size;
}

inline void bitmap_set_unify(int8_t* lhs, int8_t* rhs, const size_t bitmap_sz) {
  for (size_t i = 0; i < bitmap_sz; ++i) {
    lhs[i] = rhs[i] = lhs[i] | rhs[i];
  }
}

typedef std::vector<int64_t> ValueTuple;

class ChunkIter;

class RowSetMemoryOwner : boost::noncopyable {
public:
  void setCountDistinctDescriptors(const CountDistinctDescriptors& count_distinct_descriptors) {
    if (count_distinct_descriptors_.empty()) {
      count_distinct_descriptors_ = count_distinct_descriptors;
    }
  }

  void addCountDistinctBuffer(int8_t* count_distinct_buffer) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    count_distinct_bitmaps_.push_back(count_distinct_buffer);
  }

  void addCountDistinctSet(std::set<int64_t>* count_distinct_set) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    count_distinct_sets_.push_back(count_distinct_set);
  }

  ~RowSetMemoryOwner() {
    for (auto count_distinct_buffer : count_distinct_bitmaps_) {
      free(count_distinct_buffer);
    }
    for (auto count_distinct_set : count_distinct_sets_) {
      delete count_distinct_set;
    }
  }
private:
  CountDistinctDescriptors count_distinct_descriptors_;
  std::vector<int8_t*> count_distinct_bitmaps_;
  std::vector<std::set<int64_t>*> count_distinct_sets_;
  std::mutex state_mutex_;

  friend class QueryExecutionContext;
  friend class ResultRows;
};

inline TargetInfo target_info(const Analyzer::Expr* target_expr) {
  const auto agg_expr = dynamic_cast<const Analyzer::AggExpr*>(target_expr);
  if (!agg_expr) {
    return { false, kMIN, target_expr ? target_expr->get_type_info() : SQLTypeInfo(kBIGINT), false, false };
  }
  const auto agg_type = agg_expr->get_aggtype();
  const auto agg_arg = agg_expr->get_arg();
  if (!agg_arg) {
    CHECK_EQ(kCOUNT, agg_type);
    CHECK(!agg_expr->get_is_distinct());
    return { true, kCOUNT, SQLTypeInfo(kBIGINT), false, false };
  }
  const auto& agg_arg_ti = agg_arg->get_type_info();
  bool is_distinct { false };
  if (agg_expr->get_aggtype() == kCOUNT) {
    CHECK(agg_expr->get_is_distinct());
    CHECK(!agg_arg_ti.is_fp());
    is_distinct = true;
  }
  // TODO(alex): null support for all types
  bool skip_null = !agg_arg_ti.get_notnull() && (agg_arg_ti.is_integer() || agg_arg_ti.is_time());
  return {
    true, agg_expr->get_aggtype(),
    agg_type == kAVG ? agg_arg_ti : agg_expr->get_type_info(),
    skip_null, is_distinct
  };
}

__attribute__((always_inline))
inline double pair_to_double(const std::pair<int64_t, int64_t>& fp_pair, const bool is_int) {
  return is_int
    ? static_cast<double>(fp_pair.first) / static_cast<double>(fp_pair.second)
    : *reinterpret_cast<const double*>(&fp_pair.first) / static_cast<double>(fp_pair.second);
}

class ResultRows {
public:
  ResultRows(const std::vector<Analyzer::Expr*>& targets,
             const Executor* executor,
             const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner)
    : executor_(executor)
    , row_set_mem_owner_(row_set_mem_owner) {
    for (const auto target_expr : targets) {
      const auto agg_info = target_info(target_expr);
      targets_.push_back(agg_info);
    }
  }

  void beginRow() {
    target_values_.emplace_back();
  }

  void beginRow(const int64_t key) {
    CHECK(multi_keys_.empty());
    simple_keys_.push_back(key);
    target_values_.emplace_back();
  }

  void beginRow(const std::vector<int64_t>& key) {
    CHECK(simple_keys_.empty());
    multi_keys_.push_back(key);
    target_values_.emplace_back();
  }

  void addValues(const std::vector<int64_t>& vals) {
    for (const auto val : vals) {
      target_values_.back().emplace_back(val);
    }
  }

  void addValue(const int64_t val) {
    target_values_.back().emplace_back(val);
  }

  // used for kAVG
  void addValue(const int64_t val1, const int64_t val2) {
    target_values_.back().emplace_back(std::make_pair(val1, val2));
  }

  void addValue(const std::string& val) {
    target_values_.back().emplace_back(val);
  }

  void discardRow() {
    CHECK_NE(simple_keys_.empty(), multi_keys_.empty());
    if (!simple_keys_.empty()) {
      simple_keys_.pop_back();
    } else {
      multi_keys_.pop_back();
    }
    target_values_.pop_back();
  }

  void append(const ResultRows& more_results) {
    simple_keys_.insert(simple_keys_.end(),
      more_results.simple_keys_.begin(), more_results.simple_keys_.end());
    multi_keys_.insert(multi_keys_.end(),
      more_results.multi_keys_.begin(), more_results.multi_keys_.end());
    target_values_.insert(target_values_.end(),
      more_results.target_values_.begin(), more_results.target_values_.end());
  }

  void reduce(const ResultRows& other_results);

  void sort(const Planner::Sort* sort_plan);

  void keepFirstN(const size_t n) {
    if (n >= size()) {
      return;
    }
    target_values_.resize(n);
  }

  size_t size() const {
    return target_values_.size();
  }

  size_t colCount() const {
    return targets_.size();
  }

  bool empty() const {
    return !size();
  }

  TargetValue get(const size_t row_idx,
                const size_t col_idx,
                const bool translate_strings) const;

  SQLTypeInfo getType(const size_t col_idx) const {
    return targets_[col_idx].sql_type;
  }

  bool operator==(const ResultRows& r) const {
    if (size() != r.size()) {
      return false;
    }
    for (size_t row_idx = 0; row_idx < size(); ++row_idx) {
      for (size_t col_idx = 0; col_idx < colCount(); ++col_idx) {
        const auto lhs_val = get(row_idx, col_idx, true);
        const auto rhs_val = r.get(row_idx, col_idx, true);
        const auto lhs_pd = boost::get<std::pair<int64_t, int64_t>>(&lhs_val);
        if (lhs_pd) {
          const auto rhs_pd = boost::get<std::pair<int64_t, int64_t>>(&rhs_val);
          if (!rhs_pd) {
            return false;
          }
          if (lhs_pd->first != rhs_pd->first || lhs_pd->second != rhs_pd->second) {
            return false;
          }
        }
      }
    }
    return true;
  }
private:
  void createReductionMap() const {
    if (!as_map_.empty() || !as_unordered_map_.empty()) {
      return;
    }
    CHECK_NE(simple_keys_.empty(), multi_keys_.empty());
    for (size_t i = 0; i < simple_keys_.size(); ++i) {
      as_unordered_map_.insert(std::make_pair(simple_keys_[i], target_values_[i]));
    }
    for (size_t i = 0; i < multi_keys_.size(); ++i) {
      as_map_.insert(std::make_pair(multi_keys_[i], target_values_[i]));
    }
  }

  std::vector<TargetInfo> targets_;
  std::vector<int64_t> simple_keys_;
  typedef std::vector<int64_t> MultiKey;
  std::vector<MultiKey> multi_keys_;
  typedef boost::variant<int64_t, std::pair<int64_t, int64_t>, std::string> InternalTargetValue;
  typedef std::vector<InternalTargetValue> TargetValues;
  std::vector<TargetValues> target_values_;
  mutable std::map<MultiKey, TargetValues> as_map_;
  mutable std::unordered_map<int64_t, TargetValues> as_unordered_map_;
  const Executor* executor_;
  std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner_;
};

inline std::string row_col_to_string(const ResultRows& rows, const size_t row_idx, const size_t i) {
  const auto agg_result = rows.get(row_idx, i, true);
  const auto agg_ti = rows.getType(i);
  if (agg_ti.is_time()) {
    Datum datum;
    datum.timeval = *boost::get<int64_t>(&agg_result);
    return DatumToString(datum, agg_ti);
  }
  if (agg_ti.is_boolean()) {
    const auto bool_val = *boost::get<int64_t>(&agg_result);
    return bool_val < 0 ? "NULL" : (bool_val ? "true" : "false");
  }
  auto iptr = boost::get<int64_t>(&agg_result);
  if (iptr) {
    return std::to_string(*iptr);
  }
  auto dptr = boost::get<double>(&agg_result);
  if (dptr) {
    return std::to_string(*dptr);
  }
  auto sptr = boost::get<std::string>(&agg_result);
  CHECK(sptr);
  return *sptr;
}

class QueryExecutionContext : boost::noncopyable {
public:
  // TODO(alex): move init_agg_vals to GroupByBufferDescriptor, remove device_type
  QueryExecutionContext(
    const QueryMemoryDescriptor&,
    const std::vector<int64_t>& init_agg_vals,
    const Executor* executor,
    const ExecutorDeviceType device_type,
    const int device_id,
    const std::vector<const int8_t*>& col_buffers,
    std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner);
  ~QueryExecutionContext();

  // TOOD(alex): get rid of targets parameter
  ResultRows getRowSet(const std::vector<Analyzer::Expr*>& targets) const;
  ResultRows groupBufferToResults(const size_t i, const std::vector<Analyzer::Expr*>& targets) const;

  std::vector<int64_t*> launchGpuCode(
    const std::vector<void*>& cu_functions,
    const bool hoist_literals,
    const std::vector<int8_t>& literal_buff,
    std::vector<const int8_t*> col_buffers,
    const int64_t num_rows,
    const int64_t scan_limit,
    const std::vector<int64_t>& init_agg_vals,
    Data_Namespace::DataMgr* data_mgr,
    const unsigned block_size_x,
    const unsigned grid_size_x,
    const int device_id,
    int32_t* error_code) const;

private:
  void initGroups(int64_t* groups_buffer,
                  const int64_t* init_vals,
                  const int32_t groups_buffer_entry_count,
                  const bool keyless,
                  const size_t warp_size);

  std::vector<ssize_t> allocateCountDistinctBuffers(const bool deferred);
  int64_t allocateCountDistinctBitmap(const size_t bitmap_sz);
  int64_t allocateCountDistinctSet();

  const QueryMemoryDescriptor& query_mem_desc_;
  std::vector<int64_t> init_agg_vals_;
  const Executor* executor_;
  const ExecutorDeviceType device_type_;
  const int device_id_;
  const std::vector<const int8_t*>& col_buffers_;
  const size_t num_buffers_;

  std::vector<int64_t*> group_by_buffers_;
  std::vector<int64_t*> small_group_by_buffers_;
  std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner_;

  friend class Executor;
  friend void copy_group_by_buffers_from_gpu(
    Data_Namespace::DataMgr* data_mgr,
    const QueryExecutionContext* query_exe_context,
    const GpuQueryMemory& gpu_query_mem,
    const unsigned block_size_x,
    const unsigned grid_size_x,
    const int device_id);
};

class GroupByAndAggregate {
public:
  GroupByAndAggregate(
    Executor* executor,
    const Planner::Plan* plan,
    const Fragmenter_Namespace::QueryInfo& query_info,
    std::shared_ptr<RowSetMemoryOwner>,
    const size_t max_groups_buffer_entry_count,
    const int64_t scan_limit);

  QueryMemoryDescriptor getQueryMemoryDescriptor(const size_t max_groups_buffer_entry_count);

  // returns true iff checking the error code after every row
  // is required -- slow path group by queries for now
  bool codegen(
    llvm::Value* filter_result,
    const ExecutorDeviceType,
    const bool hoist_literals);

private:
  struct ColRangeInfo {
    const GroupByColRangeType hash_type_;
    const int64_t min;
    const int64_t max;
  };

  struct DiamondCodegen {
    DiamondCodegen(llvm::Value* cond,
                   Executor* executor,
                   const bool chain_to_next,
                   DiamondCodegen* parent = nullptr);
    ~DiamondCodegen();

    Executor* executor_;
    llvm::BasicBlock* cond_true_;
    llvm::BasicBlock* cond_false_;
    const bool chain_to_next_;
    DiamondCodegen* parent_;
  };

  llvm::Value* codegenGroupBy(
    const QueryMemoryDescriptor&,
    const ExecutorDeviceType,
    const bool hoist_literals);

  GroupByAndAggregate::ColRangeInfo getColRangeInfo(
    const std::vector<Fragmenter_Namespace::FragmentInfo>&);

  GroupByAndAggregate::ColRangeInfo getExprRangeInfo(
    const Analyzer::Expr* expr,
    const std::vector<Fragmenter_Namespace::FragmentInfo>& fragments);

  void codegenAggCalls(
    llvm::Value* agg_out_start_ptr,
    const std::vector<llvm::Value*>& agg_out_vec,
    const QueryMemoryDescriptor&,
    const ExecutorDeviceType,
    const bool hoist_literals);

  void codegenCountDistinct(
    const size_t target_idx,
    const Analyzer::Expr* target_expr,
    std::vector<llvm::Value*>& agg_args,
    const QueryMemoryDescriptor&,
    const ExecutorDeviceType,
    const bool is_group_by,
    const int32_t agg_out_off);

  std::vector<llvm::Value*> codegenAggArg(
    const Analyzer::Expr* target_expr,
    const bool hoist_literals);

  llvm::Value* emitCall(const std::string& fname, const std::vector<llvm::Value*>& args);

  Executor* executor_;
  const Planner::Plan* plan_;
  const Fragmenter_Namespace::QueryInfo& query_info_;
  std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner_;
  const size_t max_groups_buffer_entry_count_;
  const int64_t scan_limit_;
};

namespace {

inline size_t get_bit_width(const SQLTypes type) {
  switch (type) {
    case kBOOLEAN:
      return 8;
    case kSMALLINT:
      return 16;
    case kINT:
      return 32;
    case kBIGINT:
      return 64;
    case kFLOAT:
      return 32;
    case kDOUBLE:
      return 64;
    case kTIME:
    case kTIMESTAMP:
    case kDATE:
      return sizeof(time_t) * 8;
    case kTEXT:
    case kVARCHAR:
    case kCHAR:
      return 32;
    default:
      CHECK(false);
  }
}

inline std::vector<Analyzer::Expr*> get_agg_target_exprs(const Planner::Plan* plan) {
  const auto& target_list = plan->get_targetlist();
  std::vector<Analyzer::Expr*> result;
  for (auto target : target_list) {
    auto target_expr = target->get_expr();
    CHECK(target_expr);
    result.push_back(target_expr);
  }
  return result;
}

inline int64_t extract_from_datum(const Datum datum, const SQLTypeInfo& ti) {
  switch (ti.get_type()) {
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

template<class T>
inline std::vector<int8_t> get_col_byte_widths(const T& col_expr_list) {
  std::vector<int8_t> col_widths;
  for (const auto col_expr : col_expr_list) {
    if (!col_expr) {
      // row index
      col_widths.push_back(sizeof(int64_t));
    } else {
      const auto agg_info = target_info(col_expr);
      if (agg_info.sql_type.is_string() && agg_info.sql_type.get_compression() == kENCODING_NONE) {
        col_widths.push_back(sizeof(int64_t));
        col_widths.push_back(sizeof(int64_t));
        continue;
      }
      const auto col_expr_bitwidth = get_bit_width(agg_info.sql_type.get_type());
      CHECK_EQ(0, col_expr_bitwidth % 8);
      col_widths.push_back(col_expr_bitwidth / 8);
      // for average, we'll need to keep the count as well
      if (agg_info.agg_kind == kAVG) {
        CHECK(agg_info.is_agg);
        col_widths.push_back(sizeof(int64_t));
      }
    }
  }
  return col_widths;
}

#endif // QUERYENGINE_GROUPBYANDAGGREGATE_H
