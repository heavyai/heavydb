/*
 * @file    ResultRows.h
 * @author  Alex Suhan <alex@mapd.com>
 *
 * Created on May 9, 2016, 3:45 PM
 */

#ifndef QUERYENGINE_RESULTROWS_H
#define QUERYENGINE_RESULTROWS_H

#include "QueryMemoryDescriptor.h"
#include "ResultSet.h"
#include "TargetValue.h"

#include "../Analyzer/Analyzer.h"
#include "../Shared/TargetInfo.h"
#include "../StringDictionary/StringDictionaryProxy.h"

#include <boost/noncopyable.hpp>
#include <glog/logging.h>

#include <list>
#include <set>
#include <unordered_set>

struct QueryMemoryDescriptor;
struct RelAlgExecutionUnit;
class RowSetMemoryOwner;

inline int64_t bitmap_set_size(const int64_t bitmap_ptr,
                               const int target_idx,
                               const CountDistinctDescriptors& count_distinct_descriptors) {
  const auto count_distinct_desc_it = count_distinct_descriptors.find(target_idx);
  CHECK(count_distinct_desc_it != count_distinct_descriptors.end());
  if (count_distinct_desc_it->second.impl_type_ != CountDistinctImplType::Bitmap) {
    CHECK(count_distinct_desc_it->second.impl_type_ == CountDistinctImplType::StdSet);
    return reinterpret_cast<std::set<int64_t>*>(bitmap_ptr)->size();
  }
  int64_t set_size{0};
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

inline int64_t get_component(const int8_t* group_by_buffer, const size_t comp_sz, const size_t index = 0) {
  int64_t ret = std::numeric_limits<int64_t>::min();
  switch (comp_sz) {
    case 1: {
      ret = group_by_buffer[index];
      break;
    }
    case 2: {
      const int16_t* buffer_ptr = reinterpret_cast<const int16_t*>(group_by_buffer);
      ret = buffer_ptr[index];
      break;
    }
    case 4: {
      const int32_t* buffer_ptr = reinterpret_cast<const int32_t*>(group_by_buffer);
      ret = buffer_ptr[index];
      break;
    }
    case 8: {
      const int64_t* buffer_ptr = reinterpret_cast<const int64_t*>(group_by_buffer);
      ret = buffer_ptr[index];
      break;
    }
    default:
      CHECK(false);
  }
  return ret;
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

  void addCountDistinctBuffer(int8_t* count_distinct_buffer, const size_t bytes) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    count_distinct_bitmaps_.emplace_back(count_distinct_buffer, bytes);
  }

  void addCountDistinctSet(std::set<int64_t>* count_distinct_set) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    count_distinct_sets_.push_back(count_distinct_set);
  }

  const CountDistinctDescriptors& getCountDistinctDescriptors() const { return count_distinct_descriptors_; }

  void addGroupByBuffer(int64_t* group_by_buffer) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    group_by_buffers_.push_back(group_by_buffer);
  }

  std::string* addString(const std::string& str) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    strings_.emplace_back(str);
    return &strings_.back();
  }

  std::vector<int64_t>* addArray(const std::vector<int64_t>& arr) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    arrays_.emplace_back(arr);
    return &arrays_.back();
  }

  StringDictionaryProxy* addStringDict(std::shared_ptr<StringDictionary> str_dict, const int dict_id) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    auto it = str_dict_proxy_owned_.find(dict_id);
    if (it != str_dict_proxy_owned_.end()) {
      CHECK_EQ(it->second->getDictionary(), str_dict.get());
      return it->second;
    }
    StringDictionaryProxy* str_dict_proxy = new StringDictionaryProxy(str_dict);
    str_dict_proxy_owned_.emplace(dict_id, str_dict_proxy);
    return str_dict_proxy;
  }

  StringDictionaryProxy* getStringDictProxy(const int dict_id) const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    auto it = str_dict_proxy_owned_.find(dict_id);
    CHECK(it != str_dict_proxy_owned_.end());
    return it->second;
  }

  void addLiteralStringDictProxy(std::shared_ptr<StringDictionaryProxy> lit_str_dict_proxy) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    lit_str_dict_proxy_ = lit_str_dict_proxy;
  }

  void addColBuffer(const void* col_buffer) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    col_buffers_.push_back(const_cast<void*>(col_buffer));
  }

  ~RowSetMemoryOwner() {
    for (const auto& count_distinct_buffer : count_distinct_bitmaps_) {
      free(count_distinct_buffer.first);
    }
    for (auto count_distinct_set : count_distinct_sets_) {
      delete count_distinct_set;
    }
    for (auto group_by_buffer : group_by_buffers_) {
      free(group_by_buffer);
    }
    for (auto col_buffer : col_buffers_) {
      free(col_buffer);
    }

    for (auto dict_proxy : str_dict_proxy_owned_) {
      delete dict_proxy.second;
    }

    if (lit_str_dict_proxy_) {
      lit_str_dict_proxy_.reset();
      lit_str_dict_proxy_ = nullptr;
    }
  }

 private:
  CountDistinctDescriptors count_distinct_descriptors_;
  std::vector<std::pair<int8_t*, size_t>> count_distinct_bitmaps_;
  std::vector<std::set<int64_t>*> count_distinct_sets_;
  std::vector<int64_t*> group_by_buffers_;
  std::list<std::string> strings_;
  std::list<std::vector<int64_t>> arrays_;
  std::unordered_map<int, StringDictionaryProxy*> str_dict_proxy_owned_;
  std::shared_ptr<StringDictionaryProxy> lit_str_dict_proxy_;
  std::vector<void*> col_buffers_;
  mutable std::mutex state_mutex_;

  friend class ResultRows;
  friend class ResultSet;
};

class InternalRow {
 public:
  InternalRow(RowSetMemoryOwner* row_set_mem_owner) : row_set_mem_owner_(row_set_mem_owner){};

  bool operator==(const InternalRow& other) const { return row_ == other.row_; }

  bool operator<(const InternalRow& other) const { return row_ < other.row_; }

  InternalTargetValue& operator[](const size_t i) { return row_[i]; }

  const InternalTargetValue& operator[](const size_t i) const { return row_[i]; }

  size_t size() const { return row_.size(); }

 private:
  void reserve(const size_t n) { row_.reserve(n); }

  void addValue(const int64_t val) { row_.emplace_back(val); }

  // used for kAVG
  void addValue(const int64_t val1, const int64_t val2) { row_.emplace_back(val1, val2); }

  void addValue(const std::string& val) { row_.emplace_back(row_set_mem_owner_->addString(val)); }

  void addValue(const std::vector<int64_t>& val) { row_.emplace_back(row_set_mem_owner_->addArray(val)); }

  void addValue() { row_.emplace_back(); }

  std::vector<InternalTargetValue> row_;
  RowSetMemoryOwner* row_set_mem_owner_;

  friend class RowStorage;
};

class RowStorage {
 private:
  size_t size() const { return rows_.size(); }

  void clear() { rows_.clear(); }

  void reserve(const size_t n) { rows_.reserve(n); }

  void beginRow(RowSetMemoryOwner* row_set_mem_owner) { rows_.emplace_back(row_set_mem_owner); }

  void reserveRow(const size_t n) { rows_.back().reserve(n); }

  void discardRow() { rows_.pop_back(); }

  void addValue(const int64_t val) { rows_.back().addValue(val); }

  // used for kAVG
  void addValue(const int64_t val1, const int64_t val2) { rows_.back().addValue(val1, val2); }

  void addValue(const std::string& val) { rows_.back().addValue(val); }

  void addValue(const std::vector<int64_t>& val) { rows_.back().addValue(val); }

  void addValue() { rows_.back().addValue(); }

  void push_back(const InternalRow& v) { rows_.push_back(v); }

  void append(const RowStorage& other) { rows_.insert(rows_.end(), other.rows_.begin(), other.rows_.end()); }

  void truncate(const size_t n) { rows_.erase(rows_.begin() + n, rows_.end()); }

  void drop(const size_t n) {
    if (n >= rows_.size()) {
      decltype(rows_)().swap(rows_);
      return;
    }
    decltype(rows_)(rows_.begin() + n, rows_.end()).swap(rows_);
  }

  InternalRow& operator[](const size_t i) { return rows_[i]; }

  const InternalRow& operator[](const size_t i) const { return rows_[i]; }

  InternalRow& front() { return rows_.front(); }

  const InternalRow& front() const { return rows_.front(); }

  const InternalRow& back() const { return rows_.back(); }

  void top(const int64_t n, const std::function<bool(const InternalRow& lhs, const InternalRow& rhs)> compare) {
    std::make_heap(rows_.begin(), rows_.end(), compare);
    decltype(rows_) top_target_values;
    top_target_values.reserve(n);
    for (int64_t i = 0; i < n && !rows_.empty(); ++i) {
      top_target_values.push_back(rows_.front());
      std::pop_heap(rows_.begin(), rows_.end(), compare);
      rows_.pop_back();
    }
    rows_.swap(top_target_values);
  }

  void sort(const std::function<bool(const InternalRow& lhs, const InternalRow& rhs)> compare) {
    std::sort(rows_.begin(), rows_.end(), compare);
  }

  void removeDuplicates() {
    std::sort(rows_.begin(), rows_.end());
    rows_.erase(std::unique(rows_.begin(), rows_.end()), rows_.end());
  }

  std::vector<InternalRow> rows_;

  friend class ResultRows;
};

inline const Analyzer::AggExpr* cast_to_agg_expr(const Analyzer::Expr* target_expr) {
  return dynamic_cast<const Analyzer::AggExpr*>(target_expr);
}

inline const Analyzer::AggExpr* cast_to_agg_expr(const std::shared_ptr<Analyzer::Expr> target_expr) {
  return dynamic_cast<const Analyzer::AggExpr*>(target_expr.get());
}

template <class PointerType>
inline TargetInfo target_info(const PointerType target_expr) {
  const auto agg_expr = cast_to_agg_expr(target_expr);
  bool notnull = target_expr->get_type_info().get_notnull();
  if (!agg_expr) {
    return {false,
            kMIN,
            target_expr ? target_expr->get_type_info() : SQLTypeInfo(kBIGINT, notnull),
            SQLTypeInfo(kNULLT, false),
            false,
            false};
  }
  const auto agg_type = agg_expr->get_aggtype();
  const auto agg_arg = agg_expr->get_arg();
  if (!agg_arg) {
    CHECK_EQ(kCOUNT, agg_type);
    CHECK(!agg_expr->get_is_distinct());
    return {true, kCOUNT, SQLTypeInfo(kINT, notnull), SQLTypeInfo(kNULLT, false), false, false};
  }

  const auto& agg_arg_ti = agg_arg->get_type_info();
  bool is_distinct{false};
  if (agg_expr->get_aggtype() == kCOUNT) {
    is_distinct = agg_expr->get_is_distinct();
  }

  return {true,
          agg_expr->get_aggtype(),
          agg_type == kCOUNT ? SQLTypeInfo(is_distinct ? kBIGINT : kINT, notnull)
                             : (agg_type == kAVG ? agg_arg_ti : agg_expr->get_type_info()),
          agg_arg_ti,
          !agg_arg_ti.get_notnull(),
          is_distinct};
}

inline std::vector<TargetInfo> target_exprs_to_infos(const std::vector<Analyzer::Expr*>& targets) {
  std::vector<TargetInfo> target_infos;
  for (const auto target_expr : targets) {
    target_infos.push_back(target_info(target_expr));
  }
  return target_infos;
}

struct GpuQueryMemory;

class ResultRows {
 public:
  ResultRows(std::shared_ptr<ResultSet>);

  ResultRows(const QueryMemoryDescriptor& query_mem_desc,
             const std::vector<Analyzer::Expr*>& targets,
             const Executor* executor,
             const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
             const std::vector<int64_t>& init_vals,
             const ExecutorDeviceType device_type,
             int64_t* group_by_buffer = nullptr,
             const int32_t groups_buffer_entry_count = 0,
             const int64_t min_val = 0,
             const int8_t warp_count = 0,
             const int64_t queue_time_ms = 0)
      : result_set_(nullptr),
        executor_(executor),
        query_mem_desc_(query_mem_desc),
        row_set_mem_owner_(row_set_mem_owner),
        agg_init_vals_(init_vals),
        group_by_buffer_(group_by_buffer),
        groups_buffer_entry_count_(groups_buffer_entry_count),
        group_by_buffer_idx_(0),
        min_val_(min_val),
        warp_count_(warp_count),
        output_columnar_(false),
        in_place_(false),
        device_type_(device_type),
        device_id_(-1),
        crt_row_idx_(0),
        crt_row_buff_idx_(0),
        drop_first_(0),
        keep_first_(0),
        fetch_started_(false),
        in_place_buff_idx_(0),
        just_explain_(false),
        queue_time_ms_(queue_time_ms) {
    for (const auto target_expr : targets) {
      targets_.push_back(target_info(target_expr));
    }
  }

  ResultRows(const QueryMemoryDescriptor& query_mem_desc,
             const std::vector<Analyzer::Expr*>& targets,
             const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
             const std::vector<int64_t>& init_vals,
             int64_t* group_by_buffer,
             const size_t groups_buffer_entry_count,
             const bool output_columnar,
             const std::vector<std::vector<const int8_t*>>& col_buffers,
             const ExecutorDeviceType device_type,
             const int device_id);

  ResultRows(const std::string& explanation, int64_t queue_time_ms)
      : result_set_(nullptr),
        query_mem_desc_{},
        group_by_buffer_idx_(0),
        output_columnar_(false),
        in_place_(false),
        device_type_(ExecutorDeviceType::Hybrid),
        device_id_(-1),
        crt_row_idx_(0),
        crt_row_buff_idx_(0),
        drop_first_(0),
        keep_first_(0),
        fetch_started_(false),
        in_place_buff_idx_(0),
        just_explain_(true),
        explanation_(explanation),
        queue_time_ms_(queue_time_ms) {}

  ResultRows(const std::string& explanation, int64_t queue_time_ms, int64_t render_time_ms)
      : result_set_(nullptr),
        query_mem_desc_{},
        group_by_buffer_idx_(0),
        output_columnar_(false),
        in_place_(false),
        device_type_(ExecutorDeviceType::Hybrid),
        device_id_(-1),
        crt_row_idx_(0),
        crt_row_buff_idx_(0),
        drop_first_(0),
        keep_first_(0),
        fetch_started_(false),
        in_place_buff_idx_(0),
        just_explain_(true),
        explanation_(explanation),
        queue_time_ms_(queue_time_ms),
        render_time_ms_(render_time_ms) {}

  ResultRows(const std::string& explanation,
             int64_t queue_time_ms,
             int64_t render_time_ms,
             const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner)
      : result_set_(nullptr),
        query_mem_desc_{},
        row_set_mem_owner_(row_set_mem_owner),
        group_by_buffer_idx_(0),
        output_columnar_(false),
        in_place_(false),
        device_type_(ExecutorDeviceType::Hybrid),
        device_id_(-1),
        crt_row_idx_(0),
        crt_row_buff_idx_(0),
        drop_first_(0),
        keep_first_(0),
        fetch_started_(false),
        in_place_buff_idx_(0),
        just_explain_(true),
        explanation_(explanation),
        queue_time_ms_(queue_time_ms),
        render_time_ms_(render_time_ms) {}

  explicit ResultRows(const std::string& explanation)
      : result_set_(nullptr),
        query_mem_desc_{},
        group_by_buffer_idx_(0),
        output_columnar_(false),
        in_place_(false),
        device_type_(ExecutorDeviceType::Hybrid),
        device_id_(-1),
        crt_row_idx_(0),
        crt_row_buff_idx_(0),
        drop_first_(0),
        keep_first_(0),
        fetch_started_(false),
        in_place_buff_idx_(0),
        just_explain_(true),
        explanation_(explanation),
        queue_time_ms_(0) {}

  void moveToBegin() const {
    if (result_set_) {
      result_set_->moveToBegin();
      return;
    }
    crt_row_idx_ = 0;
    crt_row_buff_idx_ = 0;
    in_place_buff_idx_ = 0;
    group_by_buffer_idx_ = 0;
    fetch_started_ = false;
  }
  void beginRow() {
    CHECK(!result_set_);
    target_values_.beginRow(row_set_mem_owner_.get());
  }

  void beginRow(const int64_t key) {
    CHECK(!result_set_);
    CHECK(multi_keys_.empty());
    simple_keys_.push_back(key);
    target_values_.beginRow(row_set_mem_owner_.get());
  }

  void beginRow(const std::vector<int64_t>& key) {
    CHECK(!result_set_);
    CHECK(simple_keys_.empty());
    multi_keys_.push_back(key);
    target_values_.beginRow(row_set_mem_owner_.get());
  }

  bool reduceSingleRow(const int8_t* row_ptr,
                       const int8_t warp_count,
                       const bool is_columnar,
                       const bool replace_bitmap_ptr_with_bitmap_sz,
                       std::vector<int64_t>& agg_vals) const;

  void addKeylessGroupByBuffer(const int64_t* group_by_buffer,
                               const int32_t groups_buffer_entry_count,
                               const int64_t min_val,
                               const int8_t warp_count,
                               const bool is_columnar);

  void addValue(const int64_t val) {
    CHECK(!result_set_);
    target_values_.addValue(val);
  }

  // used for kAVG
  void addValue(const int64_t val1, const int64_t val2) {
    CHECK(!result_set_);
    target_values_.addValue(val1, val2);
  }

  void addValue(const std::string& val) {
    CHECK(!result_set_);
    target_values_.addValue(val);
  }

  void addValue(const std::vector<int64_t>& val) {
    CHECK(!result_set_);
    target_values_.addValue(val);
  }

  void addValue() {
    CHECK(!result_set_);
    target_values_.addValue();
  }

  void append(const ResultRows& more_results) {
    if (result_set_) {
      result_set_->append(*more_results.getResultSet());
    }
    simple_keys_.insert(simple_keys_.end(), more_results.simple_keys_.begin(), more_results.simple_keys_.end());
    multi_keys_.insert(multi_keys_.end(), more_results.multi_keys_.begin(), more_results.multi_keys_.end());
    target_values_.append(more_results.target_values_);
    if (in_place_) {
      in_place_group_by_buffers_.insert(in_place_group_by_buffers_.end(),
                                        more_results.in_place_group_by_buffers_.begin(),
                                        more_results.in_place_group_by_buffers_.end());
      in_place_groups_by_buffers_entry_count_.insert(in_place_groups_by_buffers_entry_count_.end(),
                                                     more_results.in_place_groups_by_buffers_entry_count_.begin(),
                                                     more_results.in_place_groups_by_buffers_entry_count_.end());
    }
  }

  void reduce(const ResultRows& other_results, const QueryMemoryDescriptor& query_mem_desc, const bool output_columnar);

  void sort(const std::list<Analyzer::OrderEntry>& order_entries, const bool remove_duplicates, const int64_t top_n);

  void keepFirstN(const size_t n) {
    if (result_set_) {
      result_set_->keepFirstN(n);
      return;
    }
    CHECK(n);
    if (in_place_ || group_by_buffer_) {
      keep_first_ = n;
      return;
    }
    if (n >= rowCount()) {
      return;
    }
    target_values_.truncate(n);
  }

  void dropFirstN(const size_t n) {
    if (result_set_) {
      result_set_->dropFirstN(n);
      return;
    }
    if (in_place_ || group_by_buffer_) {
      drop_first_ = n;
      return;
    }
    if (!n) {
      return;
    }
    target_values_.drop(n);
  }

  size_t rowCount() const {
    if (result_set_) {
      return result_set_->rowCount();
    }
    if (in_place_ || group_by_buffer_) {
      moveToBegin();
      size_t row_count{0};
      while (true) {
        auto crt_row = getNextRow(false, false);
        if (crt_row.empty()) {
          break;
        }
        ++row_count;
      }
      moveToBegin();
      return row_count;
    }
    return just_explain_ ? 1 : target_values_.size();
  }

  size_t colCount() const {
    if (result_set_) {
      return result_set_->colCount();
    }
    return just_explain_ ? 1 : targets_.size();
  }

  bool definitelyHasNoRows() const {
    if (result_set_) {
      return result_set_->definitelyHasNoRows();
    }
    if (in_place_) {
      return in_place_group_by_buffers_.empty();
    }
    return !group_by_buffer_ && !just_explain_ && !rowCount();
  }

  static bool isNull(const SQLTypeInfo& ti, const InternalTargetValue& val);

  TargetValue getRowAt(const size_t row_idx,
                       const size_t col_idx,
                       const bool translate_strings,
                       const bool decimal_to_double = true) const;

  std::vector<TargetValue> getNextRow(const bool translate_strings, const bool decimal_to_double) const;

  SQLTypeInfo getColType(const size_t col_idx) const {
    if (result_set_) {
      return result_set_->getColType(col_idx);
    }
    if (just_explain_) {
      return SQLTypeInfo(kTEXT, false);
    }
    CHECK_LT(col_idx, targets_.size());
    return targets_[col_idx].agg_kind == kAVG ? SQLTypeInfo(kDOUBLE, false) : targets_[col_idx].sql_type;
  }

  const std::vector<TargetInfo>& getTargetInfos() const { return targets_; }

  int64_t getQueueTime() const {
    if (result_set_) {
      return result_set_->getQueueTime();
    }
    return queue_time_ms_;
  }
  int64_t getRenderTime() const {
    CHECK(!result_set_);
    return render_time_ms_;
  }
  ExecutorDeviceType getDeviceType() const { return device_type_; }

  bool isInPlace() const {
    CHECK(!result_set_);
    return in_place_;
  }

  void setQueueTime(int64_t queue_time) {
    if (result_set_) {
      result_set_->setQueueTime(queue_time);
      return;
    }
    queue_time_ms_ = queue_time;
  }

  const QueryMemoryDescriptor& getQueryMemDesc() const {
    if (result_set_) {
      return result_set_->getQueryMemDesc();
    }
    return query_mem_desc_;
  }

  static void inplaceSortGpuImpl(const std::list<Analyzer::OrderEntry>&,
                                 const QueryMemoryDescriptor&,
                                 const GpuQueryMemory&,
                                 Data_Namespace::DataMgr*,
                                 const int);

  std::shared_ptr<ResultSet> getResultSet() const { return result_set_; }

  void fillOneRow(const std::vector<int64_t>& row);

  void holdChunks(const std::list<std::shared_ptr<Chunk_NS::Chunk>>& chunks) {
    if (result_set_) {
      result_set_->holdChunks(chunks);
    }
  }

  void holdLiterals(std::vector<int8_t>& literal_buff) {
    if (result_set_) {
      result_set_->holdLiterals(literal_buff);
    }
  }

  void holdChunkIterators(const std::shared_ptr<std::list<ChunkIter>> chunk_iters) {
    if (result_set_) {
      result_set_->holdChunkIterators(chunk_iters);
    }
  }

  std::shared_ptr<RowSetMemoryOwner> getRowSetMemOwner() const { return row_set_mem_owner_; }

 private:
  void reduceSingleColumn(int8_t* crt_val_i1,
                          int8_t* crt_val_i2,
                          const int8_t* new_val_i1,
                          const int8_t* new_val_i2,
                          const int64_t agg_skip_val,
                          const size_t target_idx,
                          size_t crt_byte_width = sizeof(int64_t),
                          size_t next_byte_width = sizeof(int64_t));

  void reduceDispatch(int64_t* group_by_buffer,
                      const int64_t* other_group_by_buffer,
                      const QueryMemoryDescriptor& query_mem_desc_in,
                      const size_t start,
                      const size_t end);

  void reduceInPlaceDispatch(int64_t** group_by_buffer_ptr,
                             const int64_t* other_group_by_buffer,
                             const int32_t groups_buffer_entry_count,
                             const GroupByColRangeType hash_type,
                             const QueryMemoryDescriptor& query_mem_desc_in,
                             const size_t start,
                             const size_t end);

  void reduceInPlace(int64_t** group_by_buffer_ptr,
                     const int64_t* other_group_by_buffer,
                     const int32_t groups_buffer_entry_count,
                     const int32_t other_groups_buffer_entry_count,
                     const GroupByColRangeType hash_type,
                     const QueryMemoryDescriptor& query_mem_desc_in);

  bool fetchLazyOrBuildRow(std::vector<TargetValue>& row,
                           const std::vector<std::vector<const int8_t*>>& col_buffers,
                           const std::vector<Analyzer::Expr*>& targets,
                           const bool translate_strings,
                           const bool decimal_to_double,
                           const bool fetch_lazy) const;

  void addValues(const std::vector<int64_t>& vals) {
    target_values_.reserveRow(vals.size());
    for (size_t target_idx = 0, agg_col_idx = 0; target_idx < targets_.size() && agg_col_idx < vals.size();
         ++target_idx, ++agg_col_idx) {
      const auto& agg_info = targets_[target_idx];
      if (kAVG == agg_info.agg_kind) {
        target_values_.addValue(vals[agg_col_idx], vals[agg_col_idx + 1]);
        ++agg_col_idx;
      } else {
        target_values_.addValue(vals[agg_col_idx]);
      }
    }
  }

  void discardRow() {
    CHECK_NE(simple_keys_.empty(), multi_keys_.empty());
    if (!simple_keys_.empty()) {
      simple_keys_.pop_back();
    } else {
      multi_keys_.pop_back();
    }
    target_values_.discardRow();
  }

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

  void inplaceSortGpu(const std::list<Analyzer::OrderEntry>& order_entries);

  void inplaceSortCpu(const std::list<Analyzer::OrderEntry>& order_entries);

  std::shared_ptr<ResultSet> result_set_;

  std::vector<TargetInfo> targets_;
  std::vector<int64_t> simple_keys_;
  typedef std::vector<int64_t> MultiKey;
  std::vector<MultiKey> multi_keys_;
  RowStorage target_values_;
  mutable std::map<MultiKey, InternalRow> as_map_;
  mutable std::unordered_map<int64_t, InternalRow> as_unordered_map_;
  const Executor* executor_;
  QueryMemoryDescriptor query_mem_desc_;
  std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner_;

  std::vector<int64_t> agg_init_vals_;
  int64_t* group_by_buffer_;
  size_t groups_buffer_entry_count_;
  mutable size_t group_by_buffer_idx_;
  int64_t min_val_;
  int8_t warp_count_;
  bool output_columnar_;
  bool in_place_;
  ExecutorDeviceType device_type_;
  int device_id_;
  mutable size_t crt_row_idx_;
  mutable size_t crt_row_buff_idx_;
  size_t drop_first_;
  size_t keep_first_;
  mutable bool fetch_started_;
  mutable size_t in_place_buff_idx_;
  std::vector<int32_t> in_place_groups_by_buffers_entry_count_;
  std::vector<int64_t*> in_place_group_by_buffers_;
  bool just_explain_;
  std::string explanation_;
  std::unordered_set<int64_t> unkown_top_keys_;
  int64_t queue_time_ms_;
  int64_t render_time_ms_;
};

// TODO(miyu): make some uses of this pointer able to
// hold iterator table as well and move decls elsewhere
typedef std::unique_ptr<ResultRows> RowSetPtr;

#endif  // QUERYENGINE_RESULTROWS_H
