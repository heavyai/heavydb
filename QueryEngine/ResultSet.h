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

/**
 * @file    ResultSet.h
 * @author  Alex Suhan <alex@mapd.com>
 * @brief   Basic constructors and methods of the row set interface.
 *
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 */

#ifndef QUERYENGINE_RESULTSET_H
#define QUERYENGINE_RESULTSET_H

#include "../Chunk/Chunk.h"
#include "CardinalityEstimator.h"
#include "ResultSetBufferAccessors.h"
#include "TargetValue.h"

#include "arrow/api.h"
#include "arrow/ipc/api.h"

#include <atomic>
#include <functional>
#include <list>

/*
 * Stores the underlying buffer and the meta-data for a result set. The buffer
 * format reflects the main requirements for result sets. Not all queries
 * specify a GROUP BY clause, but since it's the most important and challenging
 * case we'll focus on it. Note that the meta-data is stored separately from
 * the buffer and it's not transferred to GPU.
 *
 * 1. It has to be efficient for reduction of partial GROUP BY query results
 *    from multiple devices / cores, the cardinalities can be high. Reduction
 *    currently happens on the host.
 * 2. No conversions should be needed when buffers are transferred from GPU to
 *    host for reduction. This implies the buffer needs to be "flat", with no
 *    pointers to chase since they have no meaning in a different address space.
 * 3. Must be size-efficient.
 *
 * There are several variations of the format of a result set buffer, but the
 * most common is a sequence of entries which represent a row in the result or
 * an empty slot. One entry looks as follows:
 *
 * +-+-+-+-+-+-+-+-+-+-+-+--?--+-+-+-+-+-+-+-+-+-+-+-+-+
 * |key_0| ... |key_N-1| padding |value_0|...|value_N-1|
 * +-+-+-+-+-+-+-+-+-+-+-+--?--+-+-+-+-+-+-+-+-+-+-+-+-+
 *
 * (key_0 ... key_N-1) is a multiple component key, unique within the buffer.
 * It stores the tuple specified by the GROUP BY clause. All components have
 * the same width, 4 or 8 bytes. For the 4-byte components, 4-byte padding is
 * added if the number of components is odd. Not all entries in the buffer are
 * valid; an empty entry contains EMPTY_KEY_{64, 32} for 8-byte / 4-byte width,
 * respectively. An empty entry is ignored by subsequent operations on the
 * result set (reduction, iteration, sort etc).
 *
 * value_0 through value_N-1 are 8-byte fields which hold the columns of the
 * result, like aggregates and projected expressions. They're reduced between
 * multiple partial results for identical (key_0 ... key_N-1) tuples.
 *
 * The order of entries is decided by the type of hash used, which depends on
 * the range of the keys. For small enough ranges, a perfect hash is used. When
 * a perfect hash isn't feasible, open addressing (using MurmurHash) with linear
 * probing is used instead, with a 50% fill rate.
 */

class ResultSetStorage {
 public:
  ResultSetStorage(const std::vector<TargetInfo>& targets,
                   const QueryMemoryDescriptor& query_mem_desc,
                   int8_t* buff,
                   const bool buff_is_provided);

  void reduce(const ResultSetStorage& that,
              const std::vector<std::string>& serialized_varlen_buffer) const;

  void rewriteAggregateBufferOffsets(
      const std::vector<std::string>& serialized_varlen_buffer) const;

  int8_t* getUnderlyingBuffer() const;

  template <class KeyType>
  void moveEntriesToBuffer(int8_t* new_buff, const size_t new_entry_count) const;

  void updateEntryCount(const size_t new_entry_count) {
    query_mem_desc_.setEntryCount(new_entry_count);
  }

 private:
  void reduceEntriesNoCollisionsColWise(
      int8_t* this_buff,
      const int8_t* that_buff,
      const ResultSetStorage& that,
      const size_t start_index,
      const size_t end_index,
      const std::vector<std::string>& serialized_varlen_buffer) const;

  void copyKeyColWise(const size_t entry_idx,
                      int8_t* this_buff,
                      const int8_t* that_buff) const;

  void reduceOneEntryNoCollisionsRowWise(
      const size_t i,
      int8_t* this_buff,
      const int8_t* that_buff,
      const ResultSetStorage& that,
      const std::vector<std::string>& serialized_varlen_buffer) const;

  bool isEmptyEntry(const size_t entry_idx, const int8_t* buff) const;
  bool isEmptyEntry(const size_t entry_idx) const;
  bool isEmptyEntryColumnar(const size_t entry_idx, const int8_t* buff) const;

  void reduceOneEntryBaseline(int8_t* this_buff,
                              const int8_t* that_buff,
                              const size_t i,
                              const size_t that_entry_count,
                              const ResultSetStorage& that) const;

  void reduceOneEntrySlotsBaseline(int64_t* this_entry_slots,
                                   const int64_t* that_buff,
                                   const size_t that_entry_idx,
                                   const size_t that_entry_count,
                                   const ResultSetStorage& that) const;

  void initializeBaselineValueSlots(int64_t* this_entry_slots) const;

  void reduceOneSlotBaseline(int64_t* this_buff,
                             const size_t this_slot,
                             const int64_t* that_buff,
                             const size_t that_entry_count,
                             const size_t that_slot,
                             const TargetInfo& target_info,
                             const size_t target_logical_idx,
                             const size_t target_slot_idx,
                             const size_t init_agg_val_idx,
                             const ResultSetStorage& that) const;

  ALWAYS_INLINE
  void reduceOneSlot(int8_t* this_ptr1,
                     int8_t* this_ptr2,
                     const int8_t* that_ptr1,
                     const int8_t* that_ptr2,
                     const TargetInfo& target_info,
                     const size_t target_logical_idx,
                     const size_t target_slot_idx,
                     const size_t init_agg_val_idx,
                     const ResultSetStorage& that,
                     const std::vector<std::string>& serialized_varlen_buffer) const;

  void reduceOneCountDistinctSlot(int8_t* this_ptr1,
                                  const int8_t* that_ptr1,
                                  const size_t target_logical_idx,
                                  const ResultSetStorage& that) const;

  void fillOneEntryRowWise(const std::vector<int64_t>& entry);

  void fillOneEntryColWise(const std::vector<int64_t>& entry);

  void initializeRowWise() const;

  void initializeColWise() const;

  // TODO(alex): remove the following two methods, see comment about
  // count_distinct_sets_mapping_.
  void addCountDistinctSetPointerMapping(const int64_t remote_ptr, const int64_t ptr);

  int64_t mappedPtr(const int64_t) const;

  const std::vector<TargetInfo> targets_;
  QueryMemoryDescriptor query_mem_desc_;
  int8_t* buff_;
  const bool buff_is_provided_;
  std::vector<int64_t> target_init_vals_;
  // Provisional field used for multi-node until we improve the count distinct
  // and flatten the main group by buffer and the distinct buffers in a single,
  // contiguous buffer which we'll be able to serialize as a no-op. Used to
  // re-route the pointers in the result set received over the wire to this
  // machine address-space. Not efficient at all, just a placeholder!
  std::unordered_map<int64_t, int64_t> count_distinct_sets_mapping_;

  friend class ResultSet;
  friend class ResultSetManager;
};

namespace Analyzer {

class Expr;
class Estimator;
struct OrderEntry;

}  // namespace Analyzer

class Executor;

struct ColumnLazyFetchInfo {
  const bool is_lazily_fetched;
  const int local_col_id;
  const SQLTypeInfo type;
};

struct OneIntegerColumnRow {
  const int64_t value;
  const bool valid;
};

struct ArrowResult {
  std::vector<char> sm_handle;
  int64_t sm_size;
  std::vector<char> df_handle;
  int64_t df_size;
  int8_t* df_dev_ptr;  // Only for device memory deallocation
};

void deallocate_arrow_result(const ArrowResult& result,
                             const ExecutorDeviceType device_type,
                             const size_t device_id,
                             Data_Namespace::DataMgr* data_mgr);

class ResultSet;

class ResultSetRowIterator {
 public:
  using value_type = std::vector<TargetValue>;
  using difference_type = std::ptrdiff_t;
  using pointer = std::vector<TargetValue>*;
  using reference = std::vector<TargetValue>&;
  using iterator_category = std::input_iterator_tag;

  bool operator==(const ResultSetRowIterator& other) const {
    return result_set_ == other.result_set_ &&
           crt_row_buff_idx_ == other.crt_row_buff_idx_;
  }
  bool operator!=(const ResultSetRowIterator& other) const { return !(*this == other); }

  inline value_type operator*() const;
  inline ResultSetRowIterator& operator++(void);
  ResultSetRowIterator operator++(int) {
    ResultSetRowIterator iter(*this);
    ++(*this);
    return iter;
  }

  size_t getCurrentRowBufferIndex() const {
    if (crt_row_buff_idx_ == 0) {
      throw std::runtime_error("current row buffer iteration index is undefined");
    }
    return crt_row_buff_idx_ - 1;
  }

 private:
  const ResultSet* result_set_;
  size_t crt_row_buff_idx_;
  size_t global_entry_idx_;
  bool global_entry_idx_valid_;
  size_t fetched_so_far_;
  bool translate_strings_;
  bool decimal_to_double_;

  ResultSetRowIterator(const ResultSet* rs,
                       bool translate_strings,
                       bool decimal_to_double)
      : result_set_(rs)
      , crt_row_buff_idx_(0)
      , global_entry_idx_(0)
      , global_entry_idx_valid_(false)
      , fetched_so_far_(0)
      , translate_strings_(translate_strings)
      , decimal_to_double_(decimal_to_double){};

  ResultSetRowIterator(const ResultSet* rs) : ResultSetRowIterator(rs, false, false){};

  friend class ResultSet;
};

class TSerializedRows;

class ResultSet {
 public:
  ResultSet(const std::vector<TargetInfo>& targets,
            const ExecutorDeviceType device_type,
            const QueryMemoryDescriptor& query_mem_desc,
            const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
            const Executor* executor);

  ResultSet(const std::vector<TargetInfo>& targets,
            const std::vector<ColumnLazyFetchInfo>& lazy_fetch_info,
            const std::vector<std::vector<const int8_t*>>& col_buffers,
            const std::vector<std::vector<int64_t>>& frag_offsets,
            const std::vector<int64_t>& consistent_frag_sizes,
            const ExecutorDeviceType device_type,
            const int device_id,
            const QueryMemoryDescriptor& query_mem_desc,
            const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
            const Executor* executor);

  ResultSet(const std::shared_ptr<const Analyzer::Estimator>,
            const ExecutorDeviceType device_type,
            const int device_id,
            Data_Namespace::DataMgr* data_mgr);

  ResultSet(const std::string& explanation);

  ResultSet(int64_t queue_time_ms,
            int64_t render_time_ms,
            const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner);

  ~ResultSet();

  inline ResultSetRowIterator rowIterator(size_t from_logical_index,
                                          bool translate_strings,
                                          bool decimal_to_double) const {
    ResultSetRowIterator rowIterator(this, translate_strings, decimal_to_double);

    // move to first logical position
    ++rowIterator;

    for (size_t index = 0; index < from_logical_index; index++) {
      ++rowIterator;
    }

    return rowIterator;
  }

  inline ResultSetRowIterator rowIterator(bool translate_strings,
                                          bool decimal_to_double) const {
    return rowIterator(0, translate_strings, decimal_to_double);
  }

  ExecutorDeviceType getDeviceType() const;

  const ResultSetStorage* allocateStorage() const;

  const ResultSetStorage* allocateStorage(int8_t*, const std::vector<int64_t>&) const;

  const ResultSetStorage* allocateStorage(const std::vector<int64_t>&) const;

  void updateStorageEntryCount(const size_t new_entry_count) {
    // currently, should only be used for columnar projections
    CHECK(query_mem_desc_.didOutputColumnar());
    CHECK(query_mem_desc_.getQueryDescriptionType() == QueryDescriptionType::Projection);
    query_mem_desc_.setEntryCount(new_entry_count);
    CHECK(storage_);
    storage_->updateEntryCount(new_entry_count);
  }

  std::vector<TargetValue> getNextRow(const bool translate_strings,
                                      const bool decimal_to_double) const;

  size_t getCurrentRowBufferIndex() const;

  std::vector<TargetValue> getRowAt(const size_t index) const;

  TargetValue getRowAt(const size_t row_idx,
                       const size_t col_idx,
                       const bool translate_strings,
                       const bool decimal_to_double = true) const;

  // Specialized random access getter for result sets with a single column to
  // avoid the overhead of building a std::vector<TargetValue> result with only
  // one element. Only used by RelAlgTranslator::getInIntegerSetExpr currently.
  OneIntegerColumnRow getOneColRow(const size_t index) const;

  std::vector<TargetValue> getRowAtNoTranslations(
      const size_t index,
      const bool skip_non_lazy_columns = false) const;

  bool isRowAtEmpty(const size_t index) const;

  void sort(const std::list<Analyzer::OrderEntry>& order_entries, const size_t top_n);

  void keepFirstN(const size_t n);

  void dropFirstN(const size_t n);

  void append(ResultSet& that);

  const ResultSetStorage* getStorage() const;

  size_t colCount() const;

  SQLTypeInfo getColType(const size_t col_idx) const;

  size_t rowCount(const bool force_parallel = false) const;

  void setCachedRowCount(const size_t row_count) const;

  size_t entryCount() const;

  size_t getBufferSizeBytes(const ExecutorDeviceType device_type) const;

  bool definitelyHasNoRows() const;

  const QueryMemoryDescriptor& getQueryMemDesc() const;

  const std::vector<TargetInfo>& getTargetInfos() const;

  int8_t* getDeviceEstimatorBuffer() const;

  int8_t* getHostEstimatorBuffer() const;

  void syncEstimatorBuffer() const;

  size_t getNDVEstimator() const;

  void setQueueTime(const int64_t queue_time);

  int64_t getQueueTime() const;

  int64_t getRenderTime() const;

  void moveToBegin() const;

  bool isTruncated() const;

  // Called from the executor because in the new ResultSet we assume the 'compact' field
  // in ColWidths already contains the padding, whereas in the executor it's computed.
  // Once the buffer initialization moves to ResultSet we can remove this method.
  static QueryMemoryDescriptor fixupQueryMemoryDescriptor(const QueryMemoryDescriptor&);

  void fillOneEntry(const std::vector<int64_t>& entry) {
    CHECK(storage_);
    if (storage_->query_mem_desc_.didOutputColumnar()) {
      storage_->fillOneEntryColWise(entry);
    } else {
      storage_->fillOneEntryRowWise(entry);
    }
  }

  void initializeStorage() const;

  void holdChunks(const std::list<std::shared_ptr<Chunk_NS::Chunk>>& chunks) {
    chunks_ = chunks;
  }
  void holdChunkIterators(const std::shared_ptr<std::list<ChunkIter>> chunk_iters) {
    chunk_iters_.push_back(chunk_iters);
  }
  void holdLiterals(std::vector<int8_t>& literal_buff) {
    literal_buffers_.push_back(std::move(literal_buff));
  }

  std::shared_ptr<RowSetMemoryOwner> getRowSetMemOwner() const {
    return row_set_mem_owner_;
  }

  const std::vector<uint32_t>& getPermutationBuffer() const;
  const bool isPermutationBufferEmpty() const { return permutation_.empty(); };

  std::string serialize() const;

  static std::unique_ptr<ResultSet> unserialize(const std::string&, const Executor*);

  struct SerializedArrowOutput {
    std::shared_ptr<arrow::Buffer> schema;
    std::shared_ptr<arrow::Buffer> records;
  };

  SerializedArrowOutput getSerializedArrowOutput(
      const std::vector<std::string>& col_names,
      const int32_t first_n) const;

  ArrowResult getArrowCopy(Data_Namespace::DataMgr* data_mgr,
                           const ExecutorDeviceType device_type,
                           const size_t device_id,
                           const std::vector<std::string>& col_names,
                           const int32_t first_n) const;

  size_t getLimit();

  enum class GeoReturnType { GeoTargetValue, WktString };
  GeoReturnType getGeoReturnType() const { return geo_return_type_; }
  void setGeoReturnType(const GeoReturnType val) { geo_return_type_ = val; }

  void copyColumnIntoBuffer(const size_t column_idx,
                            int8_t* output_buffer,
                            const size_t output_buffer_size) const;

  /*
   * Determines if it is possible to directly form a ColumnarResults class from this
   * result set, bypassing the default row-wise columnarization. It is currently only
   * possible for columnar projections.
   *
   * NOTE: If there exists a permutation vector (i.e., ORDER BY), it becomes equivalent to
   * the row-wise columnarization.
   */
  bool isFastColumnarConversionPossible() const {
    return query_mem_desc_.didOutputColumnar() && permutation_.empty() &&
           query_mem_desc_.getQueryDescriptionType() == QueryDescriptionType::Projection;
  }

  const std::vector<ColumnLazyFetchInfo>& getLazyFetchInfo() const {
    return lazy_fetch_info_;
  }

  void setSeparateVarlenStorageValid(const bool val) {
    separate_varlen_storage_valid_ = val;
  }

 private:
  void advanceCursorToNextEntry(ResultSetRowIterator& iter) const;

  std::vector<TargetValue> getNextRowImpl(const bool translate_strings,
                                          const bool decimal_to_double) const;

  std::vector<TargetValue> getNextRowUnlocked(const bool translate_strings,
                                              const bool decimal_to_double) const;

  std::vector<TargetValue> getRowAt(const size_t index,
                                    const bool translate_strings,
                                    const bool decimal_to_double,
                                    const bool fixup_count_distinct_pointers,
                                    const bool skip_non_lazy_columns = false) const;

  size_t parallelRowCount() const;

  size_t advanceCursorToNextEntry() const;

  void radixSortOnGpu(const std::list<Analyzer::OrderEntry>& order_entries) const;

  void radixSortOnCpu(const std::list<Analyzer::OrderEntry>& order_entries) const;

  static bool isNull(const SQLTypeInfo& ti,
                     const InternalTargetValue& val,
                     const bool float_argument_input);

  TargetValue getTargetValueFromBufferRowwise(
      int8_t* rowwise_target_ptr,
      int8_t* keys_ptr,
      const size_t entry_buff_idx,
      const TargetInfo& target_info,
      const size_t target_logical_idx,
      const size_t slot_idx,
      const bool translate_strings,
      const bool decimal_to_double,
      const bool fixup_count_distinct_pointers) const;

  TargetValue getTargetValueFromBufferColwise(const int8_t* col_ptr,
                                              const int8_t* keys_ptr,
                                              const QueryMemoryDescriptor& query_mem_desc,
                                              const size_t local_entry_idx,
                                              const size_t global_entry_idx,
                                              const TargetInfo& target_info,
                                              const size_t target_logical_idx,
                                              const size_t slot_idx,
                                              const bool translate_strings,
                                              const bool decimal_to_double) const;

  TargetValue makeTargetValue(const int8_t* ptr,
                              const int8_t compact_sz,
                              const TargetInfo& target_info,
                              const size_t target_logical_idx,
                              const bool translate_strings,
                              const bool decimal_to_double,
                              const size_t entry_buff_idx) const;

  TargetValue makeVarlenTargetValue(const int8_t* ptr1,
                                    const int8_t compact_sz1,
                                    const int8_t* ptr2,
                                    const int8_t compact_sz2,
                                    const TargetInfo& target_info,
                                    const size_t target_logical_idx,
                                    const bool translate_strings,
                                    const size_t entry_buff_idx) const;

  struct VarlenTargetPtrPair {
    int8_t* ptr1;
    int8_t compact_sz1;
    int8_t* ptr2;
    int8_t compact_sz2;

    VarlenTargetPtrPair()
        : ptr1(nullptr), compact_sz1(0), ptr2(nullptr), compact_sz2(0) {}
  };
  TargetValue makeGeoTargetValue(const int8_t* geo_target_ptr,
                                 const size_t slot_idx,
                                 const TargetInfo& target_info,
                                 const size_t target_logical_idx,
                                 const size_t entry_buff_idx) const;

  struct StorageLookupResult {
    const ResultSetStorage* storage_ptr;
    const size_t fixedup_entry_idx;
    const size_t storage_idx;
  };

  InternalTargetValue getColumnInternal(
      const int8_t* buff,
      const size_t entry_idx,
      const size_t target_logical_idx,
      const StorageLookupResult& storage_lookup_result) const;

  InternalTargetValue getVarlenOrderEntry(const int64_t str_ptr,
                                          const size_t str_len) const;

  int64_t lazyReadInt(const int64_t ival,
                      const size_t target_logical_idx,
                      const StorageLookupResult& storage_lookup_result) const;

  std::pair<ssize_t, size_t> getStorageIndex(const size_t entry_idx) const;

  const std::vector<const int8_t*>& getColumnFrag(const size_t storge_idx,
                                                  const size_t col_logical_idx,
                                                  int64_t& global_idx) const;

  StorageLookupResult findStorage(const size_t entry_idx) const;

  struct TargetOffsets {
    const int8_t* ptr1;
    const size_t compact_sz1;
    const int8_t* ptr2;
    const size_t compact_sz2;
  };

  struct RowWiseTargetAccessor {
    RowWiseTargetAccessor(const ResultSet* result_set)
        : result_set_(result_set)
        , row_bytes_(get_row_bytes(result_set->query_mem_desc_))
        , key_width_(result_set_->query_mem_desc_.getEffectiveKeyWidth())
        , key_bytes_with_padding_(
              align_to_int64(get_key_bytes_rowwise(result_set->query_mem_desc_))) {
      initializeOffsetsForStorage();
    }

    InternalTargetValue getColumnInternal(
        const int8_t* buff,
        const size_t entry_idx,
        const size_t target_logical_idx,
        const StorageLookupResult& storage_lookup_result) const;

    void initializeOffsetsForStorage();

    inline const int8_t* get_rowwise_ptr(const int8_t* buff,
                                         const size_t entry_idx) const {
      return buff + entry_idx * row_bytes_;
    }

    std::vector<std::vector<TargetOffsets>> offsets_for_storage_;

    const ResultSet* result_set_;

    // Row-wise iteration
    const size_t row_bytes_;
    const size_t key_width_;
    const size_t key_bytes_with_padding_;
  };

  struct ColumnWiseTargetAccessor {
    ColumnWiseTargetAccessor(const ResultSet* result_set) : result_set_(result_set) {
      initializeOffsetsForStorage();
    }

    void initializeOffsetsForStorage();

    InternalTargetValue getColumnInternal(
        const int8_t* buff,
        const size_t entry_idx,
        const size_t target_logical_idx,
        const StorageLookupResult& storage_lookup_result) const;

    std::vector<std::vector<TargetOffsets>> offsets_for_storage_;

    const ResultSet* result_set_;
  };

  template <typename BUFFER_ITERATOR_TYPE>
  struct ResultSetComparator {
    using BufferIteratorType = BUFFER_ITERATOR_TYPE;

    ResultSetComparator(const std::list<Analyzer::OrderEntry>& order_entries,
                        const bool use_heap,
                        const ResultSet* result_set)
        : order_entries_(order_entries)
        , use_heap_(use_heap)
        , result_set_(result_set)
        , buffer_itr_(result_set) {}

    bool operator()(const uint32_t lhs, const uint32_t rhs) const;

    // TODO(adb): make order_entries_ a pointer
    const std::list<Analyzer::OrderEntry> order_entries_;
    const bool use_heap_;
    const ResultSet* result_set_;
    const BufferIteratorType buffer_itr_;
  };

  std::function<bool(const uint32_t, const uint32_t)> createComparator(
      const std::list<Analyzer::OrderEntry>& order_entries,
      const bool use_heap) {
    if (query_mem_desc_.didOutputColumnar()) {
      column_wise_comparator_ =
          std::make_unique<ResultSetComparator<ColumnWiseTargetAccessor>>(
              order_entries, use_heap, this);
      return [this](const uint32_t lhs, const uint32_t rhs) -> bool {
        return (*this->column_wise_comparator_)(lhs, rhs);
      };
    } else {
      row_wise_comparator_ = std::make_unique<ResultSetComparator<RowWiseTargetAccessor>>(
          order_entries, use_heap, this);
      return [this](const uint32_t lhs, const uint32_t rhs) -> bool {
        return (*this->row_wise_comparator_)(lhs, rhs);
      };
    }
  }

  static void topPermutation(
      std::vector<uint32_t>& to_sort,
      const size_t n,
      const std::function<bool(const uint32_t, const uint32_t)> compare);

  void sortPermutation(const std::function<bool(const uint32_t, const uint32_t)> compare);

  std::vector<uint32_t> initPermutationBuffer(const size_t start, const size_t step);

  void parallelTop(const std::list<Analyzer::OrderEntry>& order_entries,
                   const size_t top_n);

  void baselineSort(const std::list<Analyzer::OrderEntry>& order_entries,
                    const size_t top_n);

  void doBaselineSort(const ExecutorDeviceType device_type,
                      const std::list<Analyzer::OrderEntry>& order_entries,
                      const size_t top_n);

  bool canUseFastBaselineSort(const std::list<Analyzer::OrderEntry>& order_entries,
                              const size_t top_n);

  Data_Namespace::DataMgr* getDataManager() const;

  int getGpuCount() const;

  std::shared_ptr<arrow::RecordBatch> convertToArrow(
      const std::vector<std::string>& col_names,
      arrow::ipc::DictionaryMemo& memo,
      const int32_t first_n) const;
  std::shared_ptr<const std::vector<std::string>> getDictionary(const int dict_id) const;
  std::shared_ptr<arrow::RecordBatch> getArrowBatch(
      const std::shared_ptr<arrow::Schema>& schema,
      const int32_t first_n) const;

  ArrowResult getArrowCopyOnCpu(const std::vector<std::string>& col_names,
                                const int32_t first_n) const;
  ArrowResult getArrowCopyOnGpu(Data_Namespace::DataMgr* data_mgr,
                                const size_t device_id,
                                const std::vector<std::string>& col_names,
                                const int32_t first_n) const;

  std::string serializeProjection() const;
  void serializeVarlenAggColumn(int8_t* buf,
                                std::vector<std::string>& varlen_bufer) const;

  void serializeCountDistinctColumns(TSerializedRows&) const;

  void unserializeCountDistinctColumns(const TSerializedRows&);

  void fixupCountDistinctPointers();

  using BufferSet = std::set<int64_t>;
  void create_active_buffer_set(BufferSet& count_distinct_active_buffer_set) const;

  int64_t getDistinctBufferRefFromBufferRowwise(int8_t* rowwise_target_ptr,
                                                const TargetInfo& target_info) const;

  const std::vector<TargetInfo> targets_;
  const ExecutorDeviceType device_type_;
  const int device_id_;
  QueryMemoryDescriptor query_mem_desc_;
  mutable std::unique_ptr<ResultSetStorage> storage_;
  std::vector<std::unique_ptr<ResultSetStorage>> appended_storage_;
  mutable size_t crt_row_buff_idx_;
  mutable size_t fetched_so_far_;
  size_t drop_first_;
  size_t keep_first_;
  const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner_;
  std::vector<uint32_t> permutation_;
  int64_t queue_time_ms_;
  int64_t render_time_ms_;
  const Executor* executor_;  // TODO(alex): remove

  std::list<std::shared_ptr<Chunk_NS::Chunk>> chunks_;
  std::vector<std::shared_ptr<std::list<ChunkIter>>> chunk_iters_;
  // TODO(miyu): refine by using one buffer and
  //   setting offset instead of ptr in group by buffer.
  std::vector<std::vector<int8_t>> literal_buffers_;
  const std::vector<ColumnLazyFetchInfo> lazy_fetch_info_;
  std::vector<std::vector<std::vector<const int8_t*>>> col_buffers_;
  std::vector<std::vector<std::vector<int64_t>>> frag_offsets_;
  std::vector<std::vector<int64_t>> consistent_frag_sizes_;

  const std::shared_ptr<const Analyzer::Estimator> estimator_;
  int8_t* estimator_buffer_;
  mutable int8_t* host_estimator_buffer_;
  Data_Namespace::DataMgr* data_mgr_;

  // only used by serialization
  using SerializedVarlenBufferStorage = std::vector<std::string>;

  std::vector<SerializedVarlenBufferStorage> serialized_varlen_buffer_;
  bool separate_varlen_storage_valid_;
  std::string explanation_;
  const bool just_explain_;
  mutable std::atomic<ssize_t> cached_row_count_;
  mutable std::mutex row_iteration_mutex_;

  // only used by geo
  mutable GeoReturnType geo_return_type_;

  // comparators used for sorting (note that the actual compare function is accessed using
  // the createComparator method)
  std::unique_ptr<ResultSetComparator<RowWiseTargetAccessor>> row_wise_comparator_;
  std::unique_ptr<ResultSetComparator<ColumnWiseTargetAccessor>> column_wise_comparator_;

  friend class ResultSetManager;
  friend class ResultSetRowIterator;
};

ResultSetRowIterator::value_type ResultSetRowIterator::operator*() const {
  if (!global_entry_idx_valid_) {
    return {};
  }

  if (result_set_->just_explain_) {
    return {result_set_->explanation_};
  }

  return result_set_->getRowAt(
      global_entry_idx_, translate_strings_, decimal_to_double_, false);
}

inline ResultSetRowIterator& ResultSetRowIterator::operator++(void) {
  if (!result_set_->storage_ && !result_set_->just_explain_) {
    global_entry_idx_valid_ = false;
  } else if (result_set_->just_explain_) {
    global_entry_idx_valid_ = 0 == fetched_so_far_;
    fetched_so_far_ = 1;
  } else {
    result_set_->advanceCursorToNextEntry(*this);
  }
  return *this;
}

class ResultSetManager {
 public:
  ResultSet* reduce(std::vector<ResultSet*>&);

  std::shared_ptr<ResultSet> getOwnResultSet();

  void rewriteVarlenAggregates(ResultSet*);

 private:
  std::shared_ptr<ResultSet> rs_;
};

class RowSortException : public std::runtime_error {
 public:
  RowSortException(const std::string& cause) : std::runtime_error(cause) {}
};

int64_t lazy_decode(const ColumnLazyFetchInfo& col_lazy_fetch,
                    const int8_t* byte_stream,
                    const int64_t pos);

void fill_empty_key(void* key_ptr, const size_t key_count, const size_t key_width);

bool can_use_parallel_algorithms(const ResultSet& rows);

bool use_parallel_algorithms(const ResultSet& rows);

#endif  // QUERYENGINE_RESULTSET_H
