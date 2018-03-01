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

  void reduce(const ResultSetStorage& that) const;

  int8_t* getUnderlyingBuffer() const;

  template <class KeyType>
  void moveEntriesToBuffer(int8_t* new_buff, const size_t new_entry_count) const;

 private:
  void reduceEntriesNoCollisionsColWise(int8_t* this_buff,
                                        const int8_t* that_buff,
                                        const ResultSetStorage& that,
                                        const size_t start_index,
                                        const size_t end_index) const;

  void copyKeyColWise(const size_t entry_idx, int8_t* this_buff, const int8_t* that_buff) const;

  void reduceOneEntryNoCollisionsRowWise(const size_t i,
                                         int8_t* this_buff,
                                         const int8_t* that_buff,
                                         const ResultSetStorage& that) const;

  bool isEmptyEntry(const size_t entry_idx, const int8_t* buff) const;
  bool isEmptyEntry(const size_t entry_idx) const;

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
                     const ResultSetStorage& that) const;

  void reduceOneCountDistinctSlot(int8_t* this_ptr1,
                                  const int8_t* that_ptr1,
                                  const size_t target_logical_idx,
                                  const ResultSetStorage& that) const;

  void fillOneEntryRowWise(const std::vector<int64_t>& entry);

  void initializeRowWise() const;

  void initializeColWise() const;

  // TODO(alex): remove the following two methods, see comment about count_distinct_sets_mapping_.
  void addCountDistinctSetPointerMapping(const int64_t remote_ptr, const int64_t ptr);

  int64_t mappedPtr(const int64_t) const;

  const std::vector<TargetInfo> targets_;
  const QueryMemoryDescriptor query_mem_desc_;
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
class NDVEstimator;
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
#ifdef ENABLE_MULTIFRAG_JOIN
            const std::vector<std::vector<int64_t>>& frag_offsets,
            const std::vector<int64_t>& consistent_frag_sizes,
#endif
            const ExecutorDeviceType device_type,
            const int device_id,
            const QueryMemoryDescriptor& query_mem_desc,
            const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
            const Executor* executor);

  ResultSet(const std::shared_ptr<const Analyzer::NDVEstimator>,
            const ExecutorDeviceType device_type,
            const int device_id,
            Data_Namespace::DataMgr* data_mgr);

  ResultSet(const std::string& explanation);

  ResultSet(const std::string& image_bytes,
            int64_t queue_time_ms,
            int64_t render_time_ms,
            const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner);

  ~ResultSet();

  ExecutorDeviceType getDeviceType() const;

  const ResultSetStorage* allocateStorage() const;

  const ResultSetStorage* allocateStorage(int8_t*, const std::vector<int64_t>&) const;

  const ResultSetStorage* allocateStorage(const std::vector<int64_t>&) const;

  std::vector<TargetValue> getNextRow(const bool translate_strings, const bool decimal_to_double) const;

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

  std::vector<TargetValue> getRowAtNoTranslations(const size_t index) const;

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
    storage_->fillOneEntryRowWise(entry);
  }

  void initializeStorage() const;

  void holdChunks(const std::list<std::shared_ptr<Chunk_NS::Chunk>>& chunks) { chunks_ = chunks; }
  void holdChunkIterators(const std::shared_ptr<std::list<ChunkIter>> chunk_iters) {
    chunk_iters_.push_back(chunk_iters);
  }
  void holdLiterals(std::vector<int8_t>& literal_buff) { literal_buffers_.push_back(std::move(literal_buff)); }

  std::shared_ptr<RowSetMemoryOwner> getRowSetMemOwner() const { return row_set_mem_owner_; }

  const std::vector<uint32_t>& getPermutationBuffer() const;

  std::string serialize() const;

  static std::unique_ptr<ResultSet> unserialize(const std::string&, const Executor*);

  struct SerializedArrowOutput {
    std::shared_ptr<arrow::Buffer> schema;
    std::shared_ptr<arrow::Buffer> records;
  };

  SerializedArrowOutput getSerializedArrowOutput(const std::vector<std::string>& col_names) const;

  ArrowResult getArrowCopy(Data_Namespace::DataMgr* data_mgr,
                           const ExecutorDeviceType device_type,
                           const size_t device_id,
                           const std::vector<std::string>& col_names) const;

 private:
  std::vector<TargetValue> getNextRowImpl(const bool translate_strings, const bool decimal_to_double) const;

  std::vector<TargetValue> getNextRowUnlocked(const bool translate_strings, const bool decimal_to_double) const;

  std::vector<TargetValue> getRowAt(const size_t index,
                                    const bool translate_strings,
                                    const bool decimal_to_double,
                                    const bool fixup_count_distinct_pointers) const;

  size_t parallelRowCount() const;

  size_t advanceCursorToNextEntry() const;

  void radixSortOnGpu(const std::list<Analyzer::OrderEntry>& order_entries) const;

  void radixSortOnCpu(const std::list<Analyzer::OrderEntry>& order_entries) const;

  static bool isNull(const SQLTypeInfo& ti, const InternalTargetValue& val, const bool float_argument_input);

  TargetValue getTargetValueFromBufferRowwise(int8_t* rowwise_target_ptr,
                                              int8_t* keys_ptr,
                                              const size_t entry_buff_idx,
                                              const TargetInfo& target_info,
                                              const size_t target_logical_idx,
                                              const size_t slot_idx,
                                              const bool translate_strings,
                                              const bool decimal_to_double,
                                              const bool fixup_count_distinct_pointers) const;

  TargetValue getTargetValueFromBufferColwise(const int8_t* col1_ptr,
                                              const int8_t compact_sz1,
                                              const int8_t* col2_ptr,
                                              const int8_t compact_sz2,
                                              const size_t entry_idx,
                                              const TargetInfo& target_info,
                                              const size_t target_logical_idx,
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
                                    const size_t entry_buff_idx) const;

  struct StorageLookupResult {
    const ResultSetStorage* storage_ptr;
    const size_t fixedup_entry_idx;
    const size_t storage_idx;
  };

  InternalTargetValue getColumnInternal(const int8_t* buff,
                                        const size_t entry_idx,
                                        const size_t target_logical_idx,
                                        const StorageLookupResult& storage_lookup_result) const;

  InternalTargetValue getVarlenOrderEntry(const int64_t str_ptr, const size_t str_len) const;

  int64_t lazyReadInt(const int64_t ival,
                      const size_t target_logical_idx,
                      const StorageLookupResult& storage_lookup_result) const;

  std::pair<ssize_t, size_t> getStorageIndex(const size_t entry_idx) const;

  const std::vector<const int8_t*>& getColumnFrag(const size_t storge_idx,
                                                  const size_t col_logical_idx,
                                                  int64_t& global_idx) const;

  StorageLookupResult findStorage(const size_t entry_idx) const;

  std::function<bool(const uint32_t, const uint32_t)> createComparator(
      const std::list<Analyzer::OrderEntry>& order_entries,
      const bool use_heap) const;

  static void topPermutation(std::vector<uint32_t>& to_sort,
                             const size_t n,
                             const std::function<bool(const uint32_t, const uint32_t)> compare);

  void sortPermutation(const std::function<bool(const uint32_t, const uint32_t)> compare);

  std::vector<uint32_t> initPermutationBuffer(const size_t start, const size_t step);

  void parallelTop(const std::list<Analyzer::OrderEntry>& order_entries, const size_t top_n);

  void baselineSort(const std::list<Analyzer::OrderEntry>& order_entries, const size_t top_n);

  void doBaselineSort(const ExecutorDeviceType device_type,
                      const std::list<Analyzer::OrderEntry>& order_entries,
                      const size_t top_n);

  bool canUseFastBaselineSort(const std::list<Analyzer::OrderEntry>& order_entries, const size_t top_n);

  Data_Namespace::DataMgr* getDataManager() const;

  int getGpuCount() const;

  std::shared_ptr<arrow::RecordBatch> convertToArrow(const std::vector<std::string>& col_names,
                                                     arrow::ipc::DictionaryMemo& memo) const;
  std::shared_ptr<const std::vector<std::string>> getDictionary(const int dict_id) const;
  std::shared_ptr<arrow::RecordBatch> getArrowBatch(const std::shared_ptr<arrow::Schema>& schema) const;

  ArrowResult getArrowCopyOnCpu(const std::vector<std::string>& col_names) const;
  ArrowResult getArrowCopyOnGpu(Data_Namespace::DataMgr* data_mgr,
                                const size_t device_id,
                                const std::vector<std::string>& col_names) const;

  std::string serializeProjection() const;

  void serializeCountDistinctColumns(TSerializedRows&) const;

  void unserializeCountDistinctColumns(const TSerializedRows&);

  void fixupCountDistinctPointers();

  using BufferSet = std::set<int64_t>;
  void create_active_buffer_set(BufferSet& count_distinct_active_buffer_set) const;

  int64_t getDistinctBufferRefFromBufferRowwise(int8_t* rowwise_target_ptr, const TargetInfo& target_info) const;

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
#ifdef ENABLE_MULTIFRAG_JOIN
  std::vector<std::vector<std::vector<int64_t>>> frag_offsets_;
  std::vector<std::vector<int64_t>> consistent_frag_sizes_;
#endif

  const std::shared_ptr<const Analyzer::NDVEstimator> estimator_;
  int8_t* estimator_buffer_;
  mutable int8_t* host_estimator_buffer_;
  Data_Namespace::DataMgr* data_mgr_;

  // only used by serialization
  std::vector<std::vector<std::string>> none_encoded_strings_;
  bool none_encoded_strings_valid_;
  std::string explanation_;
  const bool just_explain_;
  mutable std::atomic<ssize_t> cached_row_count_;
  mutable std::mutex row_iteration_mutex_;

  friend class ResultSetManager;
};

class ResultSetManager {
 public:
  ResultSet* reduce(std::vector<ResultSet*>&);

  std::shared_ptr<ResultSet> getOwnResultSet();

 private:
  std::shared_ptr<ResultSet> rs_;
};

class RowSortException : public std::runtime_error {
 public:
  RowSortException(const std::string& cause) : std::runtime_error(cause) {}
};

int64_t lazy_decode(const ColumnLazyFetchInfo& col_lazy_fetch, const int8_t* byte_stream, const int64_t pos);

void fill_empty_key(void* key_ptr, const size_t key_count, const size_t key_width);

#endif  // QUERYENGINE_RESULTSET_H
