/**
 * @file    ResultSet.h
 * @author  Alex Suhan <alex@mapd.com>
 * @brief   Basic constructors and methods of the row set interface.
 *
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 */

#ifndef QUERYENGINE_RESULTSET_H
#define QUERYENGINE_RESULTSET_H

#include "CardinalityEstimator.h"
#include "ResultSetBufferAccessors.h"
#include "TargetValue.h"
#include "../Chunk/Chunk.h"

#include <list>

class ResultSetStorage {
 public:
  ResultSetStorage(const std::vector<TargetInfo>& targets,
                   const ExecutorDeviceType device_type,
                   const QueryMemoryDescriptor& query_mem_desc,
                   int8_t* buff,
                   const bool buff_is_provided);

  void reduce(const ResultSetStorage& that) const;

  int8_t* getUnderlyingBuffer() const;

  void moveEntriesToBuffer(int8_t* new_buff, const size_t new_entry_count) const;

 private:
  void reduceOneEntryNoCollisionsColWise(const size_t i, int8_t* this_buff, const int8_t* that_buff) const;

  void copyKeyColWise(const size_t entry_idx, int8_t* this_buff, const int8_t* that_buff) const;

  void reduceOneEntryNoCollisionsRowWise(const size_t i, int8_t* this_buff, const int8_t* that_buff) const;

  bool isEmptyEntry(const size_t entry_idx, const int8_t* buff) const;
  bool isEmptyEntry(const size_t entry_idx) const;

  void reduceOneEntryBaseline(int8_t* this_buff,
                              const int8_t* that_buff,
                              const size_t i,
                              const size_t that_entry_count) const;

  void reduceOneEntrySlotsBaseline(int64_t* this_entry_slots,
                                   const int64_t* that_buff,
                                   const size_t that_entry_idx,
                                   const size_t that_entry_count) const;

  void initializeBaselineValueSlots(int64_t* this_entry_slots) const;

  void reduceOneSlotBaseline(int64_t* this_buff,
                             const size_t this_slot,
                             const int64_t* that_buff,
                             const size_t that_entry_count,
                             const size_t that_slot,
                             const TargetInfo& target_info,
                             const size_t target_logical_idx,
                             const size_t target_slot_idx) const;

  void reduceOneSlot(int8_t* this_ptr1,
                     int8_t* this_ptr2,
                     const int8_t* that_ptr1,
                     const int8_t* that_ptr2,
                     const TargetInfo& target_info,
                     const size_t target_logical_idx,
                     const size_t target_slot_idx) const;

  void reduceOneCountDistinctSlot(int8_t* this_ptr1,
                                  const int8_t* that_ptr1,
                                  const TargetInfo& target_info,
                                  const size_t target_logical_idx) const;

  void fillOneEntryRowWise(const std::vector<int64_t>& entry);

  void initializeRowWise() const;

  void initializeColWise() const;

  const std::vector<TargetInfo> targets_;
  const QueryMemoryDescriptor query_mem_desc_;
  int8_t* buff_;
  const bool buff_is_provided_;
  std::vector<int64_t> target_init_vals_;

  friend class ResultSet;
  friend class ResultSetManager;
};

namespace Analyzer {

class Expr;
class NDVEstimator;
struct OrderEntry;

}  // Analyzer

class Executor;

struct ColumnLazyFetchInfo {
  const bool is_lazily_fetched;
  const int local_col_id;
  const SQLTypeInfo type;
};

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
            const ExecutorDeviceType device_type,
            const int device_id,
            const QueryMemoryDescriptor& query_mem_desc,
            const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
            const Executor* executor);

  ResultSet(const std::shared_ptr<const Analyzer::NDVEstimator>,
            const ExecutorDeviceType device_type,
            const int device_id,
            Data_Namespace::DataMgr* data_mgr);

  // Empty result set constructor
  ResultSet();

  ~ResultSet();

  const ResultSetStorage* allocateStorage() const;

  const ResultSetStorage* allocateStorage(int8_t*, const std::vector<int64_t>&) const;

  const ResultSetStorage* allocateStorage(const std::vector<int64_t>&) const;

  std::vector<TargetValue> getNextRow(const bool translate_strings, const bool decimal_to_double) const;

  void sort(const std::list<Analyzer::OrderEntry>& order_entries, const size_t top_n);

  bool isEmptyInitializer() const;

  void keepFirstN(const size_t n);

  void dropFirstN(const size_t n);

  void append(ResultSet& that);

  const ResultSetStorage* getStorage() const;

  size_t colCount() const;

  SQLTypeInfo getColType(const size_t col_idx) const;

  size_t rowCount() const;

  bool definitelyHasNoRows() const;

  const QueryMemoryDescriptor& getQueryMemDesc() const;

  const std::vector<TargetInfo>& getTargetInfos() const;

  int8_t* getDeviceEstimatorBuffer() const;

  int8_t* getHostEstimatorBuffer() const;

  void syncEstimatorBuffer() const;

  size_t getNDVEstimator() const;

  void setQueueTime(const int64_t queue_time);

  int64_t getQueueTime() const;

  void moveToBegin() const;

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

 private:
  std::vector<TargetValue> getNextRowImpl(const bool translate_strings, const bool decimal_to_double) const;

  size_t advanceCursorToNextEntry() const;

  size_t entryCount() const;

  void radixSortOnGpu(const std::list<Analyzer::OrderEntry>& order_entries) const;

  void radixSortOnCpu(const std::list<Analyzer::OrderEntry>& order_entries) const;

  static bool isNull(const SQLTypeInfo& ti, const InternalTargetValue& val);

  TargetValue getTargetValueFromBufferRowwise(const int8_t* rowwise_target_ptr,
                                              const size_t entry_buff_idx,
                                              const TargetInfo& target_info,
                                              const size_t target_logical_idx,
                                              const size_t slot_idx,
                                              const bool translate_strings,
                                              const bool decimal_to_double) const;

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

  TargetValue makeRealStringTargetValue(const int8_t* ptr1,
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

  int64_t lazyReadInt(const int64_t ival,
                      const size_t target_logical_idx,
                      const StorageLookupResult& storage_lookup_result) const;

  std::pair<ssize_t, size_t> getStorageIndex(const size_t entry_idx) const;

  StorageLookupResult findStorage(const size_t entry_idx) const;

  std::function<bool(const uint32_t, const uint32_t)> createComparator(
      const std::list<Analyzer::OrderEntry>& order_entries,
      const bool use_heap,
      const bool remove_empty_entries) const;

  static void topPermutation(std::vector<uint32_t>& to_sort,
                             const size_t n,
                             const std::function<bool(const uint32_t, const uint32_t)> compare);

  void sortPermutation(const std::function<bool(const uint32_t, const uint32_t)> compare);

  std::vector<uint32_t> initPermutationBuffer(const size_t start, const size_t step);

  void parallelTop(const std::list<Analyzer::OrderEntry>& order_entries, const size_t top_n);

  void baselineSort(const ExecutorDeviceType device_type,
                    const std::list<Analyzer::OrderEntry>& order_entries,
                    const size_t top_n);

  bool canUseFastBaselineSort(const std::list<Analyzer::OrderEntry>& order_entries, const size_t top_n);

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
  std::vector<std::vector<const int8_t*>> col_buffers_;

  const std::shared_ptr<const Analyzer::NDVEstimator> estimator_;
  int8_t* estimator_buffer_;
  mutable int8_t* host_estimator_buffer_;
  Data_Namespace::DataMgr* data_mgr_;

  friend class ResultSetManager;
};

class ResultSetManager {
 public:
  ResultSet* reduce(std::vector<ResultSet*>&);

 private:
  std::unique_ptr<ResultSet> rs_;
};

class RowSortException : public std::runtime_error {
 public:
  RowSortException(const std::string& cause) : std::runtime_error(cause) {}
};

int64_t lazy_decode(const ColumnLazyFetchInfo& col_lazy_fetch, const int8_t* byte_stream, const int64_t pos);

#endif  // QUERYENGINE_RESULTSET_H
