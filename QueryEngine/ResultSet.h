/**
 * @file    ResultSet.h
 * @author  Alex Suhan <alex@mapd.com>
 * @brief   Basic constructors and methods of the row set interface.
 *
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 */

#ifndef QUERYENGINE_RESULTSET_H
#define QUERYENGINE_RESULTSET_H

#include "ResultSetBufferAccessors.h"
#include "TargetValue.h"

#include <list>

class ResultSetStorage {
 public:
  ResultSetStorage(const std::vector<TargetInfo>& targets,
                   const ExecutorDeviceType device_type,
                   const QueryMemoryDescriptor& query_mem_desc,
                   int8_t* buff);

  void reduce(const ResultSetStorage& that) const;

  int8_t* getUnderlyingBuffer() const;

 private:
  void reduceOneEntryNoCollisionsColWise(const size_t i, int8_t* this_buff, const int8_t* that_buff) const;

  void copyKeyColWise(const size_t entry_idx, int8_t* this_buff, const int8_t* that_buff) const;

  void reduceOneEntryNoCollisionsRowWise(const size_t i, int8_t* this_buff, const int8_t* that_buff) const;

  bool isEmptyEntry(const size_t i, const int8_t* buff) const;

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
                             const size_t target_slot_idx) const;

  void reduceOneSlot(int8_t* this_ptr1,
                     int8_t* this_ptr2,
                     const int8_t* that_ptr1,
                     const int8_t* that_ptr2,
                     const TargetInfo& target_info,
                     const size_t target_slot_idx) const;

  void moveEntriesToBuffer(int8_t* new_buff, const size_t new_entry_count) const;

  void initializeRowWise() const;

  void initializeColWise() const;

  const std::vector<TargetInfo> targets_;
  const QueryMemoryDescriptor query_mem_desc_;
  int8_t* buff_;
  std::vector<int64_t> target_init_vals_;

  friend class ResultSet;
  friend class ResultSetManager;
};

namespace Analyzer {

struct OrderEntry;

}  // Analyzer

class Executor;

class ResultSet {
 public:
  ResultSet(const std::vector<TargetInfo>& targets,
            const ExecutorDeviceType device_type,
            const QueryMemoryDescriptor& query_mem_desc,
            const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
            const Executor* executor);

  // Empty result set constructor
  ResultSet();

  ~ResultSet();

  const ResultSetStorage* allocateStorage() const;

  const ResultSetStorage* allocateStorage(int8_t*) const;

  std::vector<TargetValue> getNextRow(const bool translate_strings, const bool decimal_to_double) const;

  void sort(const std::list<Analyzer::OrderEntry>& order_entries, const size_t top_n);

  bool isEmptyInitializer() const;

  void keepFirstN(const size_t n);

  void dropFirstN(const size_t n);

  void append(ResultSet& that);

 private:
  size_t advanceCursorToNextEntry() const;

  size_t entryCount() const;

  void sortOnGpu(const std::list<Analyzer::OrderEntry>& order_entries) const;

  void sortOnCpu(const std::list<Analyzer::OrderEntry>& order_entries) const;

  void fetchLazy(const std::vector<ssize_t> lazy_col_local_ids,
                 const std::vector<std::vector<const int8_t*>>& col_buffers) const;

  void initializeStorage() const;

  static bool isNull(const SQLTypeInfo& ti, const InternalTargetValue& val);

  TargetValue getTargetValueFromBufferRowwise(const int8_t* rowwise_target_ptr,
                                              const TargetInfo& target_info,
                                              const size_t slot_idx,
                                              const bool translate_strings) const;

  TargetValue getTargetValueFromBufferColwise(const int8_t* col1_ptr,
                                              const int8_t compact_sz1,
                                              const int8_t* col2_ptr,
                                              const int8_t compact_sz2,
                                              const size_t entry_idx,
                                              const TargetInfo& target_info,
                                              const bool translate_strings) const;

  TargetValue makeTargetValue(const int8_t* ptr,
                              const int8_t compact_sz,
                              const SQLTypeInfo& ti,
                              const bool translate_strings) const;

  InternalTargetValue getColumnInternal(const size_t entry_idx, const size_t col_idx) const;

  std::function<bool(const uint32_t, const uint32_t)> createComparator(
      const std::list<Analyzer::OrderEntry>& order_entries,
      const bool use_heap) const;

  void topPermutation(const size_t n, const std::function<bool(const uint32_t, const uint32_t)> compare);

  void sortPermutation(const std::function<bool(const uint32_t, const uint32_t)> compare);

  const std::vector<TargetInfo> targets_;
  const ExecutorDeviceType device_type_;
  const QueryMemoryDescriptor query_mem_desc_;
  mutable std::unique_ptr<const ResultSetStorage> storage_;
  std::vector<std::unique_ptr<const ResultSetStorage>> appended_storage_;
  mutable size_t crt_row_buff_idx_;
  size_t drop_first_;
  size_t keep_first_;
  const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner_;
  std::vector<uint32_t> permutation_;
  const Executor* executor_;  // TODO(alex): remove

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

#endif  // QUERYENGINE_RESULTSET_H
