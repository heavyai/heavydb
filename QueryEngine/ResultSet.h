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

class ResultSet {
 public:
  ResultSet(const std::vector<TargetInfo>& targets,
            const ExecutorDeviceType device_type,
            const QueryMemoryDescriptor& query_mem_desc);

  // Empty result set constructor
  ResultSet();

  ~ResultSet();

  const ResultSetStorage* allocateStorage() const;

  std::vector<TargetValue> getNextRow(const bool translate_strings, const bool decimal_to_double) const;

  void sort(const std::list<Analyzer::OrderEntry>& order_entries,
            const bool remove_duplicates,
            const int64_t top_n) const;

  bool isEmptyInitializer() const;

  void keepFirstN(const size_t n);

  void dropFirstN(const size_t n);

  void append(ResultSet& that);

 private:
  void advanceCursorToNextEntry() const;

  void sortOnGpu(const std::list<Analyzer::OrderEntry>& order_entries) const;

  void sortOnCpu(const std::list<Analyzer::OrderEntry>& order_entries) const;

  void fetchLazy(const std::vector<ssize_t> lazy_col_local_ids,
                 const std::vector<std::vector<const int8_t*>>& col_buffers) const;

  void initializeStorage() const;

  static bool isNull(const SQLTypeInfo& ti, const InternalTargetValue& val);

  const std::vector<TargetInfo> targets_;
  const ExecutorDeviceType device_type_;
  const QueryMemoryDescriptor query_mem_desc_;
  mutable std::unique_ptr<const ResultSetStorage> storage_;
  std::vector<std::unique_ptr<const ResultSetStorage>> appended_storage_;
  mutable size_t crt_row_buff_idx_;
  size_t drop_first_;
  size_t keep_first_;

  friend class ResultSetManager;
};

class ResultSetManager {
 public:
  ResultSet* reduce(std::vector<ResultSet*>&);

 private:
  std::unique_ptr<ResultSet> rs_;
};

#endif  // QUERYENGINE_RESULTSET_H
