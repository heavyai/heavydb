/*
 * Copyright 2020 OmniSci, Inc.
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
 * @file    ResultSetStorage.h
 * @author
 * @brief   Basic constructors and methods of the row set interface.
 *
 * Copyright (c) 2020 OmniSci, Inc.  All rights reserved.
 */

#ifndef QUERYENGINE_RESULTSETSTORAGE_H
#define QUERYENGINE_RESULTSETSTORAGE_H

#include "CardinalityEstimator.h"
#include "DataMgr/Chunk/Chunk.h"
#include "ResultSetBufferAccessors.h"
#include "TargetValue.h"

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

struct ReductionCode;

struct ColumnLazyFetchInfo {
  const bool is_lazily_fetched;
  const int local_col_id;
  const SQLTypeInfo type;
};

struct OneIntegerColumnRow {
  const int64_t value;
  const bool valid;
};

struct VarlenOutputInfo {
  int64_t gpu_start_address;
  int8_t* cpu_buffer_ptr;

  int8_t* computeCpuOffset(const int64_t gpu_offset_address) const;
};

class ResultSetStorage {
 private:
  ResultSetStorage(const std::vector<TargetInfo>& targets,
                   const QueryMemoryDescriptor& query_mem_desc,
                   int8_t* buff,
                   const bool buff_is_provided);

 public:
  void reduce(const ResultSetStorage& that,
              const std::vector<std::string>& serialized_varlen_buffer,
              const ReductionCode& reduction_code) const;

  void rewriteAggregateBufferOffsets(
      const std::vector<std::string>& serialized_varlen_buffer) const;

  int8_t* getUnderlyingBuffer() const;

  size_t getEntryCount() const { return query_mem_desc_.getEntryCount(); }

  template <class KeyType>
  void moveEntriesToBuffer(int8_t* new_buff, const size_t new_entry_count) const;

  template <class KeyType>
  void moveOneEntryToBuffer(const size_t entry_index,
                            int64_t* new_buff_i64,
                            const size_t new_entry_count,
                            const size_t key_count,
                            const size_t row_qw_count,
                            const int64_t* src_buff,
                            const size_t key_byte_width) const;

  void updateEntryCount(const size_t new_entry_count) {
    query_mem_desc_.setEntryCount(new_entry_count);
  }

  void reduceOneApproxQuantileSlot(int8_t* this_ptr1,
                                   const int8_t* that_ptr1,
                                   const size_t target_logical_idx,
                                   const ResultSetStorage& that) const;

  // Reduces results for a single row when using interleaved bin layouts
  static bool reduceSingleRow(const int8_t* row_ptr,
                              const int8_t warp_count,
                              const bool is_columnar,
                              const bool replace_bitmap_ptr_with_bitmap_sz,
                              std::vector<int64_t>& agg_vals,
                              const QueryMemoryDescriptor& query_mem_desc,
                              const std::vector<TargetInfo>& targets,
                              const std::vector<int64_t>& agg_init_vals);

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
  void reduceOneSlotSingleValue(int8_t* this_ptr1,
                                const TargetInfo& target_info,
                                const size_t target_slot_idx,
                                const size_t init_agg_val_idx,
                                const int8_t* that_ptr1) const;

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
                     const size_t first_slot_idx_for_target,
                     const std::vector<std::string>& serialized_varlen_buffer) const;

  void reduceOneCountDistinctSlot(int8_t* this_ptr1,
                                  const int8_t* that_ptr1,
                                  const size_t target_logical_idx,
                                  const ResultSetStorage& that) const;

  void fillOneEntryRowWise(const std::vector<int64_t>& entry);

  void fillOneEntryColWise(const std::vector<int64_t>& entry);

  void initializeRowWise() const;

  void initializeColWise() const;

  const VarlenOutputInfo* getVarlenOutputInfo() const {
    return varlen_output_info_.get();
  }

  // TODO(alex): remove the following two methods, see comment about
  // count_distinct_sets_mapping_.
  void addCountDistinctSetPointerMapping(const int64_t remote_ptr, const int64_t ptr);

  int64_t mappedPtr(const int64_t) const;

  size_t binSearchRowCount() const;

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

  // ptr to host varlen buffer and gpu address computation info
  std::shared_ptr<VarlenOutputInfo> varlen_output_info_;

  friend class ResultSet;
  friend class ResultSetManager;
};

using GroupValueInfo = std::pair<int64_t*, bool>;

namespace result_set {

int64_t lazy_decode(const ColumnLazyFetchInfo& col_lazy_fetch,
                    const int8_t* byte_stream,
                    const int64_t pos);

void fill_empty_key(void* key_ptr, const size_t key_count, const size_t key_width);

int8_t get_width_for_slot(const size_t target_slot_idx,
                          const bool float_argument_input,
                          const QueryMemoryDescriptor& query_mem_desc);

size_t get_byteoff_of_slot(const size_t slot_idx,
                           const QueryMemoryDescriptor& query_mem_desc);

GroupValueInfo get_group_value_reduction(int64_t* groups_buffer,
                                         const uint32_t groups_buffer_entry_count,
                                         const int64_t* key,
                                         const uint32_t key_count,
                                         const size_t key_width,
                                         const QueryMemoryDescriptor& query_mem_desc,
                                         const int64_t* that_buff_i64,
                                         const size_t that_entry_idx,
                                         const size_t that_entry_count,
                                         const uint32_t row_size_quad);

std::vector<int64_t> initialize_target_values_for_storage(
    const std::vector<TargetInfo>& targets);

}  // namespace result_set

#endif  // QUERYENGINE_RESULTSETSTORAGE_H
