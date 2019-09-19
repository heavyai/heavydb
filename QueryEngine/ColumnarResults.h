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

#ifndef COLUMNAR_RESULTS_H
#define COLUMNAR_RESULTS_H
#include "ResultSet.h"
#include "Shared/SqlTypesLayout.h"

#include "../Shared/checked_alloc.h"

#include <memory>
#include <unordered_map>

class ColumnarConversionNotSupported : public std::runtime_error {
 public:
  ColumnarConversionNotSupported()
      : std::runtime_error(
            "Columnar conversion not supported for variable length types") {}
};

/**
 * A helper data structure to track non-empty entries in the input buffer
 * Currently only used for direct columnarization with columnar outputs.
 * Each bank is assigned to a thread so that concurrent updates of the
 * data structure is non-blocking.
 */
class ColumnBitmap {
 public:
  ColumnBitmap(const size_t num_elements_per_bank, size_t num_banks)
      : bitmaps_(num_banks, std::vector<bool>(num_elements_per_bank, false)) {}

  inline bool get(const size_t index, const size_t bank_index) const {
    CHECK_LT(bank_index, bitmaps_.size());
    CHECK_LT(index, bitmaps_[bank_index].size());
    return bitmaps_[bank_index][index];
  }

  inline void set(const size_t index, const size_t bank_index, const bool val) {
    CHECK_LT(bank_index, bitmaps_.size());
    CHECK_LT(index, bitmaps_[bank_index].size());
    bitmaps_[bank_index][index] = val;
  }

 private:
  std::vector<std::vector<bool>> bitmaps_;
};

class ColumnarResults {
 public:
  ColumnarResults(const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                  const ResultSet& rows,
                  const size_t num_columns,
                  const std::vector<SQLTypeInfo>& target_types,
                  const bool is_parallel_execution_enforced = false);

  ColumnarResults(const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                  const int8_t* one_col_buffer,
                  const size_t num_rows,
                  const SQLTypeInfo& target_type);

  static std::unique_ptr<ColumnarResults> mergeResults(
      const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
      const std::vector<std::unique_ptr<ColumnarResults>>& sub_results);

  const std::vector<int8_t*>& getColumnBuffers() const { return column_buffers_; }

  const size_t size() const { return num_rows_; }

  const SQLTypeInfo& getColumnType(const int col_id) const {
    CHECK_GE(col_id, 0);
    CHECK_LT(static_cast<size_t>(col_id), target_types_.size());
    return target_types_[col_id];
  }

  bool isParallelConversion() const { return parallel_conversion_; }
  bool isDirectColumnarConversionPossible() const { return direct_columnar_conversion_; }

  // functions used to read content from the result set (direct columnarization, group by
  // queries)
  using ReadFunction =
      std::function<int64_t(const ResultSet&, const size_t, const size_t, const size_t)>;

  // functions used to write back contents into output column buffers (direct
  // columnarization, group by queries)
  using WriteFunction = std::function<void(const ResultSet&,
                                           const size_t,
                                           const size_t,
                                           const size_t,
                                           const size_t,
                                           const ReadFunction&)>;

 protected:
  std::vector<int8_t*> column_buffers_;
  size_t num_rows_;

 private:
  ColumnarResults(const size_t num_rows, const std::vector<SQLTypeInfo>& target_types)
      : num_rows_(num_rows), target_types_(target_types) {}
  inline void writeBackCell(const TargetValue& col_val,
                            const size_t row_idx,
                            const size_t column_idx);
  void materializeAllColumnsDirectly(const ResultSet& rows, const size_t num_columns);
  void materializeAllColumnsThroughIteration(const ResultSet& rows,
                                             const size_t num_columns);

  // Direct columnarization for group by queries (perfect hash or baseline hash)
  void materializeAllColumnsGroupBy(const ResultSet& rows, const size_t num_columns);

  // Direct columnarization for Projections (only output is columnar)
  void materializeAllColumnsProjection(const ResultSet& rows, const size_t num_columns);

  void copyAllNonLazyColumns(const std::vector<ColumnLazyFetchInfo>& lazy_fetch_info,
                             const ResultSet& rows,
                             const size_t num_columns);
  void materializeAllLazyColumns(const std::vector<ColumnLazyFetchInfo>& lazy_fetch_info,
                                 const ResultSet& rows,
                                 const size_t num_columns);

  void locateAndCountEntries(const ResultSet& rows,
                             ColumnBitmap& bitmap,
                             std::vector<size_t>& non_empty_per_thread,
                             const size_t entry_count,
                             const size_t num_threads,
                             const size_t size_per_thread) const;
  void compactAndCopyEntries(const ResultSet& rows,
                             const ColumnBitmap& bitmap,
                             const std::vector<size_t>& non_empty_per_thread,
                             const size_t num_columns,
                             const size_t entry_count,
                             const size_t num_threads,
                             const size_t size_per_thread);
  void compactAndCopyEntriesWithTargetSkipping(
      const ResultSet& rows,
      const ColumnBitmap& bitmap,
      const std::vector<size_t>& non_empty_per_thread,
      const std::vector<size_t>& global_offsets,
      const std::vector<bool>& targets_to_skip,
      const std::vector<size_t>& slot_idx_per_target_idx,
      const size_t num_columns,
      const size_t entry_count,
      const size_t num_threads,
      const size_t size_per_thread);
  void compactAndCopyEntriesWithoutTargetSkipping(
      const ResultSet& rows,
      const ColumnBitmap& bitmap,
      const std::vector<size_t>& non_empty_per_thread,
      const std::vector<size_t>& global_offsets,
      const std::vector<size_t>& slot_idx_per_target_idx,
      const size_t num_columns,
      const size_t entry_count,
      const size_t num_threads,
      const size_t size_per_thread);

  template <typename DATA_TYPE>
  void writeBackCellDirect(const ResultSet& rows,
                           const size_t input_buffer_entry_idx,
                           const size_t output_buffer_entry_idx,
                           const size_t target_idx,
                           const size_t slot_idx,
                           const ReadFunction& read_function);

  std::vector<WriteFunction> initWriteFunctions(
      const ResultSet& rows,
      const std::vector<bool>& targets_to_skip = {});

  template <QueryDescriptionType QUERY_TYPE, bool COLUMNAR_OUTPUT>
  std::vector<ReadFunction> initReadFunctions(
      const ResultSet& rows,
      const std::vector<size_t>& slot_idx_per_target_idx,
      const std::vector<bool>& targets_to_skip = {});

  std::tuple<std::vector<WriteFunction>, std::vector<ReadFunction>>
  initAllConversionFunctions(const ResultSet& rows,
                             const std::vector<size_t>& slot_idx_per_target_idx,
                             const std::vector<bool>& targets_to_skip = {});

  const std::vector<SQLTypeInfo> target_types_;
  bool parallel_conversion_;  // multi-threaded execution of columnar conversion
  bool
      direct_columnar_conversion_;  // whether columnar conversion might happen directly
                                    // with minimal ussage of result set's iterator access
};

typedef std::
    unordered_map<int, std::unordered_map<int, std::shared_ptr<const ColumnarResults>>>
        ColumnCacheMap;

#endif  // COLUMNAR_RESULTS_H
