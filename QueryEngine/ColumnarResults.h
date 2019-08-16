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
 */
class ColumnBitmap {
 public:
  ColumnBitmap(const size_t num_elements) : bitmap_(num_elements, false) {}

  inline bool get(const size_t index) const {
    CHECK(index < bitmap_.size());
    return bitmap_[index];
  }

  inline void set(const size_t index, const bool val) {
    CHECK(index < bitmap_.size());
    bitmap_[index] = val;
  }

 private:
  std::vector<bool> bitmap_;
};

class ColumnarResults {
 public:
  ColumnarResults(const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                  const ResultSet& rows,
                  const size_t num_columns,
                  const std::vector<SQLTypeInfo>& target_types);

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

  template <typename EntryT>
  EntryT getEntryAt(const size_t row_idx, const size_t column_idx) const;

  void setParallelConversion(const bool is_parallel) {
    parallel_conversion_ = is_parallel;
  }
  bool isParallelConversion() const { return parallel_conversion_; }
  bool isDirectColumnarConversionPossible() const { return direct_columnar_conversion_; }

  using ReadFunctionPerfectHash =
      std::function<int64_t(const ResultSet&, const size_t, const size_t)>;

  using WriteFunctionPerfectHash = std::function<void(const ResultSet&,
                                                      const size_t,
                                                      const size_t,
                                                      const size_t,
                                                      const size_t,
                                                      const ReadFunctionPerfectHash&)>;

 private:
  ColumnarResults(const size_t num_rows, const std::vector<SQLTypeInfo>& target_types)
      : num_rows_(num_rows), target_types_(target_types) {}
  inline void writeBackCell(const TargetValue& col_val,
                            const size_t row_idx,
                            const size_t column_idx);
  void materializeAllColumns(const ResultSet& rows, const size_t num_columns);

  // TODO(Saman): maybe refactor these into speciliazed sub-classes for each query
  // description type.

  // Direct columnarization for Perfect Hash (with columnar output)
  void materializeAllColumnsPerfectHash(const ResultSet& rows, const size_t num_columns);
  void locateAndCountEntriesPerfectHash(const ResultSet& rows,
                                        ColumnBitmap& bitmap,
                                        std::vector<size_t>& non_empty_per_thread,
                                        const size_t entry_count,
                                        const size_t num_threads,
                                        const size_t size_per_thread) const;
  void compactAndCopyEntriesPerfectHash(const ResultSet& rows,
                                        const ColumnBitmap& bitmap,
                                        const std::vector<size_t>& non_empty_per_thread,
                                        const size_t num_columns,
                                        const size_t entry_count,
                                        const size_t num_threads,
                                        const size_t size_per_thread);
  void compactAndCopyEntriesPHWithTargetSkipping(
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
  void compactAndCopyEntriesPHWithoutTargetSkipping(
      const ResultSet& rows,
      const ColumnBitmap& bitmap,
      const std::vector<size_t>& non_empty_per_thread,
      const std::vector<size_t>& global_offsets,
      const std::vector<size_t>& slot_idx_per_target_idx,
      const size_t num_columns,
      const size_t entry_count,
      const size_t num_threads,
      const size_t size_per_thread);
  template <typename DataType>
  void writeBackCellDirect(const ResultSet& rows,
                           const size_t input_buffer_entry_idx,
                           const size_t output_buffer_entry_idx,
                           const size_t target_idx,
                           const size_t slot_idx,
                           const ReadFunctionPerfectHash& read_function);
  std::vector<WriteFunctionPerfectHash> initWriteFunctionsPerfectHash(
      const ResultSet& rows,
      const std::vector<bool>& targets_to_skip = {});
  std::vector<ReadFunctionPerfectHash> initReadFunctionsPerfectHash(
      const ResultSet& rows,
      const std::vector<size_t>& slot_idx_per_target_idx,
      const std::vector<bool>& targets_to_skip = {});
  std::tuple<std::vector<WriteFunctionPerfectHash>, std::vector<ReadFunctionPerfectHash>>
  initAllConversionFunctionsPerfectHash(
      const ResultSet& rows,
      const std::vector<size_t>& slot_idx_per_target_idx,
      const std::vector<bool>& targets_to_skip = {});
  // Direct columnarization for Projections (with columnar output)
  void copyAllNonLazyColumns(const std::vector<ColumnLazyFetchInfo>& lazy_fetch_info,
                             const ResultSet& rows,
                             const size_t num_columns);
  void materializeAllLazyColumns(const std::vector<ColumnLazyFetchInfo>& lazy_fetch_info,
                                 const ResultSet& rows,
                                 const size_t num_columns);

  std::vector<int8_t*> column_buffers_;
  size_t num_rows_;
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
