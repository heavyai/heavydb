/*
 * Copyright 2022 HEAVY.AI, Inc.
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

#pragma once

#include "Analyzer/Analyzer.h"
#include "DataMgr/Chunk/Chunk.h"
#include "QueryEngine/JoinHashTable/HashJoin.h"
#include "QueryEngine/Utils/SegmentTree.h"
#include "Shared/sqltypes.h"

#include <functional>
#include <unordered_map>

// Returns true for value window functions, false otherwise.
inline bool window_function_is_value(const SqlWindowFunctionKind kind) {
  switch (kind) {
    case SqlWindowFunctionKind::LAG:
    case SqlWindowFunctionKind::LEAD:
    case SqlWindowFunctionKind::FIRST_VALUE:
    case SqlWindowFunctionKind::LAST_VALUE: {
      return true;
    }
    default: {
      return false;
    }
  }
}

// Returns true for aggregate window functions, false otherwise.
inline bool window_function_is_aggregate(const SqlWindowFunctionKind kind) {
  switch (kind) {
    case SqlWindowFunctionKind::AVG:
    case SqlWindowFunctionKind::MIN:
    case SqlWindowFunctionKind::MAX:
    case SqlWindowFunctionKind::SUM:
    case SqlWindowFunctionKind::COUNT:
    case SqlWindowFunctionKind::SUM_INTERNAL: {
      return true;
    }
    default: {
      return false;
    }
  }
}

class Executor;

struct AggregateTreeForWindowFraming {
  std::vector<int64_t*> aggregate_tree_for_integer_type_;
  std::vector<double*> aggregate_tree_for_double_type_;
  std::vector<SumAndCountPair<int64_t>*> derived_aggregate_tree_for_integer_type_;
  std::vector<SumAndCountPair<double>*> derived_aggregate_tree_for_double_type_;
};

struct WindowFrameBoundFuncArgs {
  llvm::Value* frame_start_bound_expr_lv;
  llvm::Value* frame_end_bound_expr_lv;
  llvm::Value* current_row_pos_lv;
  llvm::Value* current_col_value_lv;
  llvm::Value* current_partition_start_offset_lv;
  llvm::Value* int64_t_zero_val_lv;
  llvm::Value* num_elem_current_partition_lv;
  llvm::Value* order_key_buf_ptr_lv;
  std::string order_type_col_name;
  llvm::Value* target_partition_rowid_ptr_lv;
  llvm::Value* target_partition_sorted_rowid_ptr_lv;
  llvm::Value* null_start_pos_lv;
  llvm::Value* null_end_pos_lv;
};

struct WindowPartitionBufferPtrs {
  llvm::Value* current_partition_start_offset_lv;
  llvm::Value* num_elem_current_partition_lv;
  llvm::Value* target_partition_rowid_ptr_lv;
  llvm::Value* target_partition_sorted_rowid_ptr_lv;
};

// Per-window function context which encapsulates the logic for computing the various
// window function kinds and keeps ownership of buffers which contain the results. For
// rank functions, the code generated for the projection simply reads the values and
// writes them to the result set. For value and aggregate functions, only the iteration
// order is written to the buffer, the rest is handled by generating code in a similar way
// we do for non-window queries.
class WindowFunctionContext {
 public:
  // non-partitioned version
  WindowFunctionContext(const Analyzer::WindowFunction* window_func,
                        const size_t elem_count,
                        const ExecutorDeviceType device_type,
                        std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner);

  // partitioned version
  WindowFunctionContext(
      const Analyzer::WindowFunction* window_func,
      QueryPlanHash cache_key,
      const std::shared_ptr<HashJoin>& partitions,
      const size_t elem_count,
      const ExecutorDeviceType device_type,
      std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
      size_t aggregation_tree_fan_out = g_window_function_aggregation_tree_fanout);

  WindowFunctionContext(const WindowFunctionContext&) = delete;

  WindowFunctionContext& operator=(const WindowFunctionContext&) = delete;

  ~WindowFunctionContext();

  // Adds the order column buffer to the context and keeps ownership of it.
  void addOrderColumn(const int8_t* column,
                      const SQLTypeInfo& ti,
                      const std::vector<std::shared_ptr<Chunk_NS::Chunk>>& chunks_owner);

  void setSortedPartitionCacheKey(QueryPlanHash cache_key);

  void addColumnBufferForWindowFunctionExpression(
      const int8_t* column,
      const std::vector<std::shared_ptr<Chunk_NS::Chunk>>& chunks_owner);

  enum class WindowComparatorResult { LT, EQ, GT };
  using Comparator =
      std::function<WindowFunctionContext::WindowComparatorResult(const int64_t lhs,
                                                                  const int64_t rhs)>;

  std::vector<Comparator> createComparator(size_t partition_idx);

  // Computes the window function result to be used during the actual projection query.
  void compute(
      std::unordered_map<QueryPlanHash, size_t>& sorted_partition_key_ref_count_map,
      std::unordered_map<QueryPlanHash, std::shared_ptr<std::vector<int64_t>>>&
          sorted_partition_cache);

  // Returns a pointer to the window function associated with this context.
  const Analyzer::WindowFunction* getWindowFunction() const;

  // Returns a pointer to the output buffer of the window function result.
  const int8_t* output() const;

  // Returns a pointer to the sorted row index buffer
  const int64_t* sortedPartition() const;

  // Returns a pointer to the value field of the aggregation state.
  const int64_t* aggregateState() const;

  // Returns a pointer to the count field of the aggregation state.
  const int64_t* aggregateStateCount() const;

  // Returns a handle to the pending outputs for the aggregate window function.
  int64_t aggregateStatePendingOutputs() const;

  const int64_t* partitionStartOffset() const;

  const int64_t* partitionNumCountBuf() const;

  const std::vector<const int8_t*>& getColumnBufferForWindowFunctionExpressions() const;

  const std::vector<const int8_t*>& getOrderKeyColumnBuffers() const;

  const std::vector<SQLTypeInfo>& getOrderKeyColumnBufferTypes() const;

  int64_t** getAggregationTreesForIntegerTypeWindowExpr() const;

  double** getAggregationTreesForDoubleTypeWindowExpr() const;

  SumAndCountPair<int64_t>** getDerivedAggregationTreesForIntegerTypeWindowExpr() const;

  SumAndCountPair<double>** getDerivedAggregationTreesForDoubleTypeWindowExpr() const;

  size_t* getAggregateTreeDepth() const;

  size_t getAggregateTreeFanout() const;

  int64_t* getNullValueStartPos() const;

  int64_t* getNullValueEndPos() const;

  // Returns a pointer to the partition start bitmap.
  const int8_t* partitionStart() const;

  // Returns a pointer to the partition end bitmap.
  const int8_t* partitionEnd() const;

  // Returns the element count in the columns used by the window function.
  size_t elementCount() const;

  const int32_t* payload() const;

  const int32_t* offsets() const;

  const int32_t* counts() const;

  size_t partitionCount() const;

  const bool needsToBuildAggregateTree() const;

 private:
  // State for a window aggregate. The count field is only used for average.
  struct AggregateState {
    int64_t val;
    int64_t count;
    std::vector<void*> outputs;
    llvm::Value* row_number = nullptr;
  };

  static Comparator makeComparator(const Analyzer::ColumnVar* col_var,
                                   const int8_t* partition_values,
                                   const int32_t* partition_indices,
                                   const bool nulls_first);

  void computePartitionBuffer(const size_t partition_idx,
                              int64_t* output_for_partition_buff,
                              const Analyzer::WindowFunction* window_func);

  void sortPartition(const size_t partition_idx,
                     int64_t* output_for_partition_buff,
                     bool should_parallelize);

  void computeNullRangeOfSortedPartition(const SQLTypeInfo& order_col_ti,
                                         size_t partition_idx,
                                         const int32_t* original_col_idx_buf,
                                         const int64_t* ordered_col_idx_buf);

  void buildAggregationTreeForPartition(SqlWindowFunctionKind agg_type,
                                        size_t partition_idx,
                                        size_t partition_size,
                                        const int8_t* col_buf,
                                        const int32_t* original_rowid_buf,
                                        const int64_t* ordered_rowid_buf,
                                        const SQLTypeInfo& input_col_ti);

  void fillPartitionStart();

  void fillPartitionEnd();

  const Analyzer::WindowFunction* window_func_;
  QueryPlanHash partition_cache_key_;
  QueryPlanHash sorted_partition_cache_key_;
  // Keeps ownership of order column.
  std::vector<std::vector<std::shared_ptr<Chunk_NS::Chunk>>> order_columns_owner_;
  // Order column buffers.
  std::vector<const int8_t*> order_columns_;
  std::vector<SQLTypeInfo> order_columns_ti_;
  // Hash table which contains the partitions specified by the window.
  std::shared_ptr<HashJoin> partitions_;
  // The number of elements in the table.
  size_t elem_count_;
  // The output of the window function.
  int8_t* output_;
  std::shared_ptr<std::vector<int64_t>> sorted_partition_buf_;
  // Keeps ownership of column referenced in window function expression.
  std::vector<std::vector<std::shared_ptr<Chunk_NS::Chunk>>>
      window_func_expr_columns_owner_;
  // Column buffers used for window function expression
  std::vector<const int8_t*> window_func_expr_columns_;
  // we need to build a segment tree depending on the input column type
  std::vector<std::shared_ptr<void>> segment_trees_owned_;
  AggregateTreeForWindowFraming aggregate_trees_;
  size_t aggregate_trees_fan_out_;
  size_t* aggregate_trees_depth_;
  int64_t* ordered_partition_null_start_pos_;
  int64_t* ordered_partition_null_end_pos_;
  int64_t* partition_start_offset_;
  // Markers for partition start used to reinitialize state for aggregate window
  // functions.
  int8_t* partition_start_;
  // Markers for partition end used to reinitialize state for aggregate window
  // functions.
  int8_t* partition_end_;
  // State for aggregate function over a window.
  AggregateState aggregate_state_;
  const ExecutorDeviceType device_type_;
  std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner_;

  // For use when we do not have partitions_ hash table
  const int32_t dummy_count_;
  const int32_t dummy_offset_;
  // dummy_payload_ is only initialized if there is no partitions_ hash table
  // TODO(todd): There is no need for index buffer for non-partitioned
  // window functions, as the row to index mapping is the identity function,
  // so refactor makeComparator and ilk to allow for this
  int32_t* dummy_payload_;
};

// Keeps track of the multiple window functions in a window query.
class WindowProjectNodeContext {
 public:
  void addWindowFunctionContext(
      std::unique_ptr<WindowFunctionContext> window_function_context,
      const size_t target_index);

  // Marks the window function at the given target index as active. This simplifies the
  // code generation since it's now context sensitive. Each value window function can have
  // its own iteration order, therefore fetching a column at a given position changes
  // depending on which window function is active.
  const WindowFunctionContext* activateWindowFunctionContext(
      Executor* executor,
      const size_t target_index) const;

  // Resets the active window function, which restores the regular (non-window) codegen
  // behavior.
  static void resetWindowFunctionContext(Executor* executor);

  // Gets the current active window function.
  static WindowFunctionContext* getActiveWindowFunctionContext(Executor* executor);

  // Creates the context for a window function execution unit.
  static WindowProjectNodeContext* create(Executor* executor);

  // Retrieves the context for the active window function execution unit.
  static const WindowProjectNodeContext* get(Executor* executor);

  // Resets the active context.
  static void reset(Executor* executor);

 private:
  // A map from target index to the context associated with the window function at that
  // target index.
  std::unordered_map<size_t, std::unique_ptr<WindowFunctionContext>> window_contexts_;
};

bool window_function_is_aggregate(const SqlWindowFunctionKind kind);

bool window_function_requires_peer_handling(const Analyzer::WindowFunction* window_func);
