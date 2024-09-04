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

#ifdef HAVE_CUDA
#include "DataMgr/Allocators/CudaAllocator.h"
#endif

#include <functional>
#include <unordered_map>

// Returns true for value window functions, false otherwise.
inline bool window_function_is_value(const SqlWindowFunctionKind kind) {
  switch (kind) {
    case SqlWindowFunctionKind::LAG:
    case SqlWindowFunctionKind::LEAD:
    case SqlWindowFunctionKind::FIRST_VALUE:
    case SqlWindowFunctionKind::LAST_VALUE:
    case SqlWindowFunctionKind::NTH_VALUE:
    case SqlWindowFunctionKind::LAG_IN_FRAME:
    case SqlWindowFunctionKind::LEAD_IN_FRAME:
    case SqlWindowFunctionKind::FIRST_VALUE_IN_FRAME:
    case SqlWindowFunctionKind::LAST_VALUE_IN_FRAME:
    case SqlWindowFunctionKind::NTH_VALUE_IN_FRAME:
      return true;
    default:
      return false;
  }
}

inline bool window_function_is_value_with_frame(const SqlWindowFunctionKind kind) {
  switch (kind) {
    case SqlWindowFunctionKind::LAG_IN_FRAME:
    case SqlWindowFunctionKind::LEAD_IN_FRAME:
    case SqlWindowFunctionKind::FIRST_VALUE_IN_FRAME:
    case SqlWindowFunctionKind::LAST_VALUE_IN_FRAME:
    case SqlWindowFunctionKind::NTH_VALUE_IN_FRAME:
      return true;
    default:
      return false;
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
    case SqlWindowFunctionKind::SUM_INTERNAL:
    case SqlWindowFunctionKind::COUNT_IF:
    case SqlWindowFunctionKind::SUM_IF:
    case SqlWindowFunctionKind::CONDITIONAL_CHANGE_EVENT:
      return true;
    default:
      return false;
  }
}

inline bool window_function_conditional_aggregate(const SqlWindowFunctionKind kind) {
  switch (kind) {
    case SqlWindowFunctionKind::COUNT_IF:
    case SqlWindowFunctionKind::SUM_IF:
      return true;
    default:
      return false;
  }
}

class Executor;

namespace WindowFunctionCtx {

// When we have N window partitions, we need to build up to N aggregate trees
// (i.e., we may have empty window partitions)
// To hold them, we can allocate contiguous buffer instead of allocating N distinct
// small memory chunks b/c we know the size of window partitions, and maintain
// buffer start offsets to access one of N aggregate tree from the contiguous buffer
struct AggregateTreeForWindowFraming {
  // for aggregate trees on CPU
  int8_t* aggregate_tree_holder_{nullptr};
  // depending on aggregate data type and window function, the aggregate tree falls
  // into one of the following four types
  // todo (yoonmin): can we use template to maintain only one agg. tree buf offsets?
  std::vector<int64_t*> aggregate_tree_for_integer_type_;
  std::vector<double*> aggregate_tree_for_double_type_;
  std::vector<SumAndCountPair<int64_t>*> derived_aggregate_tree_for_integer_type_;
  std::vector<SumAndCountPair<double>*> derived_aggregate_tree_for_double_type_;
  // below two size_t* arrays need to be accessed during the query execution
  // so we maintain them per device types
  std::vector<size_t> aggregate_trees_depth_;
  std::vector<size_t> aggregate_trees_leaf_start_offset_;

  // for aggregate trees on GPU
  // todo (yoonmin) : support multi-GPU execution
  Data_Namespace::AbstractBuffer* aggregate_trees_holder_gpu_{nullptr};
  Data_Namespace::AbstractBuffer* aggregate_trees_depth_gpu_{nullptr};
  Data_Namespace::AbstractBuffer* aggregate_trees_leaf_start_offset_gpu_{nullptr};

  Data_Namespace::AbstractBuffer* aggregate_tree_for_integer_type_buf_gpu_{nullptr};
  Data_Namespace::AbstractBuffer* aggregate_tree_for_double_type_buf_gpu_{nullptr};
  Data_Namespace::AbstractBuffer* derived_aggregate_tree_for_integer_type_buf_gpu_{
      nullptr};
  Data_Namespace::AbstractBuffer* derived_aggregate_tree_for_double_type_buf_gpu_{
      nullptr};
  // casted version of aggregate trees on GPU per type
  int64_t** aggregate_tree_for_integer_type_gpu_{nullptr};
  double** aggregate_tree_for_double_type_gpu_{nullptr};
  SumAndCountPair<int64_t>** derived_aggregate_tree_for_integer_type_gpu_{nullptr};
  SumAndCountPair<double>** derived_aggregate_tree_for_double_type_gpu_{nullptr};
  Data_Namespace::DataMgr* data_mgr_;

  // common metadata of aggregate trees
  std::vector<size_t> aggregate_tree_sizes_;
  std::vector<size_t> aggregate_tree_sizes_in_byte_;
  std::vector<size_t> aggregate_tree_start_offset_;
  std::vector<std::pair<size_t, IndexPair>> aggregate_tree_infos_;

  // owned aggregate trees
  std::vector<std::shared_ptr<void>> segment_trees_owned_;

  ~AggregateTreeForWindowFraming() {
    if (aggregate_trees_holder_gpu_) {
      data_mgr_->free(aggregate_trees_holder_gpu_);
    }
    if (aggregate_trees_depth_gpu_) {
      data_mgr_->free(aggregate_trees_depth_gpu_);
    }
    if (aggregate_trees_leaf_start_offset_gpu_) {
      data_mgr_->free(aggregate_trees_leaf_start_offset_gpu_);
    }
    if (aggregate_tree_for_integer_type_buf_gpu_) {
      data_mgr_->free(aggregate_tree_for_integer_type_buf_gpu_);
    }
    if (aggregate_tree_for_double_type_buf_gpu_) {
      data_mgr_->free(aggregate_tree_for_double_type_buf_gpu_);
    }
    if (derived_aggregate_tree_for_integer_type_buf_gpu_) {
      data_mgr_->free(derived_aggregate_tree_for_integer_type_buf_gpu_);
    }
    if (derived_aggregate_tree_for_double_type_buf_gpu_) {
      data_mgr_->free(derived_aggregate_tree_for_double_type_buf_gpu_);
    }
  }

  void resizeStorageForWindowFraming(size_t partition_count) {
    // we do not need to resize agg. tree buf for GPU b/c
    // we build them on CPU regardless of what device type the query uses
    // and copy them to GPU only if we use GPU to execute the query
    // so before copying them, we can resize those vectors appropriately
    aggregate_tree_for_integer_type_.resize(partition_count);
    aggregate_tree_for_double_type_.resize(partition_count);
    derived_aggregate_tree_for_integer_type_.resize(partition_count);
    derived_aggregate_tree_for_double_type_.resize(partition_count);

    aggregate_tree_infos_.resize(partition_count);
    aggregate_tree_sizes_.resize(partition_count);
    aggregate_tree_sizes_in_byte_.resize(partition_count);
    aggregate_tree_start_offset_.resize(partition_count);
  }
};

struct WindowFrameBoundFuncLLVMArgs {
  llvm::Value* frame_start_bound_expr_lv;
  llvm::Value* frame_end_bound_expr_lv;
  llvm::Value* current_row_pos_lv;
  llvm::Value* current_col_value_lv;
  llvm::Value* current_partition_start_offset_lv;
  llvm::Value* int64_t_zero_val_lv;
  llvm::Value* int64_t_one_val_lv;
  llvm::Value* num_elem_current_partition_lv;
  llvm::Value* order_key_buf_ptr_lv;
  std::string order_type_col_name;
  llvm::Value* target_partition_rowid_ptr_lv;
  llvm::Value* target_partition_sorted_rowid_ptr_lv;
  llvm::Value* nulls_first_lv;
  llvm::Value* null_start_pos_lv;
  llvm::Value* null_end_pos_lv;
};

struct WindowPartitionBufferLLVMArgs {
  llvm::Value* current_partition_start_offset_lv;
  llvm::Value* num_elem_current_partition_lv;
  llvm::Value* target_partition_rowid_ptr_lv;
  llvm::Value* target_partition_sorted_rowid_ptr_lv;
};

struct ColumnChunkOwned {
  // Keeps ownership of order column
  std::vector<std::vector<std::shared_ptr<Chunk_NS::Chunk>>> order_columns_owner_;
  // Order column buffers.
  std::vector<const int8_t*> order_columns_;
  // Keeps ownership of column referenced in window function expression
  std::vector<std::vector<std::shared_ptr<Chunk_NS::Chunk>>>
      window_func_expr_columns_owner_;
  // Column buffers used for window function expression
  std::vector<const int8_t*> window_func_expr_columns_;

  std::vector<std::vector<std::shared_ptr<Chunk_NS::Chunk>>>
      window_func_expr_columns_for_gpu_owner_;
  std::vector<const int8_t*> window_func_expr_columns_for_gpu_;
  std::vector<std::vector<std::shared_ptr<Chunk_NS::Chunk>>> order_columns_for_gpu_owner_;
  std::vector<const int8_t*> order_columns_for_gpu_;
  std::vector<SQLTypeInfo> order_columns_ti_;
};

struct CPUExecutionCtx {
  // The output of the window function
  // this is allocated from row_set_mem_owner, so we do not need to dealloc. it
  int8_t* output_{nullptr};
  int32_t dummy_count_;
  int32_t dummy_offset_;
  // dummy_payload_ is only initialized if there is no partitions_ hash table
  // TODO(todd): There is no need for index buffer for non-partitioned
  // window functions, as the row to index mapping is the identity function,
  // so refactor makeComparator and ilk to allow for this
  int32_t* dummy_payload_{nullptr};
  // Bitmap index used to reinitialize aggregation state per partition
  int8_t* partition_start_{nullptr};
  int8_t* partition_end_{nullptr};
  size_t partition_start_sz_;
  size_t partition_end_sz_;
  // this is allocated from row_set_mem_owner, so we do not need to dealloc it
  int8_t* sorted_partition_buf_{nullptr};
  // we need to build a segment tree depending on the input column type
  std::vector<int64_t> ordered_partition_null_start_pos_;
  std::vector<int64_t> ordered_partition_null_end_pos_;
  std::vector<int64_t> partition_start_offset_;
  // this refers to either `output_` or `a unqiue ptr` so we do not dealloc it.
  int64_t* intermediate_output_buffer_{nullptr};

  ~CPUExecutionCtx() {
    if (dummy_payload_) {
      free(dummy_payload_);
    }
    if (partition_start_) {
      free(partition_start_);
    }
    if (partition_end_) {
      free(partition_end_);
    }
  }
};

struct GPUExecutionCtx {
  Data_Namespace::AbstractBuffer* output_gpu_{nullptr};
  Data_Namespace::AbstractBuffer* partitions_buf_gpu_holder_{nullptr};
  Data_Namespace::AbstractBuffer* dummy_count_gpu_{nullptr};
  Data_Namespace::AbstractBuffer* dummy_offset_gpu_{nullptr};
  Data_Namespace::AbstractBuffer* dummy_payload_gpu_{nullptr};
  Data_Namespace::AbstractBuffer* partition_start_offset_gpu_{nullptr};
  Data_Namespace::AbstractBuffer* partition_start_gpu_{nullptr};
  Data_Namespace::AbstractBuffer* partition_end_gpu_{nullptr};
  Data_Namespace::AbstractBuffer* sorted_partition_buf_gpu_{nullptr};
  Data_Namespace::AbstractBuffer* ordered_partition_null_start_pos_gpu_{nullptr};
  Data_Namespace::AbstractBuffer* ordered_partition_null_end_pos_gpu_{nullptr};
  Data_Namespace::DataMgr* data_mgr_;

  ~GPUExecutionCtx() {
    if (output_gpu_) {
      data_mgr_->free(output_gpu_);
    }
    if (partitions_buf_gpu_holder_) {
      data_mgr_->free(partitions_buf_gpu_holder_);
    }
    if (dummy_count_gpu_) {
      data_mgr_->free(dummy_count_gpu_);
    }
    if (dummy_offset_gpu_) {
      data_mgr_->free(dummy_offset_gpu_);
    }
    if (dummy_payload_gpu_) {
      data_mgr_->free(dummy_payload_gpu_);
    }
    if (partition_start_offset_gpu_) {
      data_mgr_->free(partition_start_offset_gpu_);
    }
    if (partition_start_gpu_) {
      data_mgr_->free(partition_start_gpu_);
    }
    if (partition_end_gpu_) {
      data_mgr_->free(partition_end_gpu_);
    }
    if (sorted_partition_buf_gpu_) {
      data_mgr_->free(sorted_partition_buf_gpu_);
    }
    if (ordered_partition_null_start_pos_gpu_) {
      data_mgr_->free(ordered_partition_null_start_pos_gpu_);
    }
    if (ordered_partition_null_end_pos_gpu_) {
      data_mgr_->free(ordered_partition_null_end_pos_gpu_);
    }
  }
};

};  // namespace WindowFunctionCtx

// Per-window function context which encapsulates the logic for computing the various
// window function kinds and keeps ownership of buffers which contain the results. For
// rank functions, the code generated for the projection simply reads the values and
// writes them to the result set. For value and aggregate functions, only the iteration
// order is written to the buffer, the rest is handled by generating code in a similar
// way we do for non-window queries.
class WindowFunctionContext {
 public:
  // we currently only use a single GPU to process the window function because
  // a query step having a window function expression only has a single fragment input
  // (i.e., push the window function expression down to the child projection node)
  // todo (yoonmin) : support window function execution with multi-fragmented input
  // todo (yoonmin) : support heterogeneous execution (i.e., CPU + GPU)
  static const int NUM_EXECUTION_DEVICES = 1;

  // non-partitioned version
  WindowFunctionContext(
      const Analyzer::WindowFunction* window_func,
      const size_t elem_count,
      const ExecutorDeviceType device_type,
      std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
      Data_Namespace::DataMgr* data_mgr,
      DeviceAllocator* device_allocator,
      const int device_id,
      size_t aggregation_tree_fan_out = g_window_function_aggregation_tree_fanout);

  // partitioned version
  WindowFunctionContext(
      const Analyzer::WindowFunction* window_func,
      QueryPlanHash cache_key,
      std::shared_ptr<HashJoin> hash_table,
      const size_t elem_count,
      const ExecutorDeviceType device_type,
      std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
      Data_Namespace::DataMgr* data_mgr,
      DeviceAllocator* device_allocator,
      const int device_id,
      size_t aggregation_tree_fan_out = g_window_function_aggregation_tree_fanout);

  WindowFunctionContext(const WindowFunctionContext&) = delete;

  WindowFunctionContext& operator=(const WindowFunctionContext&) = delete;

  // Adds the order column buffer to the context and keeps ownership of it.
  void addOrderColumn(const int8_t* column,
                      const SQLTypeInfo& ti,
                      const std::vector<std::shared_ptr<Chunk_NS::Chunk>>& chunks_owner,
                      ExecutorDeviceType device_type);

  void setSortedPartitionCacheKey(QueryPlanHash cache_key);

  void addColumnBufferForWindowFunctionExpression(
      const int8_t* column,
      const std::vector<std::shared_ptr<Chunk_NS::Chunk>>& chunks_owner,
      ExecutorDeviceType device_type);

  enum class WindowComparatorResult { LT, EQ, GT };
  using Comparator =
      std::function<WindowFunctionContext::WindowComparatorResult(const int64_t lhs,
                                                                  const int64_t rhs)>;

  std::vector<Comparator> createComparator(size_t partition_idx);

  // Computes the window function result to be used during the actual projection query.
  void compute(
      std::unordered_map<QueryPlanHash, size_t>& sorted_partition_key_ref_count_map,
      std::unordered_map<QueryPlanHash, int8_t*>& sorted_partition_cache,
      std::unordered_map<QueryPlanHash, WindowFunctionCtx::AggregateTreeForWindowFraming>&
          aggregate_tree_map);

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

  const size_t* getAggregateTreeDepth() const;

  const size_t* getAggregateTreeLeafStartOffset() const;

  size_t getAggregateTreeFanout() const;

  const int64_t* getNullValueStartPos() const;

  const int64_t* getNullValueEndPos() const;

  // Returns a pointer to the partition start bitmap.
  const int8_t* getPartitionStartBitmapBuf(ExecutorDeviceType const dt) const;
  void setPartitionStartBitmapSize(size_t sz);
  size_t partitionStartBitmapSize() const;

  // Returns a pointer to the partition end bitmap.
  const int8_t* getPartitionEndBitmapBuf(ExecutorDeviceType const dt) const;
  void setPartitionEndBitmapSize(size_t sz);

  // Returns the element count in the columns used by the window function.
  size_t elementCount() const;

  const int8_t* getPartitionBuf(ExecutorDeviceType const dt) const;

  const int32_t* getPayloadBuf(ExecutorDeviceType const dt) const;

  const int32_t* getOffsetBuf(ExecutorDeviceType const dt) const;

  const int32_t* getCountBuf(ExecutorDeviceType const dt) const;

  size_t getNumWindowPartition() const;

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
                                   const bool asc_ordering,
                                   const bool nulls_first);

  void computePartitionBuffer(const size_t partition_idx,
                              int64_t* output_for_partition_buff,
                              const Analyzer::WindowFunction* window_func);

  void sortPartition(const size_t partition_idx, int64_t* output_for_partition_buff);

  void computeNullRangeOfSortedPartition(const SQLTypeInfo& order_col_ti,
                                         size_t partition_idx,
                                         const int32_t* original_col_idx_buf,
                                         const int64_t* ordered_col_idx_buf);

  void buildAggregationTreeForPartition(SqlWindowFunctionKind agg_type,
                                        size_t partition_idx,
                                        size_t partition_size,
                                        const int32_t* original_rowid_buf,
                                        const int64_t* ordered_rowid_buf,
                                        const SQLTypeInfo& input_col_ti);

  void fillPartitionStart();

  void fillPartitionEnd();

  void resizeStorageForWindowFraming(bool const for_reuse = false);

  const QueryPlanHash computeAggregateTreeCacheKey() const;

  void prepareEmptyPartition(bool for_gpu);
  void preparePartitionStartOffsetBufs(bool for_gpu, size_t partition_count);
  void prepareAggregateTreeBufs(bool for_gpu, size_t const partition_count);
  ExecutorDeviceType getDeviceType() const { return device_type_; }
  bool forGpuExecution() const { return device_type_ == ExecutorDeviceType::GPU; }

  const Analyzer::WindowFunction* window_func_;
  QueryPlanHash partition_cache_key_;
  QueryPlanHash sorted_partition_cache_key_;

  // The number of elements in the table.
  size_t elem_count_;
  // an owner of the partition_ptr_ having various helper APIs
  // this can be located at either CPU or GPU
  std::shared_ptr<HashJoin> partitions_;

  std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner_;
  Data_Namespace::DataMgr* data_mgr_;
  const ExecutorDeviceType device_type_;
  const int device_id_;
  DeviceAllocator* device_allocator_;

  WindowFunctionCtx::ColumnChunkOwned column_chunk_owned_;
  WindowFunctionCtx::CPUExecutionCtx cpu_exec_ctx_;
  WindowFunctionCtx::GPUExecutionCtx gpu_exec_ctx_;
  WindowFunctionCtx::AggregateTreeForWindowFraming aggregate_trees_;
  // For use when window function does not have partitioning column(s)
  size_t aggregate_trees_fan_out_;
  // State for aggregate function over a window
  AggregateState aggregate_state_;

  mutable llvm::Value* hoisted_output_addr{nullptr};
  mutable std::mutex output_addr_lock_;
};

// Keeps track of the multiple window functions in a window query.
class WindowProjectNodeContext {
 public:
  void addWindowFunctionContext(
      std::unique_ptr<WindowFunctionContext> window_function_context,
      const size_t target_index);

  // Marks the window function at the given target index as active. This simplifies the
  // code generation since it's now context sensitive. Each value window function can
  // have its own iteration order, therefore fetching a column at a given position
  // changes depending on which window function is active.
  const WindowFunctionContext* activateWindowFunctionContext(
      Executor* executor,
      const size_t target_index) const;

#ifdef HAVE_CUDA
  void registerDeviceAllocator(int device_id, Data_Namespace::DataMgr* data_mgr);

  CudaAllocator* getDeviceAllocator(int device_id) const;
#endif

  void registerWindowPartition(size_t const target_idx,
                               std::shared_ptr<HashJoin> partition);

  HashJoin* getWindowPartition(size_t const target_idx) const;

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
  std::unordered_map<size_t, std::shared_ptr<HashJoin>> window_partition_owned_;
#ifdef HAVE_CUDA
  std::unordered_map<int, std::unique_ptr<CudaAllocator>> device_allocator_owned_;
#endif
};

bool window_function_is_aggregate(const SqlWindowFunctionKind kind);

bool window_function_requires_peer_handling(const Analyzer::WindowFunction* window_func);
