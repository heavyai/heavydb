/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "Logger/Logger.h"
#include "QueryEngine/ColumnFetcher.h"
#include "QueryEngine/Descriptors/QueryCompilationDescriptor.h"

#include "Shared/threading.h"

#include <functional>
#include <memory>
#include <vector>

class SharedKernelContext {
 public:
  using ResultConsumer = std::function<void(ResultSetPtr&&, std::vector<size_t>&&)>;

  SharedKernelContext(const std::vector<InputTableInfo>& query_infos)
      : query_infos_(query_infos)
#ifdef HAVE_TBB
      , task_group_(nullptr)
#endif
  {
  }

  const std::vector<uint64_t>& getFragOffsets();

  void addDeviceResults(ResultSetPtr&& device_results,
                        std::vector<size_t> outer_table_fragment_ids);

  std::vector<std::pair<ResultSetPtr, std::vector<size_t>>>& getFragmentResults();

  void setResultConsumer(ResultConsumer result_consumer);

  void clearResultConsumer();

  bool hasResultConsumer();

  const std::vector<InputTableInfo>& getQueryInfos() const {
    return query_infos_;
  }

  void setNumAllocatedThreads(size_t num_threads) {
    num_allocated_threads_ = num_threads;
  }

  size_t getNumAllocatedThreads() {
    return num_allocated_threads_;
  }

  std::atomic_flag dynamic_watchdog_set = ATOMIC_FLAG_INIT;

#ifdef HAVE_TBB
  auto getThreadPool() {
    return task_group_;
  }
  void setThreadPool(threading::task_group* tg) {
    task_group_ = tg;
  }
  void clearThreadExecutionContexts() {
    thread_execution_contexts_.clear();
  }
  void resetThreadExecutionContexts(const size_t num_threads) {
    thread_execution_contexts_.resize(num_threads);
  }
  auto& getExecutionContextForThread(const size_t thread_idx) {
    CHECK_LT(thread_idx, thread_execution_contexts_.size());
    return thread_execution_contexts_[thread_idx];
  }
  auto& getThreadExecutionContexts() {
    return thread_execution_contexts_;
  }
#endif  // HAVE_TBB

 private:
  std::mutex reduce_mutex_;
  ResultConsumer result_consumer_;
  std::vector<std::pair<ResultSetPtr, std::vector<size_t>>> all_fragment_results_;

  std::vector<uint64_t> all_frag_row_offsets_;
  std::mutex all_frag_row_offsets_mutex_;
  const std::vector<InputTableInfo>& query_infos_;
  const RegisteredQueryHint query_hint_;
  // the # threads to execute the query (kernel) w/ a value one by default (means serial
  // query execution). After finishing the compilation of the kernel, we will set it to a
  // proper value based on the query's status
  size_t num_allocated_threads_{1};

#ifdef HAVE_TBB
  threading::task_group* task_group_;
  // TBB runs at most one task in each arena slot at a time. Key execution contexts by
  // that same slot because reusable CPU group-by buffers use the slot as their owner.
  std::vector<std::unique_ptr<QueryExecutionContext>> thread_execution_contexts_;
#endif  // HAVE_TBB
};

class ExecutionKernel {
 public:
  ExecutionKernel(const RelAlgExecutionUnit& ra_exe_unit,
                  const ExecutorDeviceType chosen_device_type,
                  int chosen_device_id,
                  const ExecutionOptions& eo,
                  const ColumnFetcher& column_fetcher,
                  const QueryCompilationDescriptor& query_comp_desc,
                  const QueryMemoryDescriptor& query_mem_desc,
                  const FragmentsList& frag_list,
                  const ExecutorDispatchMode kernel_dispatch_mode,
                  RenderInfo* render_info,
                  const int64_t rowid_lookup_key)
      : ra_exe_unit_(ra_exe_unit)
      , chosen_device_type(chosen_device_type)
      , chosen_device_id(chosen_device_id)
      , eo(eo)
      , column_fetcher(column_fetcher)
      , query_comp_desc(query_comp_desc)
      , query_mem_desc(query_mem_desc)
      , frag_list(frag_list)
      , kernel_dispatch_mode(kernel_dispatch_mode)
      , render_info_(render_info)
      , rowid_lookup_key(rowid_lookup_key) {}

  void run(Executor* executor,
           const size_t thread_idx,
           SharedKernelContext& shared_context);

  void setDeferredSparseBaselineFilterBeforeCopy(std::vector<int64_t> preserved_keys);
  bool applyDeferredSparseBaselineFilterBeforeCopy() const {
    return apply_deferred_sparse_baseline_filter_before_copy_;
  }
  const std::vector<int64_t>& deferredSparseBaselinePreservedKeys() const {
    return deferred_sparse_baseline_preserved_keys_;
  }

  FragmentsList get_fragment_list() const { return frag_list; }
  int32_t get_chosen_device_id() const { return chosen_device_id; }
  const RelAlgExecutionUnit& ra_exe_unit_;

 private:
  const ExecutorDeviceType chosen_device_type;
  int chosen_device_id;
  const ExecutionOptions& eo;
  const ColumnFetcher& column_fetcher;
  const QueryCompilationDescriptor& query_comp_desc;
  const QueryMemoryDescriptor& query_mem_desc;
  const FragmentsList frag_list;
  const ExecutorDispatchMode kernel_dispatch_mode;
  RenderInfo* render_info_;
  const int64_t rowid_lookup_key;

  ResultSetPtr device_results_;
  bool apply_deferred_sparse_baseline_filter_before_copy_{false};
  std::vector<int64_t> deferred_sparse_baseline_preserved_keys_;

  void runImpl(Executor* executor,
               const size_t thread_idx,
               SharedKernelContext& shared_context);

  friend class KernelSubtask;
};

#ifdef HAVE_TBB
class KernelSubtask {
 public:
  KernelSubtask(ExecutionKernel& k,
                SharedKernelContext& shared_context,
                std::shared_ptr<FetchResult> fetch_result,
                std::shared_ptr<std::list<ChunkIter>> chunk_iterators,
                int64_t total_num_input_rows,
                size_t start_rowid,
                size_t num_rows_to_process,
                size_t thread_idx)
      : kernel_(k)
      , shared_context_(shared_context)
      , fetch_result_(fetch_result)
      , chunk_iterators_(chunk_iterators)
      , total_num_input_rows_(total_num_input_rows)
      , start_rowid_(start_rowid)
      , num_rows_to_process_(num_rows_to_process)
      , thread_idx_(thread_idx) {}

  void run(Executor* executor);

 private:
  void runImpl(Executor* executor);

  ExecutionKernel& kernel_;
  SharedKernelContext& shared_context_;
  std::shared_ptr<FetchResult> fetch_result_;
  std::shared_ptr<std::list<ChunkIter>> chunk_iterators_;
  int64_t total_num_input_rows_;
  size_t start_rowid_;
  size_t num_rows_to_process_;
  size_t thread_idx_;
};
#endif  // HAVE_TBB
