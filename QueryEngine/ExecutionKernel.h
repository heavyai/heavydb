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

#pragma once

#include "Logger/Logger.h"
#include "QueryEngine/ColumnFetcher.h"
#include "QueryEngine/Descriptors/QueryCompilationDescriptor.h"

class SharedKernelContext {
 public:
  SharedKernelContext(const std::vector<InputTableInfo>& query_infos)
      : query_infos_(query_infos) {}

  const std::vector<uint64_t>& getFragOffsets();

  void addDeviceResults(ResultSetPtr&& device_results,
                        std::vector<size_t> outer_table_fragment_ids);

  std::vector<std::pair<ResultSetPtr, std::vector<size_t>>>& getFragmentResults();

  const std::vector<InputTableInfo>& getQueryInfos() const { return query_infos_; }

  std::atomic_flag dynamic_watchdog_set = ATOMIC_FLAG_INIT;

 private:
  std::mutex reduce_mutex_;
  std::vector<std::pair<ResultSetPtr, std::vector<size_t>>> all_fragment_results_;

  std::vector<uint64_t> all_frag_row_offsets_;
  std::mutex all_frag_row_offsets_mutex_;
  const std::vector<InputTableInfo>& query_infos_;
  const QueryHint query_hint_;
};

class ExecutionKernel {
 public:
  ExecutionKernel(const RelAlgExecutionUnit& ra_exe_unit,
                  const ExecutorDeviceType chosen_device_type,
                  const int chosen_device_id,
                  const ExecutionOptions& eo,
                  const ColumnFetcher& column_fetcher,
                  const QueryCompilationDescriptor& query_comp_desc,
                  const QueryMemoryDescriptor& query_mem_desc,
                  const FragmentsList& frag_list,
                  const ExecutorDispatchMode kernel_dispatch_mode,
                  RenderInfo* render_info,
                  const int64_t rowid_lookup_key)
      : ra_exe_unit_(ra_exe_unit)
      , chosen_device_type_(chosen_device_type)
      , chosen_device_id_(chosen_device_id)
      , eo_(eo)
      , column_fetcher_(column_fetcher)
      , query_comp_desc_(query_comp_desc)
      , query_mem_desc_(query_mem_desc)
      , frag_list_(frag_list)
      , kernel_dispatch_mode_(kernel_dispatch_mode)
      , render_info_(render_info)
      , rowid_lookup_key_(rowid_lookup_key) {}

  void run(Executor* executor,
           const size_t thread_idx,
           SharedKernelContext& shared_context);

 private:
  const RelAlgExecutionUnit& ra_exe_unit_;
  const ExecutorDeviceType chosen_device_type_;
  const int chosen_device_id_;
  const ExecutionOptions& eo_;
  const ColumnFetcher& column_fetcher_;
  const QueryCompilationDescriptor& query_comp_desc_;
  const QueryMemoryDescriptor& query_mem_desc_;
  const FragmentsList frag_list_;
  const ExecutorDispatchMode kernel_dispatch_mode_;
  RenderInfo* render_info_;
  const int64_t rowid_lookup_key_;

  void runImpl(Executor* executor,
               const size_t thread_idx,
               SharedKernelContext& shared_context);
};
