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

#include "QueryEngine/ColumnFetcher.h"
#include "QueryEngine/Descriptors/QueryCompilationDescriptor.h"
#include "Shared/Logger.h"

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

  void run(Executor* executor, SharedKernelContext& shared_context);

 private:
  const RelAlgExecutionUnit& ra_exe_unit_;
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

  void runImpl(Executor* executor, SharedKernelContext& shared_context);
};
