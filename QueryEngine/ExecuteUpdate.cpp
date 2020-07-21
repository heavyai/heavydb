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

#include "QueryEngine/Execute.h"

#include "QueryEngine/ColumnFetcher.h"
#include "QueryEngine/Descriptors/QueryCompilationDescriptor.h"
#include "QueryEngine/Descriptors/QueryFragmentDescriptor.h"
#include "QueryEngine/ExecutionKernel.h"
#include "QueryEngine/RelAlgExecutor.h"

UpdateLogForFragment::UpdateLogForFragment(FragmentInfoType const& fragment_info,
                                           size_t const fragment_index,
                                           const std::shared_ptr<ResultSet>& rs)
    : fragment_info_(fragment_info), fragment_index_(fragment_index), rs_(rs) {
  rs->setGeoReturnType(ResultSet::GeoReturnType::GeoTargetValue);
}

std::vector<TargetValue> UpdateLogForFragment::getEntryAt(const size_t index) const {
  return rs_->getRowAtNoTranslations(index);
}

std::vector<TargetValue> UpdateLogForFragment::getTranslatedEntryAt(
    const size_t index) const {
  return rs_->getRowAt(index);
}

size_t const UpdateLogForFragment::getRowCount() const {
  return rs_->rowCount();
}

UpdateLogForFragment::FragmentInfoType const& UpdateLogForFragment::getFragmentInfo()
    const {
  return fragment_info_;
}

size_t const UpdateLogForFragment::getEntryCount() const {
  return rs_->entryCount();
}

size_t const UpdateLogForFragment::getFragmentIndex() const {
  return fragment_index_;
}

SQLTypeInfo UpdateLogForFragment::getColumnType(const size_t col_idx) const {
  return rs_->getColType(col_idx);
}

void Executor::executeUpdate(const RelAlgExecutionUnit& ra_exe_unit_in,
                             const std::vector<InputTableInfo>& table_infos,
                             const CompilationOptions& co,
                             const ExecutionOptions& eo,
                             const Catalog_Namespace::Catalog& cat,
                             std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                             const UpdateLogForFragment::Callback& cb,
                             const bool is_agg) {
  CHECK(cb);
  VLOG(1) << "Executing update/delete work unit:" << ra_exe_unit_in;

  const auto ra_exe_unit = addDeletedColumn(ra_exe_unit_in, co);
  ColumnCacheMap column_cache;

  ColumnFetcher column_fetcher(this, column_cache);
  CHECK_GT(ra_exe_unit.input_descs.size(), size_t(0));
  const auto table_id = ra_exe_unit.input_descs[0].getTableId();
  const auto& outer_fragments = table_infos.front().info.fragments;

  std::vector<FragmentsPerTable> fragments = {{0, {0}}};
  for (size_t tab_idx = 1; tab_idx < ra_exe_unit.input_descs.size(); tab_idx++) {
    int table_id = ra_exe_unit.input_descs[tab_idx].getTableId();
    CHECK_EQ(table_infos[tab_idx].table_id, table_id);
    const auto& fragmentsPerTable = table_infos[tab_idx].info.fragments;
    FragmentsPerTable entry = {table_id, {}};
    for (size_t innerFragId = 0; innerFragId < fragmentsPerTable.size(); innerFragId++) {
      entry.fragment_ids.push_back(innerFragId);
    }
    fragments.push_back(entry);
  }

  if (outer_fragments.empty()) {
    return;
  }

  const auto max_tuple_count_fragment_it = std::max_element(
      outer_fragments.begin(), outer_fragments.end(), [](const auto& a, const auto& b) {
        return a.getNumTuples() < b.getNumTuples();
      });
  CHECK(max_tuple_count_fragment_it != outer_fragments.end());
  int64_t global_max_groups_buffer_entry_guess =
      max_tuple_count_fragment_it->getNumTuples();
  if (is_agg) {
    global_max_groups_buffer_entry_guess = std::min(
        2 * global_max_groups_buffer_entry_guess, static_cast<int64_t>(100'000'000));
  }

  auto query_comp_desc = std::make_unique<QueryCompilationDescriptor>();
  std::unique_ptr<QueryMemoryDescriptor> query_mem_desc;
  {
    auto clock_begin = timer_start();
    std::lock_guard<std::mutex> compilation_lock(compilation_mutex_);
    compilation_queue_time_ms_ += timer_stop(clock_begin);

    query_mem_desc = query_comp_desc->compile(global_max_groups_buffer_entry_guess,
                                              8,
                                              /*has_cardinality_estimation=*/true,
                                              ra_exe_unit,
                                              table_infos,
                                              column_fetcher,
                                              co,
                                              eo,
                                              nullptr,
                                              this);
  }
  CHECK(query_mem_desc);

  for (size_t fragment_index = 0; fragment_index < outer_fragments.size();
       ++fragment_index) {
    const int64_t crt_fragment_tuple_count =
        outer_fragments[fragment_index].getNumTuples();
    if (crt_fragment_tuple_count == 0) {
      // nothing to update
      continue;
    }

    SharedKernelContext shared_context(table_infos);
    fragments[0] = {table_id, {fragment_index}};
    {
      ExecutionKernel current_fragment_kernel(ra_exe_unit,
                                              ExecutorDeviceType::CPU,
                                              0,
                                              eo,
                                              column_fetcher,
                                              *query_comp_desc,
                                              *query_mem_desc,
                                              fragments,
                                              ExecutorDispatchMode::KernelPerFragment,
                                              /*render_info=*/nullptr,
                                              /*rowid_lookup_key=*/-1,
                                              logger::thread_id());

      auto clock_begin = timer_start();
      std::lock_guard<std::mutex> kernel_lock(kernel_mutex_);
      kernel_queue_time_ms_ += timer_stop(clock_begin);

      current_fragment_kernel.run(this, shared_context);
    }
    const auto& proj_fragment_results = shared_context.getFragmentResults();
    if (proj_fragment_results.empty()) {
      continue;
    }
    const auto& proj_fragment_result = proj_fragment_results[0];
    const auto proj_result_set = proj_fragment_result.first;
    CHECK(proj_result_set);
    cb({outer_fragments[fragment_index], fragment_index, proj_result_set});
  }
}
