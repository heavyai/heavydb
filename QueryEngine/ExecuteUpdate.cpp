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

#include "QueryEngine/Execute.h"

#include "QueryEngine/ColumnFetcher.h"
#include "QueryEngine/Descriptors/QueryCompilationDescriptor.h"
#include "QueryEngine/Descriptors/QueryFragmentDescriptor.h"
#include "QueryEngine/ExecutionKernel.h"
#include "QueryEngine/RelAlgExecutor.h"
#include "QueryEngine/TableOptimizer.h"

extern bool g_enable_auto_metadata_update;

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

TableUpdateMetadata Executor::executeUpdate(
    const RelAlgExecutionUnit& ra_exe_unit_in,
    const std::vector<InputTableInfo>& table_infos,
    const TableDescriptor* table_desc_for_update,
    const CompilationOptions& co,
    const ExecutionOptions& eo,
    const Catalog_Namespace::Catalog& cat,
    std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
    const UpdateLogForFragment::Callback& cb,
    const bool is_agg) {
  CHECK(cb);
  CHECK(table_desc_for_update);
  VLOG(1) << "Executor " << executor_id_
          << " is executing update/delete work unit:" << ra_exe_unit_in;

  const auto [ra_exe_unit, deleted_cols_map] = addDeletedColumn(ra_exe_unit_in, co);
  ColumnCacheMap column_cache;

  ColumnFetcher column_fetcher(this, column_cache);
  CHECK_GT(ra_exe_unit.input_descs.size(), size_t(0));
  const auto& outer_table_key = ra_exe_unit.input_descs[0].getTableKey();
  CHECK_EQ(outer_table_key, table_infos.front().table_key);
  const auto& outer_fragments = table_infos.front().info.fragments;

  std::vector<FragmentsPerTable> fragments = {{{0, 0}, {0}}};
  for (size_t tab_idx = 1; tab_idx < ra_exe_unit.input_descs.size(); tab_idx++) {
    const auto& table_key = ra_exe_unit.input_descs[tab_idx].getTableKey();
    CHECK_EQ(table_infos[tab_idx].table_key, table_key);
    const auto& fragmentsPerTable = table_infos[tab_idx].info.fragments;
    FragmentsPerTable entry = {table_key, {}};
    for (size_t innerFragId = 0; innerFragId < fragmentsPerTable.size(); innerFragId++) {
      entry.fragment_ids.push_back(innerFragId);
    }
    fragments.push_back(entry);
  }

  if (outer_fragments.empty()) {
    return {};
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
    query_mem_desc = query_comp_desc->compile(global_max_groups_buffer_entry_guess,
                                              8,
                                              /*has_cardinality_estimation=*/true,
                                              ra_exe_unit,
                                              table_infos,
                                              deleted_cols_map,
                                              column_fetcher,
                                              co,
                                              eo,
                                              nullptr,
                                              this);
  }
  CHECK(query_mem_desc);

  TableUpdateMetadata table_update_metadata;
  for (size_t fragment_index = 0; fragment_index < outer_fragments.size();
       ++fragment_index) {
    const int64_t crt_fragment_tuple_count =
        outer_fragments[fragment_index].getNumTuples();
    if (crt_fragment_tuple_count == 0) {
      // nothing to update
      continue;
    }
    SharedKernelContext shared_context(table_infos);
    const auto& frag_offsets = shared_context.getFragOffsets();
    auto skip_frag = skipFragment(ra_exe_unit.input_descs[0],
                                  outer_fragments[fragment_index],
                                  ra_exe_unit.simple_quals,
                                  frag_offsets,
                                  fragment_index);
    if (skip_frag.first) {
      VLOG(2) << "Update/delete skipping fragment with table id: "
              << outer_fragments[fragment_index].physicalTableId
              << ", fragment id: " << fragment_index;
      continue;
    }
    fragments[0] = {outer_table_key, {fragment_index}};

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
                                              /*rowid_lookup_key=*/-1);

      auto clock_begin = timer_start();
      std::lock_guard<std::mutex> kernel_lock(kernel_mutex_);
      kernel_queue_time_ms_ += timer_stop(clock_begin);

      current_fragment_kernel.run(this, 0, shared_context);
    }
    const auto& proj_fragment_results = shared_context.getFragmentResults();
    if (proj_fragment_results.empty()) {
      continue;
    }
    const auto& proj_fragment_result = proj_fragment_results[0];
    const auto proj_result_set = proj_fragment_result.first;
    CHECK(proj_result_set);
    cb({outer_fragments[fragment_index], fragment_index, proj_result_set},
       table_update_metadata);
  }

  if (g_enable_auto_metadata_update) {
    auto td = cat.getMetadataForTable(table_desc_for_update->tableId);
    TableOptimizer table_optimizer{td, this, cat};
    table_optimizer.recomputeMetadataUnlocked(table_update_metadata);
  }
  return table_update_metadata;
}
