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

#include "ColumnFetcher.h"
#include "Descriptors/QueryCompilationDescriptor.h"
#include "Descriptors/QueryFragmentDescriptor.h"
#include "Execute.h"
#include "RelAlgExecutor.h"

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
  const auto ra_exe_unit = addDeletedColumn(ra_exe_unit_in);
  ColumnCacheMap column_cache;

  const auto count =
      makeExpr<Analyzer::AggExpr>(SQLTypeInfo(g_bigint_count ? kBIGINT : kINT, false),
                                  kCOUNT,
                                  nullptr,
                                  false,
                                  nullptr);
  const auto count_all_exe_unit = create_count_all_execution_unit(ra_exe_unit, count);

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

  // There could be benefit to multithread this once we see where the bottle necks really
  // are
  for (size_t fragment_index = 0; fragment_index < outer_fragments.size();
       ++fragment_index) {
    ExecutionDispatch current_fragment_execution_dispatch(
        this, ra_exe_unit, table_infos, cat, row_set_mem_owner, nullptr);

    const int64_t crt_fragment_tuple_count =
        outer_fragments[fragment_index].getNumTuples();
    int64_t max_groups_buffer_entry_guess = crt_fragment_tuple_count;
    if (is_agg) {
      max_groups_buffer_entry_guess =
          std::min(2 * max_groups_buffer_entry_guess, static_cast<int64_t>(100'000'000));
    }

    const auto execution_descriptors = current_fragment_execution_dispatch.compile(
        max_groups_buffer_entry_guess, 8, co, eo, column_fetcher, true);
    // We may want to consider in the future allowing this to execute on devices other
    // than CPU

    fragments[0] = {table_id, {fragment_index}};

    current_fragment_execution_dispatch.run(
        co.device_type_,
        0,
        eo,
        column_fetcher,
        *std::get<QueryCompilationDescriptorOwned>(execution_descriptors),
        *std::get<QueryMemoryDescriptorOwned>(execution_descriptors),
        fragments,
        ExecutorDispatchMode::KernelPerFragment,
        -1);
    const auto& proj_fragment_results =
        current_fragment_execution_dispatch.getFragmentResults();
    if (proj_fragment_results.empty()) {
      continue;
    }
    const auto& proj_fragment_result = proj_fragment_results[0];
    const auto proj_result_set = proj_fragment_result.first;
    CHECK(proj_result_set);
    cb({outer_fragments[fragment_index], fragment_index, proj_result_set});
  }
}
