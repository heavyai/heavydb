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

#include "Execute.h"
#include "QueryFragmentDescriptor.h"

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

size_t UpdateLogForFragment::count() const {
  return getEntryCount();
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
                             const InputTableInfo& table_info,
                             const CompilationOptions& co,
                             const ExecutionOptions& eo,
                             const Catalog_Namespace::Catalog& cat,
                             std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                             const UpdateLogForFragment::Callback& cb) {
  CHECK(cb);
  const auto ra_exe_unit = addDeletedColumn(ra_exe_unit_in);

  // could use std::thread::hardware_concurrency(), but some
  // slightly out-of-date compilers (gcc 4.7) implement it as always 0.
  // Play it POSIX.1 safe instead.
  int available_cpus = cpu_threads();
  auto available_gpus = get_available_gpus(cat);

  const auto context_count =
      get_context_count(co.device_type_, available_cpus, available_gpus.size());

  int error_code = 0;
  ColumnCacheMap column_cache;

  const auto count =
      makeExpr<Analyzer::AggExpr>(SQLTypeInfo(g_bigint_count ? kBIGINT : kINT, false),
                                  kCOUNT,
                                  nullptr,
                                  false,
                                  nullptr);
  const auto count_all_exe_unit = create_count_all_execution_unit(ra_exe_unit, count);

  std::vector<InputTableInfo> table_infos{table_info};
  ExecutionDispatch execution_dispatch(this,
                                       count_all_exe_unit,
                                       table_infos,
                                       cat,
                                       co,
                                       context_count,
                                       row_set_mem_owner,
                                       column_cache,
                                       &error_code,
                                       nullptr);
  execution_dispatch.compile(0, 8, eo, false);
  CHECK_EQ(size_t(1), ra_exe_unit.input_descs.size());
  const auto table_id = ra_exe_unit.input_descs[0].getTableId();
  const auto& outer_fragments = table_info.info.fragments;
  for (size_t fragment_index = 0; fragment_index < outer_fragments.size();
       ++fragment_index) {
    // We may want to consider in the future allowing this to execute on devices other
    // than CPU
    execution_dispatch.run(co.device_type_, 0, eo, {{table_id, {fragment_index}}}, 0, -1);
  }
  // Further optimization possible here to skip fragments
  CHECK_EQ(outer_fragments.size(), execution_dispatch.getFragmentResults().size());
  // There could be benefit to multithread this once we see where the bottle necks really
  // are
  for (size_t fragment_index = 0; fragment_index < outer_fragments.size();
       ++fragment_index) {
    const auto& fragment_results =
        execution_dispatch.getFragmentResults()[fragment_index];
    const auto count_result_set = fragment_results.first;
    CHECK(count_result_set);
    const auto count_row = count_result_set->getNextRow(false, false);
    CHECK_EQ(size_t(1), count_row.size());
    const auto& count_tv = count_row.front();
    const auto count_scalar_tv = boost::get<ScalarTargetValue>(&count_tv);
    CHECK(count_scalar_tv);
    const auto count_ptr = boost::get<int64_t>(count_scalar_tv);
    CHECK(count_ptr);
    ExecutionDispatch current_fragment_execution_dispatch(this,
                                                          ra_exe_unit,
                                                          table_infos,
                                                          cat,
                                                          co,
                                                          context_count,
                                                          row_set_mem_owner,
                                                          column_cache,
                                                          &error_code,
                                                          nullptr);
    current_fragment_execution_dispatch.compile(*count_ptr, 8, eo, false);
    // We may want to consider in the future allowing this to execute on devices other
    // than CPU
    current_fragment_execution_dispatch.run(
        co.device_type_, 0, eo, {FragmentsPerTable{table_id, {fragment_index}}}, 0, -1);
    const auto& proj_fragment_results =
        current_fragment_execution_dispatch.getFragmentResults()[0];
    const auto proj_result_set = proj_fragment_results.first;
    CHECK(proj_result_set);
    cb({outer_fragments[fragment_index], fragment_index, proj_result_set});
  }
}
