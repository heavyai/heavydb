/*
 * Copyright 2018 MapD Technologies, Inc.
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

#include "QueryFragmentDescriptor.h"

#include "Execute.h"

QueryFragmentDescriptor::QueryFragmentDescriptor(
    const RelAlgExecutionUnit& ra_exe_unit,
    const std::vector<InputTableInfo>& query_infos,
    Executor* executor)
    : executor_(executor) {
  const size_t input_desc_count{ra_exe_unit.input_descs.size()};
  CHECK_EQ(query_infos.size(), (input_desc_count + ra_exe_unit.extra_input_descs.size()));
  for (size_t table_idx = 0; table_idx < input_desc_count; ++table_idx) {
    const auto table_id = ra_exe_unit.input_descs[table_idx].getTableId();
    if (!selected_tables_fragments_.count(table_id)) {
      selected_tables_fragments_[ra_exe_unit.input_descs[table_idx].getTableId()] =
          &query_infos[table_idx].info.fragments;
    }
  }
}

void QueryFragmentDescriptor::buildFragmentDeviceMap(
    const RelAlgExecutionUnit& ra_exe_unit,
    const std::vector<uint64_t>& frag_offsets,
    const int device_count,
    const ExecutorDeviceType& device_type,
    const bool enable_multifrag_kernels,
    const bool enable_inner_join_fragment_skipping) {
  if (enable_multifrag_kernels) {
    buildMultifragKernelMap(ra_exe_unit,
                            frag_offsets,
                            device_count,
                            device_type,
                            enable_inner_join_fragment_skipping);
  } else {
    buildFragmentPerKernelMap(ra_exe_unit, frag_offsets, device_count, device_type);
  }
}

void QueryFragmentDescriptor::buildFragmentPerKernelMap(
    const RelAlgExecutionUnit& ra_exe_unit,
    const std::vector<uint64_t>& frag_offsets,
    const int device_count,
    const ExecutorDeviceType& device_type) {
  const auto& outer_table_desc = ra_exe_unit.input_descs.front();
  const int outer_table_id = outer_table_desc.getTableId();
  auto it = selected_tables_fragments_.find(outer_table_id);
  CHECK(it != selected_tables_fragments_.end());
  const auto outer_fragments = it->second;
  outer_fragments_size_ = outer_fragments->size();

  for (size_t i = 0; i < outer_fragments->size(); ++i) {
    const auto& fragment = (*outer_fragments)[i];
    outer_fragment_tuple_sizes_.push_back(fragment.getNumTuples());
    const auto skip_frag = executor_->skipFragment(
        outer_table_desc, fragment, ra_exe_unit.simple_quals, frag_offsets, i);
    if (skip_frag.first) {
      continue;
    }
    rowid_lookup_key_ = std::max(rowid_lookup_key_, skip_frag.second);
    const int chosen_device_count =
        device_type == ExecutorDeviceType::CPU ? 1 : device_count;
    CHECK_GT(chosen_device_count, 0);
    const auto memory_level = device_type == ExecutorDeviceType::GPU
                                  ? Data_Namespace::GPU_LEVEL
                                  : Data_Namespace::CPU_LEVEL;
    int device_id = (device_type == ExecutorDeviceType::CPU || fragment.shard == -1)
                        ? fragment.deviceIds[static_cast<int>(memory_level)]
                        : fragment.shard % chosen_device_count;

    // TODO(adb): Need to better index the kernel per fragment case
    CHECK_EQ(fragments_per_device_[i].size(), 0);
    outer_fragments_to_device_id_.insert(std::make_pair(i, device_id));

    for (size_t j = 0; j < ra_exe_unit.input_descs.size(); ++j) {
      const auto frag_ids =
          executor_->getTableFragmentIndices(ra_exe_unit,
                                             device_type,
                                             j,
                                             i,
                                             selected_tables_fragments_,
                                             executor_->getInnerTabIdToJoinCond());
      const auto table_id = ra_exe_unit.input_descs[j].getTableId();
      auto table_frags_it = selected_tables_fragments_.find(table_id);
      CHECK(table_frags_it != selected_tables_fragments_.end());

      fragments_per_device_[i].emplace_back(FragmentsPerTable{table_id, frag_ids});
    }
  }
}

void QueryFragmentDescriptor::buildMultifragKernelMap(
    const RelAlgExecutionUnit& ra_exe_unit,
    const std::vector<uint64_t>& frag_offsets,
    const int device_count,
    const ExecutorDeviceType& device_type,
    const bool enable_inner_join_fragment_skipping) {
  // Allocate all the fragments of the tables involved in the query to available
  // devices. The basic idea: the device is decided by the outer table in the
  // query (the first table in a join) and we need to broadcast the fragments
  // in the inner table to each device. Sharding will change this model.
  const auto& outer_table_desc = ra_exe_unit.input_descs.front();
  const int outer_table_id = outer_table_desc.getTableId();
  auto it = selected_tables_fragments_.find(outer_table_id);
  CHECK(it != selected_tables_fragments_.end());
  const auto outer_fragments = it->second;
  outer_fragments_size_ = outer_fragments->size();

  const auto inner_table_id_to_join_condition = executor_->getInnerTabIdToJoinCond();

  for (size_t outer_frag_id = 0; outer_frag_id < outer_fragments->size();
       ++outer_frag_id) {
    const auto& fragment = (*outer_fragments)[outer_frag_id];
    auto skip_frag = executor_->skipFragment(outer_table_desc,
                                             fragment,
                                             ra_exe_unit.simple_quals,
                                             frag_offsets,
                                             outer_frag_id);
    if (enable_inner_join_fragment_skipping &&
        (skip_frag == std::pair<bool, int64_t>(false, -1))) {
      skip_frag = executor_->skipFragmentInnerJoins(
          outer_table_desc, ra_exe_unit, fragment, frag_offsets, outer_frag_id);
    }
    if (skip_frag.first) {
      continue;
    }
    const int device_id =
        fragment.shard == -1
            ? fragment.deviceIds[static_cast<int>(Data_Namespace::GPU_LEVEL)]
            : fragment.shard % device_count;
    for (size_t j = 0; j < ra_exe_unit.input_descs.size(); ++j) {
      const auto table_id = ra_exe_unit.input_descs[j].getTableId();
      auto table_frags_it = selected_tables_fragments_.find(table_id);
      CHECK(table_frags_it != selected_tables_fragments_.end());
      const auto frag_ids =
          executor_->getTableFragmentIndices(ra_exe_unit,
                                             device_type,
                                             j,
                                             outer_frag_id,
                                             selected_tables_fragments_,
                                             inner_table_id_to_join_condition);
      if (fragments_per_device_[device_id].size() < j + 1) {
        fragments_per_device_[device_id].emplace_back(
            FragmentsPerTable{table_id, frag_ids});
      } else {
        CHECK_EQ(fragments_per_device_[device_id][j].table_id, table_id);
        auto& curr_frag_ids = fragments_per_device_[device_id][j].fragment_ids;
        for (const int frag_id : frag_ids) {
          if (std::find(curr_frag_ids.begin(), curr_frag_ids.end(), frag_id) ==
              curr_frag_ids.end()) {
            curr_frag_ids.push_back(frag_id);
          }
        }
      }
    }
    rowid_lookup_key_ = std::max(rowid_lookup_key_, skip_frag.second);
  }
}
