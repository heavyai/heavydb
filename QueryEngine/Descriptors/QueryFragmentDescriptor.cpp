/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "QueryFragmentDescriptor.h"

#include "Catalog/ColumnDescriptor.h"
#include "Catalog/TableDescriptor.h"
#include "DataMgr/DataMgr.h"
#include "QueryEngine/ErrorHandling.h"
#include "QueryEngine/Execute.h"
#include "Shared/misc.h"

#include <limits>
#include <numeric>

extern bool g_enable_result_reduction_pipeline;

namespace {

bool is_projection_execution_unit(const RelAlgExecutionUnit& ra_exe_unit) {
  return ra_exe_unit.groupby_exprs.size() == size_t(1) &&
         !ra_exe_unit.groupby_exprs.front();
}

bool can_coalesce_gpu_projection_fragments(
    const RelAlgExecutionUnit& ra_exe_unit,
    const ExecutorDeviceType device_type,
    const heavyai::QueryDescriptionType query_description_type) {
  return device_type == ExecutorDeviceType::GPU &&
         query_description_type == heavyai::QueryDescriptionType::Projection &&
         is_projection_execution_unit(ra_exe_unit) &&
         ra_exe_unit.input_descs.size() == size_t(1) && !ra_exe_unit.union_all &&
         ra_exe_unit.scan_limit > 0 && ra_exe_unit.sort_info.order_entries.empty() &&
         !ra_exe_unit.sort_info.limit && ra_exe_unit.sort_info.offset == 0;
}

bool can_coalesce_gpu_groupby_fragments(
    const RelAlgExecutionUnit& ra_exe_unit,
    const InputDescriptor& table_desc,
    const ExecutorDeviceType device_type,
    const heavyai::QueryDescriptionType query_description_type,
    const std::optional<size_t> table_desc_offset,
    const size_t max_kernel_input_rows) {
  return device_type == ExecutorDeviceType::GPU &&
         (query_description_type == heavyai::QueryDescriptionType::GroupByBaselineHash ||
          query_description_type == heavyai::QueryDescriptionType::GroupByPerfectHash) &&
         (table_desc.getSourceType() == InputSourceType::TABLE ||
          table_desc.getSourceType() == InputSourceType::RESULT) &&
         ra_exe_unit.input_descs.size() == size_t(1) && !ra_exe_unit.union_all &&
         !ra_exe_unit.groupby_exprs.empty() &&
         !is_projection_execution_unit(ra_exe_unit) && !table_desc_offset &&
         max_kernel_input_rows > 0;
}

size_t gpu_input_row_limit(const std::set<int>& device_ids,
                           const std::map<size_t, size_t>& available_gpu_mem_bytes,
                           const double gpu_input_mem_limit_percent,
                           const size_t num_bytes_for_row) {
  if (!num_bytes_for_row) {
    return std::numeric_limits<size_t>::max();
  }
  size_t row_limit = std::numeric_limits<size_t>::max();
  for (const auto device_id : device_ids) {
    auto mem_it = available_gpu_mem_bytes.find(static_cast<size_t>(device_id));
    if (mem_it == available_gpu_mem_bytes.end()) {
      continue;
    }
    const auto gpu_bytes_limit = static_cast<size_t>(static_cast<double>(mem_it->second) *
                                                     gpu_input_mem_limit_percent);
    row_limit = std::min(row_limit, gpu_bytes_limit / num_bytes_for_row);
  }
  return row_limit == 0 ? size_t(1) : row_limit;
}

bool batched_kernel_input_fits(const size_t current_outer_tuple_count,
                               const size_t candidate_outer_tuple_count,
                               const size_t max_kernel_input_rows,
                               const size_t row_multiplier) {
  if (current_outer_tuple_count >
      std::numeric_limits<size_t>::max() - candidate_outer_tuple_count) {
    return false;
  }
  const auto batched_outer_tuple_count =
      current_outer_tuple_count + candidate_outer_tuple_count;
  if (batched_outer_tuple_count > std::numeric_limits<size_t>::max() / row_multiplier) {
    return false;
  }
  return batched_outer_tuple_count * row_multiplier <= max_kernel_input_rows;
}

bool projection_output_bounded_by_outer_table(const RelAlgExecutionUnit& ra_exe_unit) {
  return !ra_exe_unit.join_quals.empty() &&
         std::all_of(ra_exe_unit.join_quals.begin(),
                     ra_exe_unit.join_quals.end(),
                     [](const auto& join_condition) {
                       return join_condition.type == JoinType::SEMI ||
                              join_condition.type == JoinType::ANTI;
                     });
}

void append_unique_fragment_ids(std::vector<size_t>& dst,
                                const std::vector<size_t>& src) {
  for (const auto fragment_id : src) {
    if (std::find(dst.begin(), dst.end(), fragment_id) == dst.end()) {
      dst.push_back(fragment_id);
    }
  }
}

bool try_append_gpu_fragment_kernel(ExecutionKernelDescriptor& dst,
                                    const ExecutionKernelDescriptor& src,
                                    const size_t max_kernel_input_rows,
                                    const size_t row_multiplier) {
  if (dst.device_id != src.device_id || !dst.outer_tuple_count ||
      !src.outer_tuple_count ||
      !batched_kernel_input_fits(*dst.outer_tuple_count,
                                 *src.outer_tuple_count,
                                 max_kernel_input_rows,
                                 row_multiplier) ||
      dst.fragments.size() != src.fragments.size()) {
    return false;
  }

  for (size_t table_idx = 0; table_idx < dst.fragments.size(); ++table_idx) {
    if (dst.fragments[table_idx].table_key != src.fragments[table_idx].table_key) {
      return false;
    }
  }

  for (size_t table_idx = 0; table_idx < dst.fragments.size(); ++table_idx) {
    append_unique_fragment_ids(dst.fragments[table_idx].fragment_ids,
                               src.fragments[table_idx].fragment_ids);
  }
  *dst.outer_tuple_count += *src.outer_tuple_count;
  return true;
}

}  // namespace

QueryFragmentDescriptor::QueryFragmentDescriptor(
    const RelAlgExecutionUnit& ra_exe_unit,
    const std::vector<InputTableInfo>& query_infos,
    const std::vector<Buffer_Namespace::MemoryInfo>& gpu_mem_infos,
    const double gpu_input_mem_limit_percent,
    std::vector<size_t> allowed_outer_fragment_indices)
    : allowed_outer_fragment_indices_(allowed_outer_fragment_indices)
    , gpu_input_mem_limit_percent_(gpu_input_mem_limit_percent) {
  const size_t input_desc_count{ra_exe_unit.input_descs.size()};
  CHECK_EQ(query_infos.size(), input_desc_count);
  for (size_t table_idx = 0; table_idx < input_desc_count; ++table_idx) {
    const auto& table_key = ra_exe_unit.input_descs[table_idx].getTableKey();
    if (!selected_tables_fragments_.count(table_key)) {
      selected_tables_fragments_[table_key] = &query_infos[table_idx].info.fragments;
    }
  }

  for (size_t device_id = 0; device_id < gpu_mem_infos.size(); device_id++) {
    const auto& gpu_mem_info = gpu_mem_infos[device_id];
    available_gpu_mem_bytes_[device_id] =
        gpu_mem_info.max_num_pages * gpu_mem_info.page_size;
  }
}

void QueryFragmentDescriptor::computeAllTablesFragments(
    std::map<shared::TableKey, const TableFragments*>& all_tables_fragments,
    const RelAlgExecutionUnit& ra_exe_unit,
    const std::vector<InputTableInfo>& query_infos) {
  for (size_t tab_idx = 0; tab_idx < ra_exe_unit.input_descs.size(); ++tab_idx) {
    const auto& table_key = ra_exe_unit.input_descs[tab_idx].getTableKey();
    CHECK_EQ(query_infos[tab_idx].table_key, table_key);
    const auto& fragments = query_infos[tab_idx].info.fragments;
    if (!all_tables_fragments.count(table_key)) {
      all_tables_fragments.insert(std::make_pair(table_key, &fragments));
    }
  }
}

void QueryFragmentDescriptor::buildFragmentKernelMap(
    const RelAlgExecutionUnit& ra_exe_unit,
    const std::vector<uint64_t>& frag_offsets,
    const std::set<int>& device_ids,
    const ExecutorDeviceType& device_type,
    const heavyai::QueryDescriptionType query_description_type,
    const size_t max_kernel_input_rows,
    const bool uses_lazy_fetch,
    const bool enable_multifrag_kernels,
    const bool enable_inner_join_fragment_skipping,
    Executor* executor) {
  // For joins, only consider the cardinality of the LHS
  // columns in the bytes per row count.
  std::set<shared::TableKey> lhs_table_keys;
  for (const auto& input_desc : ra_exe_unit.input_descs) {
    if (input_desc.getNestLevel() == 0) {
      lhs_table_keys.insert(input_desc.getTableKey());
    }
  }

  const auto num_bytes_for_row = executor->getNumBytesForFetchedRow(lhs_table_keys);

  if (ra_exe_unit.union_all) {
    buildFragmentPerKernelMapForUnion(ra_exe_unit,
                                      frag_offsets,
                                      device_ids,
                                      num_bytes_for_row,
                                      device_type,
                                      query_description_type,
                                      max_kernel_input_rows,
                                      executor);
  } else if (enable_multifrag_kernels) {
    buildMultifragKernelMap(ra_exe_unit,
                            frag_offsets,
                            device_ids,
                            num_bytes_for_row,
                            device_type,
                            query_description_type,
                            enable_inner_join_fragment_skipping,
                            executor);
  } else {
    buildFragmentPerKernelMap(ra_exe_unit,
                              frag_offsets,
                              device_ids,
                              num_bytes_for_row,
                              device_type,
                              query_description_type,
                              max_kernel_input_rows,
                              uses_lazy_fetch,
                              executor);
  }
}

void QueryFragmentDescriptor::buildFragmentPerKernelForTable(
    const TableFragments* fragments,
    const RelAlgExecutionUnit& ra_exe_unit,
    const InputDescriptor& table_desc,
    const bool is_temporary_table,
    const std::vector<uint64_t>& frag_offsets,
    const std::set<int>& device_ids,
    const size_t num_bytes_for_row,
    const ChunkMetadataVector& deleted_chunk_metadata_vec,
    const std::optional<size_t> table_desc_offset,
    const ExecutorDeviceType& device_type,
    const heavyai::QueryDescriptionType query_description_type,
    const size_t max_kernel_input_rows,
    const bool uses_lazy_fetch,
    Executor* executor) {
  const auto coalesce_gpu_projection_fragments =
      g_enable_result_reduction_pipeline && !uses_lazy_fetch &&
      can_coalesce_gpu_projection_fragments(
          ra_exe_unit, device_type, query_description_type) &&
      !table_desc_offset;
  const auto coalesce_gpu_groupby_fragments =
      g_enable_result_reduction_pipeline &&
      can_coalesce_gpu_groupby_fragments(ra_exe_unit,
                                         table_desc,
                                         device_type,
                                         query_description_type,
                                         table_desc_offset,
                                         max_kernel_input_rows);
  const auto coalesce_gpu_fragments =
      coalesce_gpu_projection_fragments || coalesce_gpu_groupby_fragments;
  auto coalescing_input_row_limit =
      coalesce_gpu_projection_fragments ? ra_exe_unit.scan_limit : max_kernel_input_rows;
  if (coalesce_gpu_groupby_fragments) {
    coalescing_input_row_limit =
        std::min(coalescing_input_row_limit,
                 gpu_input_row_limit(device_ids,
                                     available_gpu_mem_bytes_,
                                     gpu_input_mem_limit_percent_,
                                     num_bytes_for_row));
  }
  const auto coalescing_row_multiplier =
      coalesce_gpu_projection_fragments
          ? std::max<size_t>(size_t(1), ra_exe_unit.input_descs.size())
          : size_t(1);
  auto get_fragment_tuple_count = [&deleted_chunk_metadata_vec, &is_temporary_table](
                                      const auto& fragment) -> std::optional<size_t> {
    // returning std::nullopt disables execution dispatch optimizations based on tuple
    // counts as it signals to the dispatch mechanism that a reliable tuple count cannot
    // be obtained. This is the case for fragments which have deleted rows, temporary
    // table fragments, or fragments in a UNION query.
    if (is_temporary_table) {
      // 31 Mar 2021 MAT TODO I think that the fragment Tuple count should be ok
      // need to double check that at some later date
      return std::nullopt;
    }
    if (deleted_chunk_metadata_vec.empty()) {
      return fragment.getNumTuples();
    }
    const auto fragment_id = fragment.fragmentId;
    CHECK_GE(fragment_id, 0);
    if (static_cast<size_t>(fragment_id) < deleted_chunk_metadata_vec.size()) {
      const auto& chunk_metadata = deleted_chunk_metadata_vec[fragment_id];
      if (chunk_metadata.second->chunkStats.max.tinyintval == 1) {
        return std::nullopt;
      }
    }
    return fragment.getNumTuples();
  };

  for (size_t i = 0; i < fragments->size(); i++) {
    if (!allowed_outer_fragment_indices_.empty()) {
      if (std::find(allowed_outer_fragment_indices_.begin(),
                    allowed_outer_fragment_indices_.end(),
                    i) == allowed_outer_fragment_indices_.end()) {
        continue;
      }
    }

    const auto& fragment = (*fragments)[i];
    const auto skip_frag = executor->skipFragment(
        table_desc, fragment, ra_exe_unit.simple_quals, frag_offsets, i);
    if (skip_frag.first) {
      continue;
    }
    const bool can_coalesce_this_fragment =
        coalesce_gpu_fragments && skip_frag.second < 0;
    rowid_lookup_key_ = std::max(rowid_lookup_key_, skip_frag.second);
    const int chosen_device_count = device_ids.size();
    CHECK_GT(chosen_device_count, 0);
    const auto memory_level = device_type == ExecutorDeviceType::GPU
                                  ? Data_Namespace::GPU_LEVEL
                                  : Data_Namespace::CPU_LEVEL;
    // when reaching this, `fragment.deviceIds[GPU_LEVEL]` indicates a set of available
    // device ids determined by `Executor::determineAvailableDevicesToProcessQuery`
    const int device_id = fragment.deviceIds[static_cast<int>(memory_level)];
    if (device_type == ExecutorDeviceType::GPU) {
      CHECK(device_ids.find(device_id) != device_ids.end())
          << "Cannot find device_id " << device_id
          << " from pre-determined set of devices (device_ids: {"
          << ::toString(device_ids) << "})";
      checkDeviceMemoryUsage(fragment,
                             device_id,
                             num_bytes_for_row,
                             query_description_type,
                             /*is_multifrag_kernel=*/false);
    }

    ExecutionKernelDescriptor execution_kernel_desc{
        device_id, {}, get_fragment_tuple_count(fragment)};
    if (table_desc_offset) {
      const auto frag_ids =
          executor->getTableFragmentIndices(ra_exe_unit,
                                            device_type,
                                            *table_desc_offset,
                                            i,
                                            selected_tables_fragments_,
                                            executor->getInnerTabIdToJoinCond());
      const auto& table_key = ra_exe_unit.input_descs[*table_desc_offset].getTableKey();
      execution_kernel_desc.fragments.emplace_back(
          FragmentsPerTable{table_key, frag_ids});

    } else {
      for (size_t j = 0; j < ra_exe_unit.input_descs.size(); ++j) {
        const auto frag_ids =
            executor->getTableFragmentIndices(ra_exe_unit,
                                              device_type,
                                              j,
                                              i,
                                              selected_tables_fragments_,
                                              executor->getInnerTabIdToJoinCond());
        const auto& table_key = ra_exe_unit.input_descs[j].getTableKey();
        auto table_frags_it = selected_tables_fragments_.find(table_key);
        CHECK(table_frags_it != selected_tables_fragments_.end());

        execution_kernel_desc.fragments.emplace_back(
            FragmentsPerTable{table_key, frag_ids});
      }
    }

    auto itr = execution_kernels_per_device_.find(device_id);
    if (can_coalesce_this_fragment && itr != execution_kernels_per_device_.end() &&
        !itr->second.empty() &&
        try_append_gpu_fragment_kernel(itr->second.back(),
                                       execution_kernel_desc,
                                       coalescing_input_row_limit,
                                       coalescing_row_multiplier)) {
      continue;
    }
    if (itr == execution_kernels_per_device_.end()) {
      auto const pair = execution_kernels_per_device_.insert(std::make_pair(
          device_id,
          std::vector<ExecutionKernelDescriptor>{std::move(execution_kernel_desc)}));
      CHECK(pair.second);
    } else {
      itr->second.emplace_back(std::move(execution_kernel_desc));
    }
  }
}

void QueryFragmentDescriptor::buildFragmentPerKernelMapForUnion(
    const RelAlgExecutionUnit& ra_exe_unit,
    const std::vector<uint64_t>& frag_offsets,
    const std::set<int>& device_ids,
    const size_t num_bytes_for_row,
    const ExecutorDeviceType& device_type,
    const heavyai::QueryDescriptionType query_description_type,
    const size_t max_kernel_input_rows,
    Executor* executor) {
  for (size_t j = 0; j < ra_exe_unit.input_descs.size(); ++j) {
    auto const& table_desc = ra_exe_unit.input_descs[j];
    const auto& table_key = table_desc.getTableKey();
    TableFragments const* fragments = selected_tables_fragments_.at(table_key);

    auto data_mgr = executor->getDataMgr();
    ChunkMetadataVector deleted_chunk_metadata_vec;

    bool is_temporary_table = false;
    if (table_key.table_id > 0) {
      // Temporary tables will not have a table descriptor and not have deleted rows.
      CHECK_GT(table_key.db_id, 0);
      const auto td = Catalog_Namespace::get_metadata_for_table(table_key);
      CHECK(td);
      if (table_is_temporary(td)) {
        // for temporary tables, we won't have delete column metadata available. However,
        // we know the table fits in memory as it is a temporary table, so signal to the
        // lower layers that we can disregard the early out select * optimization
        is_temporary_table = true;
      } else {
        const auto deleted_cd = executor->plan_state_->getDeletedColForTable(table_key);
        if (deleted_cd) {
          ChunkKey chunk_key_prefix = {
              table_key.db_id, table_key.table_id, deleted_cd->columnId};
          data_mgr->getChunkMetadataVecForKeyPrefix(deleted_chunk_metadata_vec,
                                                    chunk_key_prefix);
        }
      }
    }

    buildFragmentPerKernelForTable(fragments,
                                   ra_exe_unit,
                                   table_desc,
                                   is_temporary_table,
                                   frag_offsets,
                                   device_ids,
                                   num_bytes_for_row,
                                   {},
                                   j,
                                   device_type,
                                   query_description_type,
                                   max_kernel_input_rows,
                                   /*uses_lazy_fetch=*/false,
                                   executor);

    std::vector<int> table_ids =
        std::accumulate(execution_kernels_per_device_[0].begin(),
                        execution_kernels_per_device_[0].end(),
                        std::vector<int>(),
                        [](auto&& vec, auto& exe_kern) {
                          vec.push_back(exe_kern.fragments[0].table_key.table_id);
                          return vec;
                        });
    VLOG(1) << "execution_kernels_per_device_.size()="
            << execution_kernels_per_device_.size()
            << " execution_kernels_per_device_[0][*].fragments[0].table_id="
            << shared::printContainer(table_ids);
  }
}

void QueryFragmentDescriptor::buildFragmentPerKernelMap(
    const RelAlgExecutionUnit& ra_exe_unit,
    const std::vector<uint64_t>& frag_offsets,
    const std::set<int>& device_ids,
    const size_t num_bytes_for_row,
    const ExecutorDeviceType& device_type,
    const heavyai::QueryDescriptionType query_description_type,
    const size_t max_kernel_input_rows,
    const bool uses_lazy_fetch,
    Executor* executor) {
  const auto& outer_table_desc = ra_exe_unit.input_descs.front();
  const auto& outer_table_key = outer_table_desc.getTableKey();
  auto it = selected_tables_fragments_.find(outer_table_key);
  CHECK(it != selected_tables_fragments_.end());
  const auto outer_fragments = it->second;
  outer_fragments_size_ = outer_fragments->size();

  ChunkMetadataVector deleted_chunk_metadata_vec;

  bool is_temporary_table = false;
  if (outer_table_key.table_id > 0) {
    CHECK_GT(outer_table_key.db_id, 0);
    const auto catalog =
        Catalog_Namespace::SysCatalog::instance().getCatalog(outer_table_key.db_id);
    CHECK(catalog);
    // Temporary tables will not have a table descriptor and not have deleted rows.
    const auto td = catalog->getMetadataForTable(outer_table_key.table_id);
    CHECK(td);
    if (table_is_temporary(td)) {
      // for temporary tables, we won't have delete column metadata available. However, we
      // know the table fits in memory as it is a temporary table, so signal to the lower
      // layers that we can disregard the early out select * optimization
      is_temporary_table = true;
    } else {
      const auto deleted_cd = catalog->getDeletedColumnIfRowsDeleted(td);
      if (deleted_cd) {
        // 01 Apr 2021 MAT TODO this code is called on logical tables (ie not the shards)
        // I wonder if this makes sense in those cases
        td->fragmenter->getFragmenterId();
        auto frags = td->fragmenter->getFragmentsForQuery().fragments;
        for (auto frag : frags) {
          auto chunk_meta_it =
              frag.getChunkMetadataMapPhysical().find(deleted_cd->columnId);
          if (chunk_meta_it != frag.getChunkMetadataMapPhysical().end()) {
            const auto& chunk_meta = chunk_meta_it->second;
            ChunkKey chunk_key_prefix = {outer_table_key.db_id,
                                         outer_table_key.table_id,
                                         deleted_cd->columnId,
                                         frag.fragmentId};
            deleted_chunk_metadata_vec.emplace_back(
                std::pair{chunk_key_prefix, chunk_meta});
          }
        }
      }
    }
  }

  buildFragmentPerKernelForTable(outer_fragments,
                                 ra_exe_unit,
                                 outer_table_desc,
                                 is_temporary_table,
                                 frag_offsets,
                                 device_ids,
                                 num_bytes_for_row,
                                 deleted_chunk_metadata_vec,
                                 std::nullopt,
                                 device_type,
                                 query_description_type,
                                 max_kernel_input_rows,
                                 uses_lazy_fetch,
                                 executor);
}

void QueryFragmentDescriptor::buildMultifragKernelMap(
    const RelAlgExecutionUnit& ra_exe_unit,
    const std::vector<uint64_t>& frag_offsets,
    const std::set<int>& device_ids,
    const size_t num_bytes_for_row,
    const ExecutorDeviceType& device_type,
    const heavyai::QueryDescriptionType query_description_type,
    const bool enable_inner_join_fragment_skipping,
    Executor* executor) {
  // Allocate all the fragments of the tables involved in the query to available
  // devices. The basic idea: the device is decided by the outer table in the
  // query (the first table in a join) and we need to broadcast the fragments
  // in the inner table to each device. Sharding will change this model.
  const auto& outer_table_desc = ra_exe_unit.input_descs.front();
  const auto& outer_table_key = outer_table_desc.getTableKey();
  auto it = selected_tables_fragments_.find(outer_table_key);
  CHECK(it != selected_tables_fragments_.end());
  const auto outer_fragments = it->second;
  outer_fragments_size_ = outer_fragments->size();

  const auto inner_table_id_to_join_condition = executor->getInnerTabIdToJoinCond();

  for (size_t outer_frag_id = 0; outer_frag_id < outer_fragments->size();
       ++outer_frag_id) {
    if (!allowed_outer_fragment_indices_.empty()) {
      if (std::find(allowed_outer_fragment_indices_.begin(),
                    allowed_outer_fragment_indices_.end(),
                    outer_frag_id) == allowed_outer_fragment_indices_.end()) {
        continue;
      }
    }

    const auto& fragment = (*outer_fragments)[outer_frag_id];
    auto skip_frag = executor->skipFragment(outer_table_desc,
                                            fragment,
                                            ra_exe_unit.simple_quals,
                                            frag_offsets,
                                            outer_frag_id);
    if (enable_inner_join_fragment_skipping &&
        (skip_frag == std::pair<bool, int64_t>(false, -1))) {
      skip_frag = executor->skipFragmentInnerJoins(
          outer_table_desc, ra_exe_unit, fragment, frag_offsets, outer_frag_id);
    }
    if (skip_frag.first) {
      continue;
    }
    // when reaching this, `fragment.deviceIds[GPU_LEVEL]` indicates a set of available
    // device ids determined by `Executor::determineAvailableDevicesToProcessQuery`
    const int chosen_device_count = device_ids.size();
    CHECK_GT(chosen_device_count, 0);
    const int device_id = fragment.deviceIds[static_cast<int>(Data_Namespace::GPU_LEVEL)];
    if (device_type == ExecutorDeviceType::GPU) {
      CHECK(device_ids.find(device_id) != device_ids.end())
          << "Cannot find device_id " << device_id
          << " from pre-determined set of devices (fragment_id: " << fragment.fragmentId
          << ", device_ids for the fragment: " << ::toString(fragment.deviceIds)
          << ", query_device_ids: " << ::toString(device_ids) << ")";
      checkDeviceMemoryUsage(fragment,
                             device_id,
                             num_bytes_for_row,
                             query_description_type,
                             /*is_multifrag_kernel=*/true);
    }
    for (size_t j = 0; j < ra_exe_unit.input_descs.size(); ++j) {
      const auto& table_key = ra_exe_unit.input_descs[j].getTableKey();
      auto table_frags_it = selected_tables_fragments_.find(table_key);
      CHECK(table_frags_it != selected_tables_fragments_.end());
      const auto frag_ids =
          executor->getTableFragmentIndices(ra_exe_unit,
                                            device_type,
                                            j,
                                            outer_frag_id,
                                            selected_tables_fragments_,
                                            inner_table_id_to_join_condition);

      if (execution_kernels_per_device_.find(device_id) ==
          execution_kernels_per_device_.end()) {
        std::vector<ExecutionKernelDescriptor> kernel_descs{
            ExecutionKernelDescriptor{device_id, FragmentsList{}, std::nullopt}};
        CHECK(
            execution_kernels_per_device_.insert(std::make_pair(device_id, kernel_descs))
                .second);
      }

      // Multifrag kernels only have one execution kernel per device. Grab the execution
      // kernel object and push back into its fragments list.
      CHECK_EQ(execution_kernels_per_device_[device_id].size(), size_t(1));
      auto& execution_kernel = execution_kernels_per_device_[device_id].front();

      auto& kernel_frag_list = execution_kernel.fragments;
      if (kernel_frag_list.size() < j + 1) {
        kernel_frag_list.emplace_back(FragmentsPerTable{table_key, frag_ids});
      } else {
        CHECK_EQ(kernel_frag_list[j].table_key, table_key);
        auto& curr_frag_ids = kernel_frag_list[j].fragment_ids;
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

namespace {

bool is_sample_query(const RelAlgExecutionUnit& ra_exe_unit) {
  const bool result = ra_exe_unit.input_descs.size() == 1 &&
                      ra_exe_unit.simple_quals.empty() && ra_exe_unit.quals.empty() &&
                      ra_exe_unit.sort_info.order_entries.empty() &&
                      ra_exe_unit.scan_limit;
  if (result) {
    CHECK_EQ(size_t(1), ra_exe_unit.groupby_exprs.size());
    CHECK(!ra_exe_unit.groupby_exprs.front());
  }
  return result;
}

}  // namespace

bool QueryFragmentDescriptor::terminateDispatchMaybe(
    size_t& tuple_count,
    const RelAlgExecutionUnit& ra_exe_unit,
    const ExecutionKernelDescriptor& kernel) const {
  const auto sample_query_limit =
      ra_exe_unit.sort_info.limit.value_or(0) + ra_exe_unit.sort_info.offset;
  if (!kernel.outer_tuple_count) {
    return false;
  } else {
    tuple_count += *kernel.outer_tuple_count;
    if (is_sample_query(ra_exe_unit) && sample_query_limit > 0 &&
        tuple_count >= sample_query_limit) {
      return true;
    }
  }
  return false;
}

std::optional<size_t> QueryFragmentDescriptor::getMaxKernelOutputRowCountEstimate(
    const RelAlgExecutionUnit& ra_exe_unit) const {
  std::optional<size_t> max_output_rows;
  const bool output_bounded_by_outer_table =
      ra_exe_unit.input_descs.size() == size_t(1) ||
      projection_output_bounded_by_outer_table(ra_exe_unit);
  if (!output_bounded_by_outer_table) {
    return std::nullopt;
  }
  for (const auto& device_kernels : execution_kernels_per_device_) {
    for (const auto& kernel : device_kernels.second) {
      if (!kernel.outer_tuple_count) {
        return std::nullopt;
      }
      const auto estimated_rows = *kernel.outer_tuple_count;
      max_output_rows = max_output_rows ? std::max(*max_output_rows, estimated_rows)
                                        : std::optional<size_t>(estimated_rows);
    }
  }
  return max_output_rows;
}

void QueryFragmentDescriptor::checkDeviceMemoryUsage(
    const Fragmenter_Namespace::FragmentInfo& fragment,
    const int device_id,
    const size_t num_bytes_for_row,
    const heavyai::QueryDescriptionType query_description_type,
    const bool is_multifrag_kernel) {
  CHECK_GE(device_id, 0);
  const size_t gpu_bytes_limit =
      available_gpu_mem_bytes_[device_id] * gpu_input_mem_limit_percent_;
  if (!g_enable_result_reduction_pipeline) {
    auto& tuple_count = tuple_count_per_device_[device_id];
    if (fragment.getNumTuples() > std::numeric_limits<size_t>::max() - tuple_count) {
      throw QueryMustRunOnCpu();
    }
    tuple_count += fragment.getNumTuples();
    if (num_bytes_for_row && tuple_count > gpu_bytes_limit / num_bytes_for_row) {
      LOG(WARNING) << "Not enough memory on device " << device_id
                   << " for input chunks totaling more than " << gpu_bytes_limit
                   << " bytes (available device memory: " << gpu_bytes_limit << " bytes)";
      throw QueryMustRunOnCpu();
    }
    return;
  }

  size_t tuple_count = fragment.getNumTuples();
  if (is_multifrag_kernel) {
    auto& accumulated_tuple_count = tuple_count_per_device_[device_id];
    if (fragment.getNumTuples() >
        std::numeric_limits<size_t>::max() - accumulated_tuple_count) {
      throw QueryExecutionError(
          ErrorCode::OUT_OF_GPU_MEM,
          "Input tuple count overflowed for the selected dispatch mode.",
          QueryExecutionProperties{query_description_type, is_multifrag_kernel});
    }
    accumulated_tuple_count += fragment.getNumTuples();
    tuple_count = accumulated_tuple_count;
  }
  const bool required_bytes_overflow =
      num_bytes_for_row &&
      tuple_count > std::numeric_limits<size_t>::max() / num_bytes_for_row;
  const size_t required_bytes = required_bytes_overflow
                                    ? std::numeric_limits<size_t>::max()
                                    : tuple_count * num_bytes_for_row;
  if (required_bytes_overflow || required_bytes > gpu_bytes_limit) {
    LOG(WARNING) << "Not enough memory on device " << device_id
                 << " for input chunks totaling " << required_bytes
                 << " bytes (available device memory: " << gpu_bytes_limit
                 << " bytes, multifrag kernel: " << is_multifrag_kernel << ")";
    throw QueryExecutionError(
        ErrorCode::OUT_OF_GPU_MEM,
        "Input chunks exceed available GPU memory for the selected dispatch mode.",
        QueryExecutionProperties{query_description_type, is_multifrag_kernel});
  }
}

std::ostream& operator<<(std::ostream& os, FragmentsPerTable const& fragments_per_table) {
  os << fragments_per_table.table_key << ", fragment_ids";
  for (size_t i = 0; i < fragments_per_table.fragment_ids.size(); ++i) {
    os << (i ? ' ' : '(') << fragments_per_table.fragment_ids[i];
  }
  return os << ')';
}
