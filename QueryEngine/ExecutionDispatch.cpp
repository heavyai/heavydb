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
#include "DynamicWatchdog.h"
#include "ErrorHandling.h"
#include "Execute.h"

#include "DataMgr/BufferMgr/BufferMgr.h"

#include <numeric>

std::mutex Executor::ExecutionDispatch::reduce_mutex_;

namespace {

bool needs_skip_result(const ResultSetPtr& res) {
  return !res || res->definitelyHasNoRows();
}

// The result set of `ra_exe_unit` needs to hold a reference to `chunk` if its
// column is part of the target expressions, result set iteration needs it alive.
bool need_to_hold_chunk(const Chunk_NS::Chunk* chunk,
                        const RelAlgExecutionUnit& ra_exe_unit) {
  CHECK(chunk->get_column_desc());
  const auto chunk_ti = chunk->get_column_desc()->columnType;
  if (chunk_ti.is_array() ||
      (chunk_ti.is_string() && chunk_ti.get_compression() == kENCODING_NONE)) {
    for (const auto target_expr : ra_exe_unit.target_exprs) {
      const auto col_var = dynamic_cast<const Analyzer::ColumnVar*>(target_expr);
      if (col_var && col_var->get_column_id() == chunk->get_column_desc()->columnId &&
          col_var->get_table_id() == chunk->get_column_desc()->tableId) {
        return true;
      }
    }
  }
  return false;
}

inline bool query_has_inner_join(const RelAlgExecutionUnit& ra_exe_unit) {
  return (std::count_if(ra_exe_unit.join_quals.begin(),
                        ra_exe_unit.join_quals.end(),
                        [](const auto& join_condition) {
                          return join_condition.type == JoinType::INNER;
                        }) > 0);
}

}  // namespace

void Executor::ExecutionDispatch::runImpl(
    const ExecutorDeviceType chosen_device_type,
    int chosen_device_id,
    const ExecutionOptions& eo,
    const ColumnFetcher& column_fetcher,
    const QueryCompilationDescriptor& query_comp_desc,
    const QueryMemoryDescriptor& query_mem_desc,
    const FragmentsList& frag_list,
    const ExecutorDispatchMode kernel_dispatch_mode,
    const int64_t rowid_lookup_key) {
  const auto memory_level = chosen_device_type == ExecutorDeviceType::GPU
                                ? Data_Namespace::GPU_LEVEL
                                : Data_Namespace::CPU_LEVEL;
  const int outer_table_id = ra_exe_unit_.input_descs[0].getTableId();
  CHECK_GE(frag_list.size(), size_t(1));
  CHECK_EQ(frag_list[0].table_id, outer_table_id);
  const auto& outer_tab_frag_ids = frag_list[0].fragment_ids;
  CHECK_GE(chosen_device_id, 0);
  CHECK_LT(chosen_device_id, max_gpu_count);
  // need to own them while query executes
  auto chunk_iterators_ptr = std::make_shared<std::list<ChunkIter>>();
  std::list<std::shared_ptr<Chunk_NS::Chunk>> chunks;
  std::unique_ptr<std::lock_guard<std::mutex>> gpu_lock;
  if (chosen_device_type == ExecutorDeviceType::GPU) {
    gpu_lock.reset(
        new std::lock_guard<std::mutex>(executor_->gpu_exec_mutex_[chosen_device_id]));
  }
  FetchResult fetch_result;
  try {
    std::map<int, const TableFragments*> all_tables_fragments;
    QueryFragmentDescriptor::computeAllTablesFragments(
        all_tables_fragments, ra_exe_unit_, query_infos_);

    fetch_result = executor_->fetchChunks(column_fetcher,
                                          ra_exe_unit_,
                                          chosen_device_id,
                                          memory_level,
                                          all_tables_fragments,
                                          frag_list,
                                          cat_,
                                          *chunk_iterators_ptr,
                                          chunks);
    if (fetch_result.num_rows.empty()) {
      return;
    }
    if (eo.with_dynamic_watchdog &&
        !dynamic_watchdog_set_.test_and_set(std::memory_order_acquire)) {
      CHECK_GT(eo.dynamic_watchdog_time_limit, 0u);
      auto cycle_budget = dynamic_watchdog_init(eo.dynamic_watchdog_time_limit);
      LOG(INFO) << "Dynamic Watchdog budget: CPU: "
                << std::to_string(eo.dynamic_watchdog_time_limit) << "ms, "
                << std::to_string(cycle_budget) << " cycles";
    }
  } catch (const OutOfMemory&) {
    throw QueryExecutionError(
        memory_level == Data_Namespace::GPU_LEVEL ? ERR_OUT_OF_GPU_MEM
                                                  : ERR_OUT_OF_CPU_MEM,
        QueryExecutionProperties{
            query_mem_desc.getQueryDescriptionType(),
            kernel_dispatch_mode == ExecutorDispatchMode::MultifragmentKernel});
    return;
  }

  const CompilationResult& compilation_result = query_comp_desc.getCompilationResult();
  std::unique_ptr<QueryExecutionContext> query_exe_context_owned;
  const bool do_render = render_info_ && render_info_->isPotentialInSituRender();

  int64_t total_num_input_rows{-1};
  if (kernel_dispatch_mode == ExecutorDispatchMode::KernelPerFragment &&
      query_mem_desc.getQueryDescriptionType() == QueryDescriptionType::Projection) {
    total_num_input_rows = 0;
    std::for_each(fetch_result.num_rows.begin(),
                  fetch_result.num_rows.end(),
                  [&total_num_input_rows](const std::vector<int64_t>& frag_row_count) {
                    total_num_input_rows = std::accumulate(frag_row_count.begin(),
                                                           frag_row_count.end(),
                                                           total_num_input_rows);
                  });
    // TODO(adb): we may want to take this early out for all queries, but we are most
    // likely to see this query pattern on the kernel per fragment path (e.g. with HAVING
    // 0=1)
    if (total_num_input_rows == 0) {
      return;
    }

    if (query_has_inner_join(ra_exe_unit_)) {
      total_num_input_rows *= ra_exe_unit_.input_descs.size();
    }
  }

  try {
    query_exe_context_owned =
        query_mem_desc.getQueryExecutionContext(ra_exe_unit_,
                                                executor_,
                                                chosen_device_type,
                                                kernel_dispatch_mode,
                                                chosen_device_id,
                                                total_num_input_rows,
                                                fetch_result.col_buffers,
                                                fetch_result.frag_offsets,
                                                row_set_mem_owner_,
                                                compilation_result.output_columnar,
                                                query_mem_desc.sortOnGpu(),
                                                do_render ? render_info_ : nullptr);
  } catch (const OutOfHostMemory& e) {
    throw QueryExecutionError(ERR_OUT_OF_CPU_MEM);
  }
  QueryExecutionContext* query_exe_context{query_exe_context_owned.get()};
  CHECK(query_exe_context);
  int32_t err{0};
  uint32_t start_rowid{0};
  if (rowid_lookup_key >= 0) {
    if (!frag_list.empty()) {
      const auto& all_frag_row_offsets = getFragOffsets();
      start_rowid = rowid_lookup_key -
                    all_frag_row_offsets[frag_list.begin()->fragment_ids.front()];
    }
  }

  ResultSetPtr device_results;
  if (ra_exe_unit_.groupby_exprs.empty()) {
    err = executor_->executePlanWithoutGroupBy(ra_exe_unit_,
                                               compilation_result,
                                               query_comp_desc.hoistLiterals(),
                                               device_results,
                                               ra_exe_unit_.target_exprs,
                                               chosen_device_type,
                                               fetch_result.col_buffers,
                                               query_exe_context,
                                               fetch_result.num_rows,
                                               fetch_result.frag_offsets,
                                               &cat_.getDataMgr(),
                                               chosen_device_id,
                                               start_rowid,
                                               ra_exe_unit_.input_descs.size(),
                                               do_render ? render_info_ : nullptr);
  } else {
    err = executor_->executePlanWithGroupBy(ra_exe_unit_,
                                            compilation_result,
                                            query_comp_desc.hoistLiterals(),
                                            device_results,
                                            chosen_device_type,
                                            fetch_result.col_buffers,
                                            outer_tab_frag_ids,
                                            query_exe_context,
                                            fetch_result.num_rows,
                                            fetch_result.frag_offsets,
                                            &cat_.getDataMgr(),
                                            chosen_device_id,
                                            ra_exe_unit_.scan_limit,
                                            start_rowid,
                                            ra_exe_unit_.input_descs.size(),
                                            do_render ? render_info_ : nullptr);
  }
  if (device_results) {
    std::list<std::shared_ptr<Chunk_NS::Chunk>> chunks_to_hold;
    for (const auto chunk : chunks) {
      if (need_to_hold_chunk(chunk.get(), ra_exe_unit_)) {
        chunks_to_hold.push_back(chunk);
      }
    }
    device_results->holdChunks(chunks_to_hold);
    device_results->holdChunkIterators(chunk_iterators_ptr);
  }
  if (err) {
    throw QueryExecutionError(err);
  }
  {
    std::lock_guard<std::mutex> lock(reduce_mutex_);
    if (!needs_skip_result(device_results)) {
      all_fragment_results_.emplace_back(std::move(device_results), outer_tab_frag_ids);
    }
  }
}

Executor::ExecutionDispatch::ExecutionDispatch(
    Executor* executor,
    const RelAlgExecutionUnit& ra_exe_unit,
    const std::vector<InputTableInfo>& query_infos,
    const Catalog_Namespace::Catalog& cat,
    const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
    RenderInfo* render_info)
    : executor_(executor)
    , ra_exe_unit_(ra_exe_unit)
    , query_infos_(query_infos)
    , cat_(cat)
    , row_set_mem_owner_(row_set_mem_owner)
    , render_info_(render_info) {
  all_fragment_results_.reserve(query_infos_.front().info.fragments.size());
}

std::tuple<QueryCompilationDescriptorOwned, QueryMemoryDescriptorOwned>
Executor::ExecutionDispatch::compile(const size_t max_groups_buffer_entry_guess,
                                     const int8_t crt_min_byte_width,
                                     const CompilationOptions& co,
                                     const ExecutionOptions& eo,
                                     const ColumnFetcher& column_fetcher,
                                     const bool has_cardinality_estimation) {
  auto query_comp_desc = std::make_unique<QueryCompilationDescriptor>();
  std::unique_ptr<QueryMemoryDescriptor> query_mem_desc;

  switch (co.device_type_) {
    case ExecutorDeviceType::CPU: {
      query_mem_desc = query_comp_desc->compile(max_groups_buffer_entry_guess,
                                                crt_min_byte_width,
                                                has_cardinality_estimation,
                                                ra_exe_unit_,
                                                query_infos_,
                                                column_fetcher,
                                                co,
                                                eo,
                                                render_info_,
                                                executor_);
    } break;
    case ExecutorDeviceType::GPU: {
      query_mem_desc = query_comp_desc->compile(max_groups_buffer_entry_guess,
                                                crt_min_byte_width,
                                                has_cardinality_estimation,
                                                ra_exe_unit_,
                                                query_infos_,
                                                column_fetcher,
                                                co,
                                                eo,
                                                render_info_,
                                                executor_);
    } break;
    default:
      UNREACHABLE();
  }

  return std::make_tuple(std::move(query_comp_desc), std::move(query_mem_desc));
}

void Executor::ExecutionDispatch::run(const ExecutorDeviceType chosen_device_type,
                                      int chosen_device_id,
                                      const ExecutionOptions& eo,
                                      const ColumnFetcher& column_fetcher,
                                      const QueryCompilationDescriptor& query_comp_desc,
                                      const QueryMemoryDescriptor& query_mem_desc,
                                      const FragmentsList& frag_list,
                                      const ExecutorDispatchMode kernel_dispatch_mode,
                                      const int64_t rowid_lookup_key) {
  try {
    runImpl(chosen_device_type,
            chosen_device_id,
            eo,
            column_fetcher,
            query_comp_desc,
            query_mem_desc,
            frag_list,
            kernel_dispatch_mode,
            rowid_lookup_key);
  } catch (const std::bad_alloc& e) {
    throw QueryExecutionError(ERR_OUT_OF_CPU_MEM, e.what());
  } catch (const OutOfHostMemory& e) {
    throw QueryExecutionError(ERR_OUT_OF_CPU_MEM, e.what());
  } catch (const OutOfRenderMemory& e) {
    throw QueryExecutionError(ERR_OUT_OF_RENDER_MEM, e.what());
  } catch (const OutOfMemory& e) {
    throw QueryExecutionError(
        ERR_OUT_OF_GPU_MEM,
        e.what(),
        QueryExecutionProperties{
            query_mem_desc.getQueryDescriptionType(),
            kernel_dispatch_mode == ExecutorDispatchMode::MultifragmentKernel});
  } catch (const ColumnarConversionNotSupported& e) {
    throw QueryExecutionError(ERR_COLUMNAR_CONVERSION_NOT_SUPPORTED, e.what());
  } catch (const TooManyLiterals& e) {
    throw QueryExecutionError(ERR_TOO_MANY_LITERALS, e.what());
  } catch (const SringConstInResultSet& e) {
    throw QueryExecutionError(ERR_STRING_CONST_IN_RESULTSET, e.what());
  } catch (const QueryExecutionError& e) {
    throw e;
  }
}

const RelAlgExecutionUnit& Executor::ExecutionDispatch::getExecutionUnit() const {
  return ra_exe_unit_;
}

const std::vector<uint64_t>& Executor::ExecutionDispatch::getFragOffsets() const {
  std::lock_guard<std::mutex> lock(all_frag_row_offsets_mutex_);
  if (all_frag_row_offsets_.empty()) {
    all_frag_row_offsets_.resize(query_infos_.front().info.fragments.size() + 1);
    for (size_t i = 1; i <= query_infos_.front().info.fragments.size(); ++i) {
      all_frag_row_offsets_[i] =
          all_frag_row_offsets_[i - 1] +
          query_infos_.front().info.fragments[i - 1].getNumTuples();
    }
  }
  return all_frag_row_offsets_;
}

std::vector<std::pair<ResultSetPtr, std::vector<size_t>>>&
Executor::ExecutionDispatch::getFragmentResults() {
  return all_fragment_results_;
}
