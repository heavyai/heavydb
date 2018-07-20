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

#include "DynamicWatchdog.h"
#include "Execute.h"
#include "ExecutionException.h"
#include "QueryFragmentDescriptor.h"

#include "DataMgr/BufferMgr/BufferMgr.h"

#include <numeric>

std::mutex Executor::ExecutionDispatch::reduce_mutex_;

namespace {

size_t get_mapped_frag_id_of_src_table(
    const std::vector<std::pair<int, size_t>>& join_dimensions,
    const int src_tab_id,
    const size_t dst_frag_id) {
  CHECK(join_dimensions.size());
  std::unordered_map<int, size_t> tab_id_to_frag_cnt;
  size_t combination_count{1};
  for (const auto& table : join_dimensions) {
    tab_id_to_frag_cnt.insert(table);
    CHECK(table.second);
    combination_count *= table.second;
  }

  auto cnt_it = tab_id_to_frag_cnt.find(src_tab_id);
  CHECK(cnt_it != tab_id_to_frag_cnt.end());
  if (size_t(1) == cnt_it->second) {
    return size_t(0);
  }
  size_t crt_frag_id{dst_frag_id};
  for (auto dim_it = join_dimensions.rbegin();
       dim_it->first != src_tab_id && dim_it != join_dimensions.rend();
       ++dim_it) {
    crt_frag_id %= combination_count;
    combination_count /= dim_it->second;
  }
  combination_count /= cnt_it->second;
  return crt_frag_id / combination_count;
}

bool needs_skip_result(const ResultPtr& res) {
  if (auto rows = boost::get<RowSetPtr>(&res)) {
    return !*rows || (*rows)->definitelyHasNoRows();
  } else if (auto tab = boost::get<IterTabPtr>(&res)) {
    // Keep empty table in result array when no error
    return !*tab;
  }

  return false;
}

}  // namespace

uint32_t Executor::ExecutionDispatch::getFragmentStride(
    const FragmentsList& frag_ids) const {
#ifdef ENABLE_MULTIFRAG_JOIN
  if (!ra_exe_unit_.inner_joins.empty()) {
    CHECK_EQ(ra_exe_unit_.input_descs.size(), frag_ids.size());
    const auto table_count = ra_exe_unit_.input_descs.size();
    uint32_t stride = 1;
    for (size_t i = 1; i < table_count; ++i) {
      if (executor_->plan_state_->join_info_.sharded_range_table_indices_.count(i)) {
        stride *= frag_ids[i].fragment_ids.size();
      }
    }
    return stride;
  }
  const bool is_hash_join = executor_->plan_state_->join_info_.join_impl_type_ ==
                                Executor::JoinImplType::HashOneToOne ||
                            executor_->plan_state_->join_info_.join_impl_type_ ==
                                Executor::JoinImplType::HashOneToMany;
  if ((is_hash_join || executor_->isOuterLoopJoin()) &&
      ra_exe_unit_.input_descs.size() == 2) {
    CHECK_EQ(frag_ids.size(), size_t(2));
    CHECK_EQ(ra_exe_unit_.input_descs.back().getTableId(), frag_ids[1].table_id);
    return static_cast<uint32_t>(frag_ids[1].fragment_ids.size());
  }
#endif
  return 1u;
}

std::vector<const ColumnarResults*> Executor::ExecutionDispatch::getAllScanColumnFrags(
    const int table_id,
    const int col_id,
    const std::map<int, const TableFragments*>& all_tables_fragments) const {
  const auto fragments_it = all_tables_fragments.find(table_id);
  CHECK(fragments_it != all_tables_fragments.end());
  const auto fragments = fragments_it->second;
  const auto frag_count = fragments->size();
  std::vector<const ColumnarResults*> results(frag_count, nullptr);
  const InputColDescriptor desc(col_id, table_id, int(0));
  CHECK(desc.getScanDesc().getSourceType() == InputSourceType::TABLE);
  auto frags_it = columnarized_ref_table_cache_.find(desc);
  if (frags_it == columnarized_ref_table_cache_.end()) {
    columnarized_ref_table_cache_.insert(std::make_pair(
        desc, std::unordered_map<CacheKey, std::unique_ptr<const ColumnarResults>>()));
    frags_it = columnarized_ref_table_cache_.find(desc);
    for (int frag_id = 0; frag_id < static_cast<int>(frag_count); ++frag_id) {
      std::list<std::shared_ptr<Chunk_NS::Chunk>> chunk_holder;
      std::list<ChunkIter> chunk_iter_holder;
      const auto& fragment = (*fragments)[frag_id];
      auto chunk_meta_it = fragment.getChunkMetadataMap().find(col_id);
      CHECK(chunk_meta_it != fragment.getChunkMetadataMap().end());
      auto col_buffer = getScanColumn(table_id,
                                      frag_id,
                                      col_id,
                                      all_tables_fragments,
                                      chunk_holder,
                                      chunk_iter_holder,
                                      Data_Namespace::CPU_LEVEL,
                                      int(0));
      frags_it->second.insert(std::make_pair(
          CacheKey{frag_id},
          boost::make_unique<ColumnarResults>(row_set_mem_owner_,
                                              col_buffer,
                                              fragment.getNumTuples(),
                                              chunk_meta_it->second.sqlType)));
    }
  }
  CHECK(frags_it != columnarized_ref_table_cache_.end());
  CHECK_EQ(frag_count, frags_it->second.size());
  for (int frag_id = 0; frag_id < static_cast<int>(frag_count); ++frag_id) {
    results[frag_id] = frags_it->second[{frag_id}].get();
  }
  return results;
}

const int8_t* Executor::ExecutionDispatch::getColumn(
    const ResultPtr& buffer,
    const int table_id,
    const int frag_id,
    const int col_id,
    const Data_Namespace::MemoryLevel memory_level,
    const int device_id) const {
  const ColumnarResults* result{nullptr};
  {
    std::lock_guard<std::mutex> columnar_conversion_guard(columnar_conversion_mutex_);
    if (columnarized_table_cache_.empty() || !columnarized_table_cache_.count(table_id)) {
      columnarized_table_cache_.insert(std::make_pair(
          table_id, std::unordered_map<int, std::shared_ptr<const ColumnarResults>>()));
    }
    auto& frag_id_to_result = columnarized_table_cache_[table_id];
    if (frag_id_to_result.empty() || !frag_id_to_result.count(frag_id)) {
      frag_id_to_result.insert(
          std::make_pair(frag_id,
                         std::shared_ptr<const ColumnarResults>(
                             columnarize_result(row_set_mem_owner_, buffer, frag_id))));
    }
    CHECK_NE(size_t(0), columnarized_table_cache_.count(table_id));
    result = columnarized_table_cache_[table_id][frag_id].get();
  }
  CHECK_GE(col_id, 0);
  return getColumn(result, col_id, &cat_.get_dataMgr(), memory_level, device_id);
}

namespace {

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

}  // namespace

void Executor::ExecutionDispatch::runImpl(const ExecutorDeviceType chosen_device_type,
                                          int chosen_device_id,
                                          const ExecutionOptions& options,
                                          const FragmentsList& frag_list,
                                          const size_t ctx_idx,
                                          const int64_t rowid_lookup_key) {
  const auto memory_level = chosen_device_type == ExecutorDeviceType::GPU
                                ? Data_Namespace::GPU_LEVEL
                                : Data_Namespace::CPU_LEVEL;
  const int outer_table_id = ra_exe_unit_.input_descs[0].getTableId();
  CHECK_GE(frag_list.size(), size_t(1));
#ifndef ENABLE_EQUIJOIN_FOLD
  CHECK_LE(frag_list.size(), size_t(2));
#endif
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
    for (size_t tab_idx = 0, tab_cnt = ra_exe_unit_.input_descs.size(); tab_idx < tab_cnt;
         ++tab_idx) {
      int table_id = ra_exe_unit_.input_descs[tab_idx].getTableId();
      CHECK_EQ(query_infos_[tab_idx].table_id, table_id);
      const auto& fragments = query_infos_[tab_idx].info.fragments;
      if (!all_tables_fragments.count(table_id)) {
        all_tables_fragments.insert(std::make_pair(table_id, &fragments));
      }
    }
    for (size_t tab_idx = 0,
                extra_tab_base = ra_exe_unit_.input_descs.size(),
                tab_cnt = ra_exe_unit_.extra_input_descs.size();
         tab_idx < tab_cnt;
         ++tab_idx) {
      int table_id = ra_exe_unit_.extra_input_descs[tab_idx].getTableId();
      const auto& fragments = query_infos_[extra_tab_base + tab_idx].info.fragments;
      all_tables_fragments.insert(std::make_pair(table_id, &fragments));
    }
    OOM_TRACE_PUSH();
    fetch_result = executor_->fetchChunks(*this,
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
    if (options.with_dynamic_watchdog &&
        !dynamic_watchdog_set_.test_and_set(std::memory_order_acquire)) {
      CHECK_GT(options.dynamic_watchdog_time_limit, 0);
      auto cycle_budget = dynamic_watchdog_init(options.dynamic_watchdog_time_limit);
      LOG(INFO) << "Dynamic Watchdog budget: CPU: "
                << std::to_string(options.dynamic_watchdog_time_limit) << "ms, "
                << std::to_string(cycle_budget) << " cycles";
    }
  } catch (const OutOfMemory&) {
    std::lock_guard<std::mutex> lock(reduce_mutex_);
    *error_code_ = ERR_OUT_OF_GPU_MEM;
    return;
  }
  const CompilationResult& compilation_result =
      chosen_device_type == ExecutorDeviceType::GPU ? compilation_result_gpu_
                                                    : compilation_result_cpu_;
  CHECK(!compilation_result.query_mem_desc.usesCachedContext() ||
        !ra_exe_unit_.scan_limit);
  std::unique_ptr<QueryExecutionContext> query_exe_context_owned;
  const bool do_render = render_info_ && render_info_->isPotentialInSituRender();
  try {
    OOM_TRACE_PUSH();
    query_exe_context_owned =
        compilation_result.query_mem_desc.usesCachedContext()
            ? nullptr
            : compilation_result.query_mem_desc.getQueryExecutionContext(
                  ra_exe_unit_,
                  executor_->plan_state_->init_agg_vals_,
                  executor_,
                  chosen_device_type,
                  chosen_device_id,
                  fetch_result.col_buffers,
                  fetch_result.iter_buffers,
                  fetch_result.frag_offsets,
                  row_set_mem_owner_,
                  compilation_result.output_columnar,
                  compilation_result.query_mem_desc.sortOnGpu(),
                  do_render ? render_info_ : nullptr);
  } catch (const OutOfHostMemory& e) {
    std::lock_guard<std::mutex> lock(reduce_mutex_);
    LOG(ERROR) << e.what();
    *error_code_ = ERR_OUT_OF_CPU_MEM;
    return;
  }
  QueryExecutionContext* query_exe_context{query_exe_context_owned.get()};
  std::unique_ptr<std::lock_guard<std::mutex>> query_ctx_lock;
  if (compilation_result.query_mem_desc.usesCachedContext()) {
    query_ctx_lock.reset(
        new std::lock_guard<std::mutex>(query_context_mutexes_[ctx_idx]));
    if (!query_contexts_[ctx_idx]) {
      try {
        OOM_TRACE_PUSH();
        query_contexts_[ctx_idx] =
            compilation_result.query_mem_desc.getQueryExecutionContext(
                ra_exe_unit_,
                executor_->plan_state_->init_agg_vals_,
                executor_,
                chosen_device_type,
                chosen_device_id,
                fetch_result.col_buffers,
                fetch_result.iter_buffers,
                fetch_result.frag_offsets,
                row_set_mem_owner_,
                compilation_result.output_columnar,
                compilation_result.query_mem_desc.sortOnGpu(),
                do_render ? render_info_ : nullptr);
      } catch (const OutOfHostMemory& e) {
        std::lock_guard<std::mutex> lock(reduce_mutex_);
        LOG(ERROR) << e.what();
        *error_code_ = ERR_OUT_OF_CPU_MEM;
        return;
      }
    }
    query_exe_context = query_contexts_[ctx_idx].get();
  }
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

  ResultPtr device_results;
  if (ra_exe_unit_.groupby_exprs.empty()) {
    OOM_TRACE_PUSH();
    err = executor_->executePlanWithoutGroupBy(ra_exe_unit_,
                                               compilation_result,
                                               co_.hoist_literals_,
                                               device_results,
                                               ra_exe_unit_.target_exprs,
                                               chosen_device_type,
                                               fetch_result.col_buffers,
                                               query_exe_context,
                                               fetch_result.num_rows,
                                               fetch_result.frag_offsets,
                                               getFragmentStride(frag_list),
                                               &cat_.get_dataMgr(),
                                               chosen_device_id,
                                               start_rowid,
                                               ra_exe_unit_.input_descs.size(),
                                               do_render ? render_info_ : nullptr);
  } else {
    OOM_TRACE_PUSH();
    err = executor_->executePlanWithGroupBy(ra_exe_unit_,
                                            compilation_result,
                                            co_.hoist_literals_,
                                            device_results,
                                            chosen_device_type,
                                            fetch_result.col_buffers,
                                            outer_tab_frag_ids,
                                            query_exe_context,
                                            fetch_result.num_rows,
                                            fetch_result.frag_offsets,
                                            getFragmentStride(frag_list),
                                            &cat_.get_dataMgr(),
                                            chosen_device_id,
                                            ra_exe_unit_.scan_limit,
                                            start_rowid,
                                            ra_exe_unit_.input_descs.size(),
                                            do_render ? render_info_ : nullptr);
  }
  if (auto rows_pp = boost::get<RowSetPtr>(&device_results)) {
    if (auto& rows_ptr = *rows_pp) {
      std::list<std::shared_ptr<Chunk_NS::Chunk>> chunks_to_hold;
      for (const auto chunk : chunks) {
        if (need_to_hold_chunk(chunk.get(), ra_exe_unit_)) {
          chunks_to_hold.push_back(chunk);
        }
      }
      rows_ptr->holdChunks(chunks_to_hold);
      rows_ptr->holdChunkIterators(chunk_iterators_ptr);
    }
  }
  {
    std::lock_guard<std::mutex> lock(reduce_mutex_);
    if (err) {
      *error_code_ = err;
    }
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
    const CompilationOptions& co,
    const size_t context_count,
    const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
    const ColumnCacheMap& column_cache,
    int32_t* error_code,
    RenderInfo* render_info)
    : executor_(executor)
    , ra_exe_unit_(ra_exe_unit)
    , query_infos_(query_infos)
    , cat_(cat)
    , co_(co)
    , query_contexts_(context_count)
    , query_context_mutexes_(context_count)
    , row_set_mem_owner_(row_set_mem_owner)
    , error_code_(error_code)
    , render_info_(render_info)
    , columnarized_table_cache_(column_cache) {
  all_fragment_results_.reserve(query_infos_.front().info.fragments.size());
}

int8_t Executor::ExecutionDispatch::compile(const Executor::JoinInfo& join_info,
                                            const size_t max_groups_buffer_entry_guess,
                                            const int8_t crt_min_byte_width,
                                            const ExecutionOptions& options,
                                            const bool has_cardinality_estimation) {
  int8_t actual_min_byte_width{MAX_BYTE_WIDTH_SUPPORTED};
  auto compile_on_cpu = [&]() {
    const CompilationOptions co_cpu{ExecutorDeviceType::CPU,
                                    co_.hoist_literals_,
                                    co_.opt_level_,
                                    co_.with_dynamic_watchdog_};

    try {
      OOM_TRACE_PUSH();
      compilation_result_cpu_ = executor_->compileWorkUnit(
          query_infos_,
          ra_exe_unit_,
          co_cpu,
          options,
          cat_.get_dataMgr().cudaMgr_,
          render_info_ && render_info_->isPotentialInSituRender() ? false : true,
          row_set_mem_owner_,
          max_groups_buffer_entry_guess,
          executor_->small_groups_buffer_entry_count_,
          crt_min_byte_width,
          join_info,
          has_cardinality_estimation,
          columnarized_table_cache_,
          render_info_);
    } catch (const CompilationRetryNoLazyFetch&) {
      OOM_TRACE_PUSH();
      if (executor_->cgen_state_->module_) {
        delete executor_->cgen_state_->module_;
      }
      compilation_result_cpu_ =
          executor_->compileWorkUnit(query_infos_,
                                     ra_exe_unit_,
                                     co_cpu,
                                     options,
                                     cat_.get_dataMgr().cudaMgr_,
                                     false,
                                     row_set_mem_owner_,
                                     max_groups_buffer_entry_guess,
                                     executor_->small_groups_buffer_entry_count_,
                                     crt_min_byte_width,
                                     join_info,
                                     has_cardinality_estimation,
                                     columnarized_table_cache_,
                                     render_info_);
    }
    actual_min_byte_width =
        compilation_result_cpu_.query_mem_desc.updateActualMinByteWidth(
            actual_min_byte_width);
  };

  if (co_.device_type_ == ExecutorDeviceType::CPU) {
    compile_on_cpu();
  }

  if (co_.device_type_ == ExecutorDeviceType::GPU) {
    const CompilationOptions co_gpu{ExecutorDeviceType::GPU,
                                    co_.hoist_literals_,
                                    co_.opt_level_,
                                    co_.with_dynamic_watchdog_};
    try {
      OOM_TRACE_PUSH();
      compilation_result_gpu_ = executor_->compileWorkUnit(
          query_infos_,
          ra_exe_unit_,
          co_gpu,
          options,
          cat_.get_dataMgr().cudaMgr_,
          render_info_ && render_info_->isPotentialInSituRender() ? false : true,
          row_set_mem_owner_,
          max_groups_buffer_entry_guess,
          executor_->small_groups_buffer_entry_count_,
          crt_min_byte_width,
          join_info,
          has_cardinality_estimation,
          columnarized_table_cache_,
          render_info_);
    } catch (const CompilationRetryNoLazyFetch&) {
      OOM_TRACE_PUSH();
      if (executor_->cgen_state_->module_) {
        delete executor_->cgen_state_->module_;
      }
      compilation_result_gpu_ =
          executor_->compileWorkUnit(query_infos_,
                                     ra_exe_unit_,
                                     co_gpu,
                                     options,
                                     cat_.get_dataMgr().cudaMgr_,
                                     false,
                                     row_set_mem_owner_,
                                     max_groups_buffer_entry_guess,
                                     executor_->small_groups_buffer_entry_count_,
                                     crt_min_byte_width,
                                     join_info,
                                     has_cardinality_estimation,
                                     columnarized_table_cache_,
                                     render_info_);
    }
    actual_min_byte_width =
        compilation_result_gpu_.query_mem_desc.updateActualMinByteWidth(
            actual_min_byte_width);
  }

  return std::max(actual_min_byte_width, crt_min_byte_width);
}

void Executor::ExecutionDispatch::run(const ExecutorDeviceType chosen_device_type,
                                      int chosen_device_id,
                                      const ExecutionOptions& options,
                                      const FragmentsList& frag_list,
                                      const size_t ctx_idx,
                                      const int64_t rowid_lookup_key) noexcept {
  try {
    runImpl(chosen_device_type,
            chosen_device_id,
            options,
            frag_list,
            ctx_idx,
            rowid_lookup_key);
  } catch (const std::bad_alloc& e) {
    std::lock_guard<std::mutex> lock(reduce_mutex_);
    LOG(ERROR) << e.what();
    *error_code_ = ERR_OUT_OF_CPU_MEM;
  } catch (const OutOfHostMemory& e) {
    std::lock_guard<std::mutex> lock(reduce_mutex_);
    LOG(ERROR) << e.what();
    *error_code_ = ERR_OUT_OF_CPU_MEM;
  } catch (const OutOfRenderMemory& e) {
    std::lock_guard<std::mutex> lock(reduce_mutex_);
    LOG(ERROR) << e.what();
    *error_code_ = ERR_OUT_OF_RENDER_MEM;
  } catch (const OutOfMemory& e) {
    std::lock_guard<std::mutex> lock(reduce_mutex_);
    LOG(ERROR) << e.what();
    *error_code_ = ERR_OUT_OF_GPU_MEM;
  } catch (const ColumnarConversionNotSupported& e) {
    std::lock_guard<std::mutex> lock(reduce_mutex_);
    *error_code_ = ERR_COLUMNAR_CONVERSION_NOT_SUPPORTED;
  } catch (const TooManyLiterals&) {
    std::lock_guard<std::mutex> lock(reduce_mutex_);
    *error_code_ = ERR_TOO_MANY_LITERALS;
  } catch (const SringConstInResultSet& e) {
    std::lock_guard<std::mutex> lock(reduce_mutex_);
    *error_code_ = ERR_STRING_CONST_IN_RESULTSET;
    LOG(INFO) << e.what();
  }
}

const int8_t* Executor::ExecutionDispatch::getScanColumn(
    const int table_id,
    const int frag_id,
    const int col_id,
    const std::map<int, const TableFragments*>& all_tables_fragments,
    std::list<std::shared_ptr<Chunk_NS::Chunk>>& chunk_holder,
    std::list<ChunkIter>& chunk_iter_holder,
    const Data_Namespace::MemoryLevel memory_level,
    const int device_id) const {
  static std::mutex varlen_chunk_mutex;  // TODO(alex): remove
  static std::mutex chunk_list_mutex;
  const auto fragments_it = all_tables_fragments.find(table_id);
  CHECK(fragments_it != all_tables_fragments.end());
  const auto fragments = fragments_it->second;
  const auto& fragment = (*fragments)[frag_id];
  if (fragment.isEmptyPhysicalFragment()) {
    return nullptr;
  }
  std::shared_ptr<Chunk_NS::Chunk> chunk;
  auto chunk_meta_it = fragment.getChunkMetadataMap().find(col_id);
  CHECK(chunk_meta_it != fragment.getChunkMetadataMap().end());
  CHECK(table_id > 0);
  auto cd = get_column_descriptor(col_id, table_id, cat_);
  CHECK(cd);
  const auto col_type =
      get_column_type(col_id, table_id, cd, executor_->temporary_tables_);
  const bool is_real_string =
      col_type.is_string() && col_type.get_compression() == kENCODING_NONE;
  const bool is_varlen =
      is_real_string ||
      col_type.is_array();  // TODO: should it be col_type.is_varlen_array() ?
  {
    ChunkKey chunk_key{
        cat_.get_currentDB().dbId, fragment.physicalTableId, col_id, fragment.fragmentId};
    std::unique_ptr<std::lock_guard<std::mutex>> varlen_chunk_lock;
    if (is_varlen) {
      varlen_chunk_lock.reset(new std::lock_guard<std::mutex>(varlen_chunk_mutex));
    }
    OOM_TRACE_PUSH(+": chunk key [" + showChunk(chunk_key) + "]");
    chunk = Chunk_NS::Chunk::getChunk(
        cd,
        &cat_.get_dataMgr(),
        chunk_key,
        memory_level,
        memory_level == Data_Namespace::CPU_LEVEL ? 0 : device_id,
        chunk_meta_it->second.numBytes,
        chunk_meta_it->second.numElements);
    std::lock_guard<std::mutex> chunk_list_lock(chunk_list_mutex);
    chunk_holder.push_back(chunk);
  }
  if (is_varlen) {
    CHECK_GT(table_id, 0);
    CHECK(chunk_meta_it != fragment.getChunkMetadataMap().end());
    chunk_iter_holder.push_back(chunk->begin_iterator(chunk_meta_it->second));
    auto& chunk_iter = chunk_iter_holder.back();
    if (memory_level == Data_Namespace::CPU_LEVEL) {
      return reinterpret_cast<int8_t*>(&chunk_iter);
    } else {
      CHECK_EQ(Data_Namespace::GPU_LEVEL, memory_level);
      auto& data_mgr = cat_.get_dataMgr();
      auto chunk_iter_gpu =
          alloc_gpu_mem(&data_mgr, sizeof(ChunkIter), device_id, nullptr);
      copy_to_gpu(&data_mgr, chunk_iter_gpu, &chunk_iter, sizeof(ChunkIter), device_id);
      return reinterpret_cast<int8_t*>(chunk_iter_gpu);
    }
  } else {
    auto ab = chunk->get_buffer();
    CHECK(ab->getMemoryPtr());
    return ab->getMemoryPtr();  // @TODO(alex) change to use ChunkIter
  }
}

#ifdef ENABLE_MULTIFRAG_JOIN
const int8_t* Executor::ExecutionDispatch::getAllScanColumnFrags(
    const int table_id,
    const int col_id,
    const std::map<int, const TableFragments*>& all_tables_fragments,
    const Data_Namespace::MemoryLevel memory_level,
    const int device_id) const {
  const auto fragments_it = all_tables_fragments.find(table_id);
  CHECK(fragments_it != all_tables_fragments.end());
  const auto fragments = fragments_it->second;
  const auto frag_count = fragments->size();
  std::vector<std::unique_ptr<ColumnarResults>> column_frags;
  const ColumnarResults* table_column = nullptr;
  const InputColDescriptor col_desc(col_id, table_id, int(0));
  CHECK(col_desc.getScanDesc().getSourceType() == InputSourceType::TABLE);
  {
    std::lock_guard<std::mutex> columnar_conversion_guard(columnar_conversion_mutex_);
    auto column_it = columnarized_scan_table_cache_.find(col_desc);
    if (column_it == columnarized_scan_table_cache_.end()) {
      for (size_t frag_id = 0; frag_id < frag_count; ++frag_id) {
        std::list<std::shared_ptr<Chunk_NS::Chunk>> chunk_holder;
        std::list<ChunkIter> chunk_iter_holder;
        const auto& fragment = (*fragments)[frag_id];
        if (fragment.isEmptyPhysicalFragment()) {
          continue;
        }
        auto chunk_meta_it = fragment.getChunkMetadataMap().find(col_id);
        CHECK(chunk_meta_it != fragment.getChunkMetadataMap().end());
        auto col_buffer = getScanColumn(table_id,
                                        static_cast<int>(frag_id),
                                        col_id,
                                        all_tables_fragments,
                                        chunk_holder,
                                        chunk_iter_holder,
                                        Data_Namespace::CPU_LEVEL,
                                        int(0));
        column_frags.push_back(
            boost::make_unique<ColumnarResults>(row_set_mem_owner_,
                                                col_buffer,
                                                fragment.getNumTuples(),
                                                chunk_meta_it->second.sqlType));
      }
      auto merged_results =
          ColumnarResults::mergeResults(row_set_mem_owner_, column_frags);
      table_column = merged_results.get();
      columnarized_scan_table_cache_.emplace(col_desc, std::move(merged_results));
    } else {
      table_column = column_it->second.get();
    }
  }
  return getColumn(table_column, 0, &cat_.get_dataMgr(), memory_level, device_id);
}
#endif

const int8_t* Executor::ExecutionDispatch::getColumn(
    const InputColDescriptor* col_desc,
    const int frag_id,
    const std::map<int, const TableFragments*>& all_tables_fragments,
    const std::map<size_t, std::vector<uint64_t>>& tab_id_to_frag_offsets,
    const Data_Namespace::MemoryLevel memory_level,
    const int device_id,
    const bool is_rowid) const {
  CHECK(col_desc);
  auto ind_col_desc = dynamic_cast<const IndirectInputColDescriptor*>(col_desc);
  if (!ind_col_desc) {
    const auto table_id = col_desc->getScanDesc().getTableId();
    return getColumn(get_temporary_table(executor_->temporary_tables_, table_id),
                     table_id,
                     frag_id,
                     col_desc->getColId(),
                     memory_level,
                     device_id);
  }

  const auto ref_table_id = ind_col_desc->getIndirectDesc().getTableId();
  const auto ref_col_id = ind_col_desc->getRefColIndex();
  const auto iter_table_id = ind_col_desc->getIterDesc().getTableId();
  const auto iter_col_id = ind_col_desc->getIterIndex();
  const auto& iter_buffer =
      get_temporary_table(executor_->temporary_tables_, iter_table_id);
  const bool ref_tab_is_result =
      ind_col_desc->getIndirectDesc().getSourceType() == InputSourceType::RESULT;

  const InputColDescriptor iter_desc(
      iter_col_id, iter_table_id, ind_col_desc->getIterDesc().getNestLevel());
  CHECK_LE(size_t(3), ra_exe_unit_.join_dimensions.size());
  const std::vector<std::pair<int, size_t>> previous_join_dims(
      ra_exe_unit_.join_dimensions.begin(),
      std::prev(ra_exe_unit_.join_dimensions.end()));
  for (const auto& table : previous_join_dims) {
    if (!table.second) {
      return nullptr;
    }
  }
  const auto ref_frag_id =
      get_mapped_frag_id_of_src_table(previous_join_dims, ref_table_id, frag_id);
  CacheKey sub_key;
  auto ref_col_id_for_cache = ref_col_id;
  const ColumnarResults* result{nullptr};
  {
    std::lock_guard<std::mutex> columnar_conversion_guard(columnar_conversion_mutex_);
    if (columnarized_table_cache_.empty() ||
        !columnarized_table_cache_.count(ref_table_id)) {
      columnarized_table_cache_.insert(std::make_pair(
          iter_table_id,
          std::unordered_map<int, std::shared_ptr<const ColumnarResults>>()));
    }
    auto& frag_id_to_iters = columnarized_table_cache_[iter_table_id];
    if (frag_id_to_iters.empty() || !frag_id_to_iters.count(frag_id)) {
      frag_id_to_iters.insert(
          std::make_pair(frag_id,
                         std::shared_ptr<const ColumnarResults>(columnarize_result(
                             row_set_mem_owner_, iter_buffer, frag_id))));
    }

    if (columnarized_ref_table_cache_.empty() ||
        !columnarized_ref_table_cache_.count(iter_desc)) {
      columnarized_ref_table_cache_.insert(std::make_pair(
          iter_desc,
          std::unordered_map<CacheKey, std::unique_ptr<const ColumnarResults>>()));
    }
    auto& frag_id_to_result = columnarized_ref_table_cache_[iter_desc];
    if (ref_tab_is_result) {
      CHECK(!is_rowid);
      const auto& ref_buffer =
          get_temporary_table(executor_->temporary_tables_, ref_table_id);
      if (columnarized_table_cache_.empty() ||
          !columnarized_table_cache_.count(ref_table_id)) {
        columnarized_table_cache_.insert(std::make_pair(
            ref_table_id,
            std::unordered_map<int, std::shared_ptr<const ColumnarResults>>()));
      }
      auto& frag_id_to_ref = columnarized_table_cache_[ref_table_id];
      if (frag_id_to_ref.empty() || !frag_id_to_ref.count(ref_frag_id)) {
        frag_id_to_ref.insert(
            std::make_pair(ref_frag_id,
                           std::shared_ptr<const ColumnarResults>(columnarize_result(
                               row_set_mem_owner_, ref_buffer, ref_frag_id))));
      }
      sub_key = {frag_id};
      if (frag_id_to_result.empty() || !frag_id_to_result.count(sub_key)) {
        frag_id_to_result.insert(std::make_pair(
            sub_key,
            ColumnarResults::createIndexedResults(row_set_mem_owner_,
                                                  *frag_id_to_ref[ref_frag_id],
                                                  *frag_id_to_iters[frag_id],
                                                  iter_col_id)));
      }
    } else {
      sub_key = {frag_id, ref_col_id};
      if (frag_id_to_result.empty() || !frag_id_to_result.count(sub_key)) {
        const auto frag_offsets_it = tab_id_to_frag_offsets.find(ref_table_id);
        CHECK(frag_offsets_it != tab_id_to_frag_offsets.end());
        const auto& frag_offsets = frag_offsets_it->second;
        CHECK_LT(ref_frag_id, frag_offsets.size());
        // TODO(miyu): Check rowid offseting
        if (is_rowid) {
          frag_id_to_result.insert(std::make_pair(
              sub_key,
              ColumnarResults::createOffsetResults(row_set_mem_owner_,
                                                   *frag_id_to_iters[frag_id],
                                                   iter_col_id,
                                                   frag_offsets[ref_frag_id])));
        } else {
#ifdef ENABLE_MULTIFRAG_JOIN
          // Each dispatch has only one fragment of outer table.
          if (ref_table_id != ra_exe_unit_.join_dimensions[0].first) {
            auto ref_frags =
                getAllScanColumnFrags(ref_table_id, ref_col_id, all_tables_fragments);
            frag_id_to_result.insert(std::make_pair(
                sub_key,
                ColumnarResults::createIndexedResults(row_set_mem_owner_,
                                                      ref_frags,
                                                      frag_offsets,
                                                      *frag_id_to_iters[frag_id],
                                                      iter_col_id)));
          } else
#endif
          {
            const auto fragments_it = all_tables_fragments.find(ref_table_id);
            CHECK(fragments_it != all_tables_fragments.end());
            const auto fragments = fragments_it->second;
            const auto& fragment = (*fragments)[ref_frag_id];
            auto chunk_meta_it = fragment.getChunkMetadataMap().find(ref_col_id);
            CHECK(chunk_meta_it != fragment.getChunkMetadataMap().end());
            std::list<std::shared_ptr<Chunk_NS::Chunk>> chunk_holder;
            std::list<ChunkIter> chunk_iter_holder;
            auto col_buffer = getScanColumn(ref_table_id,
                                            ref_frag_id,
                                            ref_col_id,
                                            all_tables_fragments,
                                            chunk_holder,
                                            chunk_iter_holder,
                                            Data_Namespace::CPU_LEVEL,
                                            device_id);
            ColumnarResults ref_values(row_set_mem_owner_,
                                       col_buffer,
                                       fragment.getNumTuples(),
                                       chunk_meta_it->second.sqlType);
            frag_id_to_result.insert(std::make_pair(
                sub_key,
                ColumnarResults::createIndexedResults(row_set_mem_owner_,
                                                      ref_values,
                                                      *frag_id_to_iters[frag_id],
                                                      iter_col_id)));
          }
        }
      }
      ref_col_id_for_cache = 0;
    }
    CHECK_NE(size_t(0), columnarized_ref_table_cache_.count(iter_desc));
    result = columnarized_ref_table_cache_[iter_desc][sub_key].get();
  }
  CHECK_GE(ref_col_id, 0);
  return getColumn(
      result, ref_col_id_for_cache, &cat_.get_dataMgr(), memory_level, device_id);
}

const int8_t* Executor::ExecutionDispatch::getColumn(
    const ColumnarResults* columnar_results,
    const int col_id,
    Data_Namespace::DataMgr* data_mgr,
    const Data_Namespace::MemoryLevel memory_level,
    const int device_id) {
  const auto& col_buffers = columnar_results->getColumnBuffers();
  CHECK_LT(static_cast<size_t>(col_id), col_buffers.size());
  if (memory_level == Data_Namespace::GPU_LEVEL) {
    const auto& col_ti = columnar_results->getColumnType(col_id);
    const auto num_bytes = columnar_results->size() * col_ti.get_size();
    OOM_TRACE_PUSH(+": device_id " + std::to_string(device_id) + ", num_bytes " +
                   std::to_string(num_bytes) + ", col_id " + std::to_string(col_id));
    auto gpu_col_buffer = alloc_gpu_mem(data_mgr, num_bytes, device_id, nullptr);
    copy_to_gpu(data_mgr, gpu_col_buffer, col_buffers[col_id], num_bytes, device_id);
    return reinterpret_cast<const int8_t*>(gpu_col_buffer);
  }
  return col_buffers[col_id];
}

std::string Executor::ExecutionDispatch::getIR(
    const ExecutorDeviceType device_type) const {
  CHECK(device_type == ExecutorDeviceType::CPU || device_type == ExecutorDeviceType::GPU);
  if (device_type == ExecutorDeviceType::CPU) {
    return compilation_result_cpu_.llvm_ir;
  }
  return compilation_result_gpu_.llvm_ir;
}

ExecutorDeviceType Executor::ExecutionDispatch::getDeviceType() const {
  return co_.device_type_;
}

const RelAlgExecutionUnit& Executor::ExecutionDispatch::getExecutionUnit() const {
  return ra_exe_unit_;
}

const QueryMemoryDescriptor& Executor::ExecutionDispatch::getQueryMemoryDescriptor()
    const {
  // TODO(alex): make query_mem_desc easily available
  return compilation_result_cpu_.native_functions.empty()
             ? compilation_result_gpu_.query_mem_desc
             : compilation_result_cpu_.query_mem_desc;
}

const bool Executor::ExecutionDispatch::outputColumnar() const {
  return compilation_result_cpu_.native_functions.empty()
             ? compilation_result_gpu_.output_columnar
             : false;
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

const std::vector<std::unique_ptr<QueryExecutionContext>>&
Executor::ExecutionDispatch::getQueryContexts() const {
  return query_contexts_;
}

std::vector<std::pair<ResultPtr, std::vector<size_t>>>&
Executor::ExecutionDispatch::getFragmentResults() {
  return all_fragment_results_;
}

std::pair<const int8_t*, size_t> Executor::ExecutionDispatch::getColumnFragment(
    Executor* executor,
    const Analyzer::ColumnVar& hash_col,
    const Fragmenter_Namespace::FragmentInfo& fragment,
    const Data_Namespace::MemoryLevel effective_mem_lvl,
    const int device_id,
    std::vector<std::shared_ptr<Chunk_NS::Chunk>>& chunks_owner,
    ColumnCacheMap& column_cache) {
  static std::mutex columnar_conversion_mutex;
  if (fragment.isEmptyPhysicalFragment()) {
    return {nullptr, 0};
  }
  auto chunk_meta_it = fragment.getChunkMetadataMap().find(hash_col.get_column_id());
  CHECK(chunk_meta_it != fragment.getChunkMetadataMap().end());
  const auto& catalog = *executor->getCatalog();
  const auto cd = get_column_descriptor_maybe(
      hash_col.get_column_id(), hash_col.get_table_id(), catalog);
  CHECK(!cd || !(cd->isVirtualCol));
  const int8_t* col_buff = nullptr;
  if (cd) {
    ChunkKey chunk_key{catalog.get_currentDB().dbId,
                       fragment.physicalTableId,
                       hash_col.get_column_id(),
                       fragment.fragmentId};
    OOM_TRACE_PUSH(+": chunk key [" + showChunk(chunk_key) + "]");
    const auto chunk = Chunk_NS::Chunk::getChunk(
        cd,
        &catalog.get_dataMgr(),
        chunk_key,
        effective_mem_lvl,
        effective_mem_lvl == Data_Namespace::CPU_LEVEL ? 0 : device_id,
        chunk_meta_it->second.numBytes,
        chunk_meta_it->second.numElements);
    chunks_owner.push_back(chunk);
    CHECK(chunk);
    auto ab = chunk->get_buffer();
    CHECK(ab->getMemoryPtr());
    col_buff = reinterpret_cast<int8_t*>(ab->getMemoryPtr());
  } else {
    const ColumnarResults* col_frag{nullptr};
    {
      std::lock_guard<std::mutex> columnar_conversion_guard(columnar_conversion_mutex);
      const auto table_id = hash_col.get_table_id();
      const auto frag_id = fragment.fragmentId;
      if (column_cache.empty() || !column_cache.count(table_id)) {
        column_cache.insert(std::make_pair(
            table_id, std::unordered_map<int, std::shared_ptr<const ColumnarResults>>()));
      }
      auto& frag_id_to_result = column_cache[table_id];
      if (frag_id_to_result.empty() || !frag_id_to_result.count(frag_id)) {
        frag_id_to_result.insert(std::make_pair(
            frag_id,
            std::shared_ptr<const ColumnarResults>(columnarize_result(
                executor->row_set_mem_owner_,
                get_temporary_table(executor->temporary_tables_, hash_col.get_table_id()),
                frag_id))));
      }
      col_frag = column_cache[table_id][frag_id].get();
    }
    col_buff = getColumn(col_frag,
                         hash_col.get_column_id(),
                         &catalog.get_dataMgr(),
                         effective_mem_lvl,
                         effective_mem_lvl == Data_Namespace::CPU_LEVEL ? 0 : device_id);
  }
  return {col_buff, fragment.getNumTuples()};
}

std::pair<const int8_t*, size_t> Executor::ExecutionDispatch::getAllColumnFragments(
    Executor* executor,
    const Analyzer::ColumnVar& hash_col,
    const std::deque<Fragmenter_Namespace::FragmentInfo>& fragments,
    std::vector<std::shared_ptr<Chunk_NS::Chunk>>& chunks_owner,
    ColumnCacheMap& column_cache) {
  CHECK(!fragments.empty());
  const size_t elem_width = hash_col.get_type_info().get_size();
  std::vector<const int8_t*> col_frags;
  std::vector<size_t> elem_counts;
  for (auto& frag : fragments) {
    const int8_t* col_frag = nullptr;
    size_t elem_count = 0;
    std::tie(col_frag, elem_count) = getColumnFragment(executor,
                                                       hash_col,
                                                       frag,
                                                       Data_Namespace::CPU_LEVEL,
                                                       0,
                                                       chunks_owner,
                                                       column_cache);
    if (col_frag == nullptr) {
      continue;
    }
    CHECK_NE(elem_count, size_t(0));
    col_frags.push_back(col_frag);
    elem_counts.push_back(elem_count);
  }
  CHECK(!col_frags.empty());
  CHECK_EQ(col_frags.size(), elem_counts.size());
  const auto total_elem_count =
      std::accumulate(elem_counts.begin(), elem_counts.end(), size_t(0));
  OOM_TRACE_PUSH(+": col_buff " + std::to_string(total_elem_count * elem_width));
  auto col_buff =
      reinterpret_cast<int8_t*>(checked_malloc(total_elem_count * elem_width));
  for (size_t i = 0, offset = 0; i < col_frags.size(); ++i) {
    memcpy(col_buff + offset, col_frags[i], elem_counts[i] * elem_width);
    offset += elem_counts[i] * elem_width;
  }
  return {col_buff, total_elem_count};
}
