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

#include "InputMetadata.h"
#include "Execute.h"

#include "../Fragmenter/Fragmenter.h"

#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>
#include <future>

extern bool g_enable_data_recycler;
extern bool g_use_chunk_metadata_cache;

InputTableInfoCache::InputTableInfoCache(Executor* executor) : executor_(executor) {}

namespace {

Fragmenter_Namespace::TableInfo copy_table_info(
    const Fragmenter_Namespace::TableInfo& table_info) {
  Fragmenter_Namespace::TableInfo table_info_copy;
  table_info_copy.chunkKeyPrefix = table_info.chunkKeyPrefix;
  table_info_copy.fragments = table_info.fragments;
  table_info_copy.setPhysicalNumTuples(table_info.getPhysicalNumTuples());
  return table_info_copy;
}

}  // namespace

Fragmenter_Namespace::TableInfo build_table_info(
    const std::vector<const TableDescriptor*>& shard_tables) {
  size_t total_number_of_tuples{0};
  Fragmenter_Namespace::TableInfo table_info_all_shards;
  for (const TableDescriptor* shard_table : shard_tables) {
    CHECK(shard_table->fragmenter);
    const auto& shard_metainfo = shard_table->fragmenter->getFragmentsForQuery();
    total_number_of_tuples += shard_metainfo.getPhysicalNumTuples();
    table_info_all_shards.fragments.reserve(table_info_all_shards.fragments.size() +
                                            shard_metainfo.fragments.size());
    table_info_all_shards.fragments.insert(table_info_all_shards.fragments.end(),
                                           shard_metainfo.fragments.begin(),
                                           shard_metainfo.fragments.end());
  }
  table_info_all_shards.setPhysicalNumTuples(total_number_of_tuples);
  return table_info_all_shards;
}

Fragmenter_Namespace::TableInfo InputTableInfoCache::getTableInfo(const int table_id) {
  const auto it = cache_.find(table_id);
  if (it != cache_.end()) {
    const auto& table_info = it->second;
    return copy_table_info(table_info);
  }
  const auto cat = executor_->getCatalog();
  CHECK(cat);
  const auto td = cat->getMetadataForTable(table_id);
  CHECK(td);
  const auto shard_tables = cat->getPhysicalTablesDescriptors(td);
  auto table_info = build_table_info(shard_tables);
  auto it_ok = cache_.emplace(table_id, copy_table_info(table_info));
  CHECK(it_ok.second);
  return copy_table_info(table_info);
}

void InputTableInfoCache::clear() {
  decltype(cache_)().swap(cache_);
}

namespace {

bool uses_int_meta(const SQLTypeInfo& col_ti) {
  return col_ti.is_integer() || col_ti.is_decimal() || col_ti.is_time() ||
         col_ti.is_boolean() ||
         (col_ti.is_string() && col_ti.get_compression() == kENCODING_DICT);
}

Fragmenter_Namespace::TableInfo synthesize_table_info(const ResultSetPtr& rows) {
  std::vector<Fragmenter_Namespace::FragmentInfo> result;
  if (rows) {
    result.resize(1);
    auto& fragment = result.front();
    fragment.fragmentId = 0;
    fragment.deviceIds.resize(3);
    fragment.resultSet = rows.get();
    fragment.resultSetMutex.reset(new std::mutex());
  }
  Fragmenter_Namespace::TableInfo table_info;
  table_info.fragments = result;
  return table_info;
}

void collect_table_infos(std::vector<InputTableInfo>& table_infos,
                         const std::vector<InputDescriptor>& input_descs,
                         Executor* executor) {
  const auto temporary_tables = executor->getTemporaryTables();
  const auto cat = executor->getCatalog();
  CHECK(cat);
  std::unordered_map<int, size_t> info_cache;
  for (const auto& input_desc : input_descs) {
    const auto table_id = input_desc.getTableId();
    const auto cached_index_it = info_cache.find(table_id);
    if (cached_index_it != info_cache.end()) {
      CHECK_LT(cached_index_it->second, table_infos.size());
      table_infos.push_back(
          {table_id, copy_table_info(table_infos[cached_index_it->second].info)});
      continue;
    }
    if (input_desc.getSourceType() == InputSourceType::RESULT) {
      CHECK_LT(table_id, 0);
      CHECK(temporary_tables);
      const auto it = temporary_tables->find(table_id);
      LOG_IF(FATAL, it == temporary_tables->end())
          << "Failed to find previous query result for node " << -table_id;
      table_infos.push_back({table_id, synthesize_table_info(it->second)});
    } else {
      CHECK(input_desc.getSourceType() == InputSourceType::TABLE);
      table_infos.push_back({table_id, executor->getTableInfo(table_id)});
    }
    CHECK(!table_infos.empty());
    info_cache.insert(std::make_pair(table_id, table_infos.size() - 1));
  }
}

}  // namespace

template <typename T>
void compute_table_function_col_chunk_stats(
    std::shared_ptr<ChunkMetadata>& chunk_metadata,
    const T* values_buffer,
    const size_t values_count,
    const T null_val) {
  T min_val{std::numeric_limits<T>::max()};
  T max_val{std::numeric_limits<T>::lowest()};
  bool has_nulls{false};
  constexpr size_t parallel_stats_compute_threshold = 20000UL;
  if (values_count < parallel_stats_compute_threshold) {
    for (size_t row_idx = 0; row_idx < values_count; ++row_idx) {
      const T cell_val = values_buffer[row_idx];
      if (cell_val == null_val) {
        has_nulls = true;
        continue;
      }
      if (cell_val < min_val) {
        min_val = cell_val;
      }
      if (cell_val > max_val) {
        max_val = cell_val;
      }
    }
  } else {
    const size_t max_thread_count = std::thread::hardware_concurrency();
    const size_t max_inputs_per_thread = 20000;
    const size_t min_grain_size = max_inputs_per_thread / 2;
    const size_t num_threads =
        std::min(max_thread_count,
                 ((values_count + max_inputs_per_thread - 1) / max_inputs_per_thread));

    std::vector<T> threads_local_mins(num_threads, std::numeric_limits<T>::max());
    std::vector<T> threads_local_maxes(num_threads, std::numeric_limits<T>::lowest());
    std::vector<bool> threads_local_has_nulls(num_threads, false);
    tbb::task_arena limited_arena(num_threads);

    limited_arena.execute([&] {
      tbb::parallel_for(
          tbb::blocked_range<size_t>(0, values_count, min_grain_size),
          [&](const tbb::blocked_range<size_t>& r) {
            const size_t start_idx = r.begin();
            const size_t end_idx = r.end();
            T local_min_val = std::numeric_limits<T>::max();
            T local_max_val = std::numeric_limits<T>::lowest();
            bool local_has_nulls = false;
            for (size_t row_idx = start_idx; row_idx < end_idx; ++row_idx) {
              const T cell_val = values_buffer[row_idx];
              if (cell_val == null_val) {
                local_has_nulls = true;
                continue;
              }
              if (cell_val < local_min_val) {
                local_min_val = cell_val;
              }
              if (cell_val > local_max_val) {
                local_max_val = cell_val;
              }
            }
            size_t thread_idx = tbb::this_task_arena::current_thread_index();
            if (local_min_val < threads_local_mins[thread_idx]) {
              threads_local_mins[thread_idx] = local_min_val;
            }
            if (local_max_val > threads_local_maxes[thread_idx]) {
              threads_local_maxes[thread_idx] = local_max_val;
            }
            if (local_has_nulls) {
              threads_local_has_nulls[thread_idx] = true;
            }
          },
          tbb::simple_partitioner());
    });

    for (size_t thread_idx = 0; thread_idx < num_threads; ++thread_idx) {
      if (threads_local_mins[thread_idx] < min_val) {
        min_val = threads_local_mins[thread_idx];
      }
      if (threads_local_maxes[thread_idx] > max_val) {
        max_val = threads_local_maxes[thread_idx];
      }
      has_nulls |= threads_local_has_nulls[thread_idx];
    }
  }
  chunk_metadata->fillChunkStats(min_val, max_val, has_nulls);
}

ChunkMetadataMap synthesize_metadata_table_function(const ResultSet* rows) {
  CHECK(rows->getQueryMemDesc().getQueryDescriptionType() ==
        QueryDescriptionType::TableFunction);
  CHECK(rows->didOutputColumnar());
  CHECK(!(rows->areAnyColumnsLazyFetched()));
  const size_t col_count = rows->colCount();
  const auto row_count = rows->entryCount();

  ChunkMetadataMap chunk_metadata_map;

  for (size_t col_idx = 0; col_idx < col_count; ++col_idx) {
    std::shared_ptr<ChunkMetadata> chunk_metadata = std::make_shared<ChunkMetadata>();
    const int8_t* columnar_buffer = const_cast<int8_t*>(rows->getColumnarBuffer(col_idx));
    const auto col_sql_type_info = rows->getColType(col_idx);
    // Here, min/max of a column of arrays, col, is defined as
    // min/max(unnest(col)). That is, if is_array is true, the
    // metadata is supposed to be syntesized for a query like `SELECT
    // UNNEST(col_of_arrays) ... GROUP BY ...`. How can we verify that
    // here?
    bool is_array = col_sql_type_info.is_array();
    const auto col_type =
        (is_array ? col_sql_type_info.get_subtype() : col_sql_type_info.get_type());
    const auto col_type_info =
        (is_array ? col_sql_type_info.get_elem_type() : col_sql_type_info);

    chunk_metadata->sqlType = col_type_info;
    chunk_metadata->numElements = row_count;

    const int8_t* values_buffer;
    size_t values_count;
    if (is_array) {
      CHECK(FlatBufferManager::isFlatBuffer(columnar_buffer));
      FlatBufferManager m{const_cast<int8_t*>(columnar_buffer)};
      chunk_metadata->numBytes = m.flatbufferSize();
      values_count = m.VarlenArray_nof_values();
      values_buffer = m.VarlenArray_values();
    } else {
      chunk_metadata->numBytes = row_count * col_type_info.get_size();
      values_count = row_count;
      values_buffer = columnar_buffer;
    }

    if (col_type != kTEXT) {
      CHECK(col_type_info.get_compression() == kENCODING_NONE);
    } else {
      CHECK(col_type_info.get_compression() == kENCODING_DICT);
      CHECK_EQ(col_type_info.get_size(), sizeof(int32_t));
    }

    switch (col_type) {
      case kBOOLEAN:
      case kTINYINT:
        compute_table_function_col_chunk_stats(
            chunk_metadata,
            values_buffer,
            values_count,
            static_cast<int8_t>(inline_fixed_encoding_null_val(col_type_info)));
        break;
      case kSMALLINT:
        compute_table_function_col_chunk_stats(
            chunk_metadata,
            reinterpret_cast<const int16_t*>(values_buffer),
            values_count,
            static_cast<int16_t>(inline_fixed_encoding_null_val(col_type_info)));
        break;
      case kINT:
      case kTEXT:
        compute_table_function_col_chunk_stats(
            chunk_metadata,
            reinterpret_cast<const int32_t*>(values_buffer),
            values_count,
            static_cast<int32_t>(inline_fixed_encoding_null_val(col_type_info)));
        break;
      case kBIGINT:
      case kTIMESTAMP:
        compute_table_function_col_chunk_stats(
            chunk_metadata,
            reinterpret_cast<const int64_t*>(values_buffer),
            values_count,
            static_cast<int64_t>(inline_fixed_encoding_null_val(col_type_info)));
        break;
      case kFLOAT:
        // For float use the typed null accessor as the generic one converts to double,
        // and do not want to risk loss of precision
        compute_table_function_col_chunk_stats(
            chunk_metadata,
            reinterpret_cast<const float*>(values_buffer),
            values_count,
            inline_fp_null_value<float>());
        break;
      case kDOUBLE:
        compute_table_function_col_chunk_stats(
            chunk_metadata,
            reinterpret_cast<const double*>(values_buffer),
            values_count,
            inline_fp_null_value<double>());
        break;
      default:
        UNREACHABLE();
    }
    chunk_metadata_map.emplace(col_idx, chunk_metadata);
  }
  return chunk_metadata_map;
}

ChunkMetadataMap synthesize_metadata(const ResultSet* rows) {
  auto timer = DEBUG_TIMER(__func__);
  ChunkMetadataMap metadata_map;

  if (rows->definitelyHasNoRows()) {
    // resultset has no valid storage, so we fill dummy metadata and return early
    std::vector<std::unique_ptr<Encoder>> decoders;
    for (size_t i = 0; i < rows->colCount(); ++i) {
      decoders.emplace_back(Encoder::Create(nullptr, rows->getColType(i)));
      const auto it_ok =
          metadata_map.emplace(i, decoders.back()->getMetadata(rows->getColType(i)));
      CHECK(it_ok.second);
    }
    return metadata_map;
  }

  std::vector<std::vector<std::unique_ptr<Encoder>>> dummy_encoders;
  const size_t worker_count =
      result_set::use_parallel_algorithms(*rows) ? cpu_threads() : 1;
  for (size_t worker_idx = 0; worker_idx < worker_count; ++worker_idx) {
    dummy_encoders.emplace_back();
    for (size_t i = 0; i < rows->colCount(); ++i) {
      const auto& col_ti = rows->getColType(i);
      dummy_encoders.back().emplace_back(Encoder::Create(nullptr, col_ti));
    }
  }

  if (rows->getQueryMemDesc().getQueryDescriptionType() ==
      QueryDescriptionType::TableFunction) {
    return synthesize_metadata_table_function(rows);
  }
  rows->moveToBegin();
  const auto do_work = [rows](const std::vector<TargetValue>& crt_row,
                              std::vector<std::unique_ptr<Encoder>>& dummy_encoders) {
    for (size_t i = 0; i < rows->colCount(); ++i) {
      const auto& col_ti = rows->getColType(i);
      const auto& col_val = crt_row[i];
      const auto scalar_col_val = boost::get<ScalarTargetValue>(&col_val);
      CHECK(scalar_col_val);
      if (uses_int_meta(col_ti)) {
        const auto i64_p = boost::get<int64_t>(scalar_col_val);
        CHECK(i64_p);
        dummy_encoders[i]->updateStats(*i64_p, *i64_p == inline_int_null_val(col_ti));
      } else if (col_ti.is_fp()) {
        switch (col_ti.get_type()) {
          case kFLOAT: {
            const auto float_p = boost::get<float>(scalar_col_val);
            CHECK(float_p);
            dummy_encoders[i]->updateStats(*float_p,
                                           *float_p == inline_fp_null_val(col_ti));
            break;
          }
          case kDOUBLE: {
            const auto double_p = boost::get<double>(scalar_col_val);
            CHECK(double_p);
            dummy_encoders[i]->updateStats(*double_p,
                                           *double_p == inline_fp_null_val(col_ti));
            break;
          }
          default:
            CHECK(false);
        }
      } else {
        throw std::runtime_error(col_ti.get_type_name() +
                                 " is not supported in temporary table.");
      }
    }
  };
  if (result_set::use_parallel_algorithms(*rows)) {
    const size_t worker_count = cpu_threads();
    std::vector<std::future<void>> compute_stats_threads;
    const auto entry_count = rows->entryCount();
    for (size_t i = 0,
                start_entry = 0,
                stride = (entry_count + worker_count - 1) / worker_count;
         i < worker_count && start_entry < entry_count;
         ++i, start_entry += stride) {
      const auto end_entry = std::min(start_entry + stride, entry_count);
      compute_stats_threads.push_back(std::async(
          std::launch::async,
          [rows, &do_work, &dummy_encoders](
              const size_t start, const size_t end, const size_t worker_idx) {
            for (size_t i = start; i < end; ++i) {
              const auto crt_row = rows->getRowAtNoTranslations(i);
              if (!crt_row.empty()) {
                do_work(crt_row, dummy_encoders[worker_idx]);
              }
            }
          },
          start_entry,
          end_entry,
          i));
    }
    for (auto& child : compute_stats_threads) {
      child.wait();
    }
    for (auto& child : compute_stats_threads) {
      child.get();
    }
  } else {
    while (true) {
      auto crt_row = rows->getNextRow(false, false);
      if (crt_row.empty()) {
        break;
      }
      do_work(crt_row, dummy_encoders[0]);
    }
  }
  rows->moveToBegin();
  for (size_t worker_idx = 1; worker_idx < worker_count; ++worker_idx) {
    CHECK_LT(worker_idx, dummy_encoders.size());
    const auto& worker_encoders = dummy_encoders[worker_idx];
    for (size_t i = 0; i < rows->colCount(); ++i) {
      dummy_encoders[0][i]->reduceStats(*worker_encoders[i]);
    }
  }
  for (size_t i = 0; i < rows->colCount(); ++i) {
    const auto it_ok =
        metadata_map.emplace(i, dummy_encoders[0][i]->getMetadata(rows->getColType(i)));
    CHECK(it_ok.second);
  }
  return metadata_map;
}

size_t get_frag_count_of_table(const int table_id, Executor* executor) {
  const auto temporary_tables = executor->getTemporaryTables();
  CHECK(temporary_tables);
  auto it = temporary_tables->find(table_id);
  if (it != temporary_tables->end()) {
    CHECK_GE(int(0), table_id);
    return size_t(1);
  } else {
    const auto table_info = executor->getTableInfo(table_id);
    return table_info.fragments.size();
  }
}

std::vector<InputTableInfo> get_table_infos(
    const std::vector<InputDescriptor>& input_descs,
    Executor* executor) {
  std::vector<InputTableInfo> table_infos;
  collect_table_infos(table_infos, input_descs, executor);
  return table_infos;
}

std::vector<InputTableInfo> get_table_infos(const RelAlgExecutionUnit& ra_exe_unit,
                                            Executor* executor) {
  std::vector<InputTableInfo> table_infos;
  collect_table_infos(table_infos, ra_exe_unit.input_descs, executor);
  return table_infos;
}

const ChunkMetadataMap& Fragmenter_Namespace::FragmentInfo::getChunkMetadataMap() const {
  if (resultSet && !synthesizedMetadataIsValid) {
    bool need_to_compute_metadata = true;
    // we disable chunk metadata recycler when filter pushdown is enabled
    // since re-executing the query invalidates the cached metdata
    // todo(yoonmin): relax this
    bool enable_chunk_metadata_cache = g_enable_data_recycler &&
                                       g_use_chunk_metadata_cache &&
                                       !g_enable_filter_push_down;
    auto executor = Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID);
    if (enable_chunk_metadata_cache) {
      std::optional<ChunkMetadataMap> cached =
          executor->getRecultSetRecyclerHolder().getCachedChunkMetadata(
              resultSet->getQueryPlanHash());
      if (cached) {
        chunkMetadataMap = *cached;
        need_to_compute_metadata = false;
      }
    }
    if (need_to_compute_metadata) {
      chunkMetadataMap = synthesize_metadata(resultSet);
      if (enable_chunk_metadata_cache && !chunkMetadataMap.empty()) {
        executor->getRecultSetRecyclerHolder().putChunkMetadataToCache(
            resultSet->getQueryPlanHash(),
            resultSet->getInputTableKeys(),
            chunkMetadataMap);
      }
    }
    synthesizedMetadataIsValid = true;
  }
  return chunkMetadataMap;
}

ChunkMetadataMap Fragmenter_Namespace::FragmentInfo::getChunkMetadataMapPhysicalCopy()
    const {
  ChunkMetadataMap metadata_map;
  for (const auto& [column_id, chunk_metadata] : chunkMetadataMap) {
    metadata_map[column_id] = std::make_shared<ChunkMetadata>(*chunk_metadata);
  }
  return metadata_map;
}

size_t Fragmenter_Namespace::FragmentInfo::getNumTuples() const {
  std::unique_ptr<std::lock_guard<std::mutex>> lock;
  if (resultSetMutex) {
    lock.reset(new std::lock_guard<std::mutex>(*resultSetMutex));
  }
  CHECK_EQ(!!resultSet, !!resultSetMutex);
  if (resultSet && !synthesizedNumTuplesIsValid) {
    numTuples = resultSet->rowCount();
    synthesizedNumTuplesIsValid = true;
  }
  return numTuples;
}

size_t Fragmenter_Namespace::TableInfo::getNumTuples() const {
  if (!fragments.empty() && fragments.front().resultSet) {
    return fragments.front().getNumTuples();
  }
  return numTuples;
}

size_t Fragmenter_Namespace::TableInfo::getNumTuplesUpperBound() const {
  if (!fragments.empty() && fragments.front().resultSet) {
    return fragments.front().resultSet->entryCount();
  }
  return numTuples;
}

size_t Fragmenter_Namespace::TableInfo::getFragmentNumTuplesUpperBound() const {
  if (!fragments.empty() && fragments.front().resultSet) {
    return fragments.front().resultSet->entryCount();
  }
  size_t fragment_num_tupples_upper_bound = 0;
  for (const auto& fragment : fragments) {
    fragment_num_tupples_upper_bound =
        std::max(fragment.getNumTuples(), fragment_num_tupples_upper_bound);
  }
  return fragment_num_tupples_upper_bound;
}
