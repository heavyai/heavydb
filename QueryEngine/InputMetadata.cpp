/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "InputMetadata.h"
#include "Execute.h"
#ifdef HAVE_CUDA
#include "GpuInitGroups.h"
#endif

#include "../Fragmenter/Fragmenter.h"

#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>
#include <future>
#include <numeric>
#include <optional>

extern bool g_enable_data_recycler;
extern bool g_use_chunk_metadata_cache;

InputTableInfoCache::InputTableInfoCache(Executor* executor) : executor_(executor) {}

Fragmenter_Namespace::TableInfo build_table_info(
    const std::vector<const TableDescriptor*>& all_physical_tds) {
  size_t total_number_of_tuples{0};
  Fragmenter_Namespace::TableInfo table_info_all_fragments;
  for (const TableDescriptor* td : all_physical_tds) {
    CHECK(td->fragmenter);
    const auto& fragment_metainfo = td->fragmenter->getFragmentsForQuery();
    total_number_of_tuples += fragment_metainfo.getPhysicalNumTuples();
    table_info_all_fragments.fragments.reserve(table_info_all_fragments.fragments.size() +
                                               fragment_metainfo.fragments.size());
    table_info_all_fragments.fragments.insert(table_info_all_fragments.fragments.end(),
                                              fragment_metainfo.fragments.begin(),
                                              fragment_metainfo.fragments.end());
  }
  table_info_all_fragments.setPhysicalNumTuples(total_number_of_tuples);
  return table_info_all_fragments;
}

Fragmenter_Namespace::TableInfo InputTableInfoCache::getTableInfo(
    const shared::TableKey& table_key) {
  const auto it = cache_.find(table_key);
  if (it != cache_.end()) {
    const auto& table_info = it->second;
    return table_info.copyTableInfo();
  }
  const auto cat = Catalog_Namespace::SysCatalog::instance().getCatalog(table_key.db_id);
  CHECK(cat);
  const auto td = cat->getMetadataForTable(table_key.table_id);
  CHECK(td);
  const auto td_per_fragment = cat->getPhysicalTablesDescriptors(td);
  auto table_info = build_table_info(td_per_fragment);
  auto it_ok = cache_.emplace(table_key, table_info.copyTableInfo());
  CHECK(it_ok.second);
  return table_info.copyTableInfo();
}

void InputTableInfoCache::updateTableInfo(
    const shared::TableKey& table_key,
    const Fragmenter_Namespace::TableInfo& table_info,
    const std::set<int>& device_ids_to_use) {
  cache_.erase(table_key);
  auto it_ok = cache_.emplace(table_key, table_info.copyTableInfo());
  CHECK(it_ok.second);
  for (auto const& frag : it_ok.first->second.fragments) {
    CHECK(device_ids_to_use.find(frag.deviceIds[MemoryLevel::GPU_LEVEL]) !=
          device_ids_to_use.end())
        << "Failed to update a device id of a fragment-" << frag.fragmentId;
  }
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

#ifdef HAVE_CUDA
std::optional<ChunkMetadataMap> synthesize_device_columnar_metadata(
    const ResultSet* rows) {
  if (!rows || rows->colCount() == 0 || rows->areAnyColumnsLazyFetched() ||
      rows->isTruncated()) {
    return std::nullopt;
  }

  struct ColumnDeviceFragments {
    SQLTypeInfo col_ti;
    SQLTypeInfo logical_ti;
    size_t elem_size{0};
    std::vector<ResultSet::DeviceColumnarBufferFragment> fragments;
  };

  std::vector<ColumnDeviceFragments> columns;
  columns.reserve(rows->colCount());
  for (size_t col_idx = 0; col_idx < rows->colCount(); ++col_idx) {
    const auto col_ti = rows->getColType(col_idx);
    const auto logical_ti = get_logical_type_info(col_ti);
    if (!uses_int_meta(col_ti) && !logical_ti.is_fp()) {
      return std::nullopt;
    }
    const auto elem_size = logical_ti.get_size();
    if (elem_size <= 0 || logical_ti.is_varlen()) {
      return std::nullopt;
    }
    std::vector<ResultSet::DeviceColumnarBufferFragment> fragments;
    if (!rows->getDeviceColumnarBufferFragments(
            col_idx, static_cast<size_t>(elem_size), fragments) ||
        fragments.empty()) {
      return std::nullopt;
    }
    columns.push_back(ColumnDeviceFragments{
        col_ti, logical_ti, static_cast<size_t>(elem_size), std::move(fragments)});
  }

  const auto row_count = rows->entryCount();
  ChunkMetadataMap metadata_map;
  for (size_t col_idx = 0; col_idx < columns.size(); ++col_idx) {
    const auto& column = columns[col_idx];
    auto encoder = Encoder::Create(nullptr, column.col_ti);
    CHECK(encoder);
    bool saw_nulls = false;
    bool saw_values = false;
    if (uses_int_meta(column.col_ti)) {
      const auto null_val = inline_int_null_val(column.col_ti);
      for (const auto& fragment : column.fragments) {
        if (!fragment.entry_count) {
          continue;
        }
        fragment.owner->waitForReadyEvent(fragment.ready_event);
        DeviceColumnFragmentStats stats;
        if (!compute_columnar_fragment_int_stats_on_device(
                fragment.buffer,
                fragment.entry_count,
                column.elem_size,
                null_val,
                fragment.device_id,
                stats,
                fragment.owner->getCudaStream())) {
          return std::nullopt;
        }
        if (stats.has_nulls) {
          saw_nulls = true;
        }
        if (stats.has_values) {
          encoder->updateStats(stats.int_min, false);
          encoder->updateStats(stats.int_max, false);
          saw_values = true;
        }
      }
      if (saw_nulls) {
        encoder->updateStats(null_val, true);
      }
    } else {
      CHECK(column.logical_ti.is_fp());
      const auto null_val = inline_fp_null_val(column.col_ti);
      for (const auto& fragment : column.fragments) {
        if (!fragment.entry_count) {
          continue;
        }
        fragment.owner->waitForReadyEvent(fragment.ready_event);
        DeviceColumnFragmentStats stats;
        if (!compute_columnar_fragment_fp_stats_on_device(
                fragment.buffer,
                fragment.entry_count,
                column.elem_size,
                null_val,
                fragment.device_id,
                stats,
                fragment.owner->getCudaStream())) {
          return std::nullopt;
        }
        if (stats.has_nulls) {
          saw_nulls = true;
        }
        if (stats.has_values) {
          encoder->updateStats(stats.fp_min, false);
          encoder->updateStats(stats.fp_max, false);
          saw_values = true;
        }
      }
      if (saw_nulls) {
        encoder->updateStats(null_val, true);
      }
    }

    auto chunk_metadata = std::make_shared<ChunkMetadata>();
    chunk_metadata->sqlType = column.col_ti;
    chunk_metadata->numElements = row_count;
    chunk_metadata->numBytes = row_count * column.elem_size;
    chunk_metadata->chunkStats = encoder->synthesizeChunkStats(column.col_ti);
    if (!saw_values) {
      chunk_metadata->chunkStats.has_nulls = saw_nulls;
    }
    const auto it_ok = metadata_map.emplace(col_idx, chunk_metadata);
    CHECK(it_ok.second);
  }

  return metadata_map;
}

#endif

Fragmenter_Namespace::TableInfo synthesize_table_info(int table_id,
                                                      const ResultSetPtr& rows,
                                                      Executor* executor) {
  std::vector<Fragmenter_Namespace::FragmentInfo> result;
  std::optional<size_t> known_physical_row_count;
  if (rows) {
    std::vector<ResultSet::DeviceColumnarFragmentInfo> device_columnar_fragments;
    const bool use_device_columnar_result_fragments =
        rows->getDeviceColumnarFragmentInfo(device_columnar_fragments);
    std::vector<size_t> columnar_fragment_row_counts;
    bool use_columnar_result_fragments = false;
    if (!use_device_columnar_result_fragments) {
      use_columnar_result_fragments =
          rows->getColumnarFragmentRowCounts(columnar_fragment_row_counts) &&
          columnar_fragment_row_counts.size() > 1;
    }
    const size_t fragment_count =
        use_device_columnar_result_fragments
            ? device_columnar_fragments.size()
            : (use_columnar_result_fragments ? columnar_fragment_row_counts.size()
                                             : size_t(1));
    if (use_device_columnar_result_fragments) {
      known_physical_row_count =
          std::accumulate(device_columnar_fragments.begin(),
                          device_columnar_fragments.end(),
                          size_t(0),
                          [](const size_t running_count,
                             const ResultSet::DeviceColumnarFragmentInfo& fragment) {
                            return running_count + fragment.entry_count;
                          });
    } else if (use_columnar_result_fragments) {
      known_physical_row_count = std::accumulate(columnar_fragment_row_counts.begin(),
                                                 columnar_fragment_row_counts.end(),
                                                 size_t(0));
    }
    result.resize(fragment_count);
    auto result_set_mutex = std::make_shared<std::mutex>();
    std::vector<int> selected_gpu_device_ids;
    if (executor->isDevicesToUseInitialized()) {
      auto const& device_ids = executor->getAvailableDevicesToProcessQuery();
      CHECK(!device_ids.empty());
      selected_gpu_device_ids.assign(device_ids.begin(), device_ids.end());
    }
    for (size_t fragment_idx = 0; fragment_idx < fragment_count; ++fragment_idx) {
      auto& fragment = result[fragment_idx];
      fragment.fragmentId = static_cast<int>(fragment_idx);
      fragment.deviceIds.resize(3);
      fragment.resultSet = rows.get();
      fragment.resultSetMutex = result_set_mutex;
      if (use_device_columnar_result_fragments) {
        fragment.setPhysicalNumTuples(
            device_columnar_fragments[fragment_idx].entry_count);
      } else if (use_columnar_result_fragments) {
        fragment.setPhysicalNumTuples(columnar_fragment_row_counts[fragment_idx]);
      } else {
        fragment.setPhysicalNumTuples(rows->rowCount());
      }
      if (use_device_columnar_result_fragments) {
        constexpr int gpu_mem_level = static_cast<int>(Data_Namespace::GPU_LEVEL);
        fragment.deviceIds[gpu_mem_level] =
            selected_gpu_device_ids.empty()
                ? device_columnar_fragments[fragment_idx].device_id
                : selected_gpu_device_ids[fragment_idx % selected_gpu_device_ids.size()];
      } else if (!selected_gpu_device_ids.empty()) {
        constexpr int gpu_mem_level = static_cast<int>(Data_Namespace::GPU_LEVEL);
        fragment.deviceIds[gpu_mem_level] =
            selected_gpu_device_ids[fragment_idx % selected_gpu_device_ids.size()];
      }
    }
    if ((use_device_columnar_result_fragments || !use_columnar_result_fragments) &&
        !result.empty() && !selected_gpu_device_ids.empty()) {
      constexpr int gpu_mem_level = static_cast<int>(Data_Namespace::GPU_LEVEL);
      VLOG(1) << "Synthesize query resultset fragment's device id to "
              << result.front().deviceIds[gpu_mem_level] << " (table_id: " << table_id
              << ")";
    }
  }
  Fragmenter_Namespace::TableInfo table_info;
  table_info.fragments = result;
  if (rows) {
    size_t row_count;
    if (known_physical_row_count) {
      row_count = *known_physical_row_count;
    } else {
      row_count = rows->rowCount();
    }
    table_info.setPhysicalNumTuples(row_count);
  }
  return table_info;
}

void collect_table_infos(std::vector<InputTableInfo>& table_infos,
                         const std::vector<InputDescriptor>& input_descs,
                         Executor* executor) {
  const auto temporary_tables = executor->getTemporaryTables();
  std::unordered_map<shared::TableKey, size_t> info_cache;
  for (const auto& input_desc : input_descs) {
    const auto& table_key = input_desc.getTableKey();
    const auto cached_index_it = info_cache.find(table_key);
    if (cached_index_it != info_cache.end()) {
      CHECK_LT(cached_index_it->second, table_infos.size());
      table_infos.push_back(
          {table_key, table_infos[cached_index_it->second].info.copyTableInfo()});
      continue;
    }

    if (input_desc.getSourceType() == InputSourceType::RESULT) {
      auto table_id = table_key.table_id;
      CHECK_LT(table_id, 0);
      CHECK(temporary_tables);
      const auto it = temporary_tables->find(table_id);
      if (it == temporary_tables->end()) {
        throw std::runtime_error("Failed to find previous query result for node " +
                                 std::to_string(-table_id));
      }
      table_infos.push_back(
          {{0, table_id}, synthesize_table_info(table_id, it->second, executor)});
    } else {
      CHECK(input_desc.getSourceType() == InputSourceType::TABLE);
      table_infos.push_back({table_key, executor->getTableInfo(table_key)});
    }
    CHECK(!table_infos.empty());
    info_cache.insert(std::make_pair(table_key, table_infos.size() - 1));
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

    // min/max of a column of a geotype is defined as the min/max of
    // all x and y coordinate values
    bool is_array = col_sql_type_info.is_array();
    bool is_geometry = col_sql_type_info.is_geometry();
    const auto col_type =
        (is_array ? col_sql_type_info.get_subtype()
                  : (is_geometry ? col_sql_type_info.get_elem_type().get_type()
                                 : col_sql_type_info.get_type()));
    const auto col_type_info =
        ((is_array || is_geometry) ? col_sql_type_info.get_elem_type()
                                   : col_sql_type_info);

    chunk_metadata->sqlType = col_type_info;
    chunk_metadata->numElements = row_count;

    const int8_t* values_buffer{nullptr};
    size_t values_count{0};
    if (FlatBufferManager::isFlatBuffer(columnar_buffer)) {
      CHECK(FlatBufferManager::isFlatBuffer(columnar_buffer));
      FlatBufferManager m{const_cast<int8_t*>(columnar_buffer)};
      chunk_metadata->numBytes = m.getBufferSize();
      if (is_geometry) {
        switch (col_sql_type_info.get_type()) {
          case kPOINT:
            // a geometry value is a pair of coordinates but its element
            // type value is a int or double, hence multiplication by 2:
            values_count = row_count * 2;
            values_buffer = m.get_values();
            break;
          case kLINESTRING:
          case kPOLYGON:
          case kMULTILINESTRING:
          case kMULTIPOLYGON: {
            values_count = m.getValuesCount();
            values_buffer = m.getValuesBuffer();
          } break;
          default:
            UNREACHABLE();
        }
      } else {
        CHECK(is_array);
        CHECK(m.isNestedArray());
        values_count = m.getValuesCount();
        values_buffer = m.getValuesBuffer();
      }
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

namespace {
union Number64 {
  double as_double;
  int64_t as_int64;
};

std::shared_ptr<ChunkMetadata> synthesize_lazy_fetch_source_column_metadata(
    const ResultSet* rows,
    const size_t column_idx) {
  const auto& column_type = rows->getColType(column_idx);
  if (!uses_int_meta(column_type) && !column_type.is_fp()) {
    return nullptr;
  }

  const auto source_metadata = rows->getLazyFetchSourceMetadata(column_idx);
  if (source_metadata.empty()) {
    return nullptr;
  }

  auto encoder = std::unique_ptr<Encoder>(Encoder::Create(nullptr, column_type));
  CHECK(encoder);
  bool saw_elements{false};
  bool saw_values{false};
  bool saw_nulls{false};
  for (const auto& source : source_metadata) {
    const auto& metadata = source.chunk_metadata;
    if (!metadata || metadata->isPlaceholder()) {
      return nullptr;
    }
    auto source_type = source.source_type;
    auto result_type = column_type;
    source_type.set_notnull(false);
    result_type.set_notnull(false);
    if (source_type != result_type) {
      return nullptr;
    }
    if (!metadata->numElements) {
      continue;
    }
    saw_elements = true;
    saw_nulls |= metadata->chunkStats.has_nulls;
    if (uses_int_meta(column_type)) {
      const auto min =
          extract_int_type_from_datum(metadata->chunkStats.min, metadata->sqlType);
      const auto max =
          extract_int_type_from_datum(metadata->chunkStats.max, metadata->sqlType);
      if (min <= max) {
        encoder->updateStats(min, false);
        encoder->updateStats(max, false);
        saw_values = true;
      } else if (!metadata->chunkStats.has_nulls) {
        return nullptr;
      }
    } else {
      const auto min =
          extract_fp_type_from_datum(metadata->chunkStats.min, metadata->sqlType);
      const auto max =
          extract_fp_type_from_datum(metadata->chunkStats.max, metadata->sqlType);
      if (min <= max) {
        encoder->updateStats(min, false);
        encoder->updateStats(max, false);
        saw_values = true;
      } else if (!metadata->chunkStats.has_nulls) {
        return nullptr;
      }
    }
  }
  if (!saw_elements || (!saw_values && !saw_nulls)) {
    return nullptr;
  }
  if (saw_nulls) {
    if (uses_int_meta(column_type)) {
      encoder->updateStats(inline_int_null_val(column_type), true);
    } else {
      encoder->updateStats(inline_fp_null_val(column_type), true);
    }
  }
  return std::make_shared<ChunkMetadata>(encoder->getMetadata(column_type));
}

std::optional<ChunkMetadataMap> synthesize_lazy_fetch_source_metadata(
    const ResultSet* rows) {
  if (!rows->hasLazyFetchSourceMetadata() ||
      rows->getQueryMemDesc().getQueryDescriptionType() ==
          QueryDescriptionType::TableFunction) {
    return std::nullopt;
  }

  std::vector<std::shared_ptr<ChunkMetadata>> source_metadata(rows->colCount());
  std::vector<bool> targets_to_skip(rows->colCount(), false);
  bool has_deferred_metadata{false};
  for (size_t column_idx = 0; column_idx < rows->colCount(); ++column_idx) {
    auto metadata = synthesize_lazy_fetch_source_column_metadata(rows, column_idx);
    if (metadata) {
      source_metadata[column_idx] = std::move(metadata);
      targets_to_skip[column_idx] = true;
      has_deferred_metadata = true;
    }
  }
  if (!has_deferred_metadata) {
    return std::nullopt;
  }
  const size_t worker_count =
      result_set::use_parallel_algorithms(*rows) ? cpu_threads() : 1;
  std::vector<std::vector<std::unique_ptr<Encoder>>> dummy_encoders;
  dummy_encoders.reserve(worker_count);
  for (size_t worker_idx = 0; worker_idx < worker_count; ++worker_idx) {
    dummy_encoders.emplace_back();
    dummy_encoders.back().reserve(rows->colCount());
    for (size_t column_idx = 0; column_idx < rows->colCount(); ++column_idx) {
      dummy_encoders.back().emplace_back(
          targets_to_skip[column_idx]
              ? nullptr
              : Encoder::Create(nullptr, rows->getColType(column_idx)));
    }
  }

  std::vector<SQLTypeInfo> row_column_types;
  row_column_types.reserve(rows->colCount());
  std::vector<Number64> column_null_values(rows->colCount());
  for (size_t column_idx = 0; column_idx < rows->colCount(); ++column_idx) {
    const auto column_type = rows->getColType(column_idx);
    row_column_types.push_back(column_type);
    if (uses_int_meta(column_type)) {
      column_null_values[column_idx].as_int64 = inline_int_null_val(column_type);
    } else if (column_type.is_fp()) {
      column_null_values[column_idx].as_double = inline_fp_null_val(column_type);
    } else {
      throw std::runtime_error(column_type.get_type_name() +
                               " is not supported in temporary table.");
    }
  }

  const auto process_row = [&row_column_types, &column_null_values, &targets_to_skip](
                               const std::vector<TargetValue>& row,
                               std::vector<std::unique_ptr<Encoder>>& encoders) {
    for (size_t column_idx = 0; column_idx < row_column_types.size(); ++column_idx) {
      if (targets_to_skip[column_idx]) {
        continue;
      }
      const auto& column_type = row_column_types[column_idx];
      const auto scalar_value = boost::get<ScalarTargetValue>(&row[column_idx]);
      CHECK(scalar_value);
      CHECK(encoders[column_idx]);
      if (uses_int_meta(column_type)) {
        const auto value = boost::get<int64_t>(scalar_value);
        CHECK(value);
        encoders[column_idx]->updateStats(
            *value, *value == column_null_values[column_idx].as_int64);
      } else {
        CHECK(column_type.is_fp());
        switch (column_type.get_type()) {
          case kFLOAT: {
            const auto value = boost::get<float>(scalar_value);
            CHECK(value);
            encoders[column_idx]->updateStats(
                *value, *value == column_null_values[column_idx].as_double);
            break;
          }
          case kDOUBLE: {
            const auto value = boost::get<double>(scalar_value);
            CHECK(value);
            encoders[column_idx]->updateStats(
                *value, *value == column_null_values[column_idx].as_double);
            break;
          }
          default:
            CHECK(false);
        }
      }
    }
  };

  const size_t entry_count = rows->entryCount();
  if (result_set::use_parallel_algorithms(*rows)) {
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, entry_count),
        [&process_row, &rows, &dummy_encoders, &targets_to_skip](
            const tbb::blocked_range<size_t>& range) {
          const size_t worker_idx = tbb::this_task_arena::current_thread_index();
          for (size_t entry_idx = range.begin(); entry_idx < range.end(); ++entry_idx) {
            const auto row = rows->getRowAtNoTranslations(entry_idx, targets_to_skip);
            if (!row.empty()) {
              process_row(row, dummy_encoders[worker_idx]);
            }
          }
        });
  } else {
    for (size_t entry_idx = 0; entry_idx < entry_count; ++entry_idx) {
      const auto row = rows->getRowAtNoTranslations(entry_idx, targets_to_skip);
      if (!row.empty()) {
        process_row(row, dummy_encoders.front());
      }
    }
  }

  for (size_t worker_idx = 1; worker_idx < worker_count; ++worker_idx) {
    for (size_t column_idx = 0; column_idx < rows->colCount(); ++column_idx) {
      if (!targets_to_skip[column_idx]) {
        dummy_encoders.front()[column_idx]->reduceStats(
            *dummy_encoders[worker_idx][column_idx]);
      }
    }
  }

  ChunkMetadataMap metadata_map;
  for (size_t column_idx = 0; column_idx < rows->colCount(); ++column_idx) {
    auto metadata = targets_to_skip[column_idx]
                        ? std::move(source_metadata[column_idx])
                        : std::make_shared<ChunkMetadata>(
                              dummy_encoders.front()[column_idx]->getMetadata(
                                  rows->getColType(column_idx)));
    const auto it_ok = metadata_map.emplace(column_idx, std::move(metadata));
    CHECK(it_ok.second);
  }
  return metadata_map;
}
}  // namespace

ChunkMetadataMap synthesize_metadata(const ResultSet* rows) {
  auto timer = DEBUG_TIMER(__func__);
  ChunkMetadataMap metadata_map;

  // If the ResultSet has no rows, fill with dummy metadata and return early.
  if (rows->definitelyHasNoRows()) {
    // resultset has no valid storage, so we fill dummy metadata and return early
    std::vector<std::unique_ptr<Encoder>> decoders;
    for (size_t i = 0; i < rows->colCount(); ++i) {
      decoders.emplace_back(Encoder::Create(nullptr, rows->getColType(i)));
      const auto it_ok =
          metadata_map.emplace(i,
                               std::make_shared<ChunkMetadata>(
                                   decoders.back()->getMetadata(rows->getColType(i))));
      CHECK(it_ok.second);
    }
    return metadata_map;
  }

#ifdef HAVE_CUDA
  if (auto device_metadata = synthesize_device_columnar_metadata(rows)) {
    return *device_metadata;
  }
#endif

  if (auto source_metadata = synthesize_lazy_fetch_source_metadata(rows)) {
    return *source_metadata;
  }

  // Create a vector of Encoder vectors for each worker.
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

  // For TableFunctions, call the optimized function we have for this format.
  if (rows->getQueryMemDesc().getQueryDescriptionType() ==
      QueryDescriptionType::TableFunction) {
    return synthesize_metadata_table_function(rows);
  }
  rows->moveToBegin();

  std::vector<SQLTypeInfo> row_col_ti;
  std::vector<Number64> col_null_vals(rows->colCount());
  for (size_t i = 0; i < rows->colCount(); i++) {
    auto const col_ti = rows->getColType(i);
    row_col_ti.push_back(col_ti);
    if (uses_int_meta(col_ti)) {
      col_null_vals[i].as_int64 = inline_int_null_val(col_ti);
    } else if (col_ti.is_fp()) {
      col_null_vals[i].as_double = inline_fp_null_val(col_ti);
    } else {
      throw std::runtime_error(col_ti.get_type_name() +
                               " is not supported in temporary table.");
    }
  }

  // Code in the do_work lambda runs for and processes each row.
  const auto do_work = [rows, &row_col_ti, &col_null_vals](
                           const std::vector<TargetValue>& crt_row,
                           std::vector<std::unique_ptr<Encoder>>& dummy_encoders) {
    for (size_t i = 0; i < rows->colCount(); ++i) {
      const auto& col_ti = row_col_ti[i];
      const auto& col_val = crt_row[i];
      const auto scalar_col_val = boost::get<ScalarTargetValue>(&col_val);
      CHECK(scalar_col_val);
      if (uses_int_meta(col_ti)) {
        const auto i64_p = boost::get<int64_t>(scalar_col_val);
        CHECK(i64_p);
        dummy_encoders[i]->updateStats(*i64_p, *i64_p == col_null_vals[i].as_int64);
      } else {
        CHECK(col_ti.is_fp());
        switch (col_ti.get_type()) {
          case kFLOAT: {
            const auto float_p = boost::get<float>(scalar_col_val);
            CHECK(float_p);
            dummy_encoders[i]->updateStats(*float_p,
                                           *float_p == col_null_vals[i].as_double);
            break;
          }
          case kDOUBLE: {
            const auto double_p = boost::get<double>(scalar_col_val);
            CHECK(double_p);
            dummy_encoders[i]->updateStats(*double_p,
                                           *double_p == col_null_vals[i].as_double);
            break;
          }
          default:
            CHECK(false);
        }
      }
    }
  };

  // Parallelize the processing using TBB if parallel algorithms are enabled.
  if (result_set::use_parallel_algorithms(*rows)) {
    const size_t entry_count = rows->entryCount();
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, entry_count),
        [&do_work, &rows, &dummy_encoders](const tbb::blocked_range<size_t>& range) {
          const size_t worker_idx = tbb::this_task_arena::current_thread_index();
          for (size_t i = range.begin(); i < range.end(); ++i) {
            const auto crt_row = rows->getRowAtNoTranslations(i);
            if (!crt_row.empty()) {
              do_work(crt_row, dummy_encoders[worker_idx]);
            }
          }
        });

  } else {
    // If parallel algorithms are not enabled, process the rows sequentially.
    while (true) {
      auto crt_row = rows->getNextRow(false, false);
      if (crt_row.empty()) {
        break;
      }
      do_work(crt_row, dummy_encoders[0]);
    }
  }
  rows->moveToBegin();

  // Reduce the results from each worker.
  for (size_t worker_idx = 1; worker_idx < worker_count; ++worker_idx) {
    CHECK_LT(worker_idx, dummy_encoders.size());
    const auto& worker_encoders = dummy_encoders[worker_idx];
    for (size_t i = 0; i < rows->colCount(); ++i) {
      dummy_encoders[0][i]->reduceStats(*worker_encoders[i]);
    }
  }
  // Add each column's results to the metadata map.
  for (size_t i = 0; i < rows->colCount(); ++i) {
    const auto it_ok =
        metadata_map.emplace(i,
                             std::make_shared<ChunkMetadata>(
                                 dummy_encoders[0][i]->getMetadata(rows->getColType(i))));
    CHECK(it_ok.second);
  }
  return metadata_map;
}

size_t get_frag_count_of_table(const shared::TableKey& table_key, Executor* executor) {
  const auto temporary_tables = executor->getTemporaryTables();
  CHECK(temporary_tables);
  auto it = temporary_tables->find(table_key.table_id);
  if (it != temporary_tables->end()) {
    CHECK_GE(int(0), table_key.table_id);
    return size_t(1);
  } else {
    const auto table_info = executor->getTableInfo(table_key);
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
          executor->getResultSetRecyclerHolder().getCachedChunkMetadata(
              resultSet->getQueryPlanHash());
      if (cached) {
        chunkMetadataMap = *cached;
        need_to_compute_metadata = false;
      }
    }
    if (need_to_compute_metadata) {
      chunkMetadataMap = synthesize_metadata(resultSet);
      if (enable_chunk_metadata_cache && !chunkMetadataMap.empty()) {
        executor->getResultSetRecyclerHolder().putChunkMetadataToCache(
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
    return std::accumulate(
        fragments.begin(), fragments.end(), size_t(0), [](const auto sum, const auto& f) {
          return sum + f.getNumTuples();
        });
  }
  return numTuples;
}

size_t Fragmenter_Namespace::TableInfo::getNumTuplesUpperBound() const {
  if (!fragments.empty() && fragments.front().resultSet) {
    if (fragments.size() == 1) {
      return fragments.front().resultSet->entryCount();
    }
    return getNumTuples();
  }
  return numTuples;
}

size_t Fragmenter_Namespace::TableInfo::getFragmentNumTuplesUpperBound() const {
  if (!fragments.empty() && fragments.front().resultSet) {
    if (fragments.size() > 1) {
      return std::accumulate(fragments.begin(),
                             fragments.end(),
                             size_t(0),
                             [](const auto max_rows, const auto& fragment) {
                               return std::max(max_rows, fragment.getNumTuples());
                             });
    }
    return fragments.front().resultSet->entryCount();
  }
  size_t fragment_num_tupples_upper_bound = 0;
  for (const auto& fragment : fragments) {
    fragment_num_tupples_upper_bound =
        std::max(fragment.getNumTuples(), fragment_num_tupples_upper_bound);
  }
  return fragment_num_tupples_upper_bound;
}

Fragmenter_Namespace::TableInfo Fragmenter_Namespace::TableInfo::copyTableInfo() const {
  Fragmenter_Namespace::TableInfo table_info_copy;
  table_info_copy.chunkKeyPrefix = chunkKeyPrefix;
  table_info_copy.fragments = fragments;
  table_info_copy.setPhysicalNumTuples(getPhysicalNumTuples());
  return table_info_copy;
}
