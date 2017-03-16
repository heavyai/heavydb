#include "InputMetadata.h"
#include "Execute.h"

#include "../Fragmenter/Fragmenter.h"

#include <future>

InputTableInfoCache::InputTableInfoCache(Executor* executor) : executor_(executor) {}

namespace {

Fragmenter_Namespace::TableInfo build_table_info(const Fragmenter_Namespace::TableInfo& table_info) {
  Fragmenter_Namespace::TableInfo table_info_copy;
  table_info_copy.chunkKeyPrefix = table_info.chunkKeyPrefix;
  table_info_copy.fragments = table_info.fragments;
  table_info_copy.numTuples = table_info.numTuples;
  return table_info_copy;
}

}  // namespace

Fragmenter_Namespace::TableInfo InputTableInfoCache::getTableInfo(const int table_id) {
  const auto it = cache_.find(table_id);
  if (it != cache_.end()) {
    const auto& table_info = it->second;
    return build_table_info(table_info);
  }
  const auto cat = executor_->getCatalog();
  CHECK(cat);
  const auto td = cat->getMetadataForTable(table_id);
  CHECK(td);
  const auto fragmenter = td->fragmenter;
  CHECK(fragmenter);
  auto table_info = fragmenter->getFragmentsForQuery();
  auto it_ok = cache_.emplace(table_id, build_table_info(table_info));
  CHECK(it_ok.second);
  return build_table_info(table_info);
}

void InputTableInfoCache::clear() {
  decltype(cache_)().swap(cache_);
}

namespace {

bool uses_int_meta(const SQLTypeInfo& col_ti) {
  return col_ti.is_integer() || col_ti.is_decimal() || col_ti.is_time() || col_ti.is_boolean() ||
         (col_ti.is_string() && col_ti.get_compression() == kENCODING_DICT);
}

std::map<int, ChunkMetadata> synthesize_metadata(const ResultRows* rows) {
  rows->moveToBegin();
  std::vector<std::vector<std::unique_ptr<Encoder>>> dummy_encoders;
  const size_t worker_count = use_parallel_algorithms(*rows) ? cpu_threads() : 1;
  for (size_t worker_idx = 0; worker_idx < worker_count; ++worker_idx) {
    dummy_encoders.emplace_back();
    for (size_t i = 0; i < rows->colCount(); ++i) {
      const auto& col_ti = rows->getColType(i);
      dummy_encoders.back().emplace_back(Encoder::Create(nullptr, col_ti));
    }
  }
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
            dummy_encoders[i]->updateStats(*float_p, *float_p == inline_fp_null_val(col_ti));
            break;
          }
          case kDOUBLE: {
            const auto double_p = boost::get<double>(scalar_col_val);
            CHECK(double_p);
            dummy_encoders[i]->updateStats(*double_p, *double_p == inline_fp_null_val(col_ti));
            break;
          }
          default:
            CHECK(false);
        }
      } else {
        throw std::runtime_error(col_ti.get_type_name() + " is not supported in temporary table.");
      }
    }
  };
  if (use_parallel_algorithms(*rows)) {
    const size_t worker_count = cpu_threads();
    std::vector<std::future<void>> compute_stats_threads;
    const auto entry_count = rows->getResultSet()->entryCount();
    for (size_t i = 0, start_entry = 0, stride = (entry_count + worker_count - 1) / worker_count;
         i < worker_count && start_entry < entry_count;
         ++i, start_entry += stride) {
      const auto end_entry = std::min(start_entry + stride, entry_count);
      const auto rs = rows->getResultSet().get();
      compute_stats_threads.push_back(
          std::async(std::launch::async,
                     [rs, &do_work, &dummy_encoders](const size_t start, const size_t end, const size_t worker_idx) {
                       for (size_t i = start; i < end; ++i) {
                         const auto crt_row = rs->getRowAtNoTranslations(i);
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
    rows->moveToBegin();
  }
  std::map<int, ChunkMetadata> metadata_map;
  for (size_t worker_idx = 1; worker_idx < worker_count; ++worker_idx) {
    CHECK_LT(worker_idx, dummy_encoders.size());
    const auto& worker_encoders = dummy_encoders[worker_idx];
    for (size_t i = 0; i < rows->colCount(); ++i) {
      dummy_encoders[0][i]->reduceStats(*worker_encoders[i]);
    }
  }
  for (size_t i = 0; i < rows->colCount(); ++i) {
    const auto it_ok = metadata_map.emplace(i, dummy_encoders[0][i]->getMetadata(rows->getColType(i)));
    CHECK(it_ok.second);
  }
  return metadata_map;
}

Fragmenter_Namespace::TableInfo synthesize_table_info(const RowSetPtr& rows) {
  std::deque<Fragmenter_Namespace::FragmentInfo> result;
  const size_t row_count = rows ? rows->rowCount() : 0;  // rows can be null only for query validation
  if (row_count) {
    result.resize(1);
    auto& fragment = result.front();
    fragment.fragmentId = 0;
    fragment.numTuples = row_count;
    fragment.deviceIds.resize(3);
    fragment.resultSet = rows.get();
  }
  Fragmenter_Namespace::TableInfo table_info;
  table_info.fragments = result;
  table_info.numTuples = row_count;
  return table_info;
}

Fragmenter_Namespace::TableInfo synthesize_table_info(const IterTabPtr& table) {
  Fragmenter_Namespace::TableInfo table_info;
  size_t total_row_count{0};  // rows can be null only for query validation
  if (!table->definitelyHasNoRows()) {
    table_info.fragments.resize(table->fragCount());
    for (size_t i = 0; i < table->fragCount(); ++i) {
      auto& fragment = table_info.fragments[i];
      fragment.fragmentId = i;
      fragment.numTuples = table->getFragAt(i).row_count;
      fragment.deviceIds.resize(3);
      total_row_count += fragment.numTuples;
    }
  }

  table_info.numTuples = total_row_count;
  return table_info;
}

void collect_table_infos(std::vector<InputTableInfo>& table_infos,
                         const std::vector<InputDescriptor>& input_descs,
                         Executor* executor) {
  const auto temporary_tables = executor->getTemporaryTables();
  const auto cat = executor->getCatalog();
  CHECK(cat);
  std::unordered_map<int, Fragmenter_Namespace::TableInfo*> info_cache;
  for (const auto& input_desc : input_descs) {
    const auto table_id = input_desc.getTableId();
    if (info_cache.count(table_id)) {
      CHECK(info_cache[table_id]);
      table_infos.push_back({table_id, build_table_info(*info_cache[table_id])});
      continue;
    }
    if (input_desc.getSourceType() == InputSourceType::RESULT) {
      CHECK_LT(table_id, 0);
      CHECK(temporary_tables);
      const auto it = temporary_tables->find(table_id);
      CHECK(it != temporary_tables->end());
      if (const auto rows = boost::get<RowSetPtr>(&it->second)) {
        CHECK(*rows);
        table_infos.push_back({table_id, synthesize_table_info(*rows)});
      } else if (const auto table = boost::get<IterTabPtr>(&it->second)) {
        CHECK(*table);
        table_infos.push_back({table_id, synthesize_table_info(*table)});
      } else {
        CHECK(false);
      }
    } else {
      CHECK(input_desc.getSourceType() == InputSourceType::TABLE);
      table_infos.push_back({table_id, executor->getTableInfo(table_id)});
    }
    info_cache.insert(std::make_pair(table_id, &table_infos.back().info));
  }
}

}  // namespace

size_t get_frag_count_of_table(const int table_id, Executor* executor) {
  const auto temporary_tables = executor->getTemporaryTables();
  CHECK(temporary_tables);
  auto it = temporary_tables->find(table_id);
  if (it != temporary_tables->end()) {
    CHECK_GE(int(0), table_id);
    CHECK(boost::get<RowSetPtr>(&it->second));
    return size_t(1);
  } else {
    const auto table_info = executor->getTableInfo(table_id);
    return table_info.fragments.size();
  }
}

std::vector<InputTableInfo> get_table_infos(const std::vector<InputDescriptor>& input_descs, Executor* executor) {
  std::vector<InputTableInfo> table_infos;
  collect_table_infos(table_infos, input_descs, executor);
  return table_infos;
}

std::vector<InputTableInfo> get_table_infos(const RelAlgExecutionUnit& ra_exe_unit, Executor* executor) {
  std::vector<InputTableInfo> table_infos;
  collect_table_infos(table_infos, ra_exe_unit.input_descs, executor);
  collect_table_infos(table_infos, ra_exe_unit.extra_input_descs, executor);
  return table_infos;
}

const std::map<int, ChunkMetadata>& Fragmenter_Namespace::FragmentInfo::getChunkMetadataMap() const {
  if (resultSet) {
    chunkMetadataMap = synthesize_metadata(resultSet);
    resultSet = nullptr;
  }
  return chunkMetadataMap;
}
