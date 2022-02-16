/*
 * Copyright 2019 OmniSci, Inc.
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

#include "TableOptimizer.h"

#include "Analyzer/Analyzer.h"
#include "LockMgr/LockMgr.h"
#include "Logger/Logger.h"
#include "QueryEngine/Execute.h"
#include "Shared/misc.h"
#include "Shared/scope.h"

TableOptimizer::TableOptimizer(const TableDescriptor* td,
                               Executor* executor,
                               SchemaProviderPtr schema_provider,
                               const Catalog_Namespace::Catalog& cat)
    : td_(td), executor_(executor), schema_provider_(schema_provider), cat_(cat) {
  CHECK(td);
}
namespace {

template <typename T>
T read_scalar_target_value(const TargetValue& tv) {
  const auto stv = boost::get<ScalarTargetValue>(&tv);
  CHECK(stv);
  const auto val_ptr = boost::get<T>(stv);
  CHECK(val_ptr);
  return *val_ptr;
}

bool set_metadata_from_results(ChunkMetadata& chunk_metadata,
                               const std::vector<TargetValue>& row,
                               const SQLTypeInfo& ti,
                               const bool has_nulls) {
  switch (ti.get_type()) {
    case kBOOLEAN:
    case kTINYINT:
    case kSMALLINT:
    case kINT:
    case kBIGINT:
    case kNUMERIC:
    case kDECIMAL:
    case kTIME:
    case kTIMESTAMP:
    case kDATE: {
      int64_t min_val = read_scalar_target_value<int64_t>(row[0]);
      int64_t max_val = read_scalar_target_value<int64_t>(row[1]);
      chunk_metadata.fillChunkStats(min_val, max_val, has_nulls);
      break;
    }
    case kFLOAT: {
      float min_val = read_scalar_target_value<float>(row[0]);
      float max_val = read_scalar_target_value<float>(row[1]);
      chunk_metadata.fillChunkStats(min_val, max_val, has_nulls);
      break;
    }
    case kDOUBLE: {
      double min_val = read_scalar_target_value<double>(row[0]);
      double max_val = read_scalar_target_value<double>(row[1]);
      chunk_metadata.fillChunkStats(min_val, max_val, has_nulls);
      break;
    }
    case kVARCHAR:
    case kCHAR:
    case kTEXT:
      if (ti.get_compression() == kENCODING_DICT) {
        int64_t min_val = read_scalar_target_value<int64_t>(row[0]);
        int64_t max_val = read_scalar_target_value<int64_t>(row[1]);
        chunk_metadata.fillChunkStats(min_val, max_val, has_nulls);
      }
      break;
    default: {
      return false;  // skip column
    }
  }
  return true;
}

RelAlgExecutionUnit build_ra_exe_unit(
    const std::shared_ptr<const InputColDescriptor> input_col_desc,
    const std::vector<Analyzer::Expr*>& target_exprs) {
  return RelAlgExecutionUnit{{input_col_desc->getScanDesc()},
                             {input_col_desc},
                             {},
                             {},
                             {},
                             {},
                             target_exprs,
                             nullptr,
                             SortInfo{{}, SortAlgorithm::Default, 0, 0},
                             0};
}

inline CompilationOptions get_compilation_options(const ExecutorDeviceType& device_type) {
  return CompilationOptions{device_type, false, ExecutorOptLevel::Default, false};
}

inline ExecutionOptions get_execution_options() {
  return ExecutionOptions{
      false, false, false, false, false, false, false, false, 0, false, false, 0, false};
}

}  // namespace

void TableOptimizer::recomputeMetadata() const {
  auto timer = DEBUG_TIMER(__func__);
  mapd_unique_lock<mapd_shared_mutex> lock(executor_->execute_mutex_);

  LOG(INFO) << "Recomputing metadata for " << td_->tableName;

  CHECK_GE(td_->tableId, 0);

  std::vector<const TableDescriptor*> table_descriptors;
  table_descriptors.push_back(td_);

  auto& data_mgr = cat_.getDataMgr();

  // acquire write lock on table data
  auto data_lock = lockmgr::TableDataLockMgr::getWriteLockForTable(cat_, td_->tableName);

  for (const auto td : table_descriptors) {
    ScopeGuard row_set_holder = [this] { executor_->row_set_mem_owner_ = nullptr; };
    executor_->row_set_mem_owner_ =
        std::make_shared<RowSetMemoryOwner>(ROW_SET_SIZE, /*num_threads=*/1);
    executor_->setSchemaProvider(schema_provider_);

    const auto table_id = td->tableId;

    // TODO(adb): Support geo
    auto col_descs = cat_.getAllColumnMetadataForTable(table_id, false, false, false);
    for (const auto& cd : col_descs) {
      recomputeColumnMetadata(td, cd, {}, {});
    }
    data_mgr.checkpoint(cat_.getCurrentDB().dbId, table_id);
    executor_->clearMetaInfoCache();
  }

  data_mgr.clearMemory(Data_Namespace::MemoryLevel::CPU_LEVEL);
  if (data_mgr.gpusPresent()) {
    data_mgr.clearMemory(Data_Namespace::MemoryLevel::GPU_LEVEL);
  }
}

void TableOptimizer::recomputeMetadataUnlocked(
    const TableUpdateMetadata& table_update_metadata) const {
  auto timer = DEBUG_TIMER(__func__);
  std::map<int, std::list<const ColumnDescriptor*>> columns_by_table_id;
  auto& columns_for_update = table_update_metadata.columns_for_metadata_update;
  for (const auto& entry : columns_for_update) {
    auto column_descriptor = entry.first;
    columns_by_table_id[column_descriptor->tableId].emplace_back(column_descriptor);
  }

  for (const auto& [table_id, columns] : columns_by_table_id) {
    auto td = cat_.getMetadataForTable(table_id);
    for (const auto cd : columns) {
      CHECK(columns_for_update.find(cd) != columns_for_update.end());
      auto fragment_indexes = getFragmentIndexes(td, columns_for_update.find(cd)->second);
      recomputeColumnMetadata(
          td, cd, Data_Namespace::MemoryLevel::CPU_LEVEL, fragment_indexes);
    }
  }
}

void TableOptimizer::recomputeColumnMetadata(
    const TableDescriptor* td,
    const ColumnDescriptor* cd,
    std::optional<Data_Namespace::MemoryLevel> memory_level,
    const std::set<size_t>& fragment_indexes) const {
  const auto ti = cd->columnType;
  if (ti.is_varlen()) {
    LOG(INFO) << "Skipping varlen column " << cd->columnName;
    return;
  }

  CHECK(!cd->isVirtualCol);
  auto column_info = cd->makeInfo(cat_.getDatabaseId());
  const auto input_col_desc = std::make_shared<const InputColDescriptor>(column_info, 0);
  const auto col_expr = makeExpr<Analyzer::ColumnVar>(column_info, 0);
  auto max_expr =
      makeExpr<Analyzer::AggExpr>(cd->columnType, kMAX, col_expr, false, nullptr);
  auto min_expr =
      makeExpr<Analyzer::AggExpr>(cd->columnType, kMIN, col_expr, false, nullptr);
  auto count_expr =
      makeExpr<Analyzer::AggExpr>(cd->columnType, kCOUNT, col_expr, false, nullptr);

  if (ti.is_string()) {
    const SQLTypeInfo fun_ti(kINT);
    const auto fun_expr = makeExpr<Analyzer::KeyForStringExpr>(col_expr);
    max_expr = makeExpr<Analyzer::AggExpr>(fun_ti, kMAX, fun_expr, false, nullptr);
    min_expr = makeExpr<Analyzer::AggExpr>(fun_ti, kMIN, fun_expr, false, nullptr);
  }
  const auto ra_exe_unit = build_ra_exe_unit(
      input_col_desc, {min_expr.get(), max_expr.get(), count_expr.get()});
  const auto table_infos = get_table_infos(ra_exe_unit, executor_);
  CHECK_EQ(table_infos.size(), size_t(1));

  const auto co = get_compilation_options(ExecutorDeviceType::CPU);
  const auto eo = get_execution_options();

  std::unordered_map</*fragment_id*/ int, ChunkStats> stats_map;

  Executor::PerFragmentCallBack compute_metadata_callback =
      [&stats_map, cd](ResultSetPtr results,
                       const Fragmenter_Namespace::FragmentInfo& fragment_info) {
        if (fragment_info.getPhysicalNumTuples() == 0) {
          // TODO(adb): Should not happen, but just to be safe...
          LOG(WARNING) << "Skipping completely empty fragment for column "
                       << cd->columnName;
          return;
        }

        const auto row = results->getNextRow(false, false);
        CHECK_EQ(row.size(), size_t(3));

        const auto& ti = cd->columnType;

        auto chunk_metadata = std::make_shared<ChunkMetadata>();
        chunk_metadata->sqlType = get_logical_type_info(ti);

        const auto count_val = read_scalar_target_value<int64_t>(row[2]);
        if (count_val == 0) {
          // Assume chunk of all nulls, bail
          return;
        }

        bool has_nulls =
            !(static_cast<size_t>(count_val) == fragment_info.getPhysicalNumTuples());

        if (!set_metadata_from_results(*chunk_metadata, row, ti, has_nulls)) {
          LOG(WARNING) << "Unable to process new metadata values for column "
                       << cd->columnName;
          return;
        }

        stats_map.emplace(
            std::make_pair(fragment_info.fragmentId, chunk_metadata->chunkStats));
      };

  executor_->executeWorkUnitPerFragment(
      ra_exe_unit, table_infos[0], co, eo, compute_metadata_callback, fragment_indexes);

  auto* fragmenter = td->fragmenter.get();
  CHECK(fragmenter);
  fragmenter->updateChunkStats(cd, stats_map, memory_level);
}

// Returns the corresponding indexes for the given fragment ids in the list of fragments
// returned by `getFragmentsForQuery()`
std::set<size_t> TableOptimizer::getFragmentIndexes(
    const TableDescriptor* td,
    const std::set<int>& fragment_ids) const {
  CHECK(td->fragmenter);
  auto table_info = td->fragmenter->getFragmentsForQuery();
  std::set<size_t> fragment_indexes;
  for (size_t i = 0; i < table_info.fragments.size(); i++) {
    if (shared::contains(fragment_ids, table_info.fragments[i].fragmentId)) {
      fragment_indexes.emplace(i);
    }
  }
  return fragment_indexes;
}
