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

// By default, when rows are deleted, vacuum fragments with a least 10% deleted rows
float g_vacuum_min_selectivity{0.1};

TableOptimizer::TableOptimizer(const TableDescriptor* td,
                               Executor* executor,
                               const Catalog_Namespace::Catalog& cat)
    : td_(td), executor_(executor), cat_(cat) {
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
  if (td_->nShards > 0) {
    const auto physical_tds = cat_.getPhysicalTablesDescriptors(td_);
    table_descriptors.insert(
        table_descriptors.begin(), physical_tds.begin(), physical_tds.end());
  } else {
    table_descriptors.push_back(td_);
  }

  auto& data_mgr = cat_.getDataMgr();

  // acquire write lock on table data
  auto data_lock = lockmgr::TableDataLockMgr::getWriteLockForTable(cat_, td_->tableName);

  for (const auto td : table_descriptors) {
    ScopeGuard row_set_holder = [this] { executor_->row_set_mem_owner_ = nullptr; };
    executor_->row_set_mem_owner_ =
        std::make_shared<RowSetMemoryOwner>(ROW_SET_SIZE, /*num_threads=*/1);
    executor_->catalog_ = &cat_;
    const auto table_id = td->tableId;
    auto stats = recomputeDeletedColumnMetadata(td);

    // TODO(adb): Support geo
    auto col_descs = cat_.getAllColumnMetadataForTable(table_id, false, false, false);
    for (const auto& cd : col_descs) {
      recomputeColumnMetadata(td, cd, stats.visible_row_count_per_fragment, {}, {});
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
    auto stats = recomputeDeletedColumnMetadata(td);
    for (const auto cd : columns) {
      CHECK(columns_for_update.find(cd) != columns_for_update.end());
      auto fragment_indexes = getFragmentIndexes(td, columns_for_update.find(cd)->second);
      recomputeColumnMetadata(td,
                              cd,
                              stats.visible_row_count_per_fragment,
                              Data_Namespace::MemoryLevel::CPU_LEVEL,
                              fragment_indexes);
    }
  }
}

// Special case handle $deleted column if it exists
// whilst handling the delete column also capture
// the number of non deleted rows per fragment
DeletedColumnStats TableOptimizer::recomputeDeletedColumnMetadata(
    const TableDescriptor* td,
    const std::set<size_t>& fragment_indexes) const {
  if (!td->hasDeletedCol) {
    return {};
  }

  auto stats = getDeletedColumnStats(td, fragment_indexes);
  auto* fragmenter = td->fragmenter.get();
  CHECK(fragmenter);
  auto cd = cat_.getDeletedColumn(td);
  fragmenter->updateChunkStats(cd, stats.chunk_stats_per_fragment, {});
  fragmenter->setNumRows(stats.total_row_count);
  return stats;
}

DeletedColumnStats TableOptimizer::getDeletedColumnStats(
    const TableDescriptor* td,
    const std::set<size_t>& fragment_indexes) const {
  if (!td->hasDeletedCol) {
    return {};
  }

  auto cd = cat_.getDeletedColumn(td);
  const auto column_id = cd->columnId;

  const auto input_col_desc =
      std::make_shared<const InputColDescriptor>(column_id, td->tableId, 0);
  const auto col_expr =
      makeExpr<Analyzer::ColumnVar>(cd->columnType, td->tableId, column_id, 0);
  const auto count_expr =
      makeExpr<Analyzer::AggExpr>(cd->columnType, kCOUNT, col_expr, false, nullptr);

  const auto ra_exe_unit = build_ra_exe_unit(input_col_desc, {count_expr.get()});
  const auto table_infos = get_table_infos(ra_exe_unit, executor_);
  CHECK_EQ(table_infos.size(), size_t(1));

  const auto co = get_compilation_options(ExecutorDeviceType::CPU);
  const auto eo = get_execution_options();

  DeletedColumnStats deleted_column_stats;
  Executor::PerFragmentCallBack compute_deleted_callback =
      [&deleted_column_stats, cd](
          ResultSetPtr results, const Fragmenter_Namespace::FragmentInfo& fragment_info) {
        // count number of tuples in $deleted as total number of tuples in table.
        if (cd->isDeletedCol) {
          deleted_column_stats.total_row_count += fragment_info.getPhysicalNumTuples();
        }
        if (fragment_info.getPhysicalNumTuples() == 0) {
          // TODO(adb): Should not happen, but just to be safe...
          LOG(WARNING) << "Skipping completely empty fragment for column "
                       << cd->columnName;
          return;
        }

        const auto row = results->getNextRow(false, false);
        CHECK_EQ(row.size(), size_t(1));

        const auto& ti = cd->columnType;

        auto chunk_metadata = std::make_shared<ChunkMetadata>();
        chunk_metadata->sqlType = get_logical_type_info(ti);

        const auto count_val = read_scalar_target_value<int64_t>(row[0]);

        // min element 0 max element 1
        std::vector<TargetValue> fakerow;

        auto num_tuples = static_cast<size_t>(count_val);

        // calculate min
        if (num_tuples == fragment_info.getPhysicalNumTuples()) {
          // nothing deleted
          // min = false;
          // max = false;
          fakerow.emplace_back(TargetValue{int64_t(0)});
          fakerow.emplace_back(TargetValue{int64_t(0)});
        } else {
          if (num_tuples == 0) {
            // everything marked as delete
            // min = true
            // max = true
            fakerow.emplace_back(TargetValue{int64_t(1)});
            fakerow.emplace_back(TargetValue{int64_t(1)});
          } else {
            // some deleted
            // min = false
            // max = true;
            fakerow.emplace_back(TargetValue{int64_t(0)});
            fakerow.emplace_back(TargetValue{int64_t(1)});
          }
        }

        // place manufacture min and max in fake row to use common infra
        if (!set_metadata_from_results(*chunk_metadata, fakerow, ti, false)) {
          LOG(WARNING) << "Unable to process new metadata values for column "
                       << cd->columnName;
          return;
        }

        deleted_column_stats.chunk_stats_per_fragment.emplace(
            std::make_pair(fragment_info.fragmentId, chunk_metadata->chunkStats));
        deleted_column_stats.visible_row_count_per_fragment.emplace(
            std::make_pair(fragment_info.fragmentId, num_tuples));
      };

  executor_->executeWorkUnitPerFragment(ra_exe_unit,
                                        table_infos[0],
                                        co,
                                        eo,
                                        cat_,
                                        compute_deleted_callback,
                                        fragment_indexes);
  return deleted_column_stats;
}

void TableOptimizer::recomputeColumnMetadata(
    const TableDescriptor* td,
    const ColumnDescriptor* cd,
    const std::unordered_map</*fragment_id*/ int, size_t>& tuple_count_map,
    std::optional<Data_Namespace::MemoryLevel> memory_level,
    const std::set<size_t>& fragment_indexes) const {
  const auto ti = cd->columnType;
  if (ti.is_varlen()) {
    LOG(INFO) << "Skipping varlen column " << cd->columnName;
    return;
  }

  const auto column_id = cd->columnId;
  const auto input_col_desc =
      std::make_shared<const InputColDescriptor>(column_id, td->tableId, 0);
  const auto col_expr =
      makeExpr<Analyzer::ColumnVar>(cd->columnType, td->tableId, column_id, 0);
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
      [&stats_map, &tuple_count_map, cd](
          ResultSetPtr results, const Fragmenter_Namespace::FragmentInfo& fragment_info) {
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

        bool has_nulls = true;  // default to wide
        auto tuple_count_itr = tuple_count_map.find(fragment_info.fragmentId);
        if (tuple_count_itr != tuple_count_map.end()) {
          has_nulls = !(static_cast<size_t>(count_val) == tuple_count_itr->second);
        } else {
          // no deleted column calc so use raw physical count
          has_nulls =
              !(static_cast<size_t>(count_val) == fragment_info.getPhysicalNumTuples());
        }

        if (!set_metadata_from_results(*chunk_metadata, row, ti, has_nulls)) {
          LOG(WARNING) << "Unable to process new metadata values for column "
                       << cd->columnName;
          return;
        }

        stats_map.emplace(
            std::make_pair(fragment_info.fragmentId, chunk_metadata->chunkStats));
      };

  executor_->executeWorkUnitPerFragment(ra_exe_unit,
                                        table_infos[0],
                                        co,
                                        eo,
                                        cat_,
                                        compute_metadata_callback,
                                        fragment_indexes);

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

void TableOptimizer::vacuumDeletedRows() const {
  auto timer = DEBUG_TIMER(__func__);
  const auto table_id = td_->tableId;
  const auto db_id = cat_.getDatabaseId();
  const auto table_lock =
      lockmgr::TableDataLockMgr::getWriteLockForTable({db_id, table_id});
  const auto table_epochs = cat_.getTableEpochs(db_id, table_id);
  const auto shards = cat_.getPhysicalTablesDescriptors(td_);
  try {
    for (const auto shard : shards) {
      vacuumFragments(shard);
    }
    cat_.checkpoint(table_id);
  } catch (...) {
    cat_.setTableEpochsLogExceptions(db_id, table_epochs);
    throw;
  }

  for (auto shard : shards) {
    cat_.removeFragmenterForTable(shard->tableId);
    cat_.getDataMgr().getGlobalFileMgr()->compactDataFiles(cat_.getDatabaseId(),
                                                           shard->tableId);
  }
}

void TableOptimizer::vacuumFragments(const TableDescriptor* td,
                                     const std::set<int>& fragment_ids) const {
  // "if not a table that supports delete return,  nothing more to do"
  const ColumnDescriptor* cd = cat_.getDeletedColumn(td);
  if (nullptr == cd) {
    return;
  }
  // vacuum chunks which show sign of deleted rows in metadata
  ChunkKey chunk_key_prefix = {cat_.getDatabaseId(), td->tableId, cd->columnId};
  ChunkMetadataVector chunk_metadata_vec;
  cat_.getDataMgr().getChunkMetadataVecForKeyPrefix(chunk_metadata_vec, chunk_key_prefix);
  for (auto& [chunk_key, chunk_metadata] : chunk_metadata_vec) {
    auto fragment_id = chunk_key[CHUNK_KEY_FRAGMENT_IDX];
    // If delete has occurred, only vacuum fragments that are in the fragment_ids set.
    // Empty fragment_ids set implies all fragments.
    if (chunk_metadata->chunkStats.max.tinyintval == 1 &&
        (fragment_ids.empty() || shared::contains(fragment_ids, fragment_id))) {
      UpdelRoll updel_roll;
      updel_roll.catalog = &cat_;
      updel_roll.logicalTableId = cat_.getLogicalTableId(td->tableId);
      updel_roll.memoryLevel = Data_Namespace::MemoryLevel::CPU_LEVEL;
      updel_roll.table_descriptor = td;
      CHECK_EQ(cd->columnId, chunk_key[CHUNK_KEY_COLUMN_IDX]);
      const auto chunk = Chunk_NS::Chunk::getChunk(cd,
                                                   &cat_.getDataMgr(),
                                                   chunk_key,
                                                   updel_roll.memoryLevel,
                                                   0,
                                                   chunk_metadata->numBytes,
                                                   chunk_metadata->numElements);
      td->fragmenter->compactRows(&cat_,
                                  td,
                                  fragment_id,
                                  td->fragmenter->getVacuumOffsets(chunk),
                                  updel_roll.memoryLevel,
                                  updel_roll);
      updel_roll.stageUpdate();
    }
  }
  td->fragmenter->resetSizesFromFragments();
}

void TableOptimizer::vacuumFragmentsAboveMinSelectivity(
    const TableUpdateMetadata& table_update_metadata) const {
  if (td_->persistenceLevel != Data_Namespace::MemoryLevel::DISK_LEVEL) {
    return;
  }
  auto timer = DEBUG_TIMER(__func__);
  std::map<const TableDescriptor*, std::set<int32_t>> fragments_to_vacuum;
  for (const auto& [table_id, fragment_ids] :
       table_update_metadata.fragments_with_deleted_rows) {
    auto td = cat_.getMetadataForTable(table_id);
    // Skip automatic vacuuming for tables with uncapped epoch
    if (td->maxRollbackEpochs == -1) {
      continue;
    }

    DeletedColumnStats deleted_column_stats;
    {
      mapd_unique_lock<mapd_shared_mutex> executor_lock(executor_->execute_mutex_);
      ScopeGuard row_set_holder = [this] { executor_->row_set_mem_owner_ = nullptr; };
      executor_->row_set_mem_owner_ =
          std::make_shared<RowSetMemoryOwner>(ROW_SET_SIZE, /*num_threads=*/1);
      deleted_column_stats =
          getDeletedColumnStats(td, getFragmentIndexes(td, fragment_ids));
      executor_->clearMetaInfoCache();
    }

    std::set<int32_t> filtered_fragment_ids;
    for (const auto [fragment_id, visible_row_count] :
         deleted_column_stats.visible_row_count_per_fragment) {
      auto total_row_count =
          td->fragmenter->getFragmentInfo(fragment_id)->getPhysicalNumTuples();
      float deleted_row_count = total_row_count - visible_row_count;
      if ((deleted_row_count / total_row_count) >= g_vacuum_min_selectivity) {
        filtered_fragment_ids.emplace(fragment_id);
      }
    }

    if (!filtered_fragment_ids.empty()) {
      fragments_to_vacuum[td] = filtered_fragment_ids;
    }
  }

  if (!fragments_to_vacuum.empty()) {
    const auto db_id = cat_.getDatabaseId();
    const auto table_lock =
        lockmgr::TableDataLockMgr::getWriteLockForTable({db_id, td_->tableId});
    const auto table_epochs = cat_.getTableEpochs(db_id, td_->tableId);
    try {
      for (const auto& [td, fragment_ids] : fragments_to_vacuum) {
        vacuumFragments(td, fragment_ids);
        VLOG(1) << "Auto-vacuumed fragments: " << shared::printContainer(fragment_ids)
                << ", table id: " << td->tableId;
      }
      cat_.checkpoint(td_->tableId);
    } catch (...) {
      cat_.setTableEpochsLogExceptions(db_id, table_epochs);
      throw;
    }
  } else {
    // Checkpoint, even when no data update occurs, in order to ensure that epochs are
    // uniformly incremented in distributed mode.
    cat_.checkpointWithAutoRollback(td_->tableId);
  }
}
