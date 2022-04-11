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
#include "DataMgr/DataMgrDataProvider.h"
#include "LockMgr/LockMgr.h"
#include "Logger/Logger.h"
#include "QueryEngine/Execute.h"
#include "Shared/misc.h"
#include "Shared/scope.h"

TableOptimizer::TableOptimizer(const TableDescriptor* td,
                               Executor* executor,
                               DataProvider* data_provider,
                               SchemaProviderPtr schema_provider,
                               const Catalog_Namespace::Catalog& cat)
    : td_(td)
    , executor_(executor)
    , data_provider_(data_provider)
    , schema_provider_(schema_provider)
    , cat_(cat) {
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
    executor_->row_set_mem_owner_ = std::make_shared<RowSetMemoryOwner>(
        data_provider_, ROW_SET_SIZE, /*num_threads=*/1);
    executor_->setSchemaProvider(schema_provider_);

    const auto table_id = td->tableId;

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
  UNREACHABLE();
}

// Returns the corresponding indexes for the given fragment ids in the list of fragments
// returned by `getFragmentsForQuery()`
std::set<size_t> TableOptimizer::getFragmentIndexes(
    const TableDescriptor* td,
    const std::set<int>& fragment_ids) const {
  UNREACHABLE();
}
