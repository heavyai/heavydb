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

#include <Analyzer/Analyzer.h>
#include <Shared/scope.h>

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
      false, false, false, false, false, false, false, false, 0, false, false, 0};
}

}  // namespace

void TableOptimizer::recomputeMetadata() const {
  INJECT_TIMER(optimizeMetadata);
  std::lock_guard<std::mutex> lock(executor_->execute_mutex_);

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

  for (const auto td : table_descriptors) {
    ScopeGuard row_set_holder = [this] { executor_->row_set_mem_owner_ = nullptr; };
    executor_->row_set_mem_owner_ = std::make_shared<RowSetMemoryOwner>();
    executor_->catalog_ = &cat_;
    const auto table_id = td->tableId;

    std::unordered_map</*fragment_id*/ int, size_t> tuple_count_map;

    // Special case handle $deleted column if it exists
    // whilst handling the delete column also capture
    // the number of non deleted rows per fragment
    if (td->hasDeletedCol) {
      auto cd = cat_.getDeletedColumn(td);
      const auto column_id = cd->columnId;

      const auto input_col_desc =
          std::make_shared<const InputColDescriptor>(column_id, table_id, 0);
      const auto col_expr =
          makeExpr<Analyzer::ColumnVar>(cd->columnType, table_id, column_id, 0);
      const auto count_expr =
          makeExpr<Analyzer::AggExpr>(cd->columnType, kCOUNT, col_expr, false, nullptr);

      const auto ra_exe_unit = build_ra_exe_unit(input_col_desc, {count_expr.get()});
      const auto table_infos = get_table_infos(ra_exe_unit, executor_);
      CHECK_EQ(table_infos.size(), size_t(1));

      const auto co = get_compilation_options(ExecutorDeviceType::CPU);
      const auto eo = get_execution_options();

      std::unordered_map</*fragment_id*/ int, ChunkStats> stats_map;

      PerFragmentCB compute_deleted_callback =
          [&stats_map, &tuple_count_map, cd](
              ResultSetPtr results,
              const Fragmenter_Namespace::FragmentInfo& fragment_info) {
            if (fragment_info.getPhysicalNumTuples() == 0) {
              // TODO(adb): Should not happen, but just to be safe...
              LOG(WARNING) << "Skipping completely empty fragment for column "
                           << cd->columnName;
              return;
            }

            const auto row = results->getNextRow(false, false);
            CHECK_EQ(row.size(), size_t(1));

            const auto& ti = cd->columnType;

            ChunkMetadata chunk_metadata;
            chunk_metadata.sqlType = get_logical_type_info(ti);

            const auto count_val = read_scalar_target_value<int64_t>(row[0]);
            if (count_val == 0) {
              // Assume chunk of all nulls, bail
              return;
            }

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
            if (!set_metadata_from_results(chunk_metadata, fakerow, ti, false)) {
              LOG(WARNING) << "Unable to process new metadata values for column "
                           << cd->columnName;
              return;
            }

            stats_map.emplace(
                std::make_pair(fragment_info.fragmentId, chunk_metadata.chunkStats));
            tuple_count_map.emplace(std::make_pair(fragment_info.fragmentId, num_tuples));
          };

      executor_->executeWorkUnitPerFragment(
          ra_exe_unit, table_infos[0], co, eo, cat_, compute_deleted_callback);

      auto* fragmenter = td->fragmenter;
      CHECK(fragmenter);
      fragmenter->updateChunkStats(cd, stats_map);
    }  // finished special handling deleted column;

    // TODO(adb): Support geo
    auto col_descs = cat_.getAllColumnMetadataForTable(table_id, false, false, false);
    for (const auto& cd : col_descs) {
      const auto ti = cd->columnType;
      const auto column_id = cd->columnId;

      if (ti.is_varlen()) {
        LOG(INFO) << "Skipping varlen column " << cd->columnName;
        continue;
      }

      const auto input_col_desc =
          std::make_shared<const InputColDescriptor>(column_id, table_id, 0);
      const auto col_expr =
          makeExpr<Analyzer::ColumnVar>(cd->columnType, table_id, column_id, 0);
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

      PerFragmentCB compute_metadata_callback =
          [&stats_map, &tuple_count_map, cd](
              ResultSetPtr results,
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

            ChunkMetadata chunk_metadata;
            chunk_metadata.sqlType = get_logical_type_info(ti);

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
              has_nulls = !(static_cast<size_t>(count_val) ==
                            fragment_info.getPhysicalNumTuples());
            }

            if (!set_metadata_from_results(chunk_metadata, row, ti, has_nulls)) {
              LOG(WARNING) << "Unable to process new metadata values for column "
                           << cd->columnName;
              return;
            }

            stats_map.emplace(
                std::make_pair(fragment_info.fragmentId, chunk_metadata.chunkStats));
          };

      executor_->executeWorkUnitPerFragment(
          ra_exe_unit, table_infos[0], co, eo, cat_, compute_metadata_callback);

      auto* fragmenter = td->fragmenter;
      CHECK(fragmenter);
      fragmenter->updateChunkStats(cd, stats_map);
    }
    data_mgr.checkpoint(cat_.getCurrentDB().dbId, table_id);
    executor_->clearMetaInfoCache();
  }

  data_mgr.clearMemory(Data_Namespace::MemoryLevel::CPU_LEVEL);
  if (data_mgr.gpusPresent()) {
    data_mgr.clearMemory(Data_Namespace::MemoryLevel::GPU_LEVEL);
  }
}

void TableOptimizer::vacuumDeletedRows() const {
  const auto table_id = td_->tableId;
  cat_.optimizeTable(td_);
  auto& data_mgr = cat_.getDataMgr();
  data_mgr.checkpoint(cat_.getCurrentDB().dbId, table_id);
}
