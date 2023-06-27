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

#pragma once

#include <future>

#include "Fragmenter/InsertOrderFragmenter.h"
#include "LockMgr/LockMgr.h"
#include "QueryEngine/CodeGenerator.h"
#include "QueryEngine/Execute.h"
#include "QueryEngine/ExternalCacheInvalidators.h"
#include "QueryEngine/InputMetadata.h"
#include "QueryEngine/TargetMetaInfo.h"
#include "Shared/UpdelRoll.h"
#include "Shared/likely.h"
#include "Shared/thread_count.h"

extern bool g_enable_auto_metadata_update;

namespace {
/**
 * Checks to see if any of the updated values match the current min/max stat for the
 * chunk.
 */
bool is_chunk_min_max_updated(const Fragmenter_Namespace::ChunkUpdateStats& update_stats,
                              int64_t min,
                              int64_t max) {
  if (update_stats.old_values_stats.min_int64t <
          update_stats.new_values_stats.min_int64t &&
      update_stats.old_values_stats.min_int64t == min) {
    return true;
  }
  if (update_stats.old_values_stats.max_int64t >
          update_stats.new_values_stats.max_int64t &&
      update_stats.old_values_stats.max_int64t == max) {
    return true;
  }
  return false;
}

bool is_chunk_min_max_updated(const Fragmenter_Namespace::ChunkUpdateStats& update_stats,
                              double min,
                              double max) {
  if (update_stats.old_values_stats.min_double <
          update_stats.new_values_stats.min_double &&
      update_stats.old_values_stats.min_double == min) {
    return true;
  }
  if (update_stats.old_values_stats.max_double >
          update_stats.new_values_stats.max_double &&
      update_stats.old_values_stats.max_double == max) {
    return true;
  }
  return false;
}

bool should_recompute_metadata(
    const std::optional<Fragmenter_Namespace::ChunkUpdateStats>& update_stats) {
  if (!g_enable_auto_metadata_update || !update_stats.has_value()) {
    return false;
  }

  CHECK(update_stats->chunk);
  CHECK(update_stats->chunk->getBuffer());
  CHECK(update_stats->chunk->getBuffer()->getEncoder());

  auto chunk_metadata = std::make_shared<ChunkMetadata>();
  update_stats->chunk->getBuffer()->getEncoder()->getMetadata(chunk_metadata);
  auto cd = update_stats.value().chunk->getColumnDesc();
  if (cd->columnType.is_fp()) {
    double min, max;
    if (cd->columnType.get_type() == kDOUBLE) {
      min = chunk_metadata->chunkStats.min.doubleval;
      max = chunk_metadata->chunkStats.max.doubleval;
    } else if (cd->columnType.get_type() == kFLOAT) {
      min = chunk_metadata->chunkStats.min.floatval;
      max = chunk_metadata->chunkStats.max.floatval;
    } else {
      min = 0;  // resolve compiler warning about uninitialized variables
      max = -1;
      UNREACHABLE();
    }
    return is_chunk_min_max_updated(update_stats.value(), min, max);
  } else {
    auto min = extract_min_stat_int_type(chunk_metadata->chunkStats, cd->columnType);
    auto max = extract_max_stat_int_type(chunk_metadata->chunkStats, cd->columnType);
    return is_chunk_min_max_updated(update_stats.value(), min, max);
  }
}
}  // namespace

class StorageIOFacility {
 public:
  using UpdateCallback = UpdateLogForFragment::Callback;

  using TableDescriptorType = TableDescriptor;
  using DeleteVictimOffsetList = std::vector<uint64_t>;
  using UpdateTargetOffsetList = std::vector<uint64_t>;
  using UpdateTargetTypeList = std::vector<TargetMetaInfo>;
  using UpdateTargetColumnNamesList = std::vector<std::string>;

  using TransactionLog =
      Fragmenter_Namespace::InsertOrderFragmenter::ModifyTransactionTracker;
  using TransactionLogPtr = std::unique_ptr<TransactionLog>;
  using ColumnValidationFunction = std::function<bool(std::string const&)>;

  class TransactionParameters {
   public:
    TransactionParameters(const TableDescriptorType* table_descriptor,
                          const Catalog_Namespace::Catalog& catalog)
        : table_descriptor_(table_descriptor)
        , table_is_temporary_(table_is_temporary(table_descriptor))
        , catalog_(catalog) {}

    virtual ~TransactionParameters() = default;

    typename StorageIOFacility::TransactionLog& getTransactionTracker() {
      return transaction_tracker_;
    }
    void finalizeTransaction(const Catalog_Namespace::Catalog& catalog) {
      auto update_occurred = transaction_tracker_.commitUpdate();
      if (!update_occurred && table_descriptor_->persistenceLevel ==
                                  Data_Namespace::MemoryLevel::DISK_LEVEL) {
        // If commitUpdate() did not checkpoint, then we need to checkpoint here in order
        // to ensure that epochs are uniformly incremented in distributed mode.
        catalog.checkpointWithAutoRollback(table_descriptor_->tableId);
      }
    }

    auto tableIsTemporary() const { return table_is_temporary_; }

    auto const* getTableDescriptor() const { return table_descriptor_; }

    const Catalog_Namespace::Catalog& getCatalog() const { return catalog_; }

    const RelAlgNode* getInputSourceNode() const { return input_source_node_; }

    void setInputSourceNode(const RelAlgNode* input_source_node) {
      input_source_node_ = input_source_node;
    }

   private:
    typename StorageIOFacility::TransactionLog transaction_tracker_;
    TableDescriptorType const* table_descriptor_;
    bool table_is_temporary_;
    const Catalog_Namespace::Catalog& catalog_;
    const RelAlgNode* input_source_node_;
  };

  struct DeleteTransactionParameters : public TransactionParameters {
   public:
    DeleteTransactionParameters(const TableDescriptorType* table_descriptor,
                                const Catalog_Namespace::Catalog& catalog)
        : TransactionParameters(table_descriptor, catalog) {}

   private:
    DeleteTransactionParameters(DeleteTransactionParameters const& other) = delete;
    DeleteTransactionParameters& operator=(DeleteTransactionParameters const& other) =
        delete;
  };

  class UpdateTransactionParameters : public TransactionParameters {
   public:
    UpdateTransactionParameters(TableDescriptorType const* table_descriptor,
                                const Catalog_Namespace::Catalog& catalog,
                                UpdateTargetColumnNamesList const& update_column_names,
                                UpdateTargetTypeList const& target_types,
                                bool varlen_update_required)
        : TransactionParameters(table_descriptor, catalog)
        , update_column_names_(update_column_names)
        , targets_meta_(target_types)
        , varlen_update_required_(varlen_update_required) {}

    auto getUpdateColumnCount() const { return update_column_names_.size(); }
    auto const& getTargetsMetaInfo() const { return targets_meta_; }
    auto getTargetsMetaInfoSize() const { return targets_meta_.size(); }
    auto const& getUpdateColumnNames() const { return update_column_names_; }
    auto isVarlenUpdateRequired() const { return varlen_update_required_; }

   private:
    UpdateTransactionParameters(UpdateTransactionParameters const& other) = delete;
    UpdateTransactionParameters& operator=(UpdateTransactionParameters const& other) =
        delete;

    UpdateTargetColumnNamesList update_column_names_;
    UpdateTargetTypeList const& targets_meta_;
    bool varlen_update_required_ = false;
  };

  StorageIOFacility(Executor* executor) : executor_(executor) {}

  StorageIOFacility::UpdateCallback yieldUpdateCallback(
      UpdateTransactionParameters& update_parameters) {
    using OffsetVector = std::vector<uint64_t>;
    using ScalarTargetValueVector = std::vector<ScalarTargetValue>;
    using RowProcessingFuturesVector = std::vector<std::future<uint64_t>>;

    if (update_parameters.isVarlenUpdateRequired()) {
      auto callback = [this, &update_parameters](
                          UpdateLogForFragment const& update_log,
                          TableUpdateMetadata& table_update_metadata) -> void {
        std::vector<const ColumnDescriptor*> columnDescriptors;
        std::vector<TargetMetaInfo> sourceMetaInfos;

        const auto& catalog = update_parameters.getCatalog();
        for (size_t idx = 0; idx < update_parameters.getUpdateColumnNames().size();
             idx++) {
          auto& column_name = update_parameters.getUpdateColumnNames()[idx];
          auto target_column =
              catalog.getMetadataForColumn(update_log.getPhysicalTableId(), column_name);
          columnDescriptors.push_back(target_column);
          sourceMetaInfos.push_back(update_parameters.getTargetsMetaInfo()[idx]);
        }

        auto td = catalog.getMetadataForTable(update_log.getPhysicalTableId());
        auto* fragmenter = td->fragmenter.get();
        CHECK(fragmenter);

        fragmenter->updateColumns(
            &catalog,
            td,
            update_log.getFragmentId(),
            sourceMetaInfos,
            columnDescriptors,
            update_log,
            update_parameters.getUpdateColumnCount(),  // last column of result set
            Data_Namespace::MemoryLevel::CPU_LEVEL,
            update_parameters.getTransactionTracker(),
            executor_);
        table_update_metadata.fragments_with_deleted_rows[td->tableId].emplace(
            update_log.getFragmentId());
      };
      return callback;
    } else if (update_parameters.tableIsTemporary()) {
      auto callback = [&update_parameters](UpdateLogForFragment const& update_log,
                                           TableUpdateMetadata&) -> void {
        auto rs = update_log.getResultSet();
        CHECK(rs->didOutputColumnar());
        CHECK(rs->isDirectColumnarConversionPossible());
        CHECK_EQ(update_parameters.getUpdateColumnCount(), size_t(1));
        CHECK_EQ(rs->colCount(), size_t(1));

        // Temporary table updates require the full projected column
        CHECK_EQ(rs->rowCount(), update_log.getRowCount());

        const auto& catalog = update_parameters.getCatalog();
        ChunkKey chunk_key_prefix{catalog.getCurrentDB().dbId,
                                  update_parameters.getTableDescriptor()->tableId};
        const auto table_lock =
            lockmgr::TableDataLockMgr::getWriteLockForTable(chunk_key_prefix);

        auto& fragment_info = update_log.getFragmentInfo();
        const auto td = catalog.getMetadataForTable(update_log.getPhysicalTableId());
        CHECK(td);
        const auto cd = catalog.getMetadataForColumn(
            td->tableId, update_parameters.getUpdateColumnNames().front());
        CHECK(cd);
        auto chunk_metadata =
            fragment_info.getChunkMetadataMapPhysical().find(cd->columnId);
        CHECK(chunk_metadata != fragment_info.getChunkMetadataMapPhysical().end());
        ChunkKey chunk_key{catalog.getCurrentDB().dbId,
                           td->tableId,
                           cd->columnId,
                           fragment_info.fragmentId};
        auto chunk = Chunk_NS::Chunk::getChunk(cd,
                                               &catalog.getDataMgr(),
                                               chunk_key,
                                               Data_Namespace::MemoryLevel::CPU_LEVEL,
                                               0,
                                               chunk_metadata->second->numBytes,
                                               chunk_metadata->second->numElements);
        CHECK(chunk);
        auto chunk_buffer = chunk->getBuffer();
        CHECK(chunk_buffer);

        auto encoder = chunk_buffer->getEncoder();
        CHECK(encoder);

        auto owned_buffer = StorageIOFacility::getRsBufferNoPadding(
            rs.get(), 0, cd->columnType, rs->rowCount());
        auto buffer = reinterpret_cast<int8_t*>(owned_buffer.get());

        const auto new_chunk_metadata =
            encoder->appendData(buffer, rs->rowCount(), cd->columnType, false, 0);
        CHECK(new_chunk_metadata);

        auto fragmenter = td->fragmenter.get();
        CHECK(fragmenter);

        // The fragmenter copy of the fragment info differs from the copy used by the
        // query engine. Update metadata in the fragmenter directly.
        auto fragment = fragmenter->getFragmentInfo(fragment_info.fragmentId);
        // TODO: we may want to put this directly in the fragmenter so we are under the
        // fragmenter lock. But, concurrent queries on the same fragmenter should not be
        // allowed in this path.

        fragment->setChunkMetadata(cd->columnId, new_chunk_metadata);
        fragment->shadowChunkMetadataMap =
            fragment->getChunkMetadataMapPhysicalCopy();  // TODO(adb): needed?

        auto& data_mgr = catalog.getDataMgr();
        if (data_mgr.gpusPresent()) {
          // flush any GPU copies of the updated chunk
          data_mgr.deleteChunksWithPrefix(chunk_key,
                                          Data_Namespace::MemoryLevel::GPU_LEVEL);
        }
      };
      return callback;
    } else {
      auto callback = [this, &update_parameters](
                          UpdateLogForFragment const& update_log,
                          TableUpdateMetadata& table_update_metadata) -> void {
        auto entries_per_column = update_log.getEntryCount();
        auto rows_per_column = update_log.getRowCount();
        if (rows_per_column == 0) {
          return;
        }

        OffsetVector column_offsets(rows_per_column);
        ScalarTargetValueVector scalar_target_values(rows_per_column);

        auto complete_entry_block_size = entries_per_column / normalized_cpu_threads();
        auto partial_row_block_size = entries_per_column % normalized_cpu_threads();
        auto usable_threads = normalized_cpu_threads();
        if (UNLIKELY(rows_per_column < (unsigned)normalized_cpu_threads())) {
          complete_entry_block_size = entries_per_column;
          partial_row_block_size = 0;
          usable_threads = 1;
        }

        std::atomic<size_t> row_idx{0};

        auto process_rows =
            [&update_parameters, &column_offsets, &scalar_target_values, &row_idx](
                auto get_entry_at_func,
                uint64_t column_index,
                uint64_t entry_start,
                uint64_t entry_count) -> uint64_t {
          uint64_t entries_processed = 0;
          for (uint64_t entry_index = entry_start;
               entry_index < (entry_start + entry_count);
               entry_index++) {
            const auto& row = get_entry_at_func(entry_index);
            if (row.empty()) {
              continue;
            }

            entries_processed++;
            size_t row_index = row_idx.fetch_add(1);

            CHECK(row.size() == update_parameters.getUpdateColumnCount() + 1);

            auto terminal_column_iter = std::prev(row.end());
            const auto frag_offset_scalar_tv =
                boost::get<ScalarTargetValue>(&*terminal_column_iter);
            CHECK(frag_offset_scalar_tv);

            column_offsets[row_index] =
                static_cast<uint64_t>(*(boost::get<int64_t>(frag_offset_scalar_tv)));
            scalar_target_values[row_index] =
                boost::get<ScalarTargetValue>(row[column_index]);
          }
          return entries_processed;
        };

        auto get_row_index =
            [complete_entry_block_size](uint64_t thread_index) -> uint64_t {
          return (thread_index * complete_entry_block_size);
        };

        const auto& catalog = update_parameters.getCatalog();
        auto const* table_descriptor =
            catalog.getMetadataForTable(update_log.getPhysicalTableId());
        auto fragment_id = update_log.getFragmentId();
        auto table_id = update_log.getPhysicalTableId();
        if (!table_descriptor) {
          const auto* input_source_node = update_parameters.getInputSourceNode();
          if (auto proj_node = dynamic_cast<const RelProject*>(input_source_node)) {
            if (proj_node->hasPushedDownWindowExpr() ||
                proj_node->hasWindowFunctionExpr()) {
              table_id = proj_node->getModifiedTableDescriptor()->tableId;
              table_descriptor = catalog.getMetadataForTable(table_id);
            }
          }
        }
        CHECK(table_descriptor);

        // Iterate over each column
        for (decltype(update_parameters.getUpdateColumnCount()) column_index = 0;
             column_index < update_parameters.getUpdateColumnCount();
             column_index++) {
          row_idx = 0;
          RowProcessingFuturesVector entry_processing_futures;
          entry_processing_futures.reserve(usable_threads);

          auto get_entry_at_func = [&update_log,
                                    &column_index](const size_t entry_index) {
            if (UNLIKELY(update_log.getColumnType(column_index).is_string())) {
              return update_log.getTranslatedEntryAt(entry_index);
            } else {
              return update_log.getEntryAt(entry_index);
            }
          };

          for (unsigned i = 0; i < static_cast<unsigned>(usable_threads); i++) {
            entry_processing_futures.emplace_back(
                std::async(std::launch::async,
                           std::forward<decltype(process_rows)>(process_rows),
                           get_entry_at_func,
                           column_index,
                           get_row_index(i),
                           complete_entry_block_size));
          }
          if (partial_row_block_size) {
            entry_processing_futures.emplace_back(
                std::async(std::launch::async,
                           std::forward<decltype(process_rows)>(process_rows),
                           get_entry_at_func,
                           column_index,
                           get_row_index(usable_threads),
                           partial_row_block_size));
          }

          for (auto& t : entry_processing_futures) {
            t.wait();
          }

          CHECK(row_idx == rows_per_column);
          const auto fragmenter = table_descriptor->fragmenter;
          CHECK(fragmenter);
          auto const* target_column = catalog.getMetadataForColumn(
              table_id, update_parameters.getUpdateColumnNames()[column_index]);
          CHECK(target_column);
          auto update_stats =
              fragmenter->updateColumn(&catalog,
                                       table_descriptor,
                                       target_column,
                                       fragment_id,
                                       column_offsets,
                                       scalar_target_values,
                                       update_log.getColumnType(column_index),
                                       Data_Namespace::MemoryLevel::CPU_LEVEL,
                                       update_parameters.getTransactionTracker());
          if (should_recompute_metadata(update_stats)) {
            table_update_metadata.columns_for_metadata_update[target_column].emplace(
                fragment_id);
          }
        }
      };
      return callback;
    }
  }

  StorageIOFacility::UpdateCallback yieldDeleteCallback(
      DeleteTransactionParameters& delete_parameters) {
    using RowProcessingFuturesVector = std::vector<std::future<uint64_t>>;

    if (delete_parameters.tableIsTemporary()) {
      auto logical_table_id = delete_parameters.getTableDescriptor()->tableId;
      const auto& catalog = delete_parameters.getCatalog();
      auto callback = [logical_table_id, &catalog](UpdateLogForFragment const& update_log,
                                                   TableUpdateMetadata&) -> void {
        auto rs = update_log.getResultSet();
        CHECK(rs->didOutputColumnar());
        CHECK(rs->isDirectColumnarConversionPossible());
        CHECK_EQ(rs->colCount(), size_t(1));

        // Temporary table updates require the full projected column
        CHECK_EQ(rs->rowCount(), update_log.getRowCount());

        const ChunkKey lock_chunk_key{catalog.getCurrentDB().dbId, logical_table_id};
        const auto table_lock =
            lockmgr::TableDataLockMgr::getWriteLockForTable(lock_chunk_key);

        auto& fragment_info = update_log.getFragmentInfo();
        const auto td = catalog.getMetadataForTable(update_log.getPhysicalTableId());
        CHECK(td);
        const auto cd = catalog.getDeletedColumn(td);
        CHECK(cd);
        CHECK(cd->columnType.get_type() == kBOOLEAN);
        auto chunk_metadata =
            fragment_info.getChunkMetadataMapPhysical().find(cd->columnId);
        CHECK(chunk_metadata != fragment_info.getChunkMetadataMapPhysical().end());
        ChunkKey chunk_key{catalog.getCurrentDB().dbId,
                           td->tableId,
                           cd->columnId,
                           fragment_info.fragmentId};
        auto chunk = Chunk_NS::Chunk::getChunk(cd,
                                               &catalog.getDataMgr(),
                                               chunk_key,
                                               Data_Namespace::MemoryLevel::CPU_LEVEL,
                                               0,
                                               chunk_metadata->second->numBytes,
                                               chunk_metadata->second->numElements);
        CHECK(chunk);
        auto chunk_buffer = chunk->getBuffer();
        CHECK(chunk_buffer);

        auto encoder = chunk_buffer->getEncoder();
        CHECK(encoder);

        auto owned_buffer = StorageIOFacility::getRsBufferNoPadding(
            rs.get(), 0, cd->columnType, rs->rowCount());
        auto buffer = reinterpret_cast<int8_t*>(owned_buffer.get());

        const auto new_chunk_metadata =
            encoder->appendData(buffer, rs->rowCount(), cd->columnType, false, 0);

        auto fragmenter = td->fragmenter.get();
        CHECK(fragmenter);

        // The fragmenter copy of the fragment info differs from the copy used by the
        // query engine. Update metadata in the fragmenter directly.
        auto fragment = fragmenter->getFragmentInfo(fragment_info.fragmentId);
        // TODO: we may want to put this directly in the fragmenter so we are under the
        // fragmenter lock. But, concurrent queries on the same fragmenter should not be
        // allowed in this path.

        fragment->setChunkMetadata(cd->columnId, new_chunk_metadata);
        fragment->shadowChunkMetadataMap =
            fragment->getChunkMetadataMapPhysicalCopy();  // TODO(adb): needed?

        auto& data_mgr = catalog.getDataMgr();
        if (data_mgr.gpusPresent()) {
          // flush any GPU copies of the updated chunk
          data_mgr.deleteChunksWithPrefix(chunk_key,
                                          Data_Namespace::MemoryLevel::GPU_LEVEL);
        }
      };
      return callback;
    } else {
      auto callback = [this, &delete_parameters](
                          UpdateLogForFragment const& update_log,
                          TableUpdateMetadata& table_update_metadata) -> void {
        auto entries_per_column = update_log.getEntryCount();
        auto rows_per_column = update_log.getRowCount();
        if (rows_per_column == 0) {
          return;
        }
        DeleteVictimOffsetList victim_offsets(rows_per_column);

        auto complete_row_block_size = entries_per_column / normalized_cpu_threads();
        auto partial_row_block_size = entries_per_column % normalized_cpu_threads();
        auto usable_threads = normalized_cpu_threads();

        if (UNLIKELY(rows_per_column < (unsigned)normalized_cpu_threads())) {
          complete_row_block_size = rows_per_column;
          partial_row_block_size = 0;
          usable_threads = 1;
        }

        std::atomic<size_t> row_idx{0};

        auto process_rows = [&update_log, &victim_offsets, &row_idx](
                                uint64_t entry_start, uint64_t entry_count) -> uint64_t {
          uint64_t entries_processed = 0;

          for (uint64_t entry_index = entry_start;
               entry_index < (entry_start + entry_count);
               entry_index++) {
            auto const row(update_log.getEntryAt(entry_index));

            if (row.empty()) {
              continue;
            }

            entries_processed++;
            size_t row_index = row_idx.fetch_add(1);

            auto terminal_column_iter = std::prev(row.end());
            const auto scalar_tv = boost::get<ScalarTargetValue>(&*terminal_column_iter);
            CHECK(scalar_tv);

            uint64_t fragment_offset =
                static_cast<uint64_t>(*(boost::get<int64_t>(scalar_tv)));
            victim_offsets[row_index] = fragment_offset;
          }
          return entries_processed;
        };

        auto get_row_index =
            [complete_row_block_size](uint64_t thread_index) -> uint64_t {
          return thread_index * complete_row_block_size;
        };

        RowProcessingFuturesVector row_processing_futures;
        row_processing_futures.reserve(usable_threads);

        for (unsigned i = 0; i < (unsigned)usable_threads; i++) {
          row_processing_futures.emplace_back(
              std::async(std::launch::async,
                         std::forward<decltype(process_rows)>(process_rows),
                         get_row_index(i),
                         complete_row_block_size));
        }
        if (partial_row_block_size) {
          row_processing_futures.emplace_back(
              std::async(std::launch::async,
                         std::forward<decltype(process_rows)>(process_rows),
                         get_row_index(usable_threads),
                         partial_row_block_size));
        }

        for (auto& t : row_processing_futures) {
          t.wait();
        }

        const auto& catalog = delete_parameters.getCatalog();
        auto const* table_descriptor =
            catalog.getMetadataForTable(update_log.getPhysicalTableId());
        CHECK(table_descriptor);
        CHECK(!table_is_temporary(table_descriptor));
        auto* fragmenter = table_descriptor->fragmenter.get();
        CHECK(fragmenter);

        auto const* deleted_column_desc = catalog.getDeletedColumn(table_descriptor);
        CHECK(deleted_column_desc);
        fragmenter->updateColumn(&catalog,
                                 table_descriptor,
                                 deleted_column_desc,
                                 update_log.getFragmentId(),
                                 victim_offsets,
                                 ScalarTargetValue(int64_t(1L)),
                                 update_log.getColumnType(0),
                                 Data_Namespace::MemoryLevel::CPU_LEVEL,
                                 delete_parameters.getTransactionTracker());
        table_update_metadata.fragments_with_deleted_rows[table_descriptor->tableId]
            .emplace(update_log.getFragmentId());
      };
      return callback;
    }
  }

 private:
  int normalized_cpu_threads() const { return cpu_threads() / 2; }

  static std::unique_ptr<int8_t[]> getRsBufferNoPadding(const ResultSet* rs,
                                                        size_t col_idx,
                                                        const SQLTypeInfo& column_type,
                                                        size_t row_count) {
    const auto padded_size = rs->getPaddedSlotWidthBytes(col_idx);
    const auto type_size = column_type.is_dict_encoded_string()
                               ? column_type.get_size()
                               : column_type.get_logical_size();

    auto rs_buffer_size = padded_size * row_count;
    auto rs_buffer = std::make_unique<int8_t[]>(rs_buffer_size);
    rs->copyColumnIntoBuffer(col_idx, rs_buffer.get(), rs_buffer_size);

    if (type_size < padded_size) {
      // else we're going to remove padding and we do it inplace in the same buffer
      // we can do updates inplace in the same buffer because type_size < padded_size
      // for some types, like kFLOAT, simple memcpy is not enough
      auto src_ptr = rs_buffer.get();
      auto dst_ptr = rs_buffer.get();
      if (column_type.is_fp()) {
        CHECK(column_type.get_type() == kFLOAT);
        CHECK(padded_size == sizeof(double));
        for (size_t i = 0; i < row_count; i++) {
          const auto old_val = *reinterpret_cast<double*>(may_alias_ptr(src_ptr));
          auto new_val = static_cast<float>(old_val);
          std::memcpy(dst_ptr, &new_val, type_size);
          dst_ptr += type_size;
          src_ptr += padded_size;
        }
      } else {
        // otherwise just take first type_size bytes from the padded value
        for (size_t i = 0; i < row_count; i++) {
          std::memcpy(dst_ptr, src_ptr, type_size);
          dst_ptr += type_size;
          src_ptr += padded_size;
        }
      }
    }
    return rs_buffer;
  }

  Executor* executor_;
};
