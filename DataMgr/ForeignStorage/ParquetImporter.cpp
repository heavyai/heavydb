/*
 * Copyright 2021 OmniSci, Inc.
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
#include "ParquetImporter.h"

#include <queue>

#include <arrow/filesystem/localfs.h>
#include <boost/filesystem.hpp>

#include "ForeignStorageException.h"
#include "FsiJsonUtils.h"
#include "LazyParquetChunkLoader.h"
#include "ParquetShared.h"
#include "Shared/misc.h"
#include "TypedParquetStorageBuffer.h"
#include "Utils/DdlUtils.h"

namespace foreign_storage {

class RowGroupIntervalTracker : public AbstractRowGroupIntervalTracker {
 public:
  RowGroupIntervalTracker(const std::set<std::string>& file_paths,
                          FileReaderMap* file_reader_cache,
                          std::shared_ptr<arrow::fs::FileSystem> file_system)
      : file_paths_(file_paths)
      , file_reader_cache_(file_reader_cache)
      , file_system_(file_system)
      , is_initialized_(false)
      , num_row_groups_(0)
      , current_row_group_index_(0)
      , current_file_iter_(file_paths_.begin()) {}

  std::optional<RowGroupInterval> getNextRowGroupInterval() override {
    advanceToNextRowGroup();
    if (filesAreExhausted()) {
      return {};
    }
    return RowGroupInterval{
        *current_file_iter_, current_row_group_index_, current_row_group_index_};
  }

 private:
  bool filesAreExhausted() { return current_file_iter_ == file_paths_.end(); }

  void advanceToNextRowGroup() {
    if (current_row_group_index_ < num_row_groups_ - 1) {
      current_row_group_index_++;
      return;
    }
    if (!is_initialized_) {
      current_file_iter_ = file_paths_.begin();
      is_initialized_ = true;
    } else {
      CHECK(!filesAreExhausted());
      current_file_iter_++;  // advance iterator
    }
    current_row_group_index_ = 0;
    if (filesAreExhausted()) {
      num_row_groups_ = 0;
    } else {
      auto file_reader =
          file_reader_cache_->getOrInsert(*current_file_iter_, file_system_);
      num_row_groups_ = file_reader->parquet_reader()->metadata()->num_row_groups();
    }
  }

  std::set<std::string> file_paths_;
  FileReaderMap* file_reader_cache_;
  std::shared_ptr<arrow::fs::FileSystem> file_system_;

  bool is_initialized_;
  int num_row_groups_;
  int current_row_group_index_;
  std::set<std::string>::const_iterator current_file_iter_;
};

class ParquetImportBatchResult : public import_export::ImportBatchResult {
 public:
  ParquetImportBatchResult() = default;
  ParquetImportBatchResult(const ForeignTable* foreign_table,
                           const int db_id,
                           const ForeignTableSchema* schema);
  ParquetImportBatchResult(ParquetImportBatchResult&& other) = default;

  Fragmenter_Namespace::InsertData getInsertData() const override;
  import_export::ImportStatus getImportStatus() const override;

  std::pair<std::map<int, Chunk_NS::Chunk>, std::map<int, StringDictionary*>>
  getChunksAndDictionaries() const;

  void populateInsertData(const std::map<int, Chunk_NS::Chunk>& chunks);

 private:
  Fragmenter_Namespace::InsertData insert_data_;
  std::map<int, std::unique_ptr<AbstractBuffer>> import_buffers_;  // holds data

  const ForeignTable* foreign_table_;
  int db_id_;
  const ForeignTableSchema* schema_;
};

void ParquetImportBatchResult::populateInsertData(
    const std::map<int, Chunk_NS::Chunk>& chunks) {
  size_t num_rows = chunks.begin()->second.getBuffer()->getEncoder()->getNumElems();
  for (const auto& [column_id, chunk] : chunks) {
    auto column_descriptor = chunk.getColumnDesc();
    CHECK(chunk.getBuffer()->getEncoder()->getNumElems() == num_rows);
    insert_data_.columnIds.emplace_back(column_id);
    auto buffer = chunk.getBuffer();
    DataBlockPtr block_ptr;
    if (column_descriptor->columnType.is_array()) {
      auto array_buffer = dynamic_cast<TypedParquetStorageBuffer<ArrayDatum>*>(buffer);
      block_ptr.arraysPtr = array_buffer->getBufferPtr();
    } else if ((column_descriptor->columnType.is_string() &&
                !column_descriptor->columnType.is_dict_encoded_string()) ||
               column_descriptor->columnType.is_geometry()) {
      auto string_buffer = dynamic_cast<TypedParquetStorageBuffer<std::string>*>(buffer);
      block_ptr.stringsPtr = string_buffer->getBufferPtr();
    } else {
      block_ptr.numbersPtr = buffer->getMemoryPtr();
    }
    insert_data_.data.emplace_back(block_ptr);
  }
  insert_data_.databaseId = db_id_;
  insert_data_.tableId = foreign_table_->tableId;
  insert_data_.is_default.assign(insert_data_.columnIds.size(), false);
  insert_data_.numRows = num_rows;
}

std::pair<std::map<int, Chunk_NS::Chunk>, std::map<int, StringDictionary*>>
ParquetImportBatchResult::getChunksAndDictionaries() const {
  std::map<int, Chunk_NS::Chunk> chunks;
  std::map<int, StringDictionary*> string_dictionaries;
  const auto catalog = Catalog_Namespace::SysCatalog::instance().getCatalog(db_id_);

  for (const auto column_descriptor : schema_->getLogicalAndPhysicalColumns()) {
    const bool is_dictionary_encoded_string_column =
        column_descriptor->columnType.is_dict_encoded_string() ||
        (column_descriptor->columnType.is_array() &&
         column_descriptor->columnType.get_elem_type().is_dict_encoded_string());

    if (is_dictionary_encoded_string_column) {
      auto dict_descriptor = catalog->getMetadataForDictUnlocked(
          column_descriptor->columnType.get_comp_param(), true);
      CHECK(dict_descriptor);
      auto string_dictionary = dict_descriptor->stringDict.get();
      string_dictionaries[column_descriptor->columnId] = string_dictionary;
    }

    Chunk_NS::Chunk chunk{column_descriptor};
    chunk.setBuffer(import_buffers_.at(column_descriptor->columnId).get());
    if (column_descriptor->columnType.is_varlen_indeed()) {
      chunk.setIndexBuffer(nullptr);  // index buffers are unused
    }
    chunk.initEncoder();
    chunks[column_descriptor->columnId] = chunk;
  }
  return {chunks, string_dictionaries};
}

ParquetImportBatchResult::ParquetImportBatchResult(const ForeignTable* foreign_table,
                                                   const int db_id,
                                                   const ForeignTableSchema* schema)
    : foreign_table_(foreign_table), db_id_(db_id), schema_(schema) {
  for (const auto column_descriptor : schema_->getLogicalAndPhysicalColumns()) {
    if (column_descriptor->columnType.is_array()) {
      import_buffers_[column_descriptor->columnId] =
          std::make_unique<TypedParquetStorageBuffer<ArrayDatum>>();
    } else if ((column_descriptor->columnType.is_string() &&
                !column_descriptor->columnType.is_dict_encoded_string()) ||
               column_descriptor->columnType.is_geometry()) {
      import_buffers_[column_descriptor->columnId] =
          std::make_unique<TypedParquetStorageBuffer<std::string>>();
    } else {
      import_buffers_[column_descriptor->columnId] =
          std::make_unique<ForeignStorageBuffer>();
    }
  }
}

Fragmenter_Namespace::InsertData ParquetImportBatchResult::getInsertData() const {
  return insert_data_;
}

import_export::ImportStatus ParquetImportBatchResult::getImportStatus() const {
  import_export::ImportStatus import_status;
  import_status.rows_completed = insert_data_.numRows;
  return import_status;
}

ParquetImporter::ParquetImporter() : db_id_(-1), foreign_table_(nullptr) {}

ParquetImporter::ParquetImporter(const int db_id,
                                 const ForeignTable* foreign_table,
                                 const UserMapping* user_mapping)
    : db_id_(db_id)
    , foreign_table_(foreign_table)
    , schema_(std::make_unique<ForeignTableSchema>(db_id, foreign_table))
    , file_reader_cache_(std::make_unique<FileReaderMap>()) {
  auto& server_options = foreign_table->foreign_server->options;
  if (server_options.find(STORAGE_TYPE_KEY)->second == LOCAL_FILE_STORAGE_TYPE) {
    file_system_ = std::make_shared<arrow::fs::LocalFileSystem>();
  } else {
    UNREACHABLE();
  }
}

std::set<std::string> ParquetImporter::getAllFilePaths() {
  auto timer = DEBUG_TIMER(__func__);
  std::set<std::string> file_paths;
  auto file_path = getFullFilePath(foreign_table_);
  auto file_info_result = file_system_->GetFileInfo(file_path);
  if (!file_info_result.ok()) {
    throw_file_access_error(file_path, file_info_result.status().message());
  } else {
    auto& file_info = file_info_result.ValueOrDie();
    if (file_info.type() == arrow::fs::FileType::NotFound) {
      throw_file_not_found_error(file_path);
    } else if (file_info.type() == arrow::fs::FileType::File) {
      file_paths.emplace(file_path);
    } else {
      CHECK_EQ(arrow::fs::FileType::Directory, file_info.type());
      arrow::fs::FileSelector file_selector{};
      file_selector.base_dir = file_path;
      file_selector.recursive = true;
      auto selector_result = file_system_->GetFileInfo(file_selector);
      if (!selector_result.ok()) {
        throw_file_access_error(file_path, selector_result.status().message());
      } else {
        auto& file_info_vector = selector_result.ValueOrDie();
        for (const auto& file_info : file_info_vector) {
          if (file_info.type() == arrow::fs::FileType::File) {
            file_paths.emplace(file_info.path());
          }
        }
      }
    }
  }
  return file_paths;
}

void ParquetImporter::populateChunkMetadata(ChunkMetadataVector& chunk_metadata_vector) {
  UNREACHABLE();
}

void ParquetImporter::populateChunkBuffers(const ChunkToBufferMap& required_buffers,
                                           const ChunkToBufferMap& optional_buffers) {
  UNREACHABLE();
}

std::string ParquetImporter::getSerializedDataWrapper() const {
  UNREACHABLE();
  return {};
}

std::unique_ptr<import_export::ImportBatchResult> ParquetImporter::getNextImportBatch() {
  if (!row_group_interval_tracker_) {
    row_group_interval_tracker_ = std::make_unique<RowGroupIntervalTracker>(
        getAllFilePaths(), file_reader_cache_.get(), file_system_);
  }

  auto import_batch_result =
      std::make_unique<ParquetImportBatchResult>(foreign_table_, db_id_, schema_.get());
  auto [chunks, string_dictionaries] = import_batch_result->getChunksAndDictionaries();

  LazyParquetChunkLoader chunk_loader(file_system_, file_reader_cache_.get());

  auto next_row_group = row_group_interval_tracker_->getNextRowGroupInterval();
  if (next_row_group.has_value()) {
    chunk_loader.loadRowGroups(*next_row_group, chunks, *schema_, string_dictionaries);
  }

  import_batch_result->populateInsertData(chunks);

  return import_batch_result;
}

void ParquetImporter::restoreDataWrapperInternals(
    const std::string& file_path,
    const ChunkMetadataVector& chunk_metadata_vector) {
  UNREACHABLE();
}

bool ParquetImporter::isRestored() const {
  UNREACHABLE();
  return {};
}

}  // namespace foreign_storage
