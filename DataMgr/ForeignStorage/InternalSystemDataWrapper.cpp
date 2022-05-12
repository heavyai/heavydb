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

#include "InternalSystemDataWrapper.h"

#include "Catalog/Catalog.h"
#include "Catalog/SysCatalog.h"
#include "ForeignStorageException.h"
#include "ForeignTableSchema.h"
#include "FsiChunkUtils.h"
#include "ImportExport/Importer.h"
#include "Shared/SysDefinitions.h"
#include "Shared/distributed.h"
#include "TextFileBufferParser.h"
#include "UserMapping.h"

namespace foreign_storage {
std::string get_db_name(int32_t db_id) {
  Catalog_Namespace::DBMetadata db_metadata;
  auto& sys_catalog = Catalog_Namespace::SysCatalog::instance();
  if (sys_catalog.getMetadataForDBById(db_id, db_metadata)) {
    return db_metadata.dbName;
  } else {
    // Database has been deleted.
    return kDeletedValueIndicator;
  }
}

std::string get_table_name(int32_t db_id, int32_t table_id) {
  auto catalog = Catalog_Namespace::SysCatalog::instance().getCatalog(db_id);
  CHECK(catalog);
  auto table_name = catalog->getTableName(table_id);
  if (table_name.has_value()) {
    return table_name.value();
  } else {
    // It is possible for the table to be concurrently deleted while querying the system
    // table.
    return kDeletedValueIndicator;
  }
}

void set_node_name(
    std::map<std::string, import_export::TypedImportBuffer*>& import_buffers) {
  if (import_buffers.find("node") != import_buffers.end()) {
    if (dist::is_leaf_node()) {
      std::string leaf_string{"Leaf " + to_string(g_distributed_leaf_idx)};
      import_buffers["node"]->addString(leaf_string);
    } else {
      import_buffers["node"]->addString("Server");
    }
  }
}

InternalSystemDataWrapper::InternalSystemDataWrapper()
    : db_id_(-1), foreign_table_(nullptr) {}

InternalSystemDataWrapper::InternalSystemDataWrapper(const int db_id,
                                                     const ForeignTable* foreign_table)
    : db_id_(db_id), foreign_table_(foreign_table) {}

void InternalSystemDataWrapper::validateServerOptions(
    const ForeignServer* foreign_server) const {
  CHECK(foreign_server->options.empty());
}

void InternalSystemDataWrapper::validateTableOptions(
    const ForeignTable* foreign_table) const {}

const std::set<std::string_view>& InternalSystemDataWrapper::getSupportedTableOptions()
    const {
  static const std::set<std::string_view> supported_table_options{};
  return supported_table_options;
}

void InternalSystemDataWrapper::validateUserMappingOptions(
    const UserMapping* user_mapping,
    const ForeignServer* foreign_server) const {
  CHECK(user_mapping->options.empty());
}

const std::set<std::string_view>&
InternalSystemDataWrapper::getSupportedUserMappingOptions() const {
  static const std::set<std::string_view> supported_user_mapping_options{};
  return supported_user_mapping_options;
}

namespace {
void initialize_chunks(std::map<ChunkKey, Chunk_NS::Chunk>& chunks,
                       const ChunkToBufferMap& buffers,
                       size_t row_count,
                       std::set<const ColumnDescriptor*>& columns_to_parse,
                       int32_t fragment_id,
                       const Catalog_Namespace::Catalog& catalog) {
  for (auto& [chunk_key, buffer] : buffers) {
    CHECK_EQ(fragment_id, chunk_key[CHUNK_KEY_FRAGMENT_IDX]);
    const auto column = catalog.getMetadataForColumn(chunk_key[CHUNK_KEY_TABLE_IDX],
                                                     chunk_key[CHUNK_KEY_COLUMN_IDX]);
    if (is_varlen_index_key(chunk_key)) {
      continue;
    }
    chunks[chunk_key] = Chunk_NS::Chunk{column};
    if (column->columnType.is_varlen_indeed()) {
      CHECK(is_varlen_data_key(chunk_key));
      size_t index_offset_size{0};
      if (column->columnType.is_string()) {
        index_offset_size = sizeof(StringOffsetT);
      } else if (column->columnType.is_array()) {
        index_offset_size = sizeof(ArrayOffsetT);
      } else {
        UNREACHABLE() << "Unexpected column type: " << column->columnType.to_string();
      }
      ChunkKey index_chunk_key = chunk_key;
      index_chunk_key[CHUNK_KEY_VARLEN_IDX] = 2;
      CHECK(buffers.find(index_chunk_key) != buffers.end());
      AbstractBuffer* index_buffer = buffers.find(index_chunk_key)->second;
      index_buffer->reserve(index_offset_size * row_count + 1);
      chunks[chunk_key].setIndexBuffer(index_buffer);
    }

    if (!column->columnType.is_varlen_indeed()) {
      buffer->reserve(column->columnType.get_size() * row_count);
    }
    chunks[chunk_key].setBuffer(buffer);
    chunks[chunk_key].initEncoder();
    columns_to_parse.emplace(column);
  }
}

void initialize_import_buffers(
    std::vector<std::unique_ptr<import_export::TypedImportBuffer>>& import_buffers,
    std::map<std::string, import_export::TypedImportBuffer*>& import_buffers_map,
    const std::set<const ColumnDescriptor*>& columns_to_parse,
    const Catalog_Namespace::Catalog& catalog) {
  for (const auto column : columns_to_parse) {
    StringDictionary* string_dictionary = nullptr;
    if (column->columnType.is_dict_encoded_string() ||
        (column->columnType.is_array() && IS_STRING(column->columnType.get_subtype()) &&
         column->columnType.get_compression() == kENCODING_DICT)) {
      auto dict_descriptor =
          catalog.getMetadataForDict(column->columnType.get_comp_param(), true);
      string_dictionary = dict_descriptor->stringDict.get();
    }
    import_buffers.emplace_back(
        std::make_unique<import_export::TypedImportBuffer>(column, string_dictionary));
    import_buffers_map[column->columnName] = import_buffers.back().get();
  }
}
}  // namespace

void InternalSystemDataWrapper::populateChunkMetadata(
    ChunkMetadataVector& chunk_metadata_vector) {
  auto& sys_catalog = Catalog_Namespace::SysCatalog::instance();
  auto catalog = sys_catalog.getCatalog(db_id_);
  CHECK(catalog);
  CHECK_EQ(catalog->name(), shared::kInfoSchemaDbName);

  initializeObjectsForTable(foreign_table_->tableName);
  if (row_count_ > static_cast<size_t>(foreign_table_->maxFragRows)) {
    throw ForeignStorageException{
        "System table size exceeds the maximum supported size."};
  }
  foreign_storage::ForeignTableSchema schema(db_id_, foreign_table_);
  for (auto column : schema.getLogicalColumns()) {
    ChunkKey chunk_key = {db_id_, foreign_table_->tableId, column->columnId, 0};
    if (column->columnType.is_varlen_indeed()) {
      chunk_key.emplace_back(1);
    }
    ForeignStorageBuffer empty_buffer;
    // Use default encoder metadata
    empty_buffer.initEncoder(column->columnType);
    auto chunk_metadata = empty_buffer.getEncoder()->getMetadata(column->columnType);
    chunk_metadata->numElements = row_count_;
    if (!column->columnType.is_varlen_indeed()) {
      chunk_metadata->numBytes = column->columnType.get_size() * row_count_;
    }
    if (column->columnType.is_array()) {
      ForeignStorageBuffer scalar_buffer;
      scalar_buffer.initEncoder(column->columnType.get_elem_type());
      auto scalar_metadata =
          scalar_buffer.getEncoder()->getMetadata(column->columnType.get_elem_type());
      chunk_metadata->chunkStats.min = scalar_metadata->chunkStats.min;
      chunk_metadata->chunkStats.max = scalar_metadata->chunkStats.max;
    }
    chunk_metadata->chunkStats.has_nulls = true;
    chunk_metadata_vector.emplace_back(chunk_key, chunk_metadata);
  }
}

void InternalSystemDataWrapper::populateChunkBuffers(
    const ChunkToBufferMap& required_buffers,
    const ChunkToBufferMap& optional_buffers,
    AbstractBuffer* delete_buffer) {
  auto timer = DEBUG_TIMER(__func__);
  CHECK(optional_buffers.empty());

  auto& sys_catalog = Catalog_Namespace::SysCatalog::instance();
  auto catalog = sys_catalog.getCatalog(db_id_);
  CHECK(catalog);
  CHECK_EQ(catalog->name(), shared::kInfoSchemaDbName);

  auto fragment_id = required_buffers.begin()->first[CHUNK_KEY_FRAGMENT_IDX];
  CHECK_EQ(fragment_id, 0)
      << "In-memory system tables are expected to have a single fragment.";

  std::map<ChunkKey, Chunk_NS::Chunk> chunks;
  std::set<const ColumnDescriptor*> columns_to_parse;
  initialize_chunks(
      chunks, required_buffers, row_count_, columns_to_parse, fragment_id, *catalog);

  // initialize import buffers from columns.
  std::vector<std::unique_ptr<import_export::TypedImportBuffer>> import_buffers;
  std::map<std::string, import_export::TypedImportBuffer*> import_buffers_map;
  initialize_import_buffers(
      import_buffers, import_buffers_map, columns_to_parse, *catalog);
  populateChunkBuffersForTable(foreign_table_->tableName, import_buffers_map);

  auto column_id_to_data_blocks_map =
      TextFileBufferParser::convertImportBuffersToDataBlocks(import_buffers);
  for (auto& [chunk_key, chunk] : chunks) {
    auto data_block_entry =
        column_id_to_data_blocks_map.find(chunk_key[CHUNK_KEY_COLUMN_IDX]);
    CHECK(data_block_entry != column_id_to_data_blocks_map.end());
    chunk.appendData(data_block_entry->second, row_count_, 0);
    chunk.setBuffer(nullptr);
    chunk.setIndexBuffer(nullptr);
  }
}

std::string InternalSystemDataWrapper::getSerializedDataWrapper() const {
  return {};
}

void InternalSystemDataWrapper::restoreDataWrapperInternals(
    const std::string& file_path,
    const ChunkMetadataVector& chunk_metadata) {}

bool InternalSystemDataWrapper::isRestored() const {
  return false;
}
}  // namespace foreign_storage
