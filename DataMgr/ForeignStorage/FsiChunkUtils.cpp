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

#include "FsiChunkUtils.h"

#include "Catalog/Catalog.h"
#include "DataMgr/ForeignStorage/ForeignStorageBuffer.h"

namespace foreign_storage {
void init_chunk_for_column(
    const ChunkKey& chunk_key,
    const std::map<ChunkKey, std::shared_ptr<ChunkMetadata>>& chunk_metadata_map,
    const std::map<ChunkKey, AbstractBuffer*>& buffers,
    Chunk_NS::Chunk& chunk) {
  auto catalog =
      Catalog_Namespace::SysCatalog::instance().getCatalog(chunk_key[CHUNK_KEY_DB_IDX]);
  CHECK(catalog);

  ChunkKey data_chunk_key = chunk_key;
  AbstractBuffer* data_buffer = nullptr;
  AbstractBuffer* index_buffer = nullptr;
  const auto column = catalog->getMetadataForColumnUnlocked(
      chunk_key[CHUNK_KEY_TABLE_IDX], chunk_key[CHUNK_KEY_COLUMN_IDX]);

  if (column->columnType.is_varlen_indeed()) {
    data_chunk_key.push_back(1);
    ChunkKey index_chunk_key = chunk_key;
    index_chunk_key.push_back(2);

    CHECK(buffers.find(data_chunk_key) != buffers.end());
    CHECK(buffers.find(index_chunk_key) != buffers.end());

    data_buffer = buffers.find(data_chunk_key)->second;
    index_buffer = buffers.find(index_chunk_key)->second;
    CHECK_EQ(data_buffer->size(), static_cast<size_t>(0));
    CHECK_EQ(index_buffer->size(), static_cast<size_t>(0));

    size_t index_offset_size{0};
    if (column->columnType.is_string()) {
      index_offset_size = sizeof(StringOffsetT);
    } else if (column->columnType.is_array()) {
      index_offset_size = sizeof(ArrayOffsetT);
    } else {
      UNREACHABLE();
    }
    CHECK(chunk_metadata_map.find(data_chunk_key) != chunk_metadata_map.end());
    index_buffer->reserve(index_offset_size *
                          (chunk_metadata_map.at(data_chunk_key)->numElements + 1));
  } else {
    data_chunk_key = chunk_key;
    CHECK(buffers.find(data_chunk_key) != buffers.end());
    data_buffer = buffers.find(data_chunk_key)->second;
  }
  CHECK(chunk_metadata_map.find(data_chunk_key) != chunk_metadata_map.end());
  data_buffer->reserve(chunk_metadata_map.at(data_chunk_key)->numBytes);

  chunk.setColumnInfo(column->makeInfo(chunk_key[CHUNK_KEY_DB_IDX]));
  chunk.setBuffer(data_buffer);
  chunk.setIndexBuffer(index_buffer);
  chunk.initEncoder();
}

std::shared_ptr<ChunkMetadata> get_placeholder_metadata(const ColumnDescriptor* column,
                                                        size_t num_elements) {
  ForeignStorageBuffer empty_buffer;
  // Use default encoder metadata as in parquet wrapper
  empty_buffer.initEncoder(column->columnType);
  auto chunk_metadata = empty_buffer.getEncoder()->getMetadata(column->columnType);
  chunk_metadata->numElements = num_elements;

  if (!column->columnType.is_varlen_indeed()) {
    chunk_metadata->numBytes = column->columnType.get_size() * num_elements;
  }
  // min/max not set by default for arrays, so get from elem type encoder
  if (column->columnType.is_array()) {
    ForeignStorageBuffer scalar_buffer;
    scalar_buffer.initEncoder(column->columnType.get_elem_type());
    auto scalar_metadata =
        scalar_buffer.getEncoder()->getMetadata(column->columnType.get_elem_type());
    chunk_metadata->chunkStats.min = scalar_metadata->chunkStats.min;
    chunk_metadata->chunkStats.max = scalar_metadata->chunkStats.max;
  }
  chunk_metadata->chunkStats.has_nulls = true;
  return chunk_metadata;
}
}  // namespace foreign_storage
