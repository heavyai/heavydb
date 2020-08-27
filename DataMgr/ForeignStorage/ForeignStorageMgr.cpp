/*
 * Copyright 2020 OmniSci, Inc.
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

#include "ForeignStorageMgr.h"

#include "Catalog/ForeignTable.h"
#include "CsvDataWrapper.h"
#include "ParquetDataWrapper.h"

namespace foreign_storage {
ForeignStorageMgr::ForeignStorageMgr(ForeignStorageCache* fsc)
    : AbstractBufferMgr(0), data_wrapper_map_({}), foreign_storage_cache_(fsc) {
  is_cache_enabled_ = (foreign_storage_cache_ != nullptr);
}

AbstractBuffer* ForeignStorageMgr::getBuffer(const ChunkKey& chunk_key,
                                             const size_t num_bytes) {
  // If this is a hit, then the buffer is allocated by the GFM currently.
  AbstractBuffer* buffer = is_cache_enabled_
                               ? foreign_storage_cache_->getCachedChunkIfExists(chunk_key)
                               : nullptr;
  if (buffer == nullptr) {
    buffer = getDataWrapper(chunk_key)->getChunkBuffer(chunk_key);
  }
  return buffer;
}

void ForeignStorageMgr::fetchBuffer(const ChunkKey& chunk_key,
                                    AbstractBuffer* destination_buffer,
                                    const size_t num_bytes) {
  CHECK(!destination_buffer->isDirty());
  bool cached = true;
  // This code is inlined from getBuffer() because we need to know if we had a cache hit
  // to know if we need to write back to the cache later.
  AbstractBuffer* buffer = is_cache_enabled_
                               ? foreign_storage_cache_->getCachedChunkIfExists(chunk_key)
                               : nullptr;
  if (buffer == nullptr) {
    cached = false;
    buffer = getDataWrapper(chunk_key)->getChunkBuffer(chunk_key);
  }

  // Read the contents of the source buffer into the dest buffer.
  size_t chunk_size = (num_bytes == 0) ? buffer->size() : num_bytes;
  destination_buffer->reserve(chunk_size);
  buffer->read(destination_buffer->getMemoryPtr() + destination_buffer->size(),
               chunk_size - destination_buffer->size(),
               destination_buffer->size(),
               destination_buffer->getType(),
               destination_buffer->getDeviceId());
  destination_buffer->setSize(chunk_size);
  destination_buffer->syncEncoder(buffer);

  // We only write back to the cache if we did not get the buffer from the cache.
  if (is_cache_enabled_ && !cached) {
    foreign_storage_cache_->cacheChunk(chunk_key, destination_buffer);
  }
}

void ForeignStorageMgr::getChunkMetadataVec(ChunkMetadataVector& chunk_metadata) {
  std::shared_lock data_wrapper_lock(data_wrapper_mutex_);
  for (auto& [table_chunk_key, data_wrapper] : data_wrapper_map_) {
    data_wrapper->populateMetadataForChunkKeyPrefix(table_chunk_key, chunk_metadata);
  }

  if (is_cache_enabled_) {
    foreign_storage_cache_->cacheMetadataVec(chunk_metadata);
  }
}

void ForeignStorageMgr::getChunkMetadataVecForKeyPrefix(
    ChunkMetadataVector& chunk_metadata,
    const ChunkKey& keyPrefix) {
  if (is_cache_enabled_ &&
      foreign_storage_cache_->hasCachedMetadataForKeyPrefix(keyPrefix)) {
    foreign_storage_cache_->getCachedMetadataVecForKeyPrefix(chunk_metadata, keyPrefix);
    return;
  }
  createDataWrapperIfNotExists(keyPrefix);
  getDataWrapper(keyPrefix)->populateMetadataForChunkKeyPrefix(keyPrefix, chunk_metadata);

  if (is_cache_enabled_) {
    foreign_storage_cache_->cacheMetadataVec(chunk_metadata);
  }
}

void ForeignStorageMgr::removeTableRelatedDS(const int db_id, const int table_id) {
  std::lock_guard data_wrapper_lock(data_wrapper_mutex_);
  data_wrapper_map_.erase({db_id, table_id});
  // Clear regardless of is_cache_enabled_
  if (is_cache_enabled_) {
    foreign_storage_cache_->clearForTablePrefix({db_id, table_id});
  }
}

MgrType ForeignStorageMgr::getMgrType() {
  return FOREIGN_STORAGE_MGR;
}

std::string ForeignStorageMgr::getStringMgrType() {
  return ToString(FOREIGN_STORAGE_MGR);
}

std::shared_ptr<ForeignDataWrapper> ForeignStorageMgr::getDataWrapper(
    const ChunkKey& chunk_key) {
  std::shared_lock data_wrapper_lock(data_wrapper_mutex_);
  ChunkKey table_key{chunk_key[0], chunk_key[1]};
  CHECK(data_wrapper_map_.find(table_key) != data_wrapper_map_.end());
  return data_wrapper_map_[table_key];
}

void ForeignStorageMgr::createDataWrapperIfNotExists(const ChunkKey& chunk_key) {
  std::lock_guard data_wrapper_lock(data_wrapper_mutex_);
  ChunkKey table_key{chunk_key[0], chunk_key[1]};
  if (data_wrapper_map_.find(table_key) == data_wrapper_map_.end()) {
    auto db_id = chunk_key[0];
    auto table_id = chunk_key[1];

    auto catalog = Catalog_Namespace::Catalog::get(db_id);
    CHECK(catalog);

    auto table = catalog->getMetadataForTableImpl(table_id, false);
    CHECK(table);

    auto foreign_table = dynamic_cast<const foreign_storage::ForeignTable*>(table);
    CHECK(foreign_table);

    if (foreign_table->foreign_server->data_wrapper_type ==
        foreign_storage::DataWrapperType::CSV) {
      data_wrapper_map_[table_key] =
          std::make_shared<CsvDataWrapper>(db_id, foreign_table);
    } else if (foreign_table->foreign_server->data_wrapper_type ==
               foreign_storage::DataWrapperType::PARQUET) {
      data_wrapper_map_[table_key] =
          std::make_shared<ParquetDataWrapper>(db_id, foreign_table);
    } else {
      throw std::runtime_error("Unsupported data wrapper");
    }
  }
}

ForeignStorageCache* ForeignStorageMgr::getForeignStorageCache() const {
  return foreign_storage_cache_;
}

void ForeignStorageMgr::refreshTablesInCache(const std::vector<ChunkKey>& table_keys) {
  if (!is_cache_enabled_) {
    return;
  }
  for (const auto& table_key : table_keys) {
    CHECK(table_key.size() == 2U);
    // Get a list of which chunks were cached for a table.
    std::vector<ChunkKey> chunk_keys =
        foreign_storage_cache_->getCachedChunksForKeyPrefix(table_key);
    foreign_storage_cache_->clearForTablePrefix(table_key);

    // Refresh metadata.
    ChunkMetadataVector metadata_vec;
    getDataWrapper(table_key)->populateMetadataForChunkKeyPrefix(table_key, metadata_vec);
    foreign_storage_cache_->cacheMetadataVec(metadata_vec);
    // Iterate through previously cached chunks and re-cache them from new wrapper.
    for (const auto& chunk_key : chunk_keys) {
      if (foreign_storage_cache_->isMetadataCached(chunk_key)) {
        foreign_storage_cache_->cacheChunk(
            chunk_key, getDataWrapper(table_key)->getChunkBuffer(chunk_key));
      }
    }
  }
}

void ForeignStorageMgr::evictTablesFromCache(const std::vector<ChunkKey>& table_keys) {
  if (!is_cache_enabled_) {
    return;
  }

  for (auto& table_key : table_keys) {
    CHECK(table_key.size() == 2U);
    foreign_storage_cache_->clearForTablePrefix(table_key);
  }
}

void ForeignStorageMgr::deleteBuffer(const ChunkKey& chunk_key, const bool purge) {
  UNREACHABLE();
}

void ForeignStorageMgr::deleteBuffersWithPrefix(const ChunkKey& chunk_key_prefix,
                                                const bool purge) {
  UNREACHABLE();
}

bool ForeignStorageMgr::isBufferOnDevice(const ChunkKey& chunk_key) {
  UNREACHABLE();
  return false;  // Added to avoid "no return statement" compiler warning
}

size_t ForeignStorageMgr::getNumChunks() {
  UNREACHABLE();
  return 0;  // Added to avoid "no return statement" compiler warning
}

AbstractBuffer* ForeignStorageMgr::createBuffer(const ChunkKey& chunk_key,
                                                const size_t page_size,
                                                const size_t initial_size) {
  UNREACHABLE();
  return nullptr;  // Added to avoid "no return statement" compiler warning
}

AbstractBuffer* ForeignStorageMgr::putBuffer(const ChunkKey& chunk_key,
                                             AbstractBuffer* source_buffer,
                                             const size_t num_bytes) {
  UNREACHABLE();
  return nullptr;  // Added to avoid "no return statement" compiler warning
}

std::string ForeignStorageMgr::printSlabs() {
  UNREACHABLE();
  return {};  // Added to avoid "no return statement" compiler warning
}

void ForeignStorageMgr::clearSlabs() {
  UNREACHABLE();
}

size_t ForeignStorageMgr::getMaxSize() {
  UNREACHABLE();
  return 0;  // Added to avoid "no return statement" compiler warning
}

size_t ForeignStorageMgr::getInUseSize() {
  UNREACHABLE();
  return 0;  // Added to avoid "no return statement" compiler warning
}

size_t ForeignStorageMgr::getAllocated() {
  UNREACHABLE();
  return 0;  // Added to avoid "no return statement" compiler warning
}

bool ForeignStorageMgr::isAllocationCapped() {
  UNREACHABLE();
  return false;  // Added to avoid "no return statement" compiler warning
}

void ForeignStorageMgr::checkpoint() {
  UNREACHABLE();
}

void ForeignStorageMgr::checkpoint(const int db_id, const int tb_id) {
  UNREACHABLE();
}

AbstractBuffer* ForeignStorageMgr::alloc(const size_t num_bytes) {
  UNREACHABLE();
  return nullptr;  // Added to avoid "no return statement" compiler warning
}

void ForeignStorageMgr::free(AbstractBuffer* buffer) {
  UNREACHABLE();
}
}  // namespace foreign_storage
