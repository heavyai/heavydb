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

#include "PersistentStorageMgr.h"
#include "Catalog/Catalog.h"
#include "DataMgr/ForeignStorage/CachingForeignStorageMgr.h"
#include "DataMgr/ForeignStorage/ForeignStorageInterface.h"
#include "MutableCachePersistentStorageMgr.h"

PersistentStorageMgr* PersistentStorageMgr::createPersistentStorageMgr(
    const std::string& data_dir,
    const size_t num_reader_threads,
    const DiskCacheConfig& config) {
  if (config.isEnabledForMutableTables()) {
    return new MutableCachePersistentStorageMgr(data_dir, num_reader_threads, config);
  } else {
    return new PersistentStorageMgr(data_dir, num_reader_threads, config);
  }
}

PersistentStorageMgr::PersistentStorageMgr(const std::string& data_dir,
                                           const size_t num_reader_threads,
                                           const DiskCacheConfig& disk_cache_config)
    : AbstractBufferMgr(0)
    , global_file_mgr_(
          std::make_unique<File_Namespace::GlobalFileMgr>(0,
                                                          data_dir,
                                                          num_reader_threads))
    , disk_cache_config_(disk_cache_config) {
  disk_cache_ =
      disk_cache_config_.isEnabled()
          ? std::make_unique<foreign_storage::ForeignStorageCache>(disk_cache_config)
          : nullptr;
  foreign_storage_mgr_ =
      disk_cache_config_.isEnabledForFSI()
          ? std::make_unique<foreign_storage::CachingForeignStorageMgr>(disk_cache_.get())
          : std::make_unique<foreign_storage::ForeignStorageMgr>();
}

AbstractBuffer* PersistentStorageMgr::createBuffer(const ChunkKey& chunk_key,
                                                   const size_t page_size,
                                                   const size_t initial_size) {
  return getStorageMgrForTableKey(chunk_key)->createBuffer(
      chunk_key, page_size, initial_size);
}

void PersistentStorageMgr::deleteBuffer(const ChunkKey& chunk_key, const bool purge) {
  getStorageMgrForTableKey(chunk_key)->deleteBuffer(chunk_key, purge);
}

void PersistentStorageMgr::deleteBuffersWithPrefix(const ChunkKey& chunk_key_prefix,
                                                   const bool purge) {
  getStorageMgrForTableKey(chunk_key_prefix)
      ->deleteBuffersWithPrefix(chunk_key_prefix, purge);
}

AbstractBuffer* PersistentStorageMgr::getBuffer(const ChunkKey& chunk_key,
                                                const size_t num_bytes) {
  return getStorageMgrForTableKey(chunk_key)->getBuffer(chunk_key, num_bytes);
}

void PersistentStorageMgr::fetchBuffer(const ChunkKey& chunk_key,
                                       AbstractBuffer* destination_buffer,
                                       const size_t num_bytes) {
  AbstractBufferMgr* mgr = getStorageMgrForTableKey(chunk_key);
  if (isChunkPrefixCacheable(chunk_key)) {
    AbstractBuffer* buffer = disk_cache_->getCachedChunkIfExists(chunk_key);
    if (buffer) {
      buffer->copyTo(destination_buffer, num_bytes);
      return;
    } else {
      mgr->fetchBuffer(chunk_key, destination_buffer, num_bytes);
      if (!isForeignStorage(chunk_key)) {
        // Foreign storage will read into cache buffers directly if enabled, so we do
        // not want to cache foreign table chunks here as they will already be cached.
        disk_cache_->cacheChunk(chunk_key, destination_buffer);
      }
      return;
    }
  }
  mgr->fetchBuffer(chunk_key, destination_buffer, num_bytes);
}

AbstractBuffer* PersistentStorageMgr::putBuffer(const ChunkKey& chunk_key,
                                                AbstractBuffer* source_buffer,
                                                const size_t num_bytes) {
  return getStorageMgrForTableKey(chunk_key)->putBuffer(
      chunk_key, source_buffer, num_bytes);
}

void PersistentStorageMgr::getChunkMetadataVecForKeyPrefix(
    ChunkMetadataVector& chunk_metadata,
    const ChunkKey& keyPrefix) {
  CHECK(has_table_prefix(keyPrefix));
  // If the disk has any cached metadata for a prefix then it is guaranteed to have all
  // metadata for that table, so we can return a complete set.  If it has no metadata,
  // then it may be that the table has no data, or that it's just not cached, so we need
  // to go to storage to check.
  if (isChunkPrefixCacheable(keyPrefix)) {
    if (disk_cache_->hasCachedMetadataForKeyPrefix(keyPrefix)) {
      disk_cache_->getCachedMetadataVecForKeyPrefix(chunk_metadata, keyPrefix);
      return;
    } else {  // if we have no cached data attempt a recovery.
      if (disk_cache_->recoverCacheForTable(chunk_metadata, get_table_key(keyPrefix))) {
        return;
      }
    }
    getStorageMgrForTableKey(keyPrefix)->getChunkMetadataVecForKeyPrefix(chunk_metadata,
                                                                         keyPrefix);
    disk_cache_->cacheMetadataVec(chunk_metadata);
  } else {
    getStorageMgrForTableKey(keyPrefix)->getChunkMetadataVecForKeyPrefix(chunk_metadata,
                                                                         keyPrefix);
  }
}

bool PersistentStorageMgr::isBufferOnDevice(const ChunkKey& chunk_key) {
  return global_file_mgr_->isBufferOnDevice(chunk_key);
}

std::string PersistentStorageMgr::printSlabs() {
  return global_file_mgr_->printSlabs();
}

void PersistentStorageMgr::clearSlabs() {
  global_file_mgr_->clearSlabs();
}

size_t PersistentStorageMgr::getMaxSize() {
  return global_file_mgr_->getMaxSize();
}

size_t PersistentStorageMgr::getInUseSize() {
  return global_file_mgr_->getInUseSize();
}

size_t PersistentStorageMgr::getAllocated() {
  return global_file_mgr_->getAllocated();
}

bool PersistentStorageMgr::isAllocationCapped() {
  return global_file_mgr_->isAllocationCapped();
}

void PersistentStorageMgr::checkpoint() {
  global_file_mgr_->checkpoint();
}

void PersistentStorageMgr::checkpoint(const int db_id, const int tb_id) {
  global_file_mgr_->checkpoint(db_id, tb_id);
}

AbstractBuffer* PersistentStorageMgr::alloc(const size_t num_bytes) {
  return global_file_mgr_->alloc(num_bytes);
}

void PersistentStorageMgr::free(AbstractBuffer* buffer) {
  global_file_mgr_->free(buffer);
}

MgrType PersistentStorageMgr::getMgrType() {
  return PERSISTENT_STORAGE_MGR;
}

std::string PersistentStorageMgr::getStringMgrType() {
  return ToString(PERSISTENT_STORAGE_MGR);
}

size_t PersistentStorageMgr::getNumChunks() {
  return global_file_mgr_->getNumChunks();
}

File_Namespace::GlobalFileMgr* PersistentStorageMgr::getGlobalFileMgr() const {
  return global_file_mgr_.get();
}

void PersistentStorageMgr::removeTableRelatedDS(const int db_id, const int table_id) {
  const ChunkKey table_key{db_id, table_id};
  if (isChunkPrefixCacheable(table_key)) {
    disk_cache_->clearForTablePrefix(table_key);
  }
  getStorageMgrForTableKey(table_key)->removeTableRelatedDS(db_id, table_id);
}

bool PersistentStorageMgr::isForeignStorage(const ChunkKey& chunk_key) const {
  CHECK(has_table_prefix(chunk_key));
  auto db_id = chunk_key[CHUNK_KEY_DB_IDX];
  auto table_id = chunk_key[CHUNK_KEY_TABLE_IDX];
  auto catalog = Catalog_Namespace::Catalog::checkedGet(db_id);

  auto table = catalog->getMetadataForTableImpl(table_id, false);
  CHECK(table);
  return table->storageType == StorageType::FOREIGN_TABLE;
}

AbstractBufferMgr* PersistentStorageMgr::getStorageMgrForTableKey(
    const ChunkKey& table_key) const {
  if (isForeignStorage(table_key)) {
    return foreign_storage_mgr_.get();
  } else {
    return global_file_mgr_.get();
  }
}

foreign_storage::ForeignStorageMgr* PersistentStorageMgr::getForeignStorageMgr() const {
  return foreign_storage_mgr_.get();
}

foreign_storage::ForeignStorageCache* PersistentStorageMgr::getDiskCache() const {
  return disk_cache_ ? disk_cache_.get() : nullptr;
}

bool PersistentStorageMgr::isChunkPrefixCacheable(const ChunkKey& chunk_prefix) const {
  CHECK(has_table_prefix(chunk_prefix));
  // If this is an Arrow FSI table then we can't cache it.
  if (ForeignStorageInterface::lookupBufferManager(chunk_prefix[CHUNK_KEY_DB_IDX],
                                                   chunk_prefix[CHUNK_KEY_TABLE_IDX])) {
    return false;
  }
  return ((disk_cache_config_.isEnabledForMutableTables() &&
           !isForeignStorage(chunk_prefix)) ||
          (disk_cache_config_.isEnabledForFSI() && isForeignStorage(chunk_prefix)));
}
