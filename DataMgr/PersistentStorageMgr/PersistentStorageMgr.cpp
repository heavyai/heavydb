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

PersistentStorageMgr::PersistentStorageMgr(const std::string& data_dir,
                                           const size_t num_reader_threads,
                                           const DiskCacheConfig& disk_cache_config)
    : AbstractBufferMgr(0)
    , global_file_mgr_(
          std::make_unique<File_Namespace::GlobalFileMgr>(0,
                                                          data_dir,
                                                          num_reader_threads)) {
  if (disk_cache_config.is_enabled) {
    disk_cache_ = std::make_unique<foreign_storage::ForeignStorageCache>(
        disk_cache_config.path, num_reader_threads, disk_cache_config.entry_limit);
    foreign_storage_mgr_ =
        std::make_unique<foreign_storage::ForeignStorageMgr>(disk_cache_.get());
  } else {
    foreign_storage_mgr_ = std::make_unique<foreign_storage::ForeignStorageMgr>();
  }
}

AbstractBuffer* PersistentStorageMgr::createBuffer(const ChunkKey& chunk_key,
                                                   const size_t page_size,
                                                   const size_t initial_size) {
  return global_file_mgr_->createBuffer(chunk_key, page_size, initial_size);
}

void PersistentStorageMgr::deleteBuffer(const ChunkKey& chunk_key, const bool purge) {
  global_file_mgr_->deleteBuffer(chunk_key, purge);
}

void PersistentStorageMgr::deleteBuffersWithPrefix(const ChunkKey& chunk_key_prefix,
                                                   const bool purge) {
  global_file_mgr_->deleteBuffersWithPrefix(chunk_key_prefix, purge);
}

AbstractBuffer* PersistentStorageMgr::getBuffer(const ChunkKey& chunk_key,
                                                const size_t num_bytes) {
  if (isForeignStorage(chunk_key)) {
    return foreign_storage_mgr_->getBuffer(chunk_key, num_bytes);
  } else {
    return global_file_mgr_->getBuffer(chunk_key, num_bytes);
  }
}

void PersistentStorageMgr::fetchBuffer(const ChunkKey& chunk_key,
                                       AbstractBuffer* destination_buffer,
                                       const size_t num_bytes) {
  if (isForeignStorage(chunk_key)) {
    foreign_storage_mgr_->fetchBuffer(chunk_key, destination_buffer, num_bytes);
  } else {
    global_file_mgr_->fetchBuffer(chunk_key, destination_buffer, num_bytes);
  }
}

AbstractBuffer* PersistentStorageMgr::putBuffer(const ChunkKey& chunk_key,
                                                AbstractBuffer* source_buffer,
                                                const size_t num_bytes) {
  return global_file_mgr_->putBuffer(chunk_key, source_buffer, num_bytes);
}

void PersistentStorageMgr::getChunkMetadataVec(ChunkMetadataVector& chunk_metadata) {
  global_file_mgr_->getChunkMetadataVec(chunk_metadata);
  foreign_storage_mgr_->getChunkMetadataVec(chunk_metadata);
}

void PersistentStorageMgr::getChunkMetadataVecForKeyPrefix(
    ChunkMetadataVector& chunk_metadata,
    const ChunkKey& keyPrefix) {
  if (isForeignStorage(keyPrefix)) {
    foreign_storage_mgr_->getChunkMetadataVecForKeyPrefix(chunk_metadata, keyPrefix);
  } else {
    global_file_mgr_->getChunkMetadataVecForKeyPrefix(chunk_metadata, keyPrefix);
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

File_Namespace::GlobalFileMgr* PersistentStorageMgr::getGlobalFileMgr() {
  return global_file_mgr_.get();
}

void PersistentStorageMgr::removeTableRelatedDS(const int db_id, const int table_id) {
  if (isForeignStorage({db_id, table_id})) {
    foreign_storage_mgr_->removeTableRelatedDS(db_id, table_id);
  }
  global_file_mgr_->removeTableRelatedDS(db_id, table_id);
}

bool PersistentStorageMgr::isForeignStorage(const ChunkKey& chunk_key) {
  auto db_id = chunk_key[0];
  auto table_id = chunk_key[1];

  auto catalog = Catalog_Namespace::Catalog::get(db_id);
  CHECK(catalog);

  auto table = catalog->getMetadataForTableImpl(table_id, false);
  CHECK(table);
  return table->storageType == StorageType::FOREIGN_TABLE;
}

foreign_storage::ForeignStorageMgr* PersistentStorageMgr::getForeignStorageMgr() const {
  return foreign_storage_mgr_.get();
}

foreign_storage::ForeignStorageCache* PersistentStorageMgr::getDiskCache() const {
  return disk_cache_ ? disk_cache_.get() : nullptr;
}
