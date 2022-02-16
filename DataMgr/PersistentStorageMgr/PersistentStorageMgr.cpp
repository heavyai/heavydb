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
#include "DataMgr/FileMgr/CachingGlobalFileMgr.h"
#include "DataMgr/ForeignStorage/ArrowForeignStorage.h"
#include "DataMgr/ForeignStorage/ForeignStorageInterface.h"

PersistentStorageMgr::PersistentStorageMgr(
    const std::string& data_dir,
    const size_t num_reader_threads,
    const File_Namespace::DiskCacheConfig& disk_cache_config)
    : AbstractBufferMgr(0), disk_cache_config_(disk_cache_config) {
  if (data_dir != "") {
    fsi_ = std::make_shared<ForeignStorageInterface>();
    ::registerArrowForeignStorage(fsi_);
    ::registerArrowCsvForeignStorage(fsi_);

    disk_cache_ =
        disk_cache_config_.isEnabled()
            ? std::make_unique<foreign_storage::ForeignStorageCache>(disk_cache_config)
            : nullptr;
    std::unique_ptr<File_Namespace::GlobalFileMgr> global_file_mgr;

    if (disk_cache_config_.isEnabledForMutableTables()) {
      CHECK(disk_cache_);
      global_file_mgr = std::make_unique<File_Namespace::CachingGlobalFileMgr>(
          0, fsi_, data_dir, num_reader_threads, disk_cache_.get());
    } else {
      global_file_mgr = std::make_unique<File_Namespace::GlobalFileMgr>(
          0, fsi_, data_dir, num_reader_threads);
    }
    mgr_by_schema_id_[0] = std::move(global_file_mgr);
  }
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
  getStorageMgrForTableKey(chunk_key)->fetchBuffer(
      chunk_key, destination_buffer, num_bytes);
}

AbstractBuffer* PersistentStorageMgr::putBuffer(const ChunkKey& chunk_key,
                                                AbstractBuffer* source_buffer,
                                                const size_t num_bytes) {
  return getStorageMgrForTableKey(chunk_key)->putBuffer(
      chunk_key, source_buffer, num_bytes);
}

void PersistentStorageMgr::getChunkMetadataVecForKeyPrefix(
    ChunkMetadataVector& chunk_metadata,
    const ChunkKey& key_prefix) {
  getStorageMgrForTableKey(key_prefix)
      ->getChunkMetadataVecForKeyPrefix(chunk_metadata, key_prefix);
}

bool PersistentStorageMgr::isBufferOnDevice(const ChunkKey& chunk_key) {
  return getGlobalFileMgr()->isBufferOnDevice(chunk_key);
}

std::string PersistentStorageMgr::printSlabs() {
  return getGlobalFileMgr()->printSlabs();
}

size_t PersistentStorageMgr::getMaxSize() {
  return getGlobalFileMgr()->getMaxSize();
}

size_t PersistentStorageMgr::getInUseSize() {
  return getGlobalFileMgr()->getInUseSize();
}

size_t PersistentStorageMgr::getAllocated() {
  return getGlobalFileMgr()->getAllocated();
}

bool PersistentStorageMgr::isAllocationCapped() {
  return getGlobalFileMgr()->isAllocationCapped();
}

void PersistentStorageMgr::checkpoint() {
  getGlobalFileMgr()->checkpoint();
}

void PersistentStorageMgr::checkpoint(const int db_id, const int tb_id) {
  getGlobalFileMgr()->checkpoint(db_id, tb_id);
}

AbstractBuffer* PersistentStorageMgr::alloc(const size_t num_bytes) {
  return getGlobalFileMgr()->alloc(num_bytes);
}

void PersistentStorageMgr::free(AbstractBuffer* buffer) {
  getGlobalFileMgr()->free(buffer);
}

MgrType PersistentStorageMgr::getMgrType() {
  return PERSISTENT_STORAGE_MGR;
}

std::string PersistentStorageMgr::getStringMgrType() {
  return ToString(PERSISTENT_STORAGE_MGR);
}

size_t PersistentStorageMgr::getNumChunks() {
  return getGlobalFileMgr()->getNumChunks();
}

File_Namespace::GlobalFileMgr* PersistentStorageMgr::getGlobalFileMgr() const {
  if (mgr_by_schema_id_.count(0)) {
    return dynamic_cast<File_Namespace::GlobalFileMgr*>(mgr_by_schema_id_.at(0).get());
  }
  return nullptr;
}

void PersistentStorageMgr::removeTableRelatedDS(const int db_id, const int table_id) {
  getStorageMgrForTableKey({db_id, table_id})->removeTableRelatedDS(db_id, table_id);
}

const DictDescriptor* PersistentStorageMgr::getDictMetadata(int db_id,
                                                            int dict_id,
                                                            bool load_dict) {
  return getStorageMgr(db_id)->getDictMetadata(db_id, dict_id, load_dict);
}

Fragmenter_Namespace::TableInfo PersistentStorageMgr::getTableMetadata(
    int db_id,
    int table_id) const {
  return getStorageMgrForTableKey({db_id, table_id})->getTableMetadata(db_id, table_id);
}

AbstractBufferMgr* PersistentStorageMgr::getStorageMgrForTableKey(
    const ChunkKey& table_key) const {
  return mgr_by_schema_id_.at(table_key[CHUNK_KEY_DB_IDX] >> 24).get();
}

AbstractBufferMgr* PersistentStorageMgr::getStorageMgr(int db_id) const {
  return mgr_by_schema_id_.at(db_id >> 24).get();
}

foreign_storage::ForeignStorageCache* PersistentStorageMgr::getDiskCache() const {
  return disk_cache_ ? disk_cache_.get() : nullptr;
}

void PersistentStorageMgr::registerDataProvider(
    int schema_id,
    std::shared_ptr<AbstractBufferMgr> provider) {
  CHECK_EQ(mgr_by_schema_id_.count(schema_id), (size_t)0);
  mgr_by_schema_id_[schema_id] = provider;
}
