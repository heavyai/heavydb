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
#include "DataMgr/FileMgr/CachingGlobalFileMgr.h"
#include "DataMgr/ForeignStorage/ArrowForeignStorage.h"
#include "DataMgr/ForeignStorage/CachingForeignStorageMgr.h"
#include "DataMgr/ForeignStorage/ForeignStorageInterface.h"

PersistentStorageMgr::PersistentStorageMgr(
    const std::string& data_dir,
    const size_t num_reader_threads,
    const File_Namespace::DiskCacheConfig& disk_cache_config)
    : AbstractBufferMgr(0), disk_cache_config_(disk_cache_config) {
  fsi_ = std::make_shared<ForeignStorageInterface>();
  ::registerArrowForeignStorage(fsi_);
  ::registerArrowCsvForeignStorage(fsi_);

  disk_cache_ =
      disk_cache_config_.isEnabled()
          ? std::make_unique<foreign_storage::ForeignStorageCache>(disk_cache_config)
          : nullptr;
  if (disk_cache_config_.isEnabledForMutableTables()) {
    CHECK(disk_cache_);
    global_file_mgr_ = std::make_unique<File_Namespace::CachingGlobalFileMgr>(
        0, fsi_, data_dir, num_reader_threads, disk_cache_.get());
  } else {
    global_file_mgr_ = std::make_unique<File_Namespace::GlobalFileMgr>(
        0, fsi_, data_dir, num_reader_threads);
  }

  if (disk_cache_config_.isEnabledForFSI()) {
    CHECK(disk_cache_);
    foreign_storage_mgr_ =
        std::make_unique<foreign_storage::CachingForeignStorageMgr>(disk_cache_.get());
  } else {
    foreign_storage_mgr_ = std::make_unique<foreign_storage::ForeignStorageMgr>();
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
  return global_file_mgr_->isBufferOnDevice(chunk_key);
}

std::string PersistentStorageMgr::printSlabs() {
  return global_file_mgr_->printSlabs();
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
  getStorageMgrForTableKey({db_id, table_id})->removeTableRelatedDS(db_id, table_id);
}

void PersistentStorageMgr::prepareTablesForExecution(const ColumnRefSet& input_cols,
                                                     const CompilationOptions& co,
                                                     const ExecutionOptions& eo,
                                                     ExecutionPhase phase) {
  ColumnRefSet gfs_input_cols;
  ColumnRefSet foreign_input_cols;
  for (auto& col : input_cols) {
    if (isForeignStorage({col.db_id, col.table_id})) {
      foreign_input_cols.insert(col);
    } else {
      gfs_input_cols.insert(col);
    }
  }
  getGlobalFileMgr()->prepareTablesForExecution(gfs_input_cols, co, eo, phase);
  getForeignStorageMgr()->prepareTablesForExecution(foreign_input_cols, co, eo, phase);
}

const DictDescriptor* PersistentStorageMgr::getDictMetadata(int db_id,
                                                            int dict_id,
                                                            bool load_dict) {
  return getGlobalFileMgr()->getDictMetadata(db_id, dict_id, load_dict);
}

bool PersistentStorageMgr::isForeignStorage(const ChunkKey& chunk_key) const {
  CHECK(has_table_prefix(chunk_key));
  auto db_id = chunk_key[CHUNK_KEY_DB_IDX];
  auto table_id = chunk_key[CHUNK_KEY_TABLE_IDX];
  auto catalog = Catalog_Namespace::SysCatalog::instance().getCatalog(db_id);

  // if catalog doesnt exist at this point we must be in an old migration.
  // Old migration can not, at this point 5.5.1, be using foreign storage
  // so this hack is to avoid the crash, when migrating old
  // catalogs that have not been upgraded over time due to issue
  // [BE-5728]
  if (!catalog) {
    return false;
  }

  auto table = catalog->getMetadataForTableImpl(table_id, false);
  // TODO: unknown tables get get there through prepareTablesForExecution. Check why.
  return table && table->storageType == StorageType::FOREIGN_TABLE;
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
