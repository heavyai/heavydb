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

#include "AbstractFileStorageDataWrapper.h"
#include "ForeignDataWrapperFactory.h"
#include "ForeignStorageException.h"
#include "ForeignTableSchema.h"

extern bool g_enable_fsi;
extern bool g_enable_s3_fsi;

namespace foreign_storage {
ForeignStorageMgr::ForeignStorageMgr() : AbstractBufferMgr(0), data_wrapper_map_({}) {}

AbstractBuffer* ForeignStorageMgr::getBuffer(const ChunkKey& chunk_key,
                                             const size_t num_bytes) {
  UNREACHABLE();
  return nullptr;  // Added to avoid "no return statement" compiler warning
}

void ForeignStorageMgr::checkIfS3NeedsToBeEnabled(const ChunkKey& chunk_key) {
  CHECK(has_table_prefix(chunk_key));
  auto catalog =
      Catalog_Namespace::SysCatalog::instance().getCatalog(chunk_key[CHUNK_KEY_DB_IDX]);
  CHECK(catalog);
  auto foreign_table = catalog->getForeignTableUnlocked(chunk_key[CHUNK_KEY_TABLE_IDX]);
  auto storage_type_entry = foreign_table->foreign_server->options.find(
      AbstractFileStorageDataWrapper::STORAGE_TYPE_KEY);

  if (storage_type_entry == foreign_table->foreign_server->options.end()) {
    // Some FSI servers such as ODBC do not have a storage_type
    return;
  }
  bool is_s3_storage_type =
      (storage_type_entry->second == AbstractFileStorageDataWrapper::S3_STORAGE_TYPE);
  if (is_s3_storage_type) {
    throw ForeignStorageException{
        "Query cannot be executed for S3 backed foreign table because AWS S3 support is "
        "currently disabled."};
  }
}

void ForeignStorageMgr::fetchBuffer(const ChunkKey& chunk_key,
                                    AbstractBuffer* destination_buffer,
                                    const size_t num_bytes) {
  checkIfS3NeedsToBeEnabled(chunk_key);
  CHECK(destination_buffer);
  CHECK(!destination_buffer->isDirty());
  // Use a temp buffer if we have no cache buffers and have one mapped for this chunk.
  if (fetchBufferIfTempBufferMapEntryExists(chunk_key, destination_buffer, num_bytes)) {
    return;
  }

  {  // Clear any temp buffers if we've moved on to a new fragment
    std::lock_guard temp_chunk_buffer_map_lock(temp_chunk_buffer_map_mutex_);
    if (temp_chunk_buffer_map_.size() > 0 &&
        temp_chunk_buffer_map_.begin()->first[CHUNK_KEY_FRAGMENT_IDX] !=
            chunk_key[CHUNK_KEY_FRAGMENT_IDX]) {
      clearTempChunkBufferMapEntriesForTableUnlocked(get_table_key(chunk_key));
    }
  }

  createAndPopulateDataWrapperIfNotExists(chunk_key);

  // TODO: Populate optional buffers as part of CSV performance improvement
  std::set<ChunkKey> chunk_keys = get_keys_set_from_table(chunk_key);
  chunk_keys.erase(chunk_key);

  // Use hints to prefetch other chunks in fragment
  ChunkToBufferMap optional_buffers;

  // Use hints to prefetch other chunks in fragment into cache
  auto& data_wrapper = *getDataWrapper(chunk_key);
  std::set<ChunkKey> optional_keys;
  getOptionalChunkKeySet(optional_keys,
                         chunk_key,
                         get_keys_set_from_table(chunk_key),
                         data_wrapper.getNonCachedParallelismLevel());
  if (optional_keys.size()) {
    {
      std::shared_lock temp_chunk_buffer_map_lock(temp_chunk_buffer_map_mutex_);
      // Erase anything already in temp_chunk_buffer_map_
      for (auto it = optional_keys.begin(); it != optional_keys.end();) {
        if (temp_chunk_buffer_map_.find(*it) != temp_chunk_buffer_map_.end()) {
          it = optional_keys.erase(it);
        } else {
          ++it;
        }
      }
    }
    if (optional_keys.size()) {
      optional_buffers = allocateTempBuffersForChunks(optional_keys);
    }
  }

  auto required_buffers = allocateTempBuffersForChunks(chunk_keys);
  required_buffers[chunk_key] = destination_buffer;
  // populate will write directly to destination_buffer so no need to copy.
  getDataWrapper(chunk_key)->populateChunkBuffers(required_buffers, optional_buffers);
}

bool ForeignStorageMgr::fetchBufferIfTempBufferMapEntryExists(
    const ChunkKey& chunk_key,
    AbstractBuffer* destination_buffer,
    size_t num_bytes) {
  AbstractBuffer* buffer{nullptr};
  {
    std::shared_lock temp_chunk_buffer_map_lock(temp_chunk_buffer_map_mutex_);
    if (temp_chunk_buffer_map_.find(chunk_key) == temp_chunk_buffer_map_.end()) {
      return false;
    }
    buffer = temp_chunk_buffer_map_[chunk_key].get();
  }
  // For the index key, calls with size 0 get 1 added as
  // empty index buffers start with one entry
  // Set to 0 here to copy entire buffer
  if (is_varlen_index_key(chunk_key) && (num_bytes == sizeof(StringOffsetT))) {
    num_bytes = 0;
  }
  CHECK(buffer);
  buffer->copyTo(destination_buffer, num_bytes);
  {
    std::lock_guard temp_chunk_buffer_map_lock(temp_chunk_buffer_map_mutex_);
    temp_chunk_buffer_map_.erase(chunk_key);
  }
  return true;
}

void ForeignStorageMgr::getChunkMetadataVecForKeyPrefix(
    ChunkMetadataVector& chunk_metadata,
    const ChunkKey& key_prefix) {
  if (!g_enable_fsi) {
    throw ForeignStorageException{
        "Query cannot be executed for foreign table because FSI is currently disabled."};
  }
  CHECK(is_table_key(key_prefix));
  checkIfS3NeedsToBeEnabled(key_prefix);
  createDataWrapperIfNotExists(key_prefix);
  getDataWrapper(key_prefix)->populateChunkMetadata(chunk_metadata);
}

void ForeignStorageMgr::removeTableRelatedDS(const int db_id, const int table_id) {
  const ChunkKey table_key{db_id, table_id};
  {
    std::lock_guard data_wrapper_lock(data_wrapper_mutex_);
    data_wrapper_map_.erase(table_key);
  }
  clearTempChunkBufferMapEntriesForTable(table_key);
}

MgrType ForeignStorageMgr::getMgrType() {
  return FOREIGN_STORAGE_MGR;
}

std::string ForeignStorageMgr::getStringMgrType() {
  return ToString(FOREIGN_STORAGE_MGR);
}

bool ForeignStorageMgr::hasDataWrapperForChunk(const ChunkKey& chunk_key) {
  std::shared_lock data_wrapper_lock(data_wrapper_mutex_);
  CHECK(has_table_prefix(chunk_key));
  ChunkKey table_key{chunk_key[CHUNK_KEY_DB_IDX], chunk_key[CHUNK_KEY_TABLE_IDX]};
  return data_wrapper_map_.find(table_key) != data_wrapper_map_.end();
}

std::shared_ptr<ForeignDataWrapper> ForeignStorageMgr::getDataWrapper(
    const ChunkKey& chunk_key) {
  std::shared_lock data_wrapper_lock(data_wrapper_mutex_);
  ChunkKey table_key{chunk_key[CHUNK_KEY_DB_IDX], chunk_key[CHUNK_KEY_TABLE_IDX]};
  CHECK(data_wrapper_map_.find(table_key) != data_wrapper_map_.end());
  return data_wrapper_map_[table_key];
}

void ForeignStorageMgr::setDataWrapper(
    const ChunkKey& table_key,
    std::shared_ptr<MockForeignDataWrapper> data_wrapper) {
  CHECK(is_table_key(table_key));
  std::lock_guard data_wrapper_lock(data_wrapper_mutex_);
  CHECK(data_wrapper_map_.find(table_key) != data_wrapper_map_.end());
  data_wrapper->setParentWrapper(data_wrapper_map_[table_key]);
  data_wrapper_map_[table_key] = data_wrapper;
}

bool ForeignStorageMgr::createDataWrapperIfNotExists(const ChunkKey& chunk_key) {
  std::lock_guard data_wrapper_lock(data_wrapper_mutex_);
  ChunkKey table_key{chunk_key[CHUNK_KEY_DB_IDX], chunk_key[CHUNK_KEY_TABLE_IDX]};
  if (data_wrapper_map_.find(table_key) == data_wrapper_map_.end()) {
    auto db_id = chunk_key[CHUNK_KEY_DB_IDX];
    auto catalog = Catalog_Namespace::SysCatalog::instance().getCatalog(db_id);
    CHECK(catalog);
    auto foreign_table = catalog->getForeignTableUnlocked(chunk_key[CHUNK_KEY_TABLE_IDX]);
    data_wrapper_map_[table_key] = ForeignDataWrapperFactory::create(
        foreign_table->foreign_server->data_wrapper_type, db_id, foreign_table);
    return true;
  }
  return false;
}

void ForeignStorageMgr::refreshTable(const ChunkKey& table_key,
                                     const bool evict_cached_entries) {
  auto catalog =
      Catalog_Namespace::SysCatalog::instance().getCatalog(table_key[CHUNK_KEY_DB_IDX]);
  CHECK(catalog);
  // Clear datawrapper unless table is non-append and evict is false
  if (evict_cached_entries ||
      !catalog->getForeignTableUnlocked(table_key[CHUNK_KEY_TABLE_IDX])->isAppendMode()) {
    clearDataWrapper(table_key);
  }
}

void ForeignStorageMgr::clearDataWrapper(const ChunkKey& table_key) {
  std::lock_guard data_wrapper_lock(data_wrapper_mutex_);
  // May not be created yet
  if (data_wrapper_map_.find(table_key) != data_wrapper_map_.end()) {
    data_wrapper_map_.erase(table_key);
  }
}

void ForeignStorageMgr::clearTempChunkBufferMapEntriesForTableUnlocked(
    const ChunkKey& table_key) {
  CHECK(is_table_key(table_key));
  auto start_it = temp_chunk_buffer_map_.lower_bound(table_key);
  ChunkKey upper_bound_prefix{table_key[CHUNK_KEY_DB_IDX],
                              table_key[CHUNK_KEY_TABLE_IDX],
                              std::numeric_limits<int>::max()};
  auto end_it = temp_chunk_buffer_map_.upper_bound(upper_bound_prefix);
  temp_chunk_buffer_map_.erase(start_it, end_it);
}

void ForeignStorageMgr::clearTempChunkBufferMapEntriesForTable(
    const ChunkKey& table_key) {
  std::lock_guard temp_chunk_buffer_map_lock(temp_chunk_buffer_map_mutex_);
  clearTempChunkBufferMapEntriesForTableUnlocked(table_key);
}

bool ForeignStorageMgr::isDatawrapperRestored(const ChunkKey& chunk_key) {
  if (!hasDataWrapperForChunk(chunk_key)) {
    return false;
  }
  return getDataWrapper(chunk_key)->isRestored();
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

void ForeignStorageMgr::createAndPopulateDataWrapperIfNotExists(
    const ChunkKey& chunk_key) {
  ChunkKey table_key = get_table_key(chunk_key);
  if (createDataWrapperIfNotExists(table_key)) {
    ChunkMetadataVector chunk_metadata;
    getDataWrapper(table_key)->populateChunkMetadata(chunk_metadata);
  }
}

std::set<ChunkKey> get_keys_set_from_table(const ChunkKey& destination_chunk_key) {
  std::set<ChunkKey> chunk_keys;
  auto db_id = destination_chunk_key[CHUNK_KEY_DB_IDX];
  auto table_id = destination_chunk_key[CHUNK_KEY_TABLE_IDX];
  auto destination_column_id = destination_chunk_key[CHUNK_KEY_COLUMN_IDX];
  auto fragment_id = destination_chunk_key[CHUNK_KEY_FRAGMENT_IDX];
  auto catalog = Catalog_Namespace::SysCatalog::instance().getCatalog(db_id);
  CHECK(catalog);
  auto foreign_table = catalog->getForeignTableUnlocked(table_id);

  ForeignTableSchema schema{db_id, foreign_table};
  auto logical_column = schema.getLogicalColumn(destination_column_id);
  auto logical_column_id = logical_column->columnId;

  for (auto column_id = logical_column_id;
       column_id <= (logical_column_id + logical_column->columnType.get_physical_cols());
       column_id++) {
    auto column = schema.getColumnDescriptor(column_id);
    if (column->columnType.is_varlen_indeed()) {
      ChunkKey data_chunk_key = {db_id, table_id, column->columnId, fragment_id, 1};
      chunk_keys.emplace(data_chunk_key);

      ChunkKey index_chunk_key{db_id, table_id, column->columnId, fragment_id, 2};
      chunk_keys.emplace(index_chunk_key);
    } else {
      ChunkKey data_chunk_key = {db_id, table_id, column->columnId, fragment_id};
      chunk_keys.emplace(data_chunk_key);
    }
  }
  return chunk_keys;
}

std::vector<ChunkKey> get_keys_vec_from_table(const ChunkKey& destination_chunk_key) {
  std::vector<ChunkKey> chunk_keys;
  auto db_id = destination_chunk_key[CHUNK_KEY_DB_IDX];
  auto table_id = destination_chunk_key[CHUNK_KEY_TABLE_IDX];
  auto destination_column_id = destination_chunk_key[CHUNK_KEY_COLUMN_IDX];
  auto fragment_id = destination_chunk_key[CHUNK_KEY_FRAGMENT_IDX];
  auto catalog = Catalog_Namespace::SysCatalog::instance().getCatalog(db_id);
  CHECK(catalog);
  auto foreign_table = catalog->getForeignTableUnlocked(table_id);

  ForeignTableSchema schema{db_id, foreign_table};
  auto logical_column = schema.getLogicalColumn(destination_column_id);
  auto logical_column_id = logical_column->columnId;

  for (auto column_id = logical_column_id;
       column_id <= (logical_column_id + logical_column->columnType.get_physical_cols());
       column_id++) {
    auto column = schema.getColumnDescriptor(column_id);
    if (column->columnType.is_varlen_indeed()) {
      ChunkKey data_chunk_key = {db_id, table_id, column->columnId, fragment_id, 1};
      chunk_keys.emplace_back(data_chunk_key);

      ChunkKey index_chunk_key{db_id, table_id, column->columnId, fragment_id, 2};
      chunk_keys.emplace_back(index_chunk_key);
    } else {
      ChunkKey data_chunk_key = {db_id, table_id, column->columnId, fragment_id};
      chunk_keys.emplace_back(data_chunk_key);
    }
  }
  return chunk_keys;
}

ChunkToBufferMap ForeignStorageMgr::allocateTempBuffersForChunks(
    const std::set<ChunkKey>& chunk_keys) {
  ChunkToBufferMap chunk_buffer_map;
  std::lock_guard temp_chunk_buffer_map_lock(temp_chunk_buffer_map_mutex_);
  for (const auto& chunk_key : chunk_keys) {
    temp_chunk_buffer_map_[chunk_key] = std::make_unique<ForeignStorageBuffer>();
    chunk_buffer_map[chunk_key] = temp_chunk_buffer_map_[chunk_key].get();
    chunk_buffer_map[chunk_key]->resetToEmpty();
  }
  return chunk_buffer_map;
}

void ForeignStorageMgr::setParallelismHints(
    const std::map<ChunkKey, std::set<ParallelismHint>>& hints_per_table) {
  std::shared_lock data_wrapper_lock(parallelism_hints_mutex_);
  parallelism_hints_per_table_ = hints_per_table;
}

void ForeignStorageMgr::getOptionalChunkKeySet(
    std::set<ChunkKey>& optional_chunk_keys,
    const ChunkKey& chunk_key,
    const std::set<ChunkKey>& required_chunk_keys,
    const ForeignDataWrapper::ParallelismLevel parallelism_level) {
  if (parallelism_level == ForeignDataWrapper::NONE) {
    return;
  }
  std::shared_lock data_wrapper_lock(parallelism_hints_mutex_);
  for (const auto& hint : parallelism_hints_per_table_[get_table_key(chunk_key)]) {
    const auto& [column_id, fragment_id] = hint;
    ChunkKey optional_chunk_key_key = get_table_key(chunk_key);
    optional_chunk_key_key.push_back(column_id);
    auto optional_chunk_key = optional_chunk_key_key;
    if (parallelism_level == ForeignDataWrapper::INTRA_FRAGMENT) {
      optional_chunk_key.push_back(chunk_key[CHUNK_KEY_FRAGMENT_IDX]);
    } else if (parallelism_level == ForeignDataWrapper::INTER_FRAGMENT) {
      optional_chunk_key.push_back(fragment_id);
    } else {
      UNREACHABLE();
    }
    std::set<ChunkKey> keys = get_keys_set_from_table(optional_chunk_key);
    for (const auto& key : keys) {
      if (required_chunk_keys.find(key) == required_chunk_keys.end()) {
        optional_chunk_keys.insert(key);
      }
    }
  }
}

}  // namespace foreign_storage
