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

#include "ForeignStorageMgr.h"

#include "AbstractFileStorageDataWrapper.h"
#include "Catalog/Catalog.h"
#include "DataMgr/ForeignStorage/FsiChunkUtils.h"
#include "ForeignDataWrapperFactory.h"
#include "ForeignStorageException.h"
#include "ForeignTableSchema.h"
#include "Shared/distributed.h"

extern bool g_enable_fsi;
extern bool g_enable_s3_fsi;

namespace {
void filter_metadata_by_leaf(ChunkMetadataVector& meta_vec, const ChunkKey& key_prefix) {
  if (!foreign_storage::is_shardable_key(key_prefix)) {
    return;
  }
  for (auto it = meta_vec.begin(); it != meta_vec.end();) {
    it = foreign_storage::fragment_maps_to_leaf(it->first) ? std::next(it)
                                                           : meta_vec.erase(it);
  }
}
}  // namespace

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
  auto foreign_table = catalog->getForeignTable(chunk_key[CHUNK_KEY_TABLE_IDX]);
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

ChunkSizeValidator::ChunkSizeValidator(const ChunkKey& chunk_key) {
  int table_id = chunk_key[CHUNK_KEY_TABLE_IDX];
  column_id_ = chunk_key[CHUNK_KEY_COLUMN_IDX];
  catalog_ =
      Catalog_Namespace::SysCatalog::instance().getCatalog(chunk_key[CHUNK_KEY_DB_IDX]);
  column_ = catalog_->getMetadataForColumn(table_id, column_id_);
  foreign_table_ = catalog_->getForeignTable(table_id);
  max_chunk_size_ = foreign_table_->maxChunkSize;
}

void ChunkSizeValidator::validateChunkSize(const AbstractBuffer* buffer) const {
  CHECK(buffer);
  int64_t actual_chunk_size = buffer->size();
  if (actual_chunk_size > max_chunk_size_) {
    throwChunkSizeViolatedError(actual_chunk_size);
  }
}

void ChunkSizeValidator::validateChunkSizes(const ChunkToBufferMap& buffers) const {
  for (const auto& [chunk_key, buffer] : buffers) {
    int64_t actual_chunk_size = buffer->size();
    if (actual_chunk_size > max_chunk_size_) {
      throwChunkSizeViolatedError(actual_chunk_size, chunk_key[CHUNK_KEY_COLUMN_IDX]);
    }
  }
}

void ChunkSizeValidator::throwChunkSizeViolatedError(const int64_t actual_chunk_size,
                                                     const int column_id) const {
  std::string column_name = column_->columnName;
  if (column_id > 0) {
    column_name =
        catalog_->getMetadataForColumn(foreign_table_->tableId, column_id)->columnName;
  }
  std::stringstream error_stream;
  error_stream << "Chunk populated by data wrapper which is " << actual_chunk_size
               << " bytes exceeds maximum byte size limit of " << max_chunk_size_ << "."
               << " Foreign table: " << foreign_table_->tableName
               << ", column name : " << column_name;
  throw ForeignStorageException(error_stream.str());
}

void ForeignStorageMgr::fetchBuffer(const ChunkKey& chunk_key,
                                    AbstractBuffer* destination_buffer,
                                    const size_t num_bytes) {
  ChunkSizeValidator chunk_size_validator(chunk_key);

  checkIfS3NeedsToBeEnabled(chunk_key);
  CHECK(destination_buffer);
  CHECK(!destination_buffer->isDirty());
  // Use a temp buffer if we have no cache buffers and have one mapped for this chunk.
  if (fetchBufferIfTempBufferMapEntryExists(chunk_key, destination_buffer, num_bytes)) {
    chunk_size_validator.validateChunkSize(destination_buffer);
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

  auto column_keys = get_column_key_set(chunk_key);

  // Use hints to prefetch other chunks in fragment
  ChunkToBufferMap optional_buffers;

  // Use hints to prefetch other chunks in fragment into cache
  auto optional_keys = getOptionalChunkKeySetAndNormalizeCache(
      chunk_key, column_keys, getDataWrapper(chunk_key)->getNonCachedParallelismLevel());
  if (optional_keys.size()) {
    optional_buffers = allocateTempBuffersForChunks(optional_keys);
  }

  // Remove the original key as it will be replaced by the destination_buffer.
  column_keys.erase(chunk_key);
  auto required_buffers = allocateTempBuffersForChunks(column_keys);
  required_buffers[chunk_key] = destination_buffer;
  // populate will write directly to destination_buffer so no need to copy.
  getDataWrapper(chunk_key)->populateChunkBuffers(required_buffers, optional_buffers);
  chunk_size_validator.validateChunkSizes(required_buffers);
  chunk_size_validator.validateChunkSizes(optional_buffers);
  updateFragmenterMetadata(required_buffers);
  updateFragmenterMetadata(optional_buffers);
}

void ForeignStorageMgr::updateFragmenterMetadata(const ChunkToBufferMap& buffers) const {
  for (const auto& [key, buffer] : buffers) {
    auto catalog =
        Catalog_Namespace::SysCatalog::instance().getCatalog(key[CHUNK_KEY_DB_IDX]);
    auto column = catalog->getMetadataForColumn(key[CHUNK_KEY_TABLE_IDX],
                                                key[CHUNK_KEY_COLUMN_IDX]);
    if (column->columnType.is_varlen_indeed() &&
        key[CHUNK_KEY_VARLEN_IDX] == 2) {  // skip over index buffers
      continue;
    }
    auto foreign_table = catalog->getForeignTable(key[CHUNK_KEY_TABLE_IDX]);
    auto fragmenter = foreign_table->fragmenter;
    if (!fragmenter) {
      continue;
    }
    auto encoder = buffer->getEncoder();
    CHECK(encoder);
    auto chunk_metadata = std::make_shared<ChunkMetadata>();
    encoder->getMetadata(chunk_metadata);
    fragmenter->updateColumnChunkMetadata(
        column, key[CHUNK_KEY_FRAGMENT_IDX], chunk_metadata);
  }
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

  if (!is_table_enabled_on_node(key_prefix)) {
    // If the table is not enabled for this node then the request should do nothing.
    return;
  }

  checkIfS3NeedsToBeEnabled(key_prefix);
  createDataWrapperIfNotExists(key_prefix);

  try {
    getDataWrapper(key_prefix)->populateChunkMetadata(chunk_metadata);
    filter_metadata_by_leaf(chunk_metadata, key_prefix);
  } catch (...) {
    eraseDataWrapper(key_prefix);
    throw;
  }
}

void ForeignStorageMgr::removeTableRelatedDS(const int db_id, const int table_id) {
  const ChunkKey table_key{db_id, table_id};
  {
    std::lock_guard data_wrapper_lock(data_wrapper_mutex_);
    if (auto mock_it = mocked_wrapper_map_.find(table_key);
        mock_it != mocked_wrapper_map_.end()) {
      mock_it->second->unsetParentWrapper();
    }
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

bool ForeignStorageMgr::hasDataWrapperForChunk(const ChunkKey& chunk_key) const {
  std::shared_lock data_wrapper_lock(data_wrapper_mutex_);
  CHECK(has_table_prefix(chunk_key));
  ChunkKey table_key{chunk_key[CHUNK_KEY_DB_IDX], chunk_key[CHUNK_KEY_TABLE_IDX]};
  return data_wrapper_map_.find(table_key) != data_wrapper_map_.end();
}

std::shared_ptr<ForeignDataWrapper> ForeignStorageMgr::getDataWrapper(
    const ChunkKey& chunk_key) const {
  std::shared_lock data_wrapper_lock(data_wrapper_mutex_);
  ChunkKey table_key{chunk_key[CHUNK_KEY_DB_IDX], chunk_key[CHUNK_KEY_TABLE_IDX]};
  CHECK(data_wrapper_map_.find(table_key) != data_wrapper_map_.end());
  return data_wrapper_map_.at(table_key);
}

void ForeignStorageMgr::setDataWrapper(
    const ChunkKey& table_key,
    std::shared_ptr<MockForeignDataWrapper> data_wrapper) {
  CHECK(is_table_key(table_key));
  std::lock_guard data_wrapper_lock(data_wrapper_mutex_);
  if (auto wrapper_iter = data_wrapper_map_.find(table_key);
      wrapper_iter != data_wrapper_map_.end()) {
    data_wrapper->setParentWrapper(wrapper_iter->second);
    data_wrapper_map_[table_key] = data_wrapper;
  }
  // If a wrapper does not yet exist, then delay setting the mock until we actually
  // create the wrapper. Preserve mock wrappers separately so they can persist the parent
  // being re-created.
  mocked_wrapper_map_[table_key] = data_wrapper;
}

void ForeignStorageMgr::createDataWrapperUnlocked(int32_t db_id, int32_t tb_id) {
  auto catalog = Catalog_Namespace::SysCatalog::instance().getCatalog(db_id);
  CHECK(catalog);
  auto foreign_table = catalog->getForeignTable(tb_id);
  ChunkKey table_key{db_id, tb_id};
  data_wrapper_map_[table_key] = ForeignDataWrapperFactory::create(
      foreign_table->foreign_server->data_wrapper_type, db_id, foreign_table);

  // If we are testing with mocks, then we want to re-wrap new wrappers with mocks if a
  // table was given a mock wrapper earlier and destroyed.
  if (auto mock_it = mocked_wrapper_map_.find(table_key);
      mock_it != mocked_wrapper_map_.end()) {
    mock_it->second->setParentWrapper(data_wrapper_map_.at(table_key));
    data_wrapper_map_[table_key] = mock_it->second;
  }
}

bool ForeignStorageMgr::createDataWrapperIfNotExists(const ChunkKey& chunk_key) {
  std::lock_guard data_wrapper_lock(data_wrapper_mutex_);
  ChunkKey table_key{chunk_key[CHUNK_KEY_DB_IDX], chunk_key[CHUNK_KEY_TABLE_IDX]};
  if (data_wrapper_map_.find(table_key) == data_wrapper_map_.end()) {
    auto [db_id, tb_id] = get_table_prefix(chunk_key);
    createDataWrapperUnlocked(db_id, tb_id);
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
      !catalog->getForeignTable(table_key[CHUNK_KEY_TABLE_IDX])->isAppendMode()) {
    eraseDataWrapper(table_key);
  }
}

void ForeignStorageMgr::eraseDataWrapper(const ChunkKey& table_key) {
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

size_t ForeignStorageMgr::getMaxSize() const {
  UNREACHABLE();
  return 0;  // Added to avoid "no return statement" compiler warning
}

size_t ForeignStorageMgr::getInUseSize() const {
  UNREACHABLE();
  return 0;  // Added to avoid "no return statement" compiler warning
}

size_t ForeignStorageMgr::getAllocated() const {
  UNREACHABLE();
  return 0;  // Added to avoid "no return statement" compiler warning
}

bool ForeignStorageMgr::isAllocationCapped() const {
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

size_t get_max_chunk_size(const ChunkKey& key) {
  auto [db_id, table_id] = get_table_prefix(key);
  auto catalog = Catalog_Namespace::SysCatalog::instance().getCatalog(db_id);
  CHECK(catalog);
  return catalog->getForeignTable(table_id)->maxChunkSize;
}

std::set<ChunkKey> get_column_key_set(const ChunkKey& destination_chunk_key) {
  std::set<ChunkKey> chunk_keys;
  auto db_id = destination_chunk_key[CHUNK_KEY_DB_IDX];
  auto table_id = destination_chunk_key[CHUNK_KEY_TABLE_IDX];
  auto destination_column_id = destination_chunk_key[CHUNK_KEY_COLUMN_IDX];
  auto fragment_id = destination_chunk_key[CHUNK_KEY_FRAGMENT_IDX];
  auto catalog = Catalog_Namespace::SysCatalog::instance().getCatalog(db_id);
  CHECK(catalog);
  auto foreign_table = catalog->getForeignTable(table_id);

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

std::vector<ChunkKey> get_column_key_vec(const ChunkKey& destination_chunk_key) {
  std::vector<ChunkKey> chunk_keys;
  auto db_id = destination_chunk_key[CHUNK_KEY_DB_IDX];
  auto table_id = destination_chunk_key[CHUNK_KEY_TABLE_IDX];
  auto destination_column_id = destination_chunk_key[CHUNK_KEY_COLUMN_IDX];
  auto fragment_id = destination_chunk_key[CHUNK_KEY_FRAGMENT_IDX];
  auto catalog = Catalog_Namespace::SysCatalog::instance().getCatalog(db_id);
  CHECK(catalog);
  auto foreign_table = catalog->getForeignTable(table_id);

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

// Defines the "<" operator to use as a comparator.
// This is similar to comparing chunks normally, but we want to give fragments a higher
// priority than columns so that if we have to exit early we prioritize same fragment
// fetching.
bool set_comp(const ChunkKey& left, const ChunkKey& right) {
  CHECK_GE(left.size(), 4ULL);
  CHECK_GE(right.size(), 4ULL);
  if ((left[CHUNK_KEY_DB_IDX] < right[CHUNK_KEY_DB_IDX]) ||
      (left[CHUNK_KEY_TABLE_IDX] < right[CHUNK_KEY_TABLE_IDX]) ||
      (left[CHUNK_KEY_FRAGMENT_IDX] < right[CHUNK_KEY_FRAGMENT_IDX]) ||
      (left[CHUNK_KEY_COLUMN_IDX] < right[CHUNK_KEY_COLUMN_IDX])) {
    return true;
  }
  if (left.size() < right.size()) {
    return true;
  }
  if (is_varlen_key(left) && is_varlen_key(right) &&
      left[CHUNK_KEY_VARLEN_IDX] < right[CHUNK_KEY_VARLEN_IDX]) {
    return true;
  }
  return false;
}

bool contains_fragment_key(const std::set<ChunkKey>& key_set,
                           const ChunkKey& target_key) {
  for (const auto& key : key_set) {
    if (get_fragment_key(target_key) == get_fragment_key(key)) {
      return true;
    }
  }
  return false;
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
  std::unique_lock data_wrapper_lock(parallelism_hints_mutex_);
  parallelism_hints_per_table_ = hints_per_table;
}

std::pair<std::set<ChunkKey, decltype(set_comp)*>,
          std::set<ChunkKey, decltype(set_comp)*>>
ForeignStorageMgr::getPrefetchSets(
    const ChunkKey& chunk_key,
    const std::set<ChunkKey>& required_chunk_keys,
    const ForeignDataWrapper::ParallelismLevel parallelism_level) const {
  std::shared_lock data_wrapper_lock(parallelism_hints_mutex_);
  auto same_fragment_keys = std::set<ChunkKey, decltype(set_comp)*>(set_comp);
  auto diff_fragment_keys = std::set<ChunkKey, decltype(set_comp)*>(set_comp);

  auto table_hints = parallelism_hints_per_table_.find(get_table_key(chunk_key));
  if (table_hints == parallelism_hints_per_table_.end()) {
    return {{}, {}};
  }
  for (const auto& hint : table_hints->second) {
    const auto& [column_id, fragment_id] = hint;
    auto optional_chunk_key = get_table_key(chunk_key);
    optional_chunk_key.push_back(column_id);
    if (parallelism_level == ForeignDataWrapper::INTRA_FRAGMENT) {
      optional_chunk_key.push_back(chunk_key[CHUNK_KEY_FRAGMENT_IDX]);
    } else if (parallelism_level == ForeignDataWrapper::INTER_FRAGMENT) {
      optional_chunk_key.push_back(fragment_id);
    } else {
      UNREACHABLE() << "Unknown parallelism level.";
    }

    CHECK(!key_does_not_shard_to_leaf(optional_chunk_key));

    if (!contains_fragment_key(required_chunk_keys, optional_chunk_key)) {
      // Do not insert an optional key if it is already a required key.
      if (optional_chunk_key[CHUNK_KEY_FRAGMENT_IDX] ==
          chunk_key[CHUNK_KEY_FRAGMENT_IDX]) {
        same_fragment_keys.emplace(optional_chunk_key);
      } else {
        diff_fragment_keys.emplace(optional_chunk_key);
      }
    }
  }
  return {same_fragment_keys, diff_fragment_keys};
}

std::set<ChunkKey> ForeignStorageMgr::getOptionalKeysWithinSizeLimit(
    const ChunkKey& chunk_key,
    const std::set<ChunkKey, decltype(set_comp)*>& same_fragment_keys,
    const std::set<ChunkKey, decltype(set_comp)*>& diff_fragment_keys) const {
  std::set<ChunkKey> optional_keys;
  for (const auto& keys : {same_fragment_keys, diff_fragment_keys}) {
    for (auto key : keys) {
      auto column_keys = get_column_key_set(key);
      for (auto column_key : column_keys) {
        optional_keys.emplace(column_key);
      }
    }
  }
  return optional_keys;
}

std::set<ChunkKey> ForeignStorageMgr::getOptionalChunkKeySetAndNormalizeCache(
    const ChunkKey& chunk_key,
    const std::set<ChunkKey>& required_chunk_keys,
    const ForeignDataWrapper::ParallelismLevel parallelism_level) {
  if (parallelism_level == ForeignDataWrapper::NONE) {
    return {};
  }

  auto [same_fragment_keys, diff_fragment_keys] =
      getPrefetchSets(chunk_key, required_chunk_keys, parallelism_level);

  auto optional_keys =
      getOptionalKeysWithinSizeLimit(chunk_key, same_fragment_keys, diff_fragment_keys);

  std::set<ChunkKey> optional_keys_to_delete;
  if (!optional_keys.empty()) {
    for (const auto& key : optional_keys) {
      if (!shared::contains(optional_keys_to_delete, key)) {
        auto key_set = get_column_key_set(key);
        auto all_keys_cached =
            std::all_of(key_set.begin(), key_set.end(), [this](const ChunkKey& key) {
              return isChunkCached(key);
            });
        // Avoid cases where the optional_keys set or cache only has a subset of the
        // column key set.
        if (all_keys_cached) {
          optional_keys_to_delete.insert(key_set.begin(), key_set.end());
        } else {
          evictChunkFromCache(key);
        }
      }
    }
  }
  for (const auto& key : optional_keys_to_delete) {
    optional_keys.erase(key);
  }
  return optional_keys;
}

size_t ForeignStorageMgr::maxFetchSize(int32_t db_id) const {
  return 0;
}

bool ForeignStorageMgr::hasMaxFetchSize() const {
  return false;
}

bool ForeignStorageMgr::isChunkCached(const ChunkKey& chunk_key) const {
  std::shared_lock temp_chunk_buffer_map_lock(temp_chunk_buffer_map_mutex_);
  return temp_chunk_buffer_map_.find(chunk_key) != temp_chunk_buffer_map_.end();
}

void ForeignStorageMgr::evictChunkFromCache(const ChunkKey& chunk_key) {
  std::unique_lock temp_chunk_buffer_map_lock(temp_chunk_buffer_map_mutex_);
  auto it = temp_chunk_buffer_map_.find(chunk_key);
  if (it != temp_chunk_buffer_map_.end()) {
    temp_chunk_buffer_map_.erase(it);
  }
}

// Determine if a wrapper is enabled on the current distributed node.
bool is_table_enabled_on_node(const ChunkKey& chunk_key) {
  CHECK(is_table_key(chunk_key));

  // Replicated tables, system tables, and non-distributed tables are on all nodes by
  // default.  Leaf nodes are on, but will filter their results later by their node index.
  if (!dist::is_distributed() || dist::is_leaf_node() ||
      is_replicated_table_chunk_key(chunk_key) || is_system_table_chunk_key(chunk_key)) {
    return true;
  }

  // If we aren't a leaf node then we are the aggregator, and the aggregator should not
  // have sharded data.
  return false;
}
}  // namespace foreign_storage
