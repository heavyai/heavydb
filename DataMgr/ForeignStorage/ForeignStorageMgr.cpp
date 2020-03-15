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

namespace foreign_storage {
ForeignStorageMgr::ForeignStorageMgr() : AbstractBufferMgr(0), data_wrapper_map_({}) {}

AbstractBuffer* ForeignStorageMgr::getBuffer(const ChunkKey& chunk_key,
                                             const size_t num_bytes) {
  return getDataWrapper(chunk_key)->getChunkBuffer(chunk_key);
}

void ForeignStorageMgr::fetchBuffer(const ChunkKey& chunk_key,
                                    AbstractBuffer* destination_buffer,
                                    const size_t num_bytes) {
  CHECK(!destination_buffer->isDirty());

  auto chunk_buffer = getDataWrapper(chunk_key)->getChunkBuffer(chunk_key);
  size_t chunk_size = (num_bytes == 0) ? chunk_buffer->size() : num_bytes;
  destination_buffer->reserve(chunk_size);
  chunk_buffer->read(destination_buffer->getMemoryPtr() + destination_buffer->size(),
                     chunk_size - destination_buffer->size(),
                     destination_buffer->size(),
                     destination_buffer->getType(),
                     destination_buffer->getDeviceId());
  destination_buffer->setSize(chunk_size);
  destination_buffer->syncEncoder(chunk_buffer);
}

void ForeignStorageMgr::getChunkMetadataVec(
    std::vector<std::pair<ChunkKey, ChunkMetadata>>& chunk_metadata) {
  std::shared_lock data_wrapper_lock(data_wrapper_mutex_);
  for (auto& [table_chunk_key, data_wrapper] : data_wrapper_map_) {
    data_wrapper->populateMetadataForChunkKeyPrefix(table_chunk_key, chunk_metadata);
  }
}

void ForeignStorageMgr::getChunkMetadataVecForKeyPrefix(
    std::vector<std::pair<ChunkKey, ChunkMetadata>>& chunk_metadata,
    const ChunkKey& keyPrefix) {
  createDataWrapperIfNotExists(keyPrefix);
  getDataWrapper(keyPrefix)->populateMetadataForChunkKeyPrefix(keyPrefix, chunk_metadata);
}

void ForeignStorageMgr::removeTableRelatedDS(const int db_id, const int table_id) {
  std::lock_guard data_wrapper_lock(data_wrapper_mutex_);
  data_wrapper_map_.erase({db_id, table_id});
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
    } else {
      throw std::runtime_error("Unsupported data wrapper");
    }
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
