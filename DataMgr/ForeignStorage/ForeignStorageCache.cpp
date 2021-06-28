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

/*
  TODO(Misiu): A lot of methods here can be made asyncronous.  It may be worth an
  investigation to determine if it's worth adding async versions of them for performance
  reasons.
*/

#include "ForeignStorageCache.h"
#include "Shared/File.h"
#include "Shared/measure.h"

namespace foreign_storage {
using read_lock = mapd_shared_lock<mapd_shared_mutex>;
using write_lock = mapd_unique_lock<mapd_shared_mutex>;

namespace {
template <typename Func, typename T>
void iterate_over_matching_prefix(Func func,
                                  T& chunk_collection,
                                  const ChunkKey& chunk_prefix) {
  ChunkKey upper_prefix(chunk_prefix);
  upper_prefix.push_back(std::numeric_limits<int>::max());
  auto end_it = chunk_collection.upper_bound(static_cast<const ChunkKey>(upper_prefix));
  for (auto chunk_it = chunk_collection.lower_bound(chunk_prefix); chunk_it != end_it;
       ++chunk_it) {
    func(*chunk_it);
  }
}

void set_metadata_for_buffer(AbstractBuffer* buffer, ChunkMetadata* meta) {
  buffer->initEncoder(meta->sqlType);
  buffer->setSize(meta->numBytes);
  buffer->getEncoder()->setNumElems(meta->numElements);
  buffer->getEncoder()->resetChunkStats(meta->chunkStats);
  buffer->setUpdated();
}
}  // namespace

ForeignStorageCache::ForeignStorageCache(const File_Namespace::DiskCacheConfig& config) {
  validatePath(config.path);
  caching_file_mgr_ = std::make_unique<File_Namespace::CachingFileMgr>(config);
}

void ForeignStorageCache::deleteBufferIfExists(const ChunkKey& chunk_key) {
  caching_file_mgr_->deleteBufferIfExists(chunk_key);
}

void ForeignStorageCache::putBuffer(const ChunkKey& key,
                                    AbstractBuffer* buf,
                                    const size_t num_bytes) {
  caching_file_mgr_->putBuffer(key, buf, num_bytes);
  CHECK(!buf->isDirty());
}

void ForeignStorageCache::checkpoint(const int32_t db_id, const int32_t tb_id) {
  caching_file_mgr_->checkpoint(db_id, tb_id);
}

File_Namespace::FileBuffer* ForeignStorageCache::getCachedChunkIfExists(
    const ChunkKey& chunk_key) {
  auto buf = caching_file_mgr_->getBufferIfExists(chunk_key);
  if (buf && (*buf)->hasDataPages()) {
    return *buf;
  }
  return nullptr;
}

bool ForeignStorageCache::isMetadataCached(const ChunkKey& chunk_key) const {
  auto buf = caching_file_mgr_->getBufferIfExists(chunk_key);
  if (buf) {
    return (*buf)->hasEncoder();
  }
  return false;
}

void ForeignStorageCache::cacheMetadataVec(const ChunkMetadataVector& metadata_vec) {
  auto timer = DEBUG_TIMER(__func__);
  if (metadata_vec.empty()) {
    return;
  }
  auto first_chunk_key = metadata_vec.begin()->first;
  for (auto& [chunk_key, metadata] : metadata_vec) {
    CHECK(in_same_table(chunk_key, first_chunk_key));
    AbstractBuffer* buf;
    AbstractBuffer* index_buffer = nullptr;
    ChunkKey index_chunk_key;
    if (is_varlen_key(chunk_key)) {
      // For variable length chunks, metadata is associated with the data chunk.
      CHECK(is_varlen_data_key(chunk_key));
      index_chunk_key = {chunk_key[CHUNK_KEY_DB_IDX],
                         chunk_key[CHUNK_KEY_TABLE_IDX],
                         chunk_key[CHUNK_KEY_COLUMN_IDX],
                         chunk_key[CHUNK_KEY_FRAGMENT_IDX],
                         2};
    }
    bool chunk_in_cache = false;
    if (!caching_file_mgr_->isBufferOnDevice(chunk_key)) {
      buf = caching_file_mgr_->createBuffer(chunk_key);

      if (!index_chunk_key.empty()) {
        CHECK(!caching_file_mgr_->isBufferOnDevice(index_chunk_key));
        index_buffer = caching_file_mgr_->createBuffer(index_chunk_key);
        CHECK(index_buffer);
      }
    } else {
      buf = caching_file_mgr_->getBuffer(chunk_key);

      if (!index_chunk_key.empty()) {
        CHECK(caching_file_mgr_->isBufferOnDevice(index_chunk_key));
        index_buffer = caching_file_mgr_->getBuffer(index_chunk_key);
        CHECK(index_buffer);
      }

      // We should have already cleared the data unless we are appending
      // If the buffer metadata has changed, we need to remove this chunk
      if (buf->getEncoder() != nullptr) {
        const std::shared_ptr<ChunkMetadata> buf_metadata =
            std::make_shared<ChunkMetadata>();
        buf->getEncoder()->getMetadata(buf_metadata);
        chunk_in_cache = *metadata.get() == *buf_metadata;
      }
    }

    if (!chunk_in_cache) {
      set_metadata_for_buffer(buf, metadata.get());
      eraseChunk(chunk_key);

      if (!index_chunk_key.empty()) {
        CHECK(index_buffer);
        index_buffer->setUpdated();
        eraseChunk(index_chunk_key);
      }
    }
  }
  caching_file_mgr_->checkpoint(first_chunk_key[CHUNK_KEY_DB_IDX],
                                first_chunk_key[CHUNK_KEY_TABLE_IDX]);
}

void ForeignStorageCache::getCachedMetadataVecForKeyPrefix(
    ChunkMetadataVector& metadata_vec,
    const ChunkKey& chunk_prefix) const {
  caching_file_mgr_->getChunkMetadataVecForKeyPrefix(metadata_vec, chunk_prefix);
}

bool ForeignStorageCache::hasCachedMetadataForKeyPrefix(
    const ChunkKey& chunk_prefix) const {
  ChunkMetadataVector meta_vec;
  caching_file_mgr_->getChunkMetadataVecForKeyPrefix(meta_vec, chunk_prefix);
  return (meta_vec.size() > 0);
}

void ForeignStorageCache::clearForTablePrefix(const ChunkKey& chunk_prefix) {
  CHECK(is_table_key(chunk_prefix));
  auto timer = DEBUG_TIMER(__func__);
  caching_file_mgr_->clearForTable(chunk_prefix[CHUNK_KEY_DB_IDX],
                                   chunk_prefix[CHUNK_KEY_TABLE_IDX]);
}

void ForeignStorageCache::clear() {
  auto timer = DEBUG_TIMER(__func__);
  // FileMgrs do not clean up after themselves nicely, so we need to close all their disk
  // resources and then re-create the CachingFileMgr to reset it.
  caching_file_mgr_->closeRemovePhysical();
  boost::filesystem::create_directory(caching_file_mgr_->getFileMgrBasePath());
  caching_file_mgr_ = caching_file_mgr_->reconstruct();
}

std::vector<ChunkKey> ForeignStorageCache::getCachedChunksForKeyPrefix(
    const ChunkKey& chunk_prefix) const {
  return caching_file_mgr_->getChunkKeysForPrefix(chunk_prefix);
}

ChunkToBufferMap ForeignStorageCache::getChunkBuffersForCaching(
    const std::vector<ChunkKey>& chunk_keys) const {
  ChunkToBufferMap chunk_buffer_map;
  for (const auto& chunk_key : chunk_keys) {
    CHECK(caching_file_mgr_->isBufferOnDevice(chunk_key));
    chunk_buffer_map[chunk_key] = caching_file_mgr_->getBuffer(chunk_key);
    auto file_buf =
        dynamic_cast<File_Namespace::FileBuffer*>(chunk_buffer_map[chunk_key]);
    CHECK(file_buf);
    CHECK(!file_buf->hasDataPages());

    // Clear all buffer metadata
    file_buf->resetToEmpty();
  }
  return chunk_buffer_map;
}

void ForeignStorageCache::eraseChunk(const ChunkKey& chunk_key) {
  caching_file_mgr_->removeChunkKeepMetadata(chunk_key);
}

std::string ForeignStorageCache::dumpCachedChunkEntries() const {
  return caching_file_mgr_->dumpKeysWithChunkData();
}

std::string ForeignStorageCache::dumpCachedMetadataEntries() const {
  return caching_file_mgr_->dumpKeysWithMetadata();
}

void ForeignStorageCache::validatePath(const std::string& base_path) const {
  // check if base_path already exists, and if not create one
  boost::filesystem::path path(base_path);
  if (boost::filesystem::exists(path)) {
    if (!boost::filesystem::is_directory(path)) {
      throw std::runtime_error{
          "cache path \"" + base_path +
          "\" is not a directory.  Please specify a valid directory "
          "with --disk_cache_path=<path>, or use the default location."};
    }
  } else {  // data directory does not exist
    if (!boost::filesystem::create_directory(path)) {
      throw std::runtime_error{
          "could not create directory at cache path \"" + base_path +
          "\".  Please specify a valid directory location "
          "with --disk_cache_path=<path> or use the default location."};
    }
  }
}

void ForeignStorageCache::cacheMetadataWithFragIdGreaterOrEqualTo(
    const ChunkMetadataVector& metadata_vec,
    const int frag_id) {
  // Only re-cache last fragment and above
  ChunkMetadataVector new_metadata_vec;
  for (const auto& chunk_metadata : metadata_vec) {
    if (chunk_metadata.first[CHUNK_KEY_FRAGMENT_IDX] >= frag_id) {
      new_metadata_vec.push_back(chunk_metadata);
    }
  }
  cacheMetadataVec(new_metadata_vec);
}

AbstractBuffer* ForeignStorageCache::getChunkBufferForPrecaching(
    const ChunkKey& chunk_key,
    bool is_new_buffer) {
  if (!is_new_buffer) {
    CHECK(caching_file_mgr_->isBufferOnDevice(chunk_key));
    return caching_file_mgr_->getBuffer(chunk_key);
  } else {
    CHECK(!caching_file_mgr_->isBufferOnDevice(chunk_key));
    return caching_file_mgr_->createBuffer(chunk_key);
  }
}

void ForeignStorageCache::storeDataWrapper(const std::string& doc,
                                           int32_t db_id,
                                           int32_t tb_id) {
  caching_file_mgr_->writeWrapperFile(doc, db_id, tb_id);
}

}  // namespace foreign_storage
