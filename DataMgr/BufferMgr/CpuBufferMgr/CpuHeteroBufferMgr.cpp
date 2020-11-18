/*
 * Copyright 2020 MapD Technologies, Inc.
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

#include <fstream>
#include <sstream>
#include <string>

#include "DataMgr/BufferMgr/CpuBufferMgr/CpuHeteroBufferMgr.h"

namespace Buffer_Namespace {

static std::string keyToString(const ChunkKey& key) {
  std::ostringstream oss;

  oss << " key: ";
  for (auto sub_key : key) {
    oss << sub_key << ",";
  }
  return oss.str();
}

CpuHeteroBufferMgr::CpuHeteroBufferMgr(const int device_id,
                                       const size_t max_buffer_size,
                                       const std::string& pmm_path,
                                       CudaMgr_Namespace::CudaMgr* cuda_mgr,
                                       const size_t page_size,
                                       AbstractBufferMgr* parent_mgr)
    : AbstractBufferMgr(device_id), parent_mgr_(parent_mgr), cuda_mgr_(cuda_mgr), mem_resource_provider_(pmm_path), page_size_(page_size), buffer_epoch_(0) {
}

CpuHeteroBufferMgr::CpuHeteroBufferMgr(const int device_id,
                                       const size_t max_buffer_size,
                                       CudaMgr_Namespace::CudaMgr* cuda_mgr,
                                       const size_t page_size,
                                       AbstractBufferMgr* parent_mgr)
    : AbstractBufferMgr(device_id), parent_mgr_(parent_mgr), cuda_mgr_(cuda_mgr), page_size_(page_size), buffer_epoch_(0) {
}

CpuHeteroBufferMgr::~CpuHeteroBufferMgr() {
  clear();
}

#ifdef HAVE_DCPMM
AbstractBuffer* CpuHeteroBufferMgr::createBuffer(BufferProperty bufProp,
                                                 const ChunkKey& key,
                                                 const size_t maxRows,
                                                 const int sqlTypeSize,
                                                 const size_t pageSize) {
  AbstractBuffer *buffer = createBuffer(bufProp, key, pageSize, maxRows * sqlTypeSize);
  buffer->setMaxRows(maxRows);
  return buffer;
}
#endif /* HAVE_DCPMM */

static MemRequirements get_mem_characteristics(BufferProperty bufProp) {
  switch (bufProp) {
    case HIGH_BDWTH:
      return MemRequirements::HIGH_BDWTH;
    case LOW_LATENCY:
      return MemRequirements::LOW_LATENCY;
  }
  return MemRequirements::CAPACITY;
}

/// Throws a runtime_error if the Chunk already exists
AbstractBuffer* CpuHeteroBufferMgr::createBuffer(BufferProperty bufProp,
                                                 const ChunkKey& chunk_key,
                                                 const size_t chunk_page_size,
                                                 const size_t initial_size) {
  size_t actual_chunk_page_size = chunk_page_size;
  if (actual_chunk_page_size == 0) {
    actual_chunk_page_size = page_size_;
  }
  std::lock_guard<chunk_index_mutex_type> index_lock(chunk_index_mutex_);
  CHECK(chunk_index_.find(chunk_key) == chunk_index_.end());

  HeteroBuffer* buffer = new HeteroBuffer(device_id_, mem_resource_provider_.get(get_mem_characteristics(bufProp)), cuda_mgr_, chunk_page_size, initial_size);

  auto res = chunk_index_.emplace(chunk_key, buffer);
  CHECK(res.second);
  
  return buffer;
}

void CpuHeteroBufferMgr::deleteBuffer(const ChunkKey& key, const bool) {
  std::unique_lock<chunk_index_mutex_type> index_lock(chunk_index_mutex_);
  auto buffer_it = chunk_index_.find(key);
  CHECK(buffer_it != chunk_index_.end());
  auto buff = buffer_it->second;
  chunk_index_.erase(buffer_it);
  index_lock.unlock();

  delete buff;
}

void CpuHeteroBufferMgr::deleteBuffersWithPrefix(const ChunkKey& keyPrefix, const bool) {
  std::unique_lock<chunk_index_mutex_type> index_lock(chunk_index_mutex_);
  auto first = chunk_index_.lower_bound(keyPrefix);
  auto last = chunk_index_.upper_bound(keyPrefix);
  while (first != last) {
    auto buff = first->second;
    delete buff;
    chunk_index_.erase(first);;
  }
}

AbstractBuffer* CpuHeteroBufferMgr::getBuffer(BufferProperty bufProp, const ChunkKey& key, const size_t numBytes) {
  std::lock_guard<global_mutex_type> lock(global_mutex_);  // granular lock
  std::unique_lock<chunk_index_mutex_type> index_lock(chunk_index_mutex_);
  auto chunk_it = chunk_index_.find(key);
  bool found_buffer = chunk_it != chunk_index_.end();

  if (found_buffer) {
    AbstractBuffer* buffer = chunk_it->second;
    buffer->pin();
    index_lock.unlock();
    
    // TODO: update last touched
    buffer_epoch_++;

    if (buffer->size() < numBytes) {
      parent_mgr_->fetchBuffer(key, buffer, numBytes);
    }
    return buffer;
  } else {
    index_lock.unlock();
    AbstractBuffer* buffer = createBuffer(bufProp, key, page_size_, numBytes);
    try {
      parent_mgr_->fetchBuffer(
          key, buffer, numBytes);
    } catch (std::runtime_error& error) {
      LOG(FATAL) << "Get chunk - Could not find chunk " << keyToString(key)
                 << " in buffer pool or parent buffer pools. Error was " << error.what();
    }
    return buffer;
  }
}

void CpuHeteroBufferMgr::fetchBuffer(const ChunkKey& key,
                                     AbstractBuffer* destBuffer,
                                     const size_t numBytes) {
  AbstractBuffer* buffer = getBuffer(BufferProperty::CAPACITY, key, numBytes);
  size_t chunk_size = numBytes == 0 ? buffer->size() : numBytes;

  destBuffer->reserve(chunk_size);
  if (buffer->isUpdated()) {
    buffer->read(destBuffer->getMemoryPtr(),
                 chunk_size,
                 0,
                 destBuffer->getType(),
                 destBuffer->getDeviceId());
  } else {
    buffer->read(destBuffer->getMemoryPtr() + destBuffer->size(),
                 chunk_size - destBuffer->size(),
                 destBuffer->size(),
                 destBuffer->getType(),
                 destBuffer->getDeviceId());
  }
  destBuffer->setSize(chunk_size);
  destBuffer->syncEncoder(buffer);
  buffer->unPin();
}
  
AbstractBuffer* CpuHeteroBufferMgr::putBuffer(const ChunkKey& key,
                                              AbstractBuffer* srcBuffer,
                                              const size_t numBytes) {
  std::unique_lock<chunk_index_mutex_type> index_lock(chunk_index_mutex_);
  auto buffer_it = chunk_index_.find(key);
  bool found_buffer = buffer_it != chunk_index_.end();
  index_lock.unlock();
  AbstractBuffer* buffer;
  if (!found_buffer) {
    buffer = createBuffer(BufferProperty::CAPACITY, key, page_size_);
  } else {
    buffer = buffer_it->second;
  }

  size_t old_buffer_size = buffer->size();
  size_t new_buffer_size = numBytes == 0 ? srcBuffer->size() : numBytes;
  CHECK(!buffer->isDirty());

  if (srcBuffer->isUpdated()) {
    //@todo use dirty flags to only flush pages of chunk that need to
    // be flushed
    buffer->write((int8_t*)srcBuffer->getMemoryPtr(),
                  new_buffer_size,
                  0,
                  srcBuffer->getType(),
                  srcBuffer->getDeviceId());
  } else if (srcBuffer->isAppended()) {
    CHECK(old_buffer_size < new_buffer_size);
    buffer->append((int8_t*)srcBuffer->getMemoryPtr() + old_buffer_size,
                   new_buffer_size - old_buffer_size,
                   srcBuffer->getType(),
                   srcBuffer->getDeviceId());
  }
  srcBuffer->clearDirtyBits();
  buffer->syncEncoder(srcBuffer);
  return buffer;
}

void CpuHeteroBufferMgr::getChunkMetadataVec(ChunkMetadataVector&) {
  LOG(FATAL) << "getChunkMetadataVec not supported for BufferMgr.";
}

void CpuHeteroBufferMgr::getChunkMetadataVecForKeyPrefix(ChunkMetadataVector&,
                                                         const ChunkKey&) {
  LOG(FATAL) << "getChunkMetadataVecForPrefix not supported for BufferMgr.";
}

bool CpuHeteroBufferMgr::isBufferOnDevice(const ChunkKey& key) {
  std::lock_guard<chunk_index_mutex_type> index_lock(chunk_index_mutex_);
  if (chunk_index_.find(key) == chunk_index_.end()) {
    return false;
  } else {
    return true;
  }
}

void CpuHeteroBufferMgr::clearSlabs() {
  std::lock_guard<chunk_index_mutex_type> index_lock(chunk_index_mutex_);
  for(auto chunk_it = chunk_index_.begin(); chunk_it != chunk_index_.end(); ++chunk_it) {
    auto buffer = chunk_it->second;
    if(buffer->getPinCount() < 1) {
      delete buffer;
      chunk_index_.erase(chunk_it);
    }
  }
}

size_t CpuHeteroBufferMgr::getMaxSize() {
  UNREACHABLE();
  return 0;  // Added to avoid "no return statement" compiler warning
}

size_t CpuHeteroBufferMgr::getInUseSize() {
  size_t in_use = 0;
  std::lock_guard<std::mutex> chunk_index_lock(chunk_index_mutex_);
  
  for (auto& index_it : chunk_index_) {
    in_use += index_it.second->reservedSize();
  }

  return in_use;
}

size_t CpuHeteroBufferMgr::getAllocated() {
  // TODO: check it is correct
  return getInUseSize();
}

bool CpuHeteroBufferMgr::isAllocationCapped() {
  // TODO: check it is correct
  return false;
}

void CpuHeteroBufferMgr::checkpoint() {
  std::lock_guard<global_mutex_type> lock(global_mutex_);  // granular lock
  std::lock_guard<chunk_index_mutex_type> chunk_index_lock(chunk_index_mutex_); 

  checkpoint(chunk_index_.begin(), chunk_index_.end());
}

void CpuHeteroBufferMgr::checkpoint(const int db_id, const int tb_id) {
  ChunkKey key_prefix;
  key_prefix.push_back(db_id);
  key_prefix.push_back(tb_id);

  std::lock_guard<global_mutex_type> lock(global_mutex_);  // granular lock
  std::lock_guard<std::mutex> chunk_index_lock(chunk_index_mutex_); 
  auto first = chunk_index_.lower_bound(key_prefix);
  auto last = chunk_index_.upper_bound(key_prefix);

  checkpoint(first, last);
}

void CpuHeteroBufferMgr::checkpoint(chunk_index_iterator first, chunk_index_iterator last) {
  for (; first != last; ++first) {
    const ChunkKey &chunk_key = first->first;
    HeteroBuffer* buffer = first->second;
    // checks that buffer is actual chunk (not just buffer) and is dirty
    if (chunk_key[0] != -1 && buffer->isDirty()) {
      parent_mgr_->putBuffer(chunk_key, buffer);
      buffer->clearDirtyBits();
    }
  }
}

void CpuHeteroBufferMgr::removeTableRelatedDS(const int db_id, const int table_id) {
  UNREACHABLE();
}

/// client is responsible for deleting memory allocated for b->mem_
AbstractBuffer* CpuHeteroBufferMgr::alloc(const size_t num_bytes) {
#if 1
  LOG(FATAL) << "Operation not supported";
  return nullptr;  // satisfy return-type warning
#else
  ChunkKey chunk_key = {-1, getBufferId()};
  return createBuffer(chunk_key, page_size_, num_bytes);
#endif
}

void CpuHeteroBufferMgr::free(AbstractBuffer* buffer) {
#if 1
  LOG(FATAL) << "Operation not supported"; 
#else
  HeteroBuffer* casted_buffer = dynamic_cast<HeteroBuffer*>(buffer);
  if (casted_buffer == 0) {
    LOG(FATAL) << "Wrong buffer type - expects base class pointer to HeteroBuffer type.";
  }
  // TODO
  //deleteBuffer(casted_buffer->seg_it_->chunk_key);
#endif
}

size_t CpuHeteroBufferMgr::getNumChunks() {
  std::lock_guard<chunk_index_mutex_type> index_lock(chunk_index_mutex_);
  return chunk_index_.size();
}

void CpuHeteroBufferMgr::clear() {
  std::lock_guard<chunk_index_mutex_type> index_lock(chunk_index_mutex_);
  for(auto& chunk : chunk_index_) {
    auto buffer = chunk.second;
    delete buffer;
  }
  chunk_index_.clear();
  buffer_epoch_ = 0;
}
} // namespace Buffer_Namespace