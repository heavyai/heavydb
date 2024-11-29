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

/**
 * @file	BufferMgr.h
 * @brief
 *
 * This file includes the class specification for the buffer manager (BufferMgr), and
 * related data structures and types.
 */

#pragma once

#define BOOST_STACKTRACE_GNU_SOURCE_NOT_REQUIRED 1

#include <iostream>
#include <list>
#include <map>
#include <mutex>
#include <optional>
#include <shared_mutex>

#include "DataMgr/AbstractBuffer.h"
#include "DataMgr/AbstractBufferMgr.h"
#include "DataMgr/BufferMgr/BufferSeg.h"
#include "Shared/boost_stacktrace.hpp"
#include "Shared/types.h"

class OutOfMemory : public std::runtime_error {
 public:
  OutOfMemory(size_t num_bytes)
      : std::runtime_error(parse_error_str("OutOfMemory", num_bytes)) {
    VLOG(1) << "Failed to allocate " << num_bytes << " bytes";
    VLOG(1) << boost::stacktrace::stacktrace();
  };

  OutOfMemory(const std::string& err) : std::runtime_error(parse_error_str(err, 0)) {
    VLOG(1) << "Failed with OutOfMemory, condition " << err;
    VLOG(1) << boost::stacktrace::stacktrace();
  };

  OutOfMemory(const std::string& err, size_t num_bytes)
      : std::runtime_error(parse_error_str(err, num_bytes)) {
    VLOG(1) << "Failed to allocate " << num_bytes << " bytes with condition " << err;
    VLOG(1) << boost::stacktrace::stacktrace();
  };

 private:
  std::string parse_error_str(const std::string& err, const size_t num_bytes = 0) {
    if (num_bytes) {
      return err + ": Failed to allocate " + std::to_string(num_bytes) + " bytes";
    } else {
      return "Failed to allocate memory with condition " + err;
    }
  }
};

class FailedToCreateFirstSlab : public OutOfMemory {
 public:
  FailedToCreateFirstSlab(size_t num_bytes)
      : OutOfMemory("FailedToCreateFirstSlab", num_bytes) {}
};

class FailedToCreateSlab : public OutOfMemory {
 public:
  FailedToCreateSlab(size_t num_bytes) : OutOfMemory("FailedToCreateSlab", num_bytes) {}
};

class TooBigForSlab : public OutOfMemory {
 public:
  TooBigForSlab(size_t num_bytes) : OutOfMemory("TooBigForSlab", num_bytes) {}
};

using namespace Data_Namespace;

namespace Buffer_Namespace {

struct MemoryData {
  size_t slab_num;
  int32_t start_page;
  size_t num_pages;
  uint32_t touch;
  ChunkKey chunk_key;
  MemStatus mem_status;
};

struct MemoryInfo {
  size_t page_size;
  size_t max_num_pages;
  size_t num_page_allocated;
  bool is_allocation_capped;
  std::vector<MemoryData> node_memory_data;
};

/**
 * @class   BufferMgr
 * @brief
 *
 * Note(s): Forbid Copying Idiom 4.1
 */

class BufferMgr : public AbstractBufferMgr {  // implements

 public:
  /// Constructs a BufferMgr object that allocates memSize bytes.
  BufferMgr(const int device_id,
            const size_t max_buffer_size,
            const size_t min_slab_size,
            const size_t max_slab_size,
            const size_t default_slab_size,
            const size_t page_size,
            AbstractBufferMgr* parent_mgr = 0);

  /// Destructor
  ~BufferMgr() override;
  std::string printSlabs() override;

  void clearSlabs();
  std::string printMap();
  void printSegs();
  size_t getInUseSize() const override;
  size_t getMaxSize() const override;
  size_t getAllocated() const override;
  size_t getMaxBufferSize() const;
  size_t getMaxSlabSize() const;
  size_t getPageSize() const;
  bool isAllocationCapped() const override;
  const std::vector<BufferList>& getSlabSegments();

  /// Creates a chunk with the specified key and page size.
  AbstractBuffer* createBuffer(const ChunkKey& key,
                               const size_t page_size = 0,
                               const size_t initial_size = 0) override;

  /// Deletes the chunk with the specified key
  void deleteBuffer(const ChunkKey& key, const bool purge = true) override;

  void deleteBuffersWithPrefix(const ChunkKey& key_prefix,
                               const bool purge = true) override;

  /// Returns the a pointer to the chunk with the specified key.
  AbstractBuffer* getBuffer(const ChunkKey& key, const size_t num_bytes = 0) override;

  /**
   * @brief Puts the contents of d into the Buffer with ChunkKey key.
   * @param key - Unique identifier for a Chunk.
   * @param d - An object representing the source data for the Chunk.
   * @return AbstractBuffer*
   */
  bool isBufferOnDevice(const ChunkKey& key) override;
  void fetchBuffer(const ChunkKey& key,
                   AbstractBuffer* dest_buffer,
                   const size_t num_bytes = 0) override;
  AbstractBuffer* putBuffer(const ChunkKey& key,
                            AbstractBuffer* d,
                            const size_t num_bytes = 0) override;
  void checkpoint() override;
  void checkpoint(const int db_id, const int tb_id) override;
  void removeTableRelatedDS(const int db_id, const int table_id) override;

  // Buffer API
  AbstractBuffer* alloc(const size_t num_bytes = 0) override;
  void free(AbstractBuffer* buffer) override;

  /// Returns the total number of bytes allocated.
  size_t size() const;
  size_t getNumChunks() override;

  BufferList::iterator reserveBuffer(BufferList::iterator& seg_it,
                                     const size_t num_bytes);
  void getChunkMetadataVecForKeyPrefix(ChunkMetadataVector& chunk_metadata_vec,
                                       const ChunkKey& key_prefix) override;

  MemoryInfo getMemoryInfo() const;

 protected:
  virtual Buffer* createBuffer(BufferList::iterator seg_it, size_t page_size) = 0;

  const size_t
      max_buffer_pool_size_;    /// max number of bytes allocated for the buffer pool
  const size_t min_slab_size_;  /// minimum size of the individual memory allocations that
                                /// compose the buffer pool (up to maxBufferSize_)
  const size_t max_slab_size_;  /// max size of the individual memory allocations that
                                /// compose the buffer pool (up to maxBufferSize_)
  const size_t
      default_slab_size_;  /// default size of the individual memory allocations that
                           /// compose the buffer pool (up to maxBufferSize_)
  const size_t page_size_;

  std::vector<int8_t*> slabs_;  /// vector of beginning memory addresses for each
                                /// allocation of the buffer pool
  std::vector<BufferList> slab_segments_;

 private:
  BufferMgr(const BufferMgr&);             // private copy constructor
  BufferMgr& operator=(const BufferMgr&);  // private assignment
  void removeSegment(BufferList::iterator& seg_it);
  BufferList::iterator findFreeBufferInSlab(const size_t slab_num,
                                            const size_t num_pages_requested);
  int getBufferId();
  virtual void addSlab(const size_t slab_size) = 0;
  virtual void freeAllMem() = 0;

  void allocateBuffer(BufferList::iterator seg_it, size_t page_size, size_t num_bytes);
  void clear();
  void reinit();
  AbstractBuffer* createBufferUnlocked(const ChunkKey& key,
                                       const size_t page_size = 0,
                                       const size_t initial_size = 0);
  void deleteBufferUnlocked(const ChunkKey& key, const bool purge = true);
  uint32_t incrementEpoch();
  void clearEpoch();
  BufferList::iterator addBufferPlaceholder(const ChunkKey& chunk_key);
  BufferList::iterator addUnsizedSegment(const ChunkKey& chunk_key);
  void eraseUnsizedSegment(const BufferList::iterator& seg_it);
  void clearUnsizedSegments();
  std::mutex& getChunkMutex(const ChunkKey& chunk_key);
  std::optional<BufferList::iterator> getChunkSegment(const ChunkKey& chunk_key) const;
  void eraseChunkSegment(const ChunkKey& chunk_key);
  void clearChunks();
  void clearSlabContainers();
  void checkpoint(const std::vector<ChunkKey>& chunk_keys);
  std::string printSlab(size_t slab_num);
  std::string printSeg(const BufferList::iterator& seg_it);

  mutable std::shared_mutex chunk_index_mutex_;
  mutable std::shared_mutex slab_mutex_;
  mutable std::shared_mutex clear_slabs_global_mutex_;
  mutable std::mutex unsized_segs_mutex_;
  mutable std::mutex buffer_id_mutex_;
  mutable std::mutex buffer_epoch_mutex_;

  std::map<ChunkKey, BufferList::iterator> chunk_index_;
  std::map<ChunkKey, std::mutex> chunk_mutex_map_;

  const size_t max_buffer_pool_num_pages_;  // max number of pages for buffer pool
  const size_t min_num_pages_per_slab_;
  const size_t max_num_pages_per_slab_;

  size_t num_pages_allocated_;
  size_t default_num_pages_per_slab_;
  size_t current_max_num_pages_per_slab_;
  bool allocations_capped_;

  AbstractBufferMgr* parent_mgr_;

  int max_buffer_id_;
  uint32_t buffer_epoch_;

  BufferList unsized_segs_;

  BufferList::iterator evict(BufferList::iterator& evict_start,
                             const size_t num_pages_requested,
                             const int slab_num);
  /**
   * @brief Gets a buffer of required size and returns an iterator to it
   *
   * If possible, this function will just select a free buffer of
   * sufficient size and use that. If not, it will evict as many
   * non-pinned but used buffers as needed to have enough space for the
   * buffer
   *
   * @return An iterator to the reserved buffer. We guarantee that this
   * buffer won't be evicted by PINNING it - caller should change this to
   * USED if applicable
   *
   */
  BufferList::iterator findFreeBuffer(size_t num_bytes);
};
}  // namespace Buffer_Namespace
