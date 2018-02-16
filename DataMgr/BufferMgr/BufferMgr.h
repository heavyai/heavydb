/*
 * Copyright 2017 MapD Technologies, Inc.
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
 * @author	Steven Stewart <steve@map-d.com>
 * @author	Todd Mostak <todd@map-d.com>
 *
 * This file includes the class specification for the buffer manager (BufferMgr), and related
 * data structures and types.
 */
#ifndef DATAMGR_MEMORY_BUFFER_BUFFERMGR_H
#define DATAMGR_MEMORY_BUFFER_BUFFERMGR_H

#include <iostream>
#include <map>
#include <list>
#include "../Shared/types.h"
#include "../AbstractBuffer.h"
#include "../AbstractBufferMgr.h"
#include "BufferSeg.h"
#include <mutex>

class OutOfMemory : public std::runtime_error {
 public:
  OutOfMemory() : std::runtime_error("OutOfMemory") { OOM_TRACE_DUMP; }

  OutOfMemory(const std::string& err) : std::runtime_error(err) { OOM_TRACE_DUMP; }
};

class FailedToCreateFirstSlab : public OutOfMemory {
 public:
  FailedToCreateFirstSlab() : OutOfMemory("FailedToCreateFirstSlab") {}
};

class FailedToCreateSlab : public OutOfMemory {
 public:
  FailedToCreateSlab() : OutOfMemory("FailedToCreateSlab") {}
};

class SlabTooBig : public OutOfMemory {
 public:
  SlabTooBig() : OutOfMemory("SlabTooBig") {}
};

using namespace Data_Namespace;

namespace Buffer_Namespace {

/**
 * @class   BufferMgr
 * @brief
 *
 * Note(s): Forbid Copying Idiom 4.1
 */

class BufferMgr : public AbstractBufferMgr {  // implements

 public:
  /// Constructs a BufferMgr object that allocates memSize bytes.
  //@todo change this to size_t
  // explicit BufferMgr(const size_t bufferSize, const size_t pageSize);
  BufferMgr(const int deviceId,
            const size_t maxBufferSize,
            const size_t maxSlabSize = 2147483648,
            const size_t pageSize = 512,
            AbstractBufferMgr* parentMgr = 0);

  /// Destructor
  virtual ~BufferMgr();

  void clear();

  std::string printSlab(size_t slabNum);
  std::string printSlabs();
  void clearSlabs();
  std::string printMap();
  void printSegs();
  std::string printSeg(BufferList::iterator& segIt);
  std::string keyToString(const ChunkKey& key);
  size_t getInUseSize();
  size_t getMaxSize();
  size_t getAllocated();
  size_t getMaxBufferSize();
  size_t getMaxSlabSize();
  size_t getPageSize();
  bool isAllocationCapped();
  const std::vector<BufferList>& getSlabSegments();

  /// Creates a chunk with the specified key and page size.
  virtual AbstractBuffer* createBuffer(const ChunkKey& key, const size_t pageSize = 0, const size_t initialSize = 0);

  /// Deletes the chunk with the specified key
  virtual void deleteBuffer(const ChunkKey& key, const bool purge = true);
  virtual void deleteBuffersWithPrefix(const ChunkKey& keyPrefix, const bool purge = true);

  /// Returns the a pointer to the chunk with the specified key.
  virtual AbstractBuffer* getBuffer(const ChunkKey& key, const size_t numBytes = 0);

  /**
   * @brief Puts the contents of d into the Buffer with ChunkKey key.
   * @param key - Unique identifier for a Chunk.
   * @param d - An object representing the source data for the Chunk.
   * @return AbstractBuffer*
   */
  virtual bool isBufferOnDevice(const ChunkKey& key);
  virtual void fetchBuffer(const ChunkKey& key, AbstractBuffer* destBuffer, const size_t numBytes = 0);
  virtual AbstractBuffer* putBuffer(const ChunkKey& key, AbstractBuffer* d, const size_t numBytes = 0);
  void checkpoint();
  void checkpoint(const int db_id, const int tb_id);

  // Buffer API
  virtual AbstractBuffer* alloc(const size_t numBytes = 0);
  virtual void free(AbstractBuffer* buffer);
  // virtual AbstractBuffer* putBuffer(AbstractBuffer *d);

  /// Returns the total number of bytes allocated.
  size_t size();
  size_t getNumChunks();

  BufferList::iterator reserveBuffer(BufferList::iterator& segIt, const size_t numBytes);
  virtual void getChunkMetadataVec(std::vector<std::pair<ChunkKey, ChunkMetadata>>& chunkMetadataVec);
  virtual void getChunkMetadataVecForKeyPrefix(std::vector<std::pair<ChunkKey, ChunkMetadata>>& chunkMetadataVec,
                                               const ChunkKey& keyPrefix);

 protected:
  std::vector<int8_t*> slabs_;  /// vector of beginning memory addresses for each allocation of the buffer pool
  std::vector<BufferList> slabSegments_;
  size_t pageSize_;

 private:
  BufferMgr(const BufferMgr&);             // private copy constructor
  BufferMgr& operator=(const BufferMgr&);  // private assignment
  void removeSegment(BufferList::iterator& segIt);
  BufferList::iterator findFreeBufferInSlab(const size_t slabNum, const size_t numPagesRequested);
  int getBufferId();
  virtual void addSlab(const size_t slabSize) = 0;
  virtual void freeAllMem() = 0;
  virtual void allocateBuffer(BufferList::iterator segIt, const size_t pageSize, const size_t numBytes) = 0;
  std::mutex chunkIndexMutex_;
  std::mutex sizedSegsMutex_;
  std::mutex unsizedSegsMutex_;
  std::mutex bufferIdMutex_;
  std::mutex globalMutex_;

  std::map<ChunkKey, BufferList::iterator> chunkIndex_;
  size_t maxBufferSize_;  /// max number of bytes allocated for the buffer pool
  size_t maxNumPages_;
  size_t numPagesAllocated_;
  size_t maxNumPagesPerSlab_;
  size_t currentMaxSlabPageSize_;
  size_t
      maxSlabSize_;  /// size of the individual memory allocations that compose the buffer pool (up to maxBufferSize_)
  bool allocationsCapped_;
  AbstractBufferMgr* parentMgr_;
  int maxBufferId_;
  unsigned int bufferEpoch_;
  // File_Namespace::FileMgr *fileMgr_;

  /// Maps sizes of free memory areas to host buffer pool memory addresses
  //@todo change this to multimap
  // std::multimap<size_t, int8_t *> freeMem_;
  BufferList unsizedSegs_;
  // std::map<size_t, int8_t *> freeMem_;

  BufferList::iterator evict(BufferList::iterator& evictStart, const size_t numPagesRequested, const int slabNum);
  BufferList::iterator findFreeBuffer(size_t numBytes);

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
};

}  // Buffer_Namespace

#endif  // DATAMGR_MEMORY_BUFFER_BUFFERMGR_H
