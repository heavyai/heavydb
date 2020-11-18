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

#pragma once

#include "DataMgr/AbstractBufferMgr.h"
#include "DataMgr/BufferMgr/CpuBufferMgr/CpuHeteroBuffer.h"

#include "HeteroMem/MemResourceProvider.h"

#include "Shared/ArenaAllocator.h"

#include <memory>
#include <map>
#include <mutex>

namespace CudaMgr_Namespace {
class CudaMgr;
}

namespace Buffer_Namespace {
class CpuHeteroBufferMgr : public AbstractBufferMgr {
public:
  CpuHeteroBufferMgr(const int device_id,
                     const size_t max_buffer_size,
                     const std::string& pmm_path,
                     CudaMgr_Namespace::CudaMgr* cuda_mgr,
                     const size_t page_size = 512,
                     AbstractBufferMgr* parent_mgr = nullptr);

  CpuHeteroBufferMgr(const int device_id,
                     const size_t max_buffer_size,
                     CudaMgr_Namespace::CudaMgr* cuda_mgr,
                     const size_t page_size = 512,
                     AbstractBufferMgr* parent_mgr = nullptr);

  ~CpuHeteroBufferMgr() override;

#ifdef HAVE_DCPMM
  AbstractBuffer* createBuffer(BufferProperty bufProp,
                               const ChunkKey& key,
                               const size_t maxRows,
                               const int sqlTypeSize,
                               const size_t pageSize = 0) override;
#endif /* HAVE_DCPMM */

  /// Creates a chunk with the specified key and page size.
  AbstractBuffer* createBuffer(BufferProperty bufProp,
                               const ChunkKey& key,
                               const size_t page_size = 0,
                               const size_t initial_size = 0) override;

  void deleteBuffer(const ChunkKey& key,
                    const bool purge = true) override;  // purge param only used in FileMgr

  void deleteBuffersWithPrefix(const ChunkKey& keyPrefix,
                               const bool purge = true) override;

  AbstractBuffer* getBuffer(BufferProperty bufProp, const ChunkKey& key, const size_t numBytes = 0) override;

  void fetchBuffer(const ChunkKey& key,
                   AbstractBuffer* destBuffer,
                   const size_t numBytes = 0) override;
  
  AbstractBuffer* putBuffer(const ChunkKey& key,
                            AbstractBuffer* srcBuffer,
                            const size_t numBytes = 0) override;

  void getChunkMetadataVec(ChunkMetadataVector& chunkMetadata) override;

  void getChunkMetadataVecForKeyPrefix(ChunkMetadataVector& chunkMetadataVec,
                                       const ChunkKey& keyPrefix) override;

  bool isBufferOnDevice(const ChunkKey& key) override;

#ifdef HAVE_DCPMM
  bool isBufferInPersistentMemory(const ChunkKey& key) override { return false; }
#endif /* HAVE_DCOMM */

  std::string printSlabs() override { return "Not Implemented"; }
  void clearSlabs() override;
  size_t getMaxSize() override;
  size_t getInUseSize() override;
  size_t getAllocated() override;
  bool isAllocationCapped() override;

  void checkpoint() override;
  void checkpoint(const int db_id, const int tb_id) override;
  void removeTableRelatedDS(const int db_id, const int table_id) override;

  // Buffer API
  AbstractBuffer* alloc(const size_t numBytes = 0) override;
  void free(AbstractBuffer* buffer) override;
  inline MgrType getMgrType() override { return CPU_HETERO_MGR; }
  inline std::string getStringMgrType() override { return ToString(CPU_HETERO_MGR); }
  size_t getNumChunks();

private:
  using HeteroBuffer = CpuHeteroBuffer;
  using global_mutex_type = std::mutex;
  using chunk_index_mutex_type = std::mutex;
  using chunk_index_type= std::map<ChunkKey, HeteroBuffer*>;

  void clear();

  using chunk_index_iterator = typename chunk_index_type::iterator;
  void checkpoint(chunk_index_iterator first, chunk_index_iterator last);
  
  AbstractBufferMgr* parent_mgr_;
  CudaMgr_Namespace::CudaMgr* cuda_mgr_;
  
  MemoryResourceProvider mem_resource_provider_;
  // std::unique_ptr<Arena> sys_allocator_;
  
  const size_t page_size_;

  global_mutex_type global_mutex_;
  chunk_index_mutex_type chunk_index_mutex_;
  chunk_index_type chunk_index_;

  size_t buffer_epoch_;
};
} // namespace Buffer_Namespace