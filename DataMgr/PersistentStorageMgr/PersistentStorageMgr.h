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

#pragma once

#include "DataMgr/AbstractBufferMgr.h"

using namespace Data_Namespace;

class PersistentStorageMgr : public AbstractBufferMgr {
 public:
  PersistentStorageMgr(const std::string& data_dir, const size_t num_reader_threads);

  AbstractBuffer* createBuffer(const ChunkKey& chunk_key,
                               const size_t page_size,
                               const size_t initial_size) override;
  AbstractBuffer* createZeroCopyBuffer(const ChunkKey& key,
                                       std::unique_ptr<AbstractDataToken> token) override;
  void deleteBuffer(const ChunkKey& chunk_key, const bool purge) override;
  void deleteBuffersWithPrefix(const ChunkKey& chunk_key_prefix,
                               const bool purge) override;
  AbstractBuffer* getBuffer(const ChunkKey& chunk_key, const size_t num_bytes) override;
  std::unique_ptr<AbstractDataToken> getZeroCopyBufferMemory(const ChunkKey& key,
                                                             size_t numBytes) override;
  void fetchBuffer(const ChunkKey& chunk_key,
                   AbstractBuffer* destination_buffer,
                   const size_t num_bytes) override;
  void getChunkMetadataVecForKeyPrefix(ChunkMetadataVector& chunk_metadata,
                                       const ChunkKey& chunk_key_prefix) override;
  bool isBufferOnDevice(const ChunkKey& chunk_key) override;
  std::string printSlabs() override;
  size_t getMaxSize() override;
  size_t getInUseSize() override;
  size_t getAllocated() override;
  bool isAllocationCapped() override;
  AbstractBuffer* alloc(const size_t num_bytes) override;
  void free(AbstractBuffer* buffer) override;
  MgrType getMgrType() override;
  std::string getStringMgrType() override;
  size_t getNumChunks() override;

  const DictDescriptor* getDictMetadata(int dict_id, bool load_dict = true);

  TableFragmentsInfo getTableMetadata(int db_id, int table_id) const override;

  void registerDataProvider(int schema_id, std::shared_ptr<AbstractBufferMgr>);

  std::shared_ptr<AbstractBufferMgr> getDataProvider(int schema_id) const;

 protected:
  bool isForeignStorage(const ChunkKey& chunk_key) const;
  AbstractBufferMgr* getStorageMgrForTableKey(const ChunkKey& table_key) const;
  AbstractBufferMgr* getStorageMgr(int db_id) const;
  bool hasStorageMgr(int db_id) const;
  bool isChunkPrefixCacheable(const ChunkKey& chunk_prefix) const;
  int recoverDataWrapperIfCachedAndGetHighestFragId(const ChunkKey& table_key);

  std::unordered_map<int, std::shared_ptr<AbstractBufferMgr>> mgr_by_schema_id_;
};
