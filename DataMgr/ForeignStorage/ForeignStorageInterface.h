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

#include "../AbstractBufferMgr.h"
#include "Catalog/Catalog.h"

#include <atomic>
#include <unordered_map>

struct ForeignStorageColumnBuffer {
  const ChunkKey chunk_key;
  const SQLTypeInfo sql_type;
  const std::vector<int8_t> buff;
};

class PersistentForeignStorageInterface {
 public:
  virtual ~PersistentForeignStorageInterface() {}

  virtual void append(const std::vector<ForeignStorageColumnBuffer>& column_buffers) = 0;
  virtual void read(const ChunkKey& chunk_key,
                    const SQLTypeInfo& sql_type,
                    int8_t* dest,
                    const size_t num_bytes) = 0;
  virtual int8_t* tryZeroCopy(const ChunkKey& chunk_key,
                              const SQLTypeInfo& sql_type,
                              const size_t num_bytes) {
    return nullptr;
  }
  virtual void prepareTable(const int /*db_id*/,
                            const std::string& type,
                            TableDescriptor& /*td*/,
                            std::list<ColumnDescriptor>& /*cols*/) {}
  virtual void registerTable(Catalog_Namespace::Catalog* catalog,
                             std::pair<int, int> table_key,
                             const std::string& type,
                             const TableDescriptor& td,
                             const std::list<ColumnDescriptor>& cols,
                             Data_Namespace::AbstractBufferMgr* mgr) = 0;
  virtual std::string getType() const = 0;
};

class ForeignStorageBuffer : public Data_Namespace::AbstractBuffer {
 public:
  ForeignStorageBuffer(const ChunkKey& chunk_key,
                       PersistentForeignStorageInterface* persistent_foreign_storage);

  void read(int8_t* const dst,
            const size_t numBytes,
            const size_t offset = 0,
            const Data_Namespace::MemoryLevel dstBufferType = Data_Namespace::CPU_LEVEL,
            const int dstDeviceId = -1) override;

  void append(int8_t* src,
              const size_t numBytes,
              const Data_Namespace::MemoryLevel srcBufferType = Data_Namespace::CPU_LEVEL,
              const int deviceId = -1) override;

  Data_Namespace::MemoryLevel getType() const override {
    return Data_Namespace::MemoryLevel::DISK_LEVEL;
  };

  std::vector<int8_t> moveBuffer() { return std::move(buff_); }

  void write(int8_t* src,
             const size_t numBytes,
             const size_t offset = 0,
             const Data_Namespace::MemoryLevel srcBufferType = Data_Namespace::CPU_LEVEL,
             const int srcDeviceId = -1) override {
    CHECK(false);
  }

  void reserve(size_t numBytes) override { CHECK(false); }

  int8_t* getMemoryPtr() override {
    CHECK(false);
    return nullptr;
  }

  size_t pageCount() const override {
    CHECK(false);
    return 0;
  }

  size_t pageSize() const override {
    CHECK(false);
    return 0;
  }

  size_t reservedSize() const override {
    CHECK(false);
    return 0;
  }

  int8_t* tryZeroCopy(const size_t numBytes);

 private:
  const ChunkKey chunk_key_;
  PersistentForeignStorageInterface* persistent_foreign_storage_;
  std::vector<int8_t> buff_;
};

class ForeignStorageBufferMgr : public Data_Namespace::AbstractBufferMgr {
 public:
  ForeignStorageBufferMgr(const int db_id,
                          const int table_id,
                          PersistentForeignStorageInterface* persistent_foreign_storage);

  void checkpoint() override;

  Data_Namespace::AbstractBuffer* createBuffer(const ChunkKey& key,
                                               const size_t pageSize = 0,
                                               const size_t initialSize = 0) override;

  Data_Namespace::AbstractBuffer* getBuffer(const ChunkKey& key,
                                            const size_t numBytes = 0) override;

  void fetchBuffer(const ChunkKey& key,
                   Data_Namespace::AbstractBuffer* destBuffer,
                   const size_t numBytes = 0) override;

  void getChunkMetadataVecForKeyPrefix(ChunkMetadataVector& chunkMetadataVec,
                                       const ChunkKey& keyPrefix) override;
  std::string getStringMgrType() override { return ToString(FILE_MGR); }

  size_t getNumChunks() override {
    mapd_shared_lock<mapd_shared_mutex> chunk_index_write_lock(chunk_index_mutex_);
    return chunk_index_.size();
  }

  void deleteBuffer(const ChunkKey& key, const bool purge = true) override {
    CHECK(false);
  }

  void deleteBuffersWithPrefix(const ChunkKey& keyPrefix,
                               const bool purge = true) override {
    CHECK(false);
  }

  Data_Namespace::AbstractBuffer* putBuffer(const ChunkKey& key,
                                            Data_Namespace::AbstractBuffer* srcBuffer,
                                            const size_t numBytes = 0) override {
    CHECK(false);
    return nullptr;
  }

  bool isBufferOnDevice(const ChunkKey& key) override {
    CHECK(false);
    return false;
  }

  std::string printSlabs() override {
    CHECK(false);
    return "";
  }

  size_t getMaxSize() override {
    CHECK(false);
    return 0;
  }

  size_t getInUseSize() override {
    CHECK(false);
    return 0;
  }

  size_t getAllocated() override {
    CHECK(false);
    return 0;
  }

  bool isAllocationCapped() override {
    CHECK(false);
    return false;
  }

  void checkpoint(const int db_id, const int tb_id) override { CHECK(false); }

  // Buffer API
  Data_Namespace::AbstractBuffer* alloc(const size_t numBytes = 0) override {
    CHECK(false);
    return nullptr;
  }

  void free(Data_Namespace::AbstractBuffer* buffer) override { CHECK(false); }

  MgrType getMgrType() override {
    CHECK(false);
    return FILE_MGR;
  }

  void removeTableRelatedDS(const int db_id, const int table_id) override {
    UNREACHABLE();
  }

  const DictDescriptor* getDictMetadata(int db_id,
                                        int dict_id,
                                        bool load_dict = true) override {
    UNREACHABLE();
  }

  Fragmenter_Namespace::TableInfo getTableInfo(int db_id, int table_id) const override {
    UNREACHABLE();
  }

 private:
  PersistentForeignStorageInterface* persistent_foreign_storage_;
  std::map<ChunkKey, std::unique_ptr<ForeignStorageBuffer>> chunk_index_;
  mutable mapd_shared_mutex chunk_index_mutex_;
};

class ForeignStorageInterface {
 public:
  ForeignStorageInterface() {}
  ~ForeignStorageInterface() {}

  ForeignStorageInterface(const ForeignStorageInterface& other) = delete;
  ForeignStorageInterface(ForeignStorageInterface&& other) = delete;

  ForeignStorageInterface& operator=(const ForeignStorageInterface& other) = delete;
  ForeignStorageInterface& operator=(ForeignStorageInterface&& other) = delete;

  Data_Namespace::AbstractBufferMgr* lookupBufferManager(const int db_id,
                                                         const int table_id);

  void registerPersistentStorageInterface(
      std::unique_ptr<PersistentForeignStorageInterface> persistent_foreign_storage);

  //! prepare table options and modify columns
  void prepareTable(const int db_id,
                    TableDescriptor& td,
                    std::list<ColumnDescriptor>& cols);
  //! ids are created
  void registerTable(Catalog_Namespace::Catalog* catalog,
                     const TableDescriptor& td,
                     const std::list<ColumnDescriptor>& cols);

 private:
  std::unordered_map<std::string, std::unique_ptr<PersistentForeignStorageInterface>>
      persistent_storage_interfaces_;
  std::map<std::pair<int, int>, PersistentForeignStorageInterface*>
      table_persistent_storage_interface_map_;
  std::map<std::pair<int, int>, std::unique_ptr<ForeignStorageBufferMgr>> managers_map_;
  std::mutex persistent_storage_interfaces_mutex_;
};
