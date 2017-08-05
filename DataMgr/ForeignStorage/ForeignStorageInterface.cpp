/*
 * Copyright 2019 OmniSci, Inc.
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

#include "ForeignStorageInterface.h"
#include "Shared/StringTransform.h"

namespace {

class ForeignStorageBuffer : public Data_Namespace::AbstractBuffer {
 public:
  ForeignStorageBuffer(const ChunkKey& chunk_key,
                       PersistentForeignStorageInterface* persistent_foreign_storage)
      : Data_Namespace::AbstractBuffer(0)
      , chunk_key_(chunk_key)
      , persistent_foreign_storage_(persistent_foreign_storage) {}

  void read(int8_t* const dst,
            const size_t numBytes,
            const size_t offset = 0,
            const Data_Namespace::MemoryLevel dstBufferType = Data_Namespace::CPU_LEVEL,
            const int dstDeviceId = -1) override {
    CHECK_EQ(size_t(0), offset);
    CHECK_EQ(-1, dstDeviceId);
    persistent_foreign_storage_->read(chunk_key_, sql_type, dst, numBytes);
  }

  void append(int8_t* src,
              const size_t numBytes,
              const Data_Namespace::MemoryLevel srcBufferType = Data_Namespace::CPU_LEVEL,
              const int deviceId = -1) override {
    is_dirty_ = true;
    is_appended_ = true;
    buff_.insert(buff_.end(), src, src + numBytes);
    size_ += numBytes;
  }

  Data_Namespace::MemoryLevel getType() const override {
    return Data_Namespace::MemoryLevel::DISK_LEVEL;
  };

  size_t size() const override { return size_; }

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

 private:
  const ChunkKey chunk_key_;
  PersistentForeignStorageInterface* persistent_foreign_storage_;
  std::vector<int8_t> buff_;
};

class ForeignStorageBufferMgr : public Data_Namespace::AbstractBufferMgr {
 public:
  ForeignStorageBufferMgr(const int db_id,
                          const int table_id,
                          PersistentForeignStorageInterface* persistent_foreign_storage)
      : AbstractBufferMgr(0)
      , db_id_(db_id)
      , table_id_(table_id)
      , persistent_foreign_storage_(persistent_foreign_storage) {}

  void checkpoint() override {
    // TODO(alex)
    std::vector<ForeignStorageColumnBuffer> column_buffers;
    for (auto& kv : chunk_index_) {
      const auto buffer = kv.second->moveBuffer();
      column_buffers.emplace_back(
          ForeignStorageColumnBuffer{kv.first, kv.second->sql_type, buffer});
    }
    persistent_foreign_storage_->append(column_buffers);
  }

  Data_Namespace::AbstractBuffer* createBuffer(const ChunkKey& key,
                                               const size_t pageSize = 0,
                                               const size_t initialSize = 0) override {
    mapd_unique_lock<mapd_shared_mutex> chunk_index_write_lock(chunk_index_mutex_);
    const auto it_ok = chunk_index_.emplace(
        key, new ForeignStorageBuffer(key, persistent_foreign_storage_));
    CHECK(it_ok.second);
    return it_ok.first->second.get();
  }

  Data_Namespace::AbstractBuffer* getBuffer(const ChunkKey& key,
                                            const size_t numBytes = 0) override {
    mapd_shared_lock<mapd_shared_mutex> chunk_index_write_lock(chunk_index_mutex_);
    const auto it = chunk_index_.find(key);
    CHECK(it != chunk_index_.end());
    return it->second.get();
  }

  void fetchBuffer(const ChunkKey& key,
                   Data_Namespace::AbstractBuffer* destBuffer,
                   const size_t numBytes = 0) override {
    CHECK(numBytes);
    destBuffer->reserve(numBytes);
    auto file_buffer = getBuffer(key, numBytes);
    file_buffer->read(destBuffer->getMemoryPtr(), numBytes);
    destBuffer->setSize(numBytes);
  }

  void getChunkMetadataVecForKeyPrefix(
      std::vector<std::pair<ChunkKey, ChunkMetadata>>& chunkMetadataVec,
      const ChunkKey& keyPrefix) override {
    mapd_unique_lock<mapd_shared_mutex> chunk_index_write_lock(
        chunk_index_mutex_);  // is this guarding the right structure?  it look slike we
                              // oly read here for chunk
    auto chunk_it = chunk_index_.lower_bound(keyPrefix);
    if (chunk_it == chunk_index_.end()) {
      return;  // throw?
    }
    while (chunk_it != chunk_index_.end() &&
           std::search(chunk_it->first.begin(),
                       chunk_it->first.begin() + keyPrefix.size(),
                       keyPrefix.begin(),
                       keyPrefix.end()) != chunk_it->first.begin() + keyPrefix.size()) {
      if (chunk_it->second->has_encoder) {
        ChunkMetadata chunkMetadata;
        chunk_it->second->encoder->getMetadata(chunkMetadata);
        chunkMetadataVec.push_back(std::make_pair(chunk_it->first, chunkMetadata));
      }
      chunk_it++;
    }
  }

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

  void getChunkMetadataVec(
      std::vector<std::pair<ChunkKey, ChunkMetadata>>& chunkMetadata) override {
    CHECK(false);
  }

  bool isBufferOnDevice(const ChunkKey& key) override {
    CHECK(false);
    return false;
  }

  std::string printSlabs() override {
    CHECK(false);
    return "";
  }

  void clearSlabs() override { CHECK(false); }

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

 private:
  const int db_id_;
  const int table_id_;
  PersistentForeignStorageInterface* persistent_foreign_storage_;
  std::map<ChunkKey, std::unique_ptr<ForeignStorageBuffer>> chunk_index_;
  mutable mapd_shared_mutex chunk_index_mutex_;
};

}  // namespace

Data_Namespace::AbstractBufferMgr* ForeignStorageInterface::lookupBufferManager(
    const int db_id,
    const int table_id) {
  std::lock_guard<std::mutex> persistent_storage_interfaces_lock(
      persistent_storage_interfaces_mutex_);
  const auto it =
      table_persistent_storage_interface_map_.find(std::make_pair(db_id, table_id));
  if (it == table_persistent_storage_interface_map_.end()) {
    return nullptr;
  }
  // TODO(alex): don't leak
  return new ForeignStorageBufferMgr(db_id, table_id, it->second);
}

void ForeignStorageInterface::registerPersistentStorageInterface(
    PersistentForeignStorageInterface* persistent_foreign_storage) {
  std::lock_guard<std::mutex> persistent_storage_interfaces_lock(
      persistent_storage_interfaces_mutex_);
  const auto it_ok = persistent_storage_interfaces_.emplace(
      persistent_foreign_storage->getType(), persistent_foreign_storage);
  CHECK(it_ok.second);
}

void ForeignStorageInterface::registerTable(const int db_id,
                                            const int table_id,
                                            const std::string& type) {
  std::lock_guard<std::mutex> persistent_storage_interfaces_lock(
      persistent_storage_interfaces_mutex_);
  const auto it = persistent_storage_interfaces_.find(type);
  CHECK(it != persistent_storage_interfaces_.end());
  const auto it_ok = table_persistent_storage_interface_map_.emplace(
      std::make_pair(db_id, table_id), it->second.get());
  CHECK(it_ok.second);
}

std::unordered_map<std::string, std::unique_ptr<PersistentForeignStorageInterface>>
    ForeignStorageInterface::persistent_storage_interfaces_;
std::map<std::pair<int, int>, PersistentForeignStorageInterface*>
    ForeignStorageInterface::table_persistent_storage_interface_map_;
std::mutex ForeignStorageInterface::persistent_storage_interfaces_mutex_;
