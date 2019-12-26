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
    // this check fails if we create table, drop it and create again
    // CHECK(it_ok.second);
    return it_ok.first->second;
  }

  Data_Namespace::AbstractBuffer* getBuffer(const ChunkKey& key,
                                            const size_t numBytes = 0) override {
    mapd_shared_lock<mapd_shared_mutex> chunk_index_write_lock(chunk_index_mutex_);
    const auto it = chunk_index_.find(key);
    CHECK(it != chunk_index_.end());
    return it->second;
  }

  void fetchBuffer(const ChunkKey& key,
                   Data_Namespace::AbstractBuffer* destBuffer,
                   const size_t numBytes = 0) override {
    CHECK(numBytes);
    destBuffer->reserve(numBytes);
    auto file_buffer = getBuffer(key, numBytes);
    file_buffer->read(destBuffer->getMemoryPtr(), numBytes);
    destBuffer->setSize(numBytes);
    destBuffer->syncEncoder(file_buffer);
  }

  void getChunkMetadataVecForKeyPrefix(
      std::vector<std::pair<ChunkKey, ChunkMetadata>>& chunkMetadataVec,
      const ChunkKey& keyPrefix) override {
    mapd_unique_lock<mapd_shared_mutex> chunk_index_write_lock(
        chunk_index_mutex_);  // is this guarding the right structure?  it look slike we
                              // oly read here for chunk
    auto chunk_it = chunk_index_.lower_bound(keyPrefix);
    if (chunk_it == chunk_index_.end()) {
      CHECK(false);  // throw?
    }

    while (chunk_it != chunk_index_.end() &&
           std::search(chunk_it->first.begin(),
                       chunk_it->first.begin() + keyPrefix.size(),
                       keyPrefix.begin(),
                       keyPrefix.end()) != chunk_it->first.begin() + keyPrefix.size()) {
      const auto& chunk_key = chunk_it->first;
      if (chunk_key.size() == 5) {
        if (chunk_key[4] == 1) {
          const auto& buffer = *chunk_it->second;
          auto type = buffer.sql_type;
          auto size = buffer.size();
          auto subkey = chunk_key;
          subkey[4] = 2;
          auto& index_buf = *(chunk_index_.find(subkey)->second);
          auto bs = index_buf.size() / index_buf.sql_type.get_size();
          ChunkMetadata m{type, size, bs, ChunkStats{}};
          chunkMetadataVec.emplace_back(chunk_key, m);
        }
      } else {
        const auto& buffer = *chunk_it->second;
        ChunkMetadata m{buffer.sql_type};
        buffer.encoder->getMetadata(m);
        chunkMetadataVec.emplace_back(chunk_key, m);
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
  std::map<ChunkKey, ForeignStorageBuffer*> chunk_index_;
  mutable mapd_shared_mutex chunk_index_mutex_;
};

}  // namespace

Data_Namespace::AbstractBufferMgr* ForeignStorageInterface::lookupBufferManager(
    const int db_id,
    const int table_id) {
  static std::map<std::pair<int, int>, Data_Namespace::AbstractBufferMgr*> managers_map_;

  auto key = std::make_pair(db_id, table_id);
  if (managers_map_.count(key)) {
    return managers_map_[key];
  }

  std::lock_guard<std::mutex> persistent_storage_interfaces_lock(
      persistent_storage_interfaces_mutex_);
  const auto it = table_persistent_storage_interface_map_.find(key);
  if (it == table_persistent_storage_interface_map_.end()) {
    return nullptr;
  }
  const auto it_ok = managers_map_.emplace(
      key, new ForeignStorageBufferMgr(db_id, table_id, it->second));
  CHECK(it_ok.second);
  return it_ok.first->second;
}

void ForeignStorageInterface::registerPersistentStorageInterface(
    PersistentForeignStorageInterface* persistent_foreign_storage) {
  std::lock_guard<std::mutex> persistent_storage_interfaces_lock(
      persistent_storage_interfaces_mutex_);
  const auto it_ok = persistent_storage_interfaces_.emplace(
      persistent_foreign_storage->getType(), persistent_foreign_storage);
  CHECK(it_ok.second);
}

std::pair<std::string, std::string> parseStorageType(const std::string& type) {
  size_t sep = type.find_first_of(':'), sep2 = sep != std::string::npos ? sep + 1 : sep;
  auto res = std::make_pair(type.substr(0, sep), type.substr(sep2));
  return res;
}

void ForeignStorageInterface::prepareTable(const int db_id,
                                           TableDescriptor& td,
                                           std::list<ColumnDescriptor>& cols) {
  auto type = parseStorageType(td.storageType);
  std::unique_lock<std::mutex> persistent_storage_interfaces_lock(
      persistent_storage_interfaces_mutex_);
  const auto it = persistent_storage_interfaces_.find(type.first);
  CHECK(it != persistent_storage_interfaces_.end());
  auto p = it->second;
  persistent_storage_interfaces_lock.unlock();
  p->prepareTable(db_id, type.second, td, cols);
}

void ForeignStorageInterface::registerTable(Catalog_Namespace::Catalog* catalog,
                                            const TableDescriptor& td,
                                            const std::list<ColumnDescriptor>& cols) {
  const int table_id = td.tableId;
  auto type = parseStorageType(td.storageType);

  std::unique_lock<std::mutex> persistent_storage_interfaces_lock(
      persistent_storage_interfaces_mutex_);
  const auto it = persistent_storage_interfaces_.find(type.first);
  CHECK(it != persistent_storage_interfaces_.end());

  auto db_id = catalog->getCurrentDB().dbId;
  const auto it_ok = table_persistent_storage_interface_map_.emplace(
      std::make_pair(db_id, table_id), it->second);
  // this check fails if we create table, drop it and create again
  // CHECK(it_ok.second);
  persistent_storage_interfaces_lock.unlock();
  it_ok.first->second->registerTable(catalog,
                                     it_ok.first->first,
                                     type.second,
                                     td,
                                     cols,
                                     lookupBufferManager(db_id, table_id));
}

void ForeignStorageInterface::destroy() {
  persistent_storage_interfaces_.clear();
}

std::unordered_map<std::string, PersistentForeignStorageInterface*>
    ForeignStorageInterface::persistent_storage_interfaces_;
std::map<std::pair<int, int>, PersistentForeignStorageInterface*>
    ForeignStorageInterface::table_persistent_storage_interface_map_;
std::mutex ForeignStorageInterface::persistent_storage_interfaces_mutex_;
