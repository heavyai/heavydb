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

#pragma once

#include "DataMgr/AbstractBuffer.h"
#include "DataMgr/Allocators/CudaAllocator.h"
#include "QueryEngine/JoinHashTable/HashJoin.h"
#include "QueryEngine/JoinHashTable/HashTable.h"

class BaselineHashTableEntryInfo : public HashTableEntryInfo {
 public:
  BaselineHashTableEntryInfo(size_t num_hash_entries,
                             size_t num_keys,
                             size_t rowid_size_in_bytes,
                             size_t num_join_keys,
                             size_t join_key_size_in_byte,
                             HashType layout,
                             bool for_window_framing = false)
      : HashTableEntryInfo(num_hash_entries,
                           num_keys,
                           rowid_size_in_bytes,
                           layout,
                           for_window_framing)
      , num_join_keys_(num_join_keys)
      , join_key_size_in_byte_(join_key_size_in_byte) {}

  size_t computeTotalNumSlots() const override { return num_hash_entries_; }

  size_t computeKeySize() const {
    auto const entry_cnt = (num_join_keys_ + (layout_ == HashType::OneToOne ? 1 : 0));
    return entry_cnt * join_key_size_in_byte_;
  }

  size_t computeNumAdditionalSlotsForOneToManyJoin() const {
    return HashJoin::layoutRequiresAdditionalBuffers(layout_)
               ? 2 * num_hash_entries_ + (num_keys_ * (1 + (for_window_framing_)))
               : 0;
  }

  size_t computeHashTableSize() const override {
    return computeTotalNumSlots() * computeKeySize() +
           computeNumAdditionalSlotsForOneToManyJoin() * rowid_size_in_bytes_;
  }

  size_t getNumJoinKeys() const { return num_join_keys_; }

  size_t getJoinKeysSize() const { return join_key_size_in_byte_; }

 private:
  size_t const num_join_keys_;
  size_t const join_key_size_in_byte_;
};

class BaselineHashTable : public HashTable {
 public:
  // CPU + GPU constructor
  BaselineHashTable(MemoryLevel memory_level,
                    BaselineHashTableEntryInfo hash_table_entry_info,
                    size_t max_slab_size,
                    Data_Namespace::DataMgr* data_mgr = nullptr,
                    const int device_id = -1)
      : memory_level_(memory_level)
      , hash_table_entry_info_(hash_table_entry_info)
      , data_mgr_(data_mgr)
      , device_id_(device_id) {
    auto const hash_table_size = hash_table_entry_info.computeHashTableSize();
    if (memory_level_ == Data_Namespace::GPU_LEVEL) {
#ifdef HAVE_CUDA
      CHECK(data_mgr_);
      gpu_hash_table_buff_ =
          CudaAllocator::allocGpuAbstractBuffer(data_mgr_, hash_table_size, device_id_);
#else
      UNREACHABLE();
#endif
    } else {
      CHECK(!data_mgr_);  // we do not need `data_mgr` for CPU hash table
      if (hash_table_size > max_slab_size) {
        std::ostringstream oss;
        oss << "Failed to allocate a join hash table on CPU memory: the size ("
            << hash_table_size << " bytes) exceeded the system limit (" << max_slab_size
            << " bytes)";
        throw std::runtime_error(oss.str());
      }
      allocator_.reset(new CpuMgrArenaAllocator());
      cpu_hash_table_buff_ =
          reinterpret_cast<int32_t*>(allocator_->allocate(hash_table_size));
    }
    printInitLog();
  }

  ~BaselineHashTable() override {
#ifdef HAVE_CUDA
    if (gpu_hash_table_buff_) {
      CHECK(data_mgr_);
      data_mgr_->free(gpu_hash_table_buff_);
    }
#endif
  }

  int8_t* getGpuBuffer() const override {
    return gpu_hash_table_buff_ ? gpu_hash_table_buff_->getMemoryPtr() : nullptr;
  }

  size_t getHashTableBufferSize(const ExecutorDeviceType device_type) const override {
    return hash_table_entry_info_.computeHashTableSize();
  }

  int8_t* getCpuBuffer() override {
    return reinterpret_cast<int8_t*>(cpu_hash_table_buff_);
  }

  HashType getLayout() const override {
    return hash_table_entry_info_.getHashTableLayout();
  }
  size_t getEntryCount() const override {
    return hash_table_entry_info_.getNumHashEntries();
  }
  size_t getEmittedKeysCount() const override {
    return hash_table_entry_info_.getNumKeys();
  }
  size_t getRowIdSize() const override {
    return hash_table_entry_info_.getRowIdSizeInBytes();
  }
  BaselineHashTableEntryInfo getHashTableEntryInfo() const {
    return hash_table_entry_info_;
  }

  void printInitLog() {
    std::string device_str = memory_level_ == MemoryLevel::GPU_LEVEL ? "GPU" : "CPU";
    std::string layout_str =
        hash_table_entry_info_.getHashTableLayout() == HashType::OneToOne ? "OneToOne"
                                                                          : "OneToMany";
    std::ostringstream oss;
    oss << "Initialize a " << device_str << " baseline hash table";
    if (memory_level_ == MemoryLevel::GPU_LEVEL) {
      CHECK_GE(device_id_, 0);
      oss << " on device-" << device_id_;
    }
    oss << " with join type " << layout_str
        << ", hash table size: " << hash_table_entry_info_.computeHashTableSize()
        << " bytes"
        << ", # hash entries: " << hash_table_entry_info_.getNumHashEntries()
        << ", # entries stored in the payload buffer: "
        << hash_table_entry_info_.getNumKeys()
        << ", rowid size: " << hash_table_entry_info_.getRowIdSizeInBytes() << " bytes";
    VLOG(1) << oss.str();
  }

 private:
  int32_t* cpu_hash_table_buff_{nullptr};
  Data_Namespace::AbstractBuffer* gpu_hash_table_buff_{nullptr};

  MemoryLevel memory_level_;
  BaselineHashTableEntryInfo hash_table_entry_info_;
  Data_Namespace::DataMgr* data_mgr_;
  const int device_id_;
  std::unique_ptr<Arena> allocator_{nullptr};
};
