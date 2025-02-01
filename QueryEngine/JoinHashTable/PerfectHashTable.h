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

#include <DataMgr/BufferMgr/CpuBufferMgr/CpuBufferMgr.h>
#include <memory>

#include "DataMgr/Allocators/CpuMgrArenaAllocator.h"
#include "DataMgr/Allocators/CudaAllocator.h"
#include "QueryEngine/JoinHashTable/HashTable.h"

class PerfectHashTableEntryInfo : public HashTableEntryInfo {
 public:
  PerfectHashTableEntryInfo(size_t num_hash_entries,
                            size_t num_keys,
                            size_t rowid_size_in_bytes,
                            HashType layout,
                            bool for_window_framing = false)
      : HashTableEntryInfo(num_hash_entries,
                           num_keys,
                           rowid_size_in_bytes,
                           layout,
                           for_window_framing) {}

  size_t computeTotalNumSlots() const override {
    return layout_ == HashType::OneToOne
               ? num_hash_entries_
               : (2 * num_hash_entries_) + ((1 + for_window_framing_) * num_keys_);
  }

  size_t computeHashTableSize() const override {
    return computeTotalNumSlots() * rowid_size_in_bytes_;
  }
};

class PerfectHashTable : public HashTable {
 public:
  // CPU + GPU constructor
  PerfectHashTable(const ExecutorDeviceType device_type,
                   PerfectHashTableEntryInfo hash_table_entry_info,
                   size_t max_slab_size,
                   BucketizedHashEntryInfo const& hash_entry_info,
                   size_t column_num_elems,
                   Data_Namespace::DataMgr* data_mgr = nullptr,
                   const int device_id = -1)
      : hash_table_entry_info_(hash_table_entry_info)
      , hash_entry_info_(hash_entry_info)
      , column_num_elems_(column_num_elems)
      , data_mgr_(data_mgr)
      , device_id_(device_id) {
    if (device_type == ExecutorDeviceType::CPU) {
      size_t const cpu_hash_table_size =
          sizeof(int32_t) * hash_table_entry_info.computeTotalNumSlots();
      if (cpu_hash_table_size > max_slab_size) {
        std::ostringstream oss;
        oss << "Failed to allocate a join hash table on CPU memory: the size ("
            << cpu_hash_table_size << " bytes) exceeded the system limit ("
            << max_slab_size << " bytes)";
        throw std::runtime_error(oss.str());
      }
      allocator_.reset(new CpuMgrArenaAllocator());
      cpu_hash_table_buff_ =
          reinterpret_cast<int32_t*>(allocator_->allocate(cpu_hash_table_size));
    }
    printInitLog(device_type);
  }

  ~PerfectHashTable() override {
#ifdef HAVE_CUDA
    if (gpu_hash_table_buff_) {
      CHECK(data_mgr_);
      data_mgr_->free(gpu_hash_table_buff_);
    }
#endif
  }

  void allocateGpuMemory(const size_t num_entries) {
    CHECK_GE(device_id_, 0);
    CHECK(!gpu_hash_table_buff_);
    CHECK(data_mgr_);
    auto const buf_size = num_entries * hash_table_entry_info_.getRowIdSizeInBytes();
    gpu_hash_table_buff_ =
        CudaAllocator::allocGpuAbstractBuffer(data_mgr_, buf_size, device_id_);
  }

  size_t getHashTableBufferSize(const ExecutorDeviceType device_type) const override {
    if (device_type == ExecutorDeviceType::CPU) {
      return hash_table_entry_info_.computeHashTableSize();
    } else {
      return gpu_hash_table_buff_ ? gpu_hash_table_buff_->reservedSize() : 0;
    }
  }

  HashType getLayout() const override {
    return hash_table_entry_info_.getHashTableLayout();
  }

  int8_t* getCpuBuffer() override {
    return reinterpret_cast<int8_t*>(cpu_hash_table_buff_);
  }

  int8_t* getGpuBuffer() const override {
    return gpu_hash_table_buff_ ? gpu_hash_table_buff_->getMemoryPtr() : nullptr;
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

  void setHashEntryInfo(BucketizedHashEntryInfo const& hash_entry_info) {
    hash_entry_info_ = hash_entry_info;
  }

  void setColumnNumElems(size_t elem) {
    column_num_elems_ = elem;
  }

  BucketizedHashEntryInfo getHashEntryInfo() const {
    return hash_entry_info_;
  }

  size_t getColumnNumElems() const {
    return column_num_elems_;
  }

  PerfectHashTableEntryInfo getHashTableEntryInfo() const {
    return hash_table_entry_info_;
  }

  void printInitLog(ExecutorDeviceType device_type) {
    std::string device_str = device_type == ExecutorDeviceType::CPU ? "CPU" : "GPU";
    std::string layout_str =
        hash_table_entry_info_.getHashTableLayout() == HashType::OneToOne ? "OneToOne"
                                                                          : "OneToMany";
    std::ostringstream oss;
    oss << "Initialize a " << device_type << " perfect join hash table";
    if (device_type == ExecutorDeviceType::GPU) {
      oss << " on device-" << device_id_;
    }
    oss << ", join type " << layout_str
        << ", # hash entries: " << hash_table_entry_info_.getNumHashEntries()
        << ", # entries stored in the payload buffer: "
        << hash_table_entry_info_.getNumKeys()
        << ", hash table size : " << hash_table_entry_info_.computeHashTableSize()
        << " bytes";
    VLOG(1) << oss.str();
  }

 private:
  Data_Namespace::AbstractBuffer* gpu_hash_table_buff_{nullptr};
  int32_t* cpu_hash_table_buff_{nullptr};
  PerfectHashTableEntryInfo hash_table_entry_info_;
  BucketizedHashEntryInfo hash_entry_info_;
  size_t column_num_elems_;
  Data_Namespace::DataMgr* data_mgr_;
  int device_id_;
  std::unique_ptr<Arena> allocator_{nullptr};
};
