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

#include <memory>
#include <vector>

#include "DataMgr/Allocators/CudaAllocator.h"
#include "QueryEngine/JoinHashTable/HashTable.h"

class PerfectHashTable : public HashTable {
 public:
  // CPU + GPU constructor
  PerfectHashTable(Data_Namespace::DataMgr* data_mgr,
                   const HashType layout,
                   const ExecutorDeviceType device_type,
                   const size_t entry_count,
                   const size_t emitted_keys_count,
                   const bool for_window_framing = false)
      : data_mgr_(data_mgr)
      , layout_(layout)
      , entry_count_(entry_count)
      , emitted_keys_count_(emitted_keys_count) {
    std::string device_str = "GPU";
    if (device_type == ExecutorDeviceType::CPU) {
      device_str = "CPU";
      cpu_hash_table_buff_size_ =
          layout_ == HashType::OneToOne
              ? entry_count_
              : 2 * entry_count_ + ((1 + for_window_framing) * emitted_keys_count_);
      cpu_hash_table_buff_.reset(new int32_t[cpu_hash_table_buff_size_]);
    }
    VLOG(1) << "Initialize a " << device_str << " perfect hash table for join type "
            << ::toString(layout) << " # hash entries: " << entry_count
            << ", # entries stored in the payload buffer: " << emitted_keys_count
            << ", hash table size : " << cpu_hash_table_buff_size_ * 4 << " Bytes";
  }

  ~PerfectHashTable() override {
    if (gpu_hash_table_buff_) {
      CHECK(data_mgr_);
      data_mgr_->free(gpu_hash_table_buff_);
    }
  }

  size_t gpuReservedSize() const {
    CHECK(gpu_hash_table_buff_);
    return gpu_hash_table_buff_->reservedSize();
  }

  void allocateGpuMemory(const size_t entries, const int device_id) {
    CHECK_GE(device_id, 0);
    CHECK(!gpu_hash_table_buff_);
    gpu_hash_table_buff_ = CudaAllocator::allocGpuAbstractBuffer(
        data_mgr_, entries * sizeof(int32_t), device_id);
  }

  size_t getHashTableBufferSize(const ExecutorDeviceType device_type) const override {
    if (device_type == ExecutorDeviceType::CPU) {
      return cpu_hash_table_buff_size_ *
             sizeof(decltype(cpu_hash_table_buff_)::element_type);
    } else {
      return gpu_hash_table_buff_ ? gpu_hash_table_buff_->reservedSize() : 0;
    }
  }

  HashType getLayout() const override { return layout_; }

  int8_t* getCpuBuffer() override {
    return reinterpret_cast<int8_t*>(cpu_hash_table_buff_.get());
  }

  int8_t* getGpuBuffer() const override {
    return gpu_hash_table_buff_ ? gpu_hash_table_buff_->getMemoryPtr() : nullptr;
  }

  size_t getEntryCount() const override { return entry_count_; }

  size_t getEmittedKeysCount() const override { return emitted_keys_count_; }

  void setHashEntryInfo(BucketizedHashEntryInfo& hash_entry_info) {
    hash_entry_info_ = hash_entry_info;
  }

  void setColumnNumElems(size_t elem) { column_num_elems_ = elem; }

  BucketizedHashEntryInfo getHashEntryInfo() const { return hash_entry_info_; }

  size_t getColumnNumElems() const { return column_num_elems_; }

 private:
  Data_Namespace::AbstractBuffer* gpu_hash_table_buff_{nullptr};
  Data_Namespace::DataMgr* data_mgr_;
  std::unique_ptr<int32_t[]> cpu_hash_table_buff_;
  size_t cpu_hash_table_buff_size_;

  HashType layout_;
  size_t entry_count_;         // number of keys in the hash table
  size_t emitted_keys_count_;  // number of keys emitted across all rows

  BucketizedHashEntryInfo hash_entry_info_;
  size_t column_num_elems_;
};
