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

#include "Catalog/Catalog.h"
#include "DataMgr/AbstractBuffer.h"
#include "DataMgr/Allocators/CudaAllocator.h"
#include "QueryEngine/CompilationOptions.h"
#include "QueryEngine/JoinHashTable/HashJoin.h"

class BaselineHashTable {
 public:
  // CPU constructor
  BaselineHashTable(const Catalog_Namespace::Catalog* catalog,
                    HashJoin::HashType layout,
                    const size_t entry_count,
                    const size_t emitted_keys_count,
                    const size_t hash_table_size)
      : device_id_(0)
      , catalog_(catalog)
      , layout_(layout)
      , entry_count_(entry_count)
      , emitted_keys_count_(emitted_keys_count) {
    cpu_hash_table_buff_.resize(hash_table_size);
  }

  // GPU constructor
  BaselineHashTable(const Catalog_Namespace::Catalog* catalog,
                    HashJoin::HashType layout,
                    const size_t entry_count,
                    const size_t emitted_keys_count,
                    const size_t hash_table_size,
                    const size_t device_id)
      : device_id_(device_id)
      , catalog_(catalog)
      , layout_(layout)
      , entry_count_(entry_count)
      , emitted_keys_count_(emitted_keys_count) {
#ifdef HAVE_CUDA
    CHECK(catalog_);
    auto& data_mgr = catalog_->getDataMgr();
    gpu_hash_table_buff_ =
        CudaAllocator::allocGpuAbstractBuffer(&data_mgr, hash_table_size, device_id_);
#else
    UNREACHABLE();
#endif
  }

  ~BaselineHashTable() {
#ifdef HAVE_CUDA
    CHECK(catalog_);
    auto& data_mgr = catalog_->getDataMgr();
    if (gpu_hash_table_buff_) {
      data_mgr.free(gpu_hash_table_buff_);
    }
#endif
  }

  Data_Namespace::AbstractBuffer* getGpuBuffer() const { return gpu_hash_table_buff_; }

  size_t getHashTableBufferSize(const ExecutorDeviceType device_type) const {
    if (device_type == ExecutorDeviceType::CPU) {
      return cpu_hash_table_buff_.size() *
             sizeof(decltype(cpu_hash_table_buff_)::value_type);
    } else {
      const auto gpu_buff = getGpuBuffer();
      return gpu_buff ? gpu_buff->reservedSize() : 0;
    }
  }

  int8_t* getCpuBuffer() { return cpu_hash_table_buff_.data(); }
  size_t getCpuBufferSize() { return cpu_hash_table_buff_.size(); }

  auto getLayout() const { return layout_; }
  size_t getEntryCount() const { return entry_count_; }
  size_t getEmittedKeysCount() const { return emitted_keys_count_; }

 private:
  std::vector<int8_t> cpu_hash_table_buff_;
  Data_Namespace::AbstractBuffer* gpu_hash_table_buff_{nullptr};
  const size_t device_id_;

  // TODO: only required for cuda
  const Catalog_Namespace::Catalog* catalog_;

  HashJoin::HashType layout_;
  // size_t key_component_count_;
  // size_t key_component_width_;
  size_t entry_count_;         // number of keys in the hash table
  size_t emitted_keys_count_;  // number of keys emitted across all rows
};
