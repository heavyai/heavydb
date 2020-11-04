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

#include <memory>
#include <vector>

class PerfectHashTable {
 public:
  PerfectHashTable(const Catalog_Namespace::Catalog* catalog) : catalog_(catalog) {}

  size_t size() const {
    CHECK(cpu_hash_table_buff_);
    return cpu_hash_table_buff_->size();
  }

  int32_t* data() const {
    CHECK(cpu_hash_table_buff_);
    return cpu_hash_table_buff_->data();
  }

  constexpr size_t elementSize() const {
    return sizeof(decltype(cpu_hash_table_buff_)::element_type::value_type);
  }

#ifdef HAVE_CUDA
  ~PerfectHashTable() {
    CHECK(catalog_);
    auto& data_mgr = catalog_->getDataMgr();
    if (gpu_hash_table_buff_) {
      data_mgr.free(gpu_hash_table_buff_);
    }
  }

  int8_t* gpuBufferPtr() const {
    CHECK(gpu_hash_table_buff_);
    return gpu_hash_table_buff_->getMemoryPtr();
  }

  size_t gpuReservedSize() const {
    CHECK(gpu_hash_table_buff_);
    return gpu_hash_table_buff_->reservedSize();
  }

  void allocateGpuMemory(const size_t entries, const int device_id) {
    CHECK(catalog_);
    auto& data_mgr = catalog_->getDataMgr();
    CHECK_GE(device_id, 0);
    CHECK(!gpu_hash_table_buff_);
    gpu_hash_table_buff_ = CudaAllocator::allocGpuAbstractBuffer(
        &data_mgr, entries * sizeof(int32_t), device_id);
  }
#endif

  void setHashTableBuffer(std::shared_ptr<std::vector<int32_t>> hash_table_buff) {
    cpu_hash_table_buff_ = hash_table_buff;
  }

 private:
#ifdef HAVE_CUDA
  Data_Namespace::AbstractBuffer* gpu_hash_table_buff_{nullptr};
#endif  // HAVE_CUDA
  const Catalog_Namespace::Catalog* catalog_;
  std::shared_ptr<std::vector<int32_t>> cpu_hash_table_buff_;
  const size_t device_id_{0};
};
