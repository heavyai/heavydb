/*
 * Copyright 2023 HEAVY.AI, Inc.
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

#include "CpuMgrArenaAllocator.h"

#include "Catalog/SysCatalog.h"

namespace {
Data_Namespace::DataMgr& get_data_mgr_instance() {
  const auto& sys_catalog = Catalog_Namespace::SysCatalog::instance();
  CHECK(sys_catalog.isInitialized());
  return sys_catalog.getDataMgr();
}
}  // namespace

CpuMgrArenaAllocator::CpuMgrArenaAllocator()
    : data_mgr_(get_data_mgr_instance()), size_(0) {}

CpuMgrArenaAllocator::~CpuMgrArenaAllocator() {
  for (auto buffer : allocated_buffers_) {
    data_mgr_.free(buffer);
  }
}

void* CpuMgrArenaAllocator::allocate(size_t num_bytes) {
  if (num_bytes == 0) {
    return nullptr;
  }
  AbstractBuffer* buffer = nullptr;
  try {
    buffer = data_mgr_.alloc(Data_Namespace::CPU_LEVEL, 0, num_bytes);
  } catch (const OutOfMemory& e) {
    LOG(ERROR) << e.what();
    throw OutOfHostMemory(num_bytes);
  }
  CHECK(buffer);
  allocated_buffers_.emplace_back(buffer);

  auto mem_ptr = buffer->getMemoryPtr();
  CHECK(mem_ptr);
  size_ += num_bytes;
  return mem_ptr;
}

void* CpuMgrArenaAllocator::allocateAndZero(const size_t num_bytes) {
  auto ret = allocate(num_bytes);
  std::memset(ret, 0, num_bytes);
  return ret;
}

size_t CpuMgrArenaAllocator::bytesUsed() const {
  return size_;
}

size_t CpuMgrArenaAllocator::totalBytes() const {
  return size_;
}

Arena::MemoryType CpuMgrArenaAllocator::getMemoryType() const {
  return Arena::MemoryType::DRAM;
}
