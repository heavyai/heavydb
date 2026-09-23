/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "CpuMgrArenaAllocator.h"

#include "Catalog/SysCatalog.h"

bool g_disable_cpu_mem_pool_import_buffers{false};

namespace {
Data_Namespace::DataMgr& get_data_mgr_instance() {
  const auto& sys_catalog = Catalog_Namespace::SysCatalog::instance();
  CHECK(sys_catalog.isInitialized());
  return sys_catalog.getDataMgr();
}
}  // namespace

CpuMgrBaseAllocator::CpuMgrBaseAllocator() : data_mgr_(&get_data_mgr_instance()) {}

CpuMgrBaseAllocator::~CpuMgrBaseAllocator() {
  deallocateAll();
}

void* CpuMgrBaseAllocator::allocate(const size_t num_bytes) {
  if (!num_bytes) {
    return nullptr;
  }

  AbstractBuffer* buffer = nullptr;
  try {
    buffer = data_mgr_->alloc(Data_Namespace::CPU_LEVEL, 0, num_bytes);
  } catch (const OutOfMemory& e) {
    LOG(ERROR) << e.what();
    throw OutOfHostMemory(num_bytes);
  }
  CHECK(buffer);
  allocated_buffers_.emplace_back(buffer);

  auto mem_ptr = allocated_buffers_.back()->getMemoryPtr();
  CHECK(mem_ptr);
  if (auto p = static_cast<void*>(mem_ptr)) {
    return p;
  }

  throw std::bad_alloc();
  return {};
}

void CpuMgrBaseAllocator::deallocate(void* p) {
  CHECK(p);  // Should never be called on a nullptr
  try {
    auto it = std::find_if(
        allocated_buffers_.begin(), allocated_buffers_.end(), [p](AbstractBuffer* b) {
          return p == static_cast<void*>(b->getMemoryPtr());
        });
    CHECK(it != allocated_buffers_.end());
    data_mgr_->free(*it);
    allocated_buffers_.erase(it);
  } catch (std::exception& e) {
    throw std::runtime_error("Encountered exception while freeing DataMgr cpu buffer: " +
                             std::string{e.what()});
  }
}

void CpuMgrBaseAllocator::deallocateAll() {
  while (allocated_buffers_.size()) {
    data_mgr_->free(allocated_buffers_.front());
    allocated_buffers_.pop_front();
  }
}

CpuMgrArenaAllocator::CpuMgrArenaAllocator() : size_(0) {}

void* CpuMgrArenaAllocator::allocate(size_t num_bytes) {
  auto mem_ptr = base_allocator_.allocate(num_bytes);
  size_ += num_bytes;
  return mem_ptr;
}

void* CpuMgrArenaAllocator::allocateAndZero(const size_t num_bytes) {
  auto ret = allocate(num_bytes);
  if (!ret) {
    return ret;
  }
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
