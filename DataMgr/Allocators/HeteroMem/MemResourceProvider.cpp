// SPDX-License-Identifier: BSD-2-Clause
/* Copyright (C) 2020 Intel Corporation. */

#include "DataMgr/Allocators/HeteroMem/MemResourceProvider.h"

#include "Logger/Logger.h"

// MemKind Allocator
#include "DataMgr/Allocators/HeteroMem/MemResources/memory_resources.h"

#include <memory>

#include <unistd.h>
#include <cstdint>
#include <filesystem>
#include "memkind.h"

#ifdef __APPLE__
#include <sys/sysctl.h>
#include <sys/types.h>
#endif

namespace Buffer_Namespace {
using pmem_memory_resource_type = libmemkind::pmem::memory_resource;
using static_kind_memory_resource_type = libmemkind::static_kind::memory_resource;

MemoryResourceProvider::MemoryResourceProvider()
    : dram_mem_resource_(new static_kind_memory_resource_type(libmemkind::kinds::REGULAR))
    , mem_resources_(3, dram_mem_resource_.get()) {
  if (memkind_check_available(MEMKIND_DAX_KMEM_ALL) == MEMKIND_SUCCESS) {
    pmem_mem_resource_ = std::make_unique<static_kind_memory_resource_type>(
        libmemkind::kinds::DAX_KMEM_ALL);
    mem_resources_[CAPACITY] = pmem_mem_resource_.get();
    CHECK(mem_resources_[CAPACITY]);
    LOG(INFO) << "KMEM DAX nodes are detected - will use it as a capacity pool";
    pmem_memory_size = SIZE_MAX;
  }
  initAvailableDRAMSize();
}

MemoryResourceProvider::MemoryResourceProvider(const std::string& pmem_path, size_t size)
    : dram_mem_resource_(new static_kind_memory_resource_type(libmemkind::kinds::REGULAR))
    , mem_resources_(3, dram_mem_resource_.get()) {
  initAvailableDRAMSize();
  initPmm(pmem_path, size);
}

std::pmr::memory_resource* MemoryResourceProvider::get(const MemRequirements& req) const {
  CHECK(req < mem_resources_.size());
  CHECK(mem_resources_[req]);
  return mem_resources_[req];
}

size_t MemoryResourceProvider::getAvailableMemorySize(
    std::pmr::memory_resource* mem_resource) const {
  if (dram_mem_resource_.get() == mem_resource) {
    return dram_memory_size;
  }

  if (pmem_mem_resource_.get() == mem_resource) {
    return pmem_memory_size;
  }

  throw std::runtime_error("Memory type unidentified");
}

MemType MemoryResourceProvider::getMemoryType(
    std::pmr::memory_resource* memory_resource) const {
  if (dram_mem_resource_.get() == memory_resource) {
    return DRAM;
  }
  if (pmem_mem_resource_.get() == memory_resource) {
    return PMEM;
  }

  throw std::runtime_error("Memory type unidentified");
}

std::vector<std::pmr::memory_resource*> MemoryResourceProvider::getOrderByBandwidth()
    const {
  CHECK(dram_mem_resource_);
  std::vector<std::pmr::memory_resource*> res;
  // TODO: Once we have HBM, we need to add it to the resulting vector first.

  res.emplace_back(dram_mem_resource_.get());
  if (pmem_mem_resource_)
    res.emplace_back(pmem_mem_resource_.get());

  return res;
}

void MemoryResourceProvider::initPmm(const std::string& pmem_path, size_t size) {
  if (isDAXPath(pmem_path)) {
    LOG(INFO) << pmem_path << " is on DAX-enabled file system.";
  } else {
    LOG(WARNING) << pmem_path << " is not on DAX-enabled file system.";
  }

  if (size == 0) {
    initAvailablePMEMSize(pmem_path);
  } else {
    pmem_memory_size = size;
  }

  pmem_mem_resource_ =
      std::make_unique<pmem_memory_resource_type>(pmem_path, pmem_memory_size);
  mem_resources_[CAPACITY] = pmem_mem_resource_.get();
  CHECK(mem_resources_[CAPACITY]);
}

bool MemoryResourceProvider::isDAXPath(const std::string& path) const {
  int status = memkind_check_dax_path(path.c_str());
  if (status) {
    return false;
  }

  return true;
}

void MemoryResourceProvider::initAvailablePMEMSize(std::string path) {
  std::error_code ec;
  const std::filesystem::space_info si = std::filesystem::space(path, ec);
  if (!ec) {
    pmem_memory_size = static_cast<size_t>(si.capacity);
  } else {
    LOG(FATAL) << "Invalid pmem path: impossible to determine size of the pmem memory";
    return;
  }
}

void MemoryResourceProvider::initAvailableDRAMSize() {
// TODO: this implementation is incorrect for hetero memory
// Memkind will provide proper implementation soon.
// Will update this function once we have correct implementation from Memkind.
#ifdef __APPLE__
  int mib[2];
  size_t physical_memory;
  size_t length;
  // Get the Physical memory size
  mib[0] = CTL_HW;
  mib[1] = HW_MEMSIZE;
  length = sizeof(size_t);
  sysctl(mib, 2, &physical_memory, &length, NULL, 0);
  dram_memory_size = physical_memory;
#elif defined(_MSC_VER)
  MEMORYSTATUSEX status;
  status.dwLength = sizeof(status);
  GlobalMemoryStatusEx(&status);
  dram_memory_size = status.ullTotalPhys;
#else  // Linux
  long pages = sysconf(_SC_PHYS_PAGES);
  long page_size = sysconf(_SC_PAGE_SIZE);
  dram_memory_size = pages * page_size;
#endif
}
}  // namespace Buffer_Namespace