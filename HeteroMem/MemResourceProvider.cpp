// SPDX-License-Identifier: BSD-2-Clause
/* Copyright (C) 2020 Intel Corporation. */

#include "HeteroMem/MemResourceProvider.h"

#include "Shared/Logger.h"

// MemKind Allocator
#include "HeteroMem/MemResources/memory_resources.h"

#include <memory>

namespace Buffer_Namespace {
MemoryResourceProvider::MemoryResourceProvider() : mem_resources_(3, std::pmr::get_default_resource())
{
}

MemoryResourceProvider::MemoryResourceProvider(const std::string& pmmPath) : mem_resources_(3, std::pmr::get_default_resource())
{
    initPmm(pmmPath);
}

std::pmr::memory_resource* MemoryResourceProvider::get(const MemRequirements& req) const
{
  CHECK(req < mem_resources_.size());
  CHECK(mem_resources_[req]);
  return mem_resources_[req];
}

void MemoryResourceProvider::initPmm(const std::string& pmmPath)
{
  using pmem_memory_resource_type = libmemkind::pmem::memory_resource;

  std::ifstream pmem_dirs_file(pmmPath);
  if (pmem_dirs_file.is_open()) {
    std::string line;
    while (!pmem_dirs_file.eof()) {
      std::getline(pmem_dirs_file, line);
      if (!line.empty()) {
        std::stringstream ss;
        std::string path;
        size_t size;

        ss << line;
        ss >> path;
        ss >> size;

        // TODO: need to support multiple directories.
        if(pmem_mem_resource_.get()) {
          LOG(FATAL) << "For now OmniSciDB does not support more than one directory for PMM volatile" << std::endl;
          return;
        }
        
        pmem_mem_resource_ = std::make_unique<pmem_memory_resource_type>(path, size * 1024 * 1024 * 1024);
        mem_resources_[CAPACITY] = pmem_mem_resource_.get();
        CHECK(mem_resources_[CAPACITY]);
      }
    }
    pmem_dirs_file.close();
  }
  else{
    LOG(FATAL) << "Unable to open file " << pmmPath << std::endl;
    return;
  }

}

} // namespace Buffer_Namespace