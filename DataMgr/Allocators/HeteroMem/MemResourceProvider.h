// SPDX-License-Identifier: BSD-2-Clause
/* Copyright (C) 2020 Intel Corporation. */

#pragma once

#include <memory_resource>
#include <string>
#include <vector>

namespace Buffer_Namespace {

enum MemRequirements { CAPACITY, HIGH_BDWTH, LOW_LATENCY };

enum MemType { DRAM, PMEM, HBM };

class MemoryResourceProvider {
 public:
  MemoryResourceProvider();

  explicit MemoryResourceProvider(const std::string& pmem_path, size_t size);

  std::pmr::memory_resource* get(const MemRequirements& req) const;
  MemType getMemoryType(std::pmr::memory_resource* other) const;
  size_t getAvailableMemorySize(std::pmr::memory_resource* mem_resource) const;
  std::vector<std::pmr::memory_resource*> getOrderByBandwidth() const;

 private:
  void initPmm(const std::string& pmem_path, size_t size);

  bool isDAXPath(const std::string& path) const;

  void initAvailablePMEMSize(std::string path);
  void initAvailableDRAMSize();

  using mem_resources_storage_type = std::vector<std::pmr::memory_resource*>;

  std::unique_ptr<std::pmr::memory_resource> dram_mem_resource_;
  std::unique_ptr<std::pmr::memory_resource> pmem_mem_resource_;
  size_t dram_memory_size;
  size_t pmem_memory_size;

  mem_resources_storage_type mem_resources_;
};

}  // namespace Buffer_Namespace
