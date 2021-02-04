// SPDX-License-Identifier: BSD-2-Clause
/* Copyright (C) 2020 Intel Corporation. */

#pragma once

#include <memory_resource>
#include <string>
#include <vector>

namespace Buffer_Namespace {

enum MemRequirements { CAPACITY, HIGH_BDWTH, LOW_LATENCY };

class MemoryResourceProvider {
 public:
  MemoryResourceProvider();

  MemoryResourceProvider(const std::string& pmmPath);

  std::pmr::memory_resource* get(const MemRequirements& req) const;

 private:
  /*
   * the pmm_path is the file containing the persistent memory file folder
   * each line in the file is a directory pathname, for example:
   * /mnt/ad1/omnisci	0		// use all availabe space
   * /mnt/ad3/omnisci	128		// use no more than 128GB
   */
  void initPmm(const std::string& pmmPath);

  std::vector<std::pmr::memory_resource*> mem_resources_;

  std::unique_ptr<std::pmr::memory_resource> pmem_mem_resource_;
};

}  // namespace Buffer_Namespace