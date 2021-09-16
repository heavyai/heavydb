/*
 * Copyright 2021 Omnisci Inc.
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

#include "DataMgr/BufferMgr/CpuBufferMgr/CpuBufferMgr.h"

namespace CudaMgr_Namespace {
class CudaMgr;
}

namespace Data_Namespace {
constexpr size_t numCpuTiers = 2;
enum CpuTier { DRAM = 0, PMEM = 1 };
using CpuTierSizeVector = std::vector<size_t>;
}  // namespace Data_Namespace

namespace Buffer_Namespace {

class TieredCpuBufferMgr : public CpuBufferMgr {
 public:
  TieredCpuBufferMgr(const int device_id,
                     const size_t total_size,
                     CudaMgr_Namespace::CudaMgr* cuda_mgr,
                     const size_t min_slab_size,
                     const size_t max_slab_size,
                     const size_t page_size,
                     const CpuTierSizeVector& cpu_tier_sizes,
                     AbstractBufferMgr* parent_mgr = nullptr);

  ~TieredCpuBufferMgr() override {
    // The destruction of the allocators automatically frees all memory
  }

  // Needed for testing to replace allocators with Mocks.
  std::vector<std::pair<std::unique_ptr<Arena>, const size_t>>& getAllocators() {
    return allocators_;
  }

  inline MgrType getMgrType() override { return TIERED_CPU_MGR; }
  inline std::string getStringMgrType() override { return ToString(TIERED_CPU_MGR); }
  Arena* getAllocatorForSlab(int32_t slab_num) const;
  std::string dump() const;

 private:
  void addSlab(const size_t slab_size) override;
  void freeAllMem() override;
  void initializeMem() override;

  // A vector of allocators (order in vector denotes priority for use).  These allocators
  // should represent various tiers of memory we intend to use, such as DRAM, PMEM, and
  // HBMEM.  The size specifies the maximum space is allowed for each allocator.
  std::vector<std::pair<std::unique_ptr<Arena>, const size_t>> allocators_;
  // Map to track which slabs were created by which allocator (may not be necessary
  // later).
  std::map<int32_t, Arena*> slab_to_allocator_map_;
};

}  // namespace Buffer_Namespace
