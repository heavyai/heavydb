/*
 * SPDX-FileCopyrightText: Copyright (c) 2014-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "DataMgr/BufferMgr/BufferMgr.h"

#include "DataMgr/Allocators/ArenaAllocator.h"

namespace CudaMgr_Namespace {
class CudaMgr;
}

namespace Buffer_Namespace {

class CpuBufferMgr : public BufferMgr {
 public:
  CpuBufferMgr(const int device_id,
               const size_t max_buffer_pool_size,
               CudaMgr_Namespace::CudaMgr* cuda_mgr,
               const size_t min_slab_size,
               const size_t max_slab_size,
               const size_t default_slab_size,
               const size_t page_size,
               AbstractBufferMgr* parent_mgr = nullptr)
      : BufferMgr(device_id,
                  max_buffer_pool_size,
                  min_slab_size,
                  max_slab_size,
                  default_slab_size,
                  page_size,
                  parent_mgr)
      , cuda_mgr_(cuda_mgr) {
    initializeMem();
  }

  ~CpuBufferMgr() override {
    /* the destruction of the allocator automatically frees all memory */
  }

  inline MgrType getMgrType() override { return CPU_MGR; }
  inline std::string getStringMgrType() override { return ToString(CPU_MGR); }

  // Used for testing.
  void setAllocator(std::unique_ptr<DramArena> allocator) {
    allocator_ = std::move(allocator);
  }

  struct CpuBufferMgrMemoryUsage {
    size_t allocated;
    size_t in_use;
  };

  CpuBufferMgrMemoryUsage getMemoryUsage() const {
    return {getAllocated(), getInUseSize()};
  }

 protected:
  void addSlab(const size_t slab_size) override;
  void freeAllMem() override;
  Buffer* createBuffer(BufferList::iterator seg_it, size_t page_size) override;

  virtual void initializeMem();

  CudaMgr_Namespace::CudaMgr* cuda_mgr_;

 private:
  std::unique_ptr<DramArena> allocator_;
};

std::ostream& operator<<(std::ostream& os,
                         const CpuBufferMgr::CpuBufferMgrMemoryUsage& bm);
}  // namespace Buffer_Namespace
