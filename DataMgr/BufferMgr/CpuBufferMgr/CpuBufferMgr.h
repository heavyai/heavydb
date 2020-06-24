/*
 * Copyright 2017 MapD Technologies, Inc.
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
               const size_t page_size,
               AbstractBufferMgr* parent_mgr = nullptr)
      : BufferMgr(device_id,
                  max_buffer_pool_size,
                  min_slab_size,
                  max_slab_size,
                  page_size,
                  parent_mgr)
      , cuda_mgr_(cuda_mgr)
      , allocator_(std::make_unique<Arena>(/*min_block_size=*/max_slab_size +
                                           kArenaBlockOverhead)) {}

  ~CpuBufferMgr() {
    /* the destruction of the allocator automatically frees all memory */
  }

  inline MgrType getMgrType() override { return CPU_MGR; }
  inline std::string getStringMgrType() override { return ToString(CPU_MGR); }

 private:
  void addSlab(const size_t slab_size) override;
  void freeAllMem() override;
  void allocateBuffer(BufferList::iterator segment_iter,
                      const size_t page_size,
                      const size_t initial_size) override;

  CudaMgr_Namespace::CudaMgr* cuda_mgr_;
  std::unique_ptr<Arena> allocator_;
};

}  // namespace Buffer_Namespace
