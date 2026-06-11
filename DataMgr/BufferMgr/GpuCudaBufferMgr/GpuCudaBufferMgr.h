/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "DataMgr/BufferMgr/BufferMgr.h"

namespace CudaMgr_Namespace {
class CudaMgr;
}

namespace Buffer_Namespace {

class GpuCudaBufferMgr : public BufferMgr {
 public:
  GpuCudaBufferMgr(const int device_id,
                   const size_t max_buffer_pool_size,
                   CudaMgr_Namespace::CudaMgr* cuda_mgr,
                   const size_t min_slab_size,
                   const size_t max_slab_size,
                   const size_t default_slab_size,
                   const size_t page_size,
                   AbstractBufferMgr* parent_mgr = 0);
  inline MgrType getMgrType() override { return GPU_MGR; }
  inline std::string getStringMgrType() override { return ToString(GPU_MGR); }
  ~GpuCudaBufferMgr() override;

 private:
  void addSlab(const size_t slab_size) override;
  void freeAllMem() override;
  Buffer* createBuffer(BufferList::iterator seg_it, size_t page_size) override;
  CudaMgr_Namespace::CudaMgr* cuda_mgr_;
};

}  // namespace Buffer_Namespace
