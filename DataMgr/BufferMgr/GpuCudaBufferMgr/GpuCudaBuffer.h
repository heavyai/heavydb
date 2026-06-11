/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "DataMgr/BufferMgr/Buffer.h"

namespace CudaMgr_Namespace {
class CudaMgr;
}
namespace Buffer_Namespace {

class GpuCudaBuffer : public Buffer {
 public:
  GpuCudaBuffer(BufferMgr* bm,
                BufferList::iterator seg_it,
                const int device_id,
                CudaMgr_Namespace::CudaMgr* cuda_mgr,
                const size_t page_size = 512);
  inline Data_Namespace::MemoryLevel getType() const override { return GPU_LEVEL; }

 private:
  void readData(int8_t* const dst,
                const size_t num_bytes,
                const size_t offset = 0,
                const MemoryLevel dst_buffer_type = CPU_LEVEL,
                const int dst_devic_id = -1) override;
  void writeData(int8_t* const src,
                 const size_t num_bytes,
                 const size_t offset = 0,
                 const MemoryLevel src_buffer_type = CPU_LEVEL,
                 const int src_device_id = -1) override;

  CudaMgr_Namespace::CudaMgr* cuda_mgr_;
};
}  // namespace Buffer_Namespace
