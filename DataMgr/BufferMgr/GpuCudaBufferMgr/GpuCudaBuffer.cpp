/*
 * SPDX-FileCopyrightText: Copyright (c) 2014-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DataMgr/BufferMgr/GpuCudaBufferMgr/GpuCudaBuffer.h"

#include <cassert>

#include "CudaMgr/CudaMgr.h"
#include "Logger/Logger.h"

namespace Buffer_Namespace {

GpuCudaBuffer::GpuCudaBuffer(BufferMgr* bm,
                             BufferList::iterator seg_it,
                             const int device_id,
                             CudaMgr_Namespace::CudaMgr* cuda_mgr,
                             const size_t page_size)
    : Buffer(bm, seg_it, device_id, page_size), cuda_mgr_(cuda_mgr) {}

void GpuCudaBuffer::readData(int8_t* const dst,
                             const size_t num_bytes,
                             const size_t offset,
                             const MemoryLevel dst_buffer_type,
                             const int dst_device_id) {
  if (dst_buffer_type == CPU_LEVEL) {
    cuda_mgr_->copyDeviceToHost(
        dst, mem_ + offset, num_bytes, "CudaBuffer");  // need to replace 0 with gpu num
  } else if (dst_buffer_type == GPU_LEVEL) {
    cuda_mgr_->copyDeviceToDevice(
        dst, mem_ + offset, num_bytes, dst_device_id, device_id_, "CudaBuffer");

  } else {
    LOG(FATAL) << "Unsupported buffer type";
  }
}

void GpuCudaBuffer::writeData(int8_t* const src,
                              const size_t num_bytes,
                              const size_t offset,
                              const MemoryLevel src_buffer_type,
                              const int src_device_id) {
  if (src_buffer_type == CPU_LEVEL) {
    cuda_mgr_->copyHostToDevice(mem_ + offset,
                                src,
                                num_bytes,
                                device_id_,
                                "CudaBuffer");  // need to replace 0 with gpu num

  } else if (src_buffer_type == GPU_LEVEL) {
    CHECK_GE(src_device_id, 0);
    cuda_mgr_->copyDeviceToDevice(
        mem_ + offset, src, num_bytes, device_id_, src_device_id, "CudaBuffer");
  } else {
    LOG(FATAL) << "Unsupported buffer type";
  }
}

}  // namespace Buffer_Namespace
