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

#include "DataMgr/BufferMgr/GpuCudaBufferMgr/GpuCudaBuffer.h"

#include <cassert>

#include "CudaMgr/CudaMgr.h"
#include "Shared/Logger.h"

namespace Buffer_Namespace {

GpuCudaBuffer::GpuCudaBuffer(BufferMgr* bm,
                             BufferList::iterator seg_it,
                             const int device_id,
                             CudaMgr_Namespace::CudaMgr* cuda_mgr,
                             const size_t page_size,
                             const size_t num_bytes)
    : Buffer(bm, seg_it, device_id, page_size, num_bytes), cuda_mgr_(cuda_mgr) {}

void GpuCudaBuffer::readData(int8_t* const dst,
                             const size_t num_bytes,
                             const size_t offset,
                             const MemoryLevel dst_buffer_type,
                             const int dst_device_id) {
  if (dst_buffer_type == CPU_LEVEL) {
    cuda_mgr_->copyDeviceToHost(
        dst, mem_ + offset, num_bytes, device_id_);  // need to replace 0 with gpu num
  } else if (dst_buffer_type == GPU_LEVEL) {
    cuda_mgr_->copyDeviceToDevice(
        dst, mem_ + offset, num_bytes, dst_device_id, device_id_);

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
    // std::cout << "Writing to GPU from source CPU" << std::endl;

    cuda_mgr_->copyHostToDevice(
        mem_ + offset, src, num_bytes, device_id_);  // need to replace 0 with gpu num

  } else if (src_buffer_type == GPU_LEVEL) {
    // std::cout << "Writing to GPU from source GPU" << std::endl;
    assert(src_device_id >= 0);
    cuda_mgr_->copyDeviceToDevice(
        mem_ + offset, src, num_bytes, device_id_, src_device_id);
  } else {
    LOG(FATAL) << "Unsupported buffer type";
  }
}

}  // namespace Buffer_Namespace
