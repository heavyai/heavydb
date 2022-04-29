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

#include "DataMgr/BufferMgr/GpuBufferMgr/GpuBufferMgr.h"
#ifdef HAVE_CUDA
#include "CudaMgr/CudaMgr.h"
#endif

#include "DataMgr/BufferMgr/GpuBufferMgr/GpuBuffer.h"
#include "Logger/Logger.h"

namespace Buffer_Namespace {

GpuBufferMgr::GpuBufferMgr(const int device_id,
                           const size_t max_buffer_pool_size,
                           GpuMgr* gpu_mgr,
                           const size_t min_slab_size,
                           const size_t max_slab_size,
                           const size_t page_size,
                           AbstractBufferMgr* parent_mgr)
    : BufferMgr(device_id,
                max_buffer_pool_size,
                min_slab_size,
                max_slab_size,
                page_size,
                parent_mgr)
    , gpu_mgr_(gpu_mgr) {}

GpuBufferMgr::~GpuBufferMgr() {
  try {
    gpu_mgr_->synchronizeDevices();
    freeAllMem();
#ifdef HAVE_CUDA
  } catch (const CudaMgr_Namespace::CudaErrorException& e) {
    if (e.getStatus() == CUDA_ERROR_DEINITIALIZED) {
      // TODO(adb / asuhan): Verify cuModuleUnload removes the context
      return;
    }
#endif
  } catch (const std::runtime_error& e) {
    LOG(ERROR) << "GPU Error: " << e.what();
  }
}

void GpuBufferMgr::addSlab(const size_t slab_size) {
  slabs_.resize(slabs_.size() + 1);
  try {
    slabs_.back() = gpu_mgr_->allocateDeviceMem(slab_size, device_id_);
  } catch (std::runtime_error& error) {
    slabs_.resize(slabs_.size() - 1);
    throw FailedToCreateSlab(slab_size);
  }
  slab_segments_.resize(slab_segments_.size() + 1);
  slab_segments_[slab_segments_.size() - 1].push_back(
      BufferSeg(0, slab_size / page_size_));
}

void GpuBufferMgr::freeAllMem() {
  for (auto buf_it = slabs_.begin(); buf_it != slabs_.end(); ++buf_it) {
    gpu_mgr_->freeDeviceMem(*buf_it);
  }
}

void GpuBufferMgr::allocateBuffer(BufferList::iterator seg_it,
                                  const size_t page_size,
                                  const size_t initial_size) {
  new GpuBuffer(this,
                seg_it,
                device_id_,
                gpu_mgr_,
                page_size,
                initial_size);  // this line is admittedly a bit weird
                                // but the segment iterator passed into
                                // buffer takes the address of the new
                                // Buffer in its buffer member
}

}  // namespace Buffer_Namespace
