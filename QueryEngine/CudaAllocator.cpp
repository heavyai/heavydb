/*
 * Copyright 2018 MapD Technologies, Inc.
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

#include "CudaAllocator.h"

#include <CudaMgr/CudaMgr.h>
#include <DataMgr/DataMgr.h>
#include <Shared/types.h>
#include "Rendering/RenderAllocator.h"

#include <glog/logging.h>

CudaAllocator::CudaAllocator(Data_Namespace::DataMgr* data_mgr, const int device_id)
    : data_mgr_(data_mgr) {
  CHECK(data_mgr_);
#ifdef HAVE_CUDA
  const auto cuda_mgr = data_mgr_->getCudaMgr();
  CHECK(cuda_mgr);
  cuda_mgr->setContext(device_id);
#endif  // HAVE_CUDA
}

CUdeviceptr CudaAllocator::alloc(const size_t num_bytes,
                                 const int device_id,
                                 RenderAllocator* render_allocator) const {
  if (render_allocator) {
    return reinterpret_cast<CUdeviceptr>(render_allocator->alloc(num_bytes));
  }
  OOM_TRACE_PUSH(+": device_id " + std::to_string(device_id) + ", num_bytes " +
                 std::to_string(num_bytes));
  auto ab = allocGpuAbstractBuffer(num_bytes, device_id);
  return reinterpret_cast<CUdeviceptr>(ab->getMemoryPtr());
}

void CudaAllocator::free(Data_Namespace::AbstractBuffer* ab) const {
  data_mgr_->free(ab);
}

void CudaAllocator::copyToDevice(CUdeviceptr dst,
                                 const void* src,
                                 const size_t num_bytes,
                                 const int device_id) const {
  const auto cuda_mgr = data_mgr_->getCudaMgr();
  CHECK(cuda_mgr);
  cuda_mgr->copyHostToDevice(reinterpret_cast<int8_t*>(dst),
                             static_cast<const int8_t*>(src),
                             num_bytes,
                             device_id);
}

void CudaAllocator::copyFromDevice(void* dst,
                                   const CUdeviceptr src,
                                   const size_t num_bytes,
                                   const int device_id) const {
  const auto cuda_mgr = data_mgr_->getCudaMgr();
  CHECK(cuda_mgr);
  cuda_mgr->copyDeviceToHost(static_cast<int8_t*>(dst),
                             reinterpret_cast<const int8_t*>(src),
                             num_bytes,
                             device_id);
}

void CudaAllocator::zeroDeviceMem(int8_t* device_ptr,
                                  const size_t num_bytes,
                                  const int device_id) const {
  const auto cuda_mgr = data_mgr_->getCudaMgr();
  CHECK(cuda_mgr);
  cuda_mgr->zeroDeviceMem(device_ptr, num_bytes, device_id);
}

void CudaAllocator::setDeviceMem(int8_t* device_ptr,
                                 unsigned char uc,
                                 const size_t num_bytes,
                                 const int device_id) const {
  const auto cuda_mgr = data_mgr_->getCudaMgr();
  CHECK(cuda_mgr);
  cuda_mgr->setDeviceMem(device_ptr, uc, num_bytes, device_id);
}

Data_Namespace::AbstractBuffer* CudaAllocator::allocGpuAbstractBuffer(
    const size_t num_bytes,
    const int device_id) const {
  auto ab = data_mgr_->alloc(Data_Namespace::GPU_LEVEL, device_id, num_bytes);
  CHECK_EQ(ab->getPinCount(), 1);
  return ab;
}