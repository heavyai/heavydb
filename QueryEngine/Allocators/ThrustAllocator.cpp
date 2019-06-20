/*
 * Copyright 2019 OmniSci, Inc.
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

#include "ThrustAllocator.h"
#include "CudaAllocator.h"
#include "Shared/Logger.h"

#include <CudaMgr/CudaMgr.h>
#include <DataMgr/DataMgr.h>

int8_t* ThrustAllocator::allocate(std::ptrdiff_t num_bytes) {
#ifdef HAVE_CUDA
  if (!data_mgr_) {  // only for unit tests
    CUdeviceptr ptr;
    const auto err = cuMemAlloc(&ptr, num_bytes);
    CHECK_EQ(CUDA_SUCCESS, err);
    return reinterpret_cast<int8_t*>(ptr);
  }
#endif  // HAVE_CUDA
  Data_Namespace::AbstractBuffer* ab =
      CudaAllocator::allocGpuAbstractBuffer(data_mgr_, num_bytes, device_id_);
  int8_t* raw_ptr = reinterpret_cast<int8_t*>(ab->getMemoryPtr());
  CHECK(!raw_to_ab_ptr_.count(raw_ptr));
  raw_to_ab_ptr_.insert(std::make_pair(raw_ptr, ab));
  return raw_ptr;
}

void ThrustAllocator::deallocate(int8_t* ptr, size_t num_bytes) {
#ifdef HAVE_CUDA
  if (!data_mgr_) {  // only for unit tests
    const auto err = cuMemFree(reinterpret_cast<CUdeviceptr>(ptr));
    CHECK_EQ(CUDA_SUCCESS, err);
    return;
  }
#endif  // HAVE_CUDA
  PtrMapperType::iterator ab_it = raw_to_ab_ptr_.find(ptr);
  CHECK(ab_it != raw_to_ab_ptr_.end());
  data_mgr_->free(ab_it->second);
  raw_to_ab_ptr_.erase(ab_it);
}

int8_t* ThrustAllocator::allocateScopedBuffer(std::ptrdiff_t num_bytes) {
#ifdef HAVE_CUDA
  if (!data_mgr_) {  // only for unit tests
    CUdeviceptr ptr;
    const auto err = cuMemAlloc(&ptr, num_bytes);
    CHECK_EQ(CUDA_SUCCESS, err);
    default_alloc_scoped_buffers_.push_back(reinterpret_cast<int8_t*>(ptr));
    return reinterpret_cast<int8_t*>(ptr);
  }
#endif  // HAVE_CUDA
  Data_Namespace::AbstractBuffer* ab =
      CudaAllocator::allocGpuAbstractBuffer(data_mgr_, num_bytes, device_id_);
  scoped_buffers_.push_back(ab);
  return reinterpret_cast<int8_t*>(ab->getMemoryPtr());
}

ThrustAllocator::~ThrustAllocator() {
  for (auto ab : scoped_buffers_) {
    data_mgr_->free(ab);
  }
#ifdef HAVE_CUDA
  for (auto ptr : default_alloc_scoped_buffers_) {
    const auto err = cuMemFree(reinterpret_cast<CUdeviceptr>(ptr));
    CHECK_EQ(CUDA_SUCCESS, err);
  }
#endif  // HAVE_CUDA
}
