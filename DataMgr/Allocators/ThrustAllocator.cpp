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

#include "DataMgr/Allocators/ThrustAllocator.h"

#define BOOST_STACKTRACE_GNU_SOURCE_NOT_REQUIRED 1
#include <boost/stacktrace.hpp>

#include <cstdint>

#include "CudaMgr/CudaMgr.h"
#include "DataMgr/Allocators/CudaAllocator.h"
#include "DataMgr/DataMgr.h"
#include "Logger/Logger.h"

int8_t* ThrustAllocator::allocate(std::ptrdiff_t num_bytes) {
  VLOG(1) << "Thrust allocation: Device #" << device_id_ << " Allocation #"
          << ++num_allocations_ << ": " << num_bytes << " bytes";
#ifdef HAVE_CUDA
  if (!data_mgr_) {  // only for unit tests
    CUdeviceptr ptr;
    const auto err = cuMemAlloc(&ptr, num_bytes);
    CHECK_EQ(CUDA_SUCCESS, err);
    return reinterpret_cast<int8_t*>(ptr);
  }
  Data_Namespace::AbstractBuffer* ab =
      CudaAllocator::allocGpuAbstractBuffer(data_mgr_, num_bytes, device_id_);
#else
  Data_Namespace::AbstractBuffer* ab =
      data_mgr_->alloc(MemoryLevel::CPU_LEVEL, device_id_, num_bytes);
  CHECK_EQ(ab->getPinCount(), 1);
#endif  // HAVE_CUDA
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
  Data_Namespace::AbstractBuffer* ab =
      CudaAllocator::allocGpuAbstractBuffer(data_mgr_, num_bytes, device_id_);
#else
  Data_Namespace::AbstractBuffer* ab =
      data_mgr_->alloc(MemoryLevel::CPU_LEVEL, device_id_, num_bytes);
  CHECK_EQ(ab->getPinCount(), 1);
#endif  // HAVE_CUDA
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
  if (!raw_to_ab_ptr_.empty()) {
    LOG(ERROR) << "Not all GPU buffers deallocated before destruction of Thrust "
                  "allocator for device "
               << device_id_ << ". Remaining buffers: ";
    for (auto& kv : raw_to_ab_ptr_) {
      auto& ab = kv.second;
      CHECK(ab);
      LOG(ERROR) << (ab->pageCount() * ab->pageSize()) / (1024. * 1024.) << " MB";
    }
    VLOG(1) << boost::stacktrace::stacktrace();
  }
}
