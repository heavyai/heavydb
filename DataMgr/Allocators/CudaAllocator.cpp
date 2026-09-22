/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DataMgr/Allocators/CudaAllocator.h"

#include <CudaMgr/CudaMgr.h>
#include <CudaMgr/CudaShared.h>
#include <DataMgr/DataMgr.h>
#include <Logger/Logger.h>
#include <Shared/types.h>

#include <mutex>
#include <utility>

class CudaReadyEventPool {
 public:
#ifdef HAVE_CUDA
  CudaReadyEventPool(const CUcontext context, const int device_id)
      : context_(context), device_id_(device_id) {
    CHECK(context_);
  }

  ~CudaReadyEventPool() {
    CUcontext original_context{nullptr};
    if (cuCtxGetCurrent(&original_context) != CUDA_SUCCESS ||
        cuCtxSetCurrent(context_) != CUDA_SUCCESS) {
      return;
    }
    for (const auto event : available_events_) {
      const auto status = cuEventDestroy(event);
      if (status != CUDA_SUCCESS && status != CUDA_ERROR_DEINITIALIZED) {
        VLOG(1) << "Failed to destroy pooled CUDA ready event: device=" << device_id_
                << " error=" << CudaMgr_Namespace::error_message(status);
      }
    }
    cuCtxSetCurrent(original_context);
  }

  CudaReadyEventHandle acquire() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!available_events_.empty()) {
      const auto event = available_events_.back();
      available_events_.pop_back();
      return event;
    }
    CudaReadyEventHandle event{nullptr};
    CudaMgr_Namespace::check_error(cuEventCreate(&event, CU_EVENT_DISABLE_TIMING));
    return event;
  }

  void release(const CudaReadyEventHandle event) noexcept {
    if (!event) {
      return;
    }
    try {
      std::lock_guard<std::mutex> lock(mutex_);
      available_events_.push_back(event);
      return;
    } catch (...) {
    }

    CUcontext original_context{nullptr};
    if (cuCtxGetCurrent(&original_context) != CUDA_SUCCESS ||
        cuCtxSetCurrent(context_) != CUDA_SUCCESS) {
      return;
    }
    cuEventDestroy(event);
    cuCtxSetCurrent(original_context);
  }

 private:
  CUcontext context_;
  int device_id_;
  std::mutex mutex_;
  std::vector<CudaReadyEventHandle> available_events_;
#else
  CudaReadyEventHandle acquire() {
    return nullptr;
  }
  void release(const CudaReadyEventHandle) noexcept {}
#endif
};

CudaStreamReadyEvent::CudaStreamReadyEvent(std::shared_ptr<CudaReadyEventPool> event_pool,
                                           const int device_id)
    : event_pool_(std::move(event_pool)), device_id_(device_id), event_(nullptr) {
  CHECK(event_pool_);
  event_ = event_pool_->acquire();
}

CudaStreamReadyEvent::~CudaStreamReadyEvent() {
  if (event_pool_) {
    event_pool_->release(event_);
  }
}

CudaAllocator::CudaAllocator(Data_Namespace::DataMgr* data_mgr,
                             const int device_id,
                             CUstream cuda_stream)
    : data_mgr_(data_mgr), device_id_(device_id), cuda_stream_(cuda_stream) {
  CHECK(data_mgr_);
#ifdef HAVE_CUDA
  const auto cuda_mgr = data_mgr_->getCudaMgr();
  CHECK(cuda_mgr);
  cuda_mgr->setContext(device_id);
  ready_event_pool_ = std::make_shared<CudaReadyEventPool>(
      cuda_mgr->getDeviceContexts().at(device_id), device_id);
#endif  // HAVE_CUDA
}

CudaAllocator::~CudaAllocator() {
  CHECK(data_mgr_);
  for (auto& buffer_ptr : owned_buffers_) {
    data_mgr_->free(buffer_ptr);
  }
}

Data_Namespace::AbstractBuffer* CudaAllocator::allocGpuAbstractBuffer(
    Data_Namespace::DataMgr* data_mgr,
    const size_t num_bytes,
    const int device_id) {
  CHECK(data_mgr);
  Data_Namespace::AbstractBuffer* ab{nullptr};
  try {
    ab = data_mgr->alloc(Data_Namespace::GPU_LEVEL, device_id, num_bytes);
  } catch (const std::exception& e) {
    LOG(WARNING) << "DataMgr GPU allocation failed: device=GPU:" << device_id
                 << " bytes=" << num_bytes << " error=" << e.what();
    throw;
  }
  CHECK_EQ(ab->getPinCount(), 1);
#ifdef HAVE_CUDA
  auto cuda_mgr = data_mgr->getCudaMgr();
  CHECK(cuda_mgr);
  if (cuda_mgr->logMemoryActivity()) {
    VLOG(1) << "DataMgr allocating GPU memory: " << num_bytes
            << " bytes (address: " << static_cast<void*>(ab->getMemoryPtr())
            << ", device-" << device_id << ")";
  }
#endif
  return ab;
}

void CudaAllocator::freeGpuAbstractBuffer(Data_Namespace::DataMgr* data_mgr,
                                          Data_Namespace::AbstractBuffer* ab) {
  CHECK(data_mgr);
#ifdef HAVE_CUDA
  auto cuda_mgr = data_mgr->getCudaMgr();
  CHECK(cuda_mgr);
  const bool log_memory_activity = cuda_mgr->logMemoryActivity();
  const auto allocation_size = log_memory_activity ? ab->size() : size_t(0);
  const auto allocation_address =
      log_memory_activity ? static_cast<void*>(ab->getMemoryPtr()) : nullptr;
  const auto allocation_device = log_memory_activity ? ab->getDeviceId() : 0;
#endif
  data_mgr->free(ab);
#ifdef HAVE_CUDA
  if (log_memory_activity) {
    VLOG(1) << "DataMgr freeing GPU memory: " << allocation_size
            << " bytes (address: " << allocation_address << ", device-"
            << allocation_device << ")";
  }
#endif
}

int8_t* CudaAllocator::alloc(const size_t num_bytes) {
  CHECK(data_mgr_);
  owned_buffers_.emplace_back(
      CudaAllocator::allocGpuAbstractBuffer(data_mgr_, num_bytes, device_id_));
  return owned_buffers_.back()->getMemoryPtr();
}

void CudaAllocator::free(Data_Namespace::AbstractBuffer* ab) const {
#ifdef HAVE_CUDA
  auto cuda_mgr = data_mgr_->getCudaMgr();
  CHECK(cuda_mgr);
  const bool log_memory_activity =
      ab->getType() == MemoryLevel::GPU_LEVEL && cuda_mgr->logMemoryActivity();
  const auto allocation_size = log_memory_activity ? ab->size() : size_t(0);
  const auto allocation_address =
      log_memory_activity ? static_cast<void*>(ab->getMemoryPtr()) : nullptr;
  const auto allocation_device = log_memory_activity ? ab->getDeviceId() : 0;
#endif
  data_mgr_->free(ab);
#ifdef HAVE_CUDA
  if (log_memory_activity) {
    VLOG(1) << "DataMgr freeing GPU memory: " << allocation_size
            << " bytes (address: " << allocation_address << ", device-"
            << allocation_device << ")";
  }
#endif
}

void CudaAllocator::copyToDevice(void* device_dst,
                                 const void* host_src,
                                 const size_t num_bytes,
                                 std::optional<std::string_view> tag) const {
  const auto cuda_mgr = data_mgr_->getCudaMgr();
  CHECK(cuda_mgr);
  cuda_mgr->copyHostToDevice(
      (int8_t*)device_dst, (int8_t*)host_src, num_bytes, device_id_, tag, cuda_stream_);
}

void CudaAllocator::copyFromDevice(void* host_dst,
                                   const void* device_src,
                                   const size_t num_bytes,
                                   std::optional<std::string_view> tag) const {
  const auto cuda_mgr = data_mgr_->getCudaMgr();
  CHECK(cuda_mgr);
  cuda_mgr->copyDeviceToHost(
      (int8_t*)host_dst, (int8_t*)device_src, num_bytes, tag, cuda_stream_);
}

void CudaAllocator::zeroDeviceMem(int8_t* device_ptr, const size_t num_bytes) const {
  const auto cuda_mgr = data_mgr_->getCudaMgr();
  CHECK(cuda_mgr);
  cuda_mgr->zeroDeviceMem(device_ptr, num_bytes, device_id_, cuda_stream_);
}

void CudaAllocator::setDeviceMem(int8_t* device_ptr,
                                 unsigned char uc,
                                 const size_t num_bytes) const {
  const auto cuda_mgr = data_mgr_->getCudaMgr();
  CHECK(cuda_mgr);
  cuda_mgr->setDeviceMem(device_ptr, uc, num_bytes, device_id_, cuda_stream_);
}

std::shared_ptr<CudaStreamReadyEvent> CudaAllocator::recordReadyEvent() const {
#ifdef HAVE_CUDA
  const auto cuda_mgr = data_mgr_->getCudaMgr();
  CHECK(cuda_mgr);
  cuda_mgr->setContext(device_id_);
  CHECK(ready_event_pool_);
  auto ready_event =
      std::make_shared<CudaStreamReadyEvent>(ready_event_pool_, device_id_);
  CudaMgr_Namespace::check_error(cuEventRecord(ready_event->event(), cuda_stream_));
  return ready_event;
#else
  return nullptr;
#endif
}

void CudaAllocator::waitForReadyEvent(
    std::shared_ptr<CudaStreamReadyEvent> ready_event) const {
#ifdef HAVE_CUDA
  if (!ready_event || !ready_event->event()) {
    return;
  }
  const auto cuda_mgr = data_mgr_->getCudaMgr();
  CHECK(cuda_mgr);
  cuda_mgr->setContext(device_id_);
  CudaMgr_Namespace::check_error(
      cuStreamWaitEvent(cuda_stream_, ready_event->event(), 0));
#else
  (void)ready_event;
#endif
}

void CudaAllocator::rollbackAllocationsTo(const size_t checkpoint) noexcept {
  if (checkpoint > owned_buffers_.size()) {
    LOG(ERROR) << "Invalid CUDA allocator rollback checkpoint: checkpoint=" << checkpoint
               << " allocation_count=" << owned_buffers_.size()
               << " device=" << device_id_;
    return;
  }
  if (checkpoint == owned_buffers_.size()) {
    return;
  }

#ifdef HAVE_CUDA
  try {
    const auto cuda_mgr = data_mgr_->getCudaMgr();
    CHECK(cuda_mgr);
    cuda_mgr->setContext(device_id_);
    if (cuda_stream_) {
      CudaMgr_Namespace::check_error(cuStreamSynchronize(cuda_stream_));
    } else {
      cuda_mgr->synchronizeDevice(device_id_);
    }
  } catch (const std::exception& error) {
    // Keeping the allocations pinned is safer than making pages reusable while failed
    // asynchronous reducer work may still reference them.
    LOG(ERROR) << "Could not synchronize CUDA allocator before rollback: device="
               << device_id_ << " error=" << error.what();
    return;
  }
#endif

  while (owned_buffers_.size() > checkpoint) {
    auto* buffer = owned_buffers_.back();
    try {
      data_mgr_->free(buffer);
      owned_buffers_.pop_back();
    } catch (const std::exception& error) {
      LOG(ERROR) << "Could not release CUDA allocation during rollback: device="
                 << device_id_ << " error=" << error.what();
      return;
    }
  }
}
