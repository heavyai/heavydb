/*
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "CudaMgr/CudaMgr.h"

#include <algorithm>
#include <iostream>
#include <limits>
#include <new>
#include <stdexcept>

#include <boost/filesystem.hpp>

#include "Logger/Logger.h"
#include "QueryEngine/NvidiaKernel.h"
#include "Shared/scope.h"

bool g_enable_gpu_dynamic_smem{true};
std::string g_peer_copy_mode_name{"direct"};
int g_peer_copy_mode{CudaMgr_Namespace::kPeerCopyModeDirect};
size_t g_peer_copy_staging_buffer_size{256 * 1024 * 1024};

namespace CudaMgr_Namespace {

CudaMgr::CudaMgr(const int num_gpus, const int start_gpu)
    : start_gpu_(start_gpu)
    , min_shared_memory_per_block_for_all_devices(0)
    , min_num_mps_for_all_devices(0)
    , log_memory_activity_(false)
    , device_memory_allocation_map_{std::make_unique<DeviceMemoryAllocationMap>()} {
  checkError(cuInit(0));
  checkError(cuDeviceGetCount(&device_count_));

  if (num_gpus > 0) {  // numGpus <= 0 will just use number of gpus found
    device_count_ = std::min(device_count_, num_gpus);
  } else {
    // if we are using all gpus we cannot start on a gpu other than 0
    CHECK_EQ(start_gpu_, 0);
  }

  // Fill device properties and initialize group/contexts
  fillDeviceProperties();
  initDeviceGroup();
  createDeviceContexts();

  CUcontext original_context{nullptr};
  checkError(cuCtxGetCurrent(&original_context));
  ScopeGuard restore_context = [original_context] {
    const auto status = cuCtxSetCurrent(original_context);
    if (status != CUDA_SUCCESS && status != CUDA_ERROR_DEINITIALIZED) {
      LOG(ERROR) << "Failed to restore CUDA context after transfer-stream creation: "
                 << error_message(status);
    }
  };
  device_transfer_streams_.resize(device_count_);
  try {
    for (int device_id = 0; device_id < device_count_; ++device_id) {
      setContext(device_id);
      checkError(
          cuStreamCreate(&device_transfer_streams_[device_id], CU_STREAM_NON_BLOCKING));
    }
  } catch (...) {
    for (int device_id = 0; device_id < device_count_; ++device_id) {
      if (device_transfer_streams_[device_id]) {
        setContext(device_id);
        cuStreamDestroy(device_transfer_streams_[device_id]);
      }
    }
    device_transfer_streams_.clear();
    throw;
  }

  logDeviceProperties();

  // warm up the GPU JIT
  LOG(INFO) << "Warming up the GPU JIT Compiler... (this may take several seconds)";
  setContext(0);
  nvidia_jit_warmup();
  LOG(INFO) << "GPU JIT Compiler initialized.";

  jump_buffer_transfer_mgr_ =
      std::make_unique<JumpBufferTransferMgr>(device_count_, device_contexts_);
}

void CudaMgr::initDeviceGroup() {
  CHECK(device_properties_initialized_);
  for (int device_id = 0; device_id < device_count_; device_id++) {
    device_group_.push_back(
        {device_id, device_id + start_gpu_, device_properties_[device_id].uuid});
  }
}

CudaMgr::~CudaMgr() {
  try {
    releasePeerCopyStagingBuffers();
    // We don't want to remove the cudaMgr before all other processes have cleaned up.
    // This should be enforced by the lifetime policies, but take this lock to be safe.
    std::lock_guard<std::mutex> device_lock(device_mutex_);
    jump_buffer_transfer_mgr_.reset();

    synchronizeDevices();

    for (int device_id = 0; device_id < device_count_; ++device_id) {
      if (!device_transfer_streams_[device_id]) {
        continue;
      }
      setContext(device_id);
      const auto status = cuStreamDestroy(device_transfer_streams_[device_id]);
      if (status != CUDA_SUCCESS && status != CUDA_ERROR_DEINITIALIZED) {
        LOG(ERROR) << "Failed to destroy CUDA transfer stream: device=" << device_id
                   << " error=" << error_message(status);
      }
    }
    device_transfer_streams_.clear();

    CHECK(peer_copy_staging_buffers_.empty());
    CHECK(peer_copyable_device_allocations_.empty());
    CHECK(getDeviceMemoryAllocationMap().mapEmpty());
    device_memory_allocation_map_ = nullptr;

    for (int d = 0; d < device_count_; ++d) {
      checkError(cuCtxDestroy(device_contexts_[d]));
    }
  } catch (const CudaErrorException& e) {
    if (e.getStatus() == CUDA_ERROR_DEINITIALIZED) {
      // TODO(adb / asuhan): Verify cuModuleUnload removes the context
      return;
    }
    LOG(ERROR) << "CUDA Error: " << e.what();
  } catch (const std::runtime_error& e) {
    LOG(ERROR) << "CUDA Error: " << e.what();
  }
}

void CudaMgr::startBackgroundJumpBufferAllocation() {
  if (jump_buffer_transfer_mgr_) {
    jump_buffer_transfer_mgr_->startBackgroundAllocations();
  }
}

size_t CudaMgr::computePaddedBufferSize(size_t buf_size, size_t granularity) const {
  if (granularity == 0 ||
      buf_size > std::numeric_limits<size_t>::max() - (granularity - size_t(1))) {
    throw std::overflow_error("CUDA allocation size overflow");
  }
  return (((buf_size + (granularity - 1)) / granularity) * granularity);
}

size_t CudaMgr::getGranularity(const int device_num) const {
  CHECK_GE(device_num, 0);
  CHECK_LT(device_num, device_count_);
  CUmemAllocationProp allocation_prop{};
  allocation_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  allocation_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  allocation_prop.location.id = device_properties_[device_num].device;
  size_t granularity{};
  checkError(cuMemGetAllocationGranularity(
      &granularity, &allocation_prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
  return granularity;
}

void CudaMgr::synchronizeDevices() const {
  for (int d = 0; d < device_count_; ++d) {
    synchronizeDevice(d);
  }
}

void CudaMgr::synchronizeDevice(const int device_num) const {
  CHECK_GE(device_num, 0);
  CHECK_LT(device_num, device_count_);
  CUcontext original_context{nullptr};
  checkError(cuCtxGetCurrent(&original_context));
  ScopeGuard restore_context = [original_context] {
    const auto status = cuCtxSetCurrent(original_context);
    if (status != CUDA_SUCCESS && status != CUDA_ERROR_DEINITIALIZED) {
      LOG(ERROR) << "Failed to restore CUDA context after device synchronization: "
                 << error_message(status);
    }
  };
  setContext(device_num);
  checkError(cuCtxSynchronize());
}

void CudaMgr::synchronizeStream(CUstream cuda_stream) const {
  if (!cuda_stream) {
    synchronizeDevices();
    return;
  }
  checkError(cuStreamSynchronize(cuda_stream));
}

CUstream CudaMgr::getDeviceTransferStream(const int device_num) const {
  CHECK_GE(device_num, 0);
  CHECK_LT(device_num, device_count_);
  CHECK_EQ(device_transfer_streams_.size(), static_cast<size_t>(device_count_));
  CHECK(device_transfer_streams_[device_num]);
  return device_transfer_streams_[device_num];
}

bool CudaMgr::canAccessPeer(const int dest_device_num, const int src_device_num) const {
  CHECK_GE(dest_device_num, 0);
  CHECK_LT(dest_device_num, device_count_);
  CHECK_GE(src_device_num, 0);
  CHECK_LT(src_device_num, device_count_);

  if (dest_device_num == src_device_num) {
    return true;
  }

  int can_access_peer{0};
  checkError(cuDeviceCanAccessPeer(&can_access_peer,
                                   device_properties_[dest_device_num].device,
                                   device_properties_[src_device_num].device));
  return can_access_peer != 0;
}

void CudaMgr::copyHostToDevice(int8_t* device_ptr,
                               const int8_t* host_ptr,
                               const size_t num_bytes,
                               const int device_num,
                               std::optional<std::string_view> tag,
                               CUstream cuda_stream) {
  bool jump_buffer_used{false};
  if (jump_buffer_transfer_mgr_->shouldUseForHostToDeviceTransfer(num_bytes)) {
    jump_buffer_used = jump_buffer_transfer_mgr_->copyHostToDevice(
        device_ptr, host_ptr, num_bytes, device_num, cuda_stream);
  }

  if (!jump_buffer_used) {
    copyHostToDeviceDirect(device_ptr, host_ptr, num_bytes, device_num, tag, cuda_stream);
    return;
  }

  if (log_memory_activity_ && tag) {
    VLOG(1) << "CUDA H2D, tag: " << *tag
            << ", host address: " << static_cast<const void*>(host_ptr)
            << ", size: " << num_bytes << ", device: " << device_num
            << ",  device address: " << static_cast<const void*>(device_ptr);
  }
}

void CudaMgr::copyHostToDeviceDirect(int8_t* device_ptr,
                                     const int8_t* host_ptr,
                                     const size_t num_bytes,
                                     const int device_num,
                                     std::optional<std::string_view> tag,
                                     CUstream cuda_stream) {
  (void)tag;
  setContext(device_num);
  if (!cuda_stream) {
    checkError(
        cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(device_ptr), host_ptr, num_bytes));
  } else {
    checkError(cuMemcpyHtoDAsync(
        reinterpret_cast<CUdeviceptr>(device_ptr), host_ptr, num_bytes, cuda_stream));
    checkError(cuStreamSynchronize(cuda_stream));
  }
}

void CudaMgr::copyHostToDevice2DDirect(int8_t* device_ptr,
                                       const size_t destination_pitch,
                                       const int8_t* host_ptr,
                                       const size_t source_pitch,
                                       const size_t width_bytes,
                                       const size_t height,
                                       const int device_num,
                                       std::optional<std::string_view> tag,
                                       CUstream cuda_stream) {
  (void)tag;
  if (width_bytes == 0 || height == 0) {
    return;
  }

  setContext(device_num);
  CUDA_MEMCPY2D copy{};
  copy.srcMemoryType = CU_MEMORYTYPE_HOST;
  copy.srcHost = host_ptr;
  copy.srcPitch = source_pitch;
  copy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
  copy.dstDevice = reinterpret_cast<CUdeviceptr>(device_ptr);
  copy.dstPitch = destination_pitch;
  copy.WidthInBytes = width_bytes;
  copy.Height = height;

  if (!cuda_stream) {
    checkError(cuMemcpy2D(&copy));
  } else {
    checkError(cuMemcpy2DAsync(&copy, cuda_stream));
    checkError(cuStreamSynchronize(cuda_stream));
  }
}

bool CudaMgr::copyHostToDeviceFromPinnedProducer(
    int8_t* device_ptr,
    const size_t num_bytes,
    const int device_num,
    std::optional<std::string_view> tag,
    const std::function<void(int8_t* host_ptr, size_t num_bytes, size_t offset)>&
        producer,
    CUstream cuda_stream) {
  (void)tag;
  const auto transfer_used =
      jump_buffer_transfer_mgr_->copyHostToDeviceFromPinnedProducer(
          device_ptr, num_bytes, device_num, cuda_stream, producer);
  return transfer_used;
}

int CudaMgr::getDeviceNumFromDevicePtr(CUdeviceptr cu_device_ptr,
                                       const size_t allocated_mem_bytes) {
  std::lock_guard<std::mutex> device_lock(device_mutex_);
  auto const [allocation_base, allocation] =
      getDeviceMemoryAllocationMap().getAllocation(cu_device_ptr);
  CHECK_GE(cu_device_ptr, allocation_base);
  const auto allocation_offset = cu_device_ptr - allocation_base;
  CHECK_LE(allocation_offset, allocation.size);
  CHECK_LE(allocated_mem_bytes, allocation.size - allocation_offset);
  CHECK_GE(allocation.device_num, 0);
  return allocation.device_num;
}

bool CudaMgr::isDeviceMemoryPointer(CUdeviceptr cu_device_ptr,
                                    const size_t allocated_mem_bytes) {
  std::lock_guard<std::mutex> device_lock(device_mutex_);
  return getDeviceMemoryAllocationMap().containsAllocation(cu_device_ptr,
                                                           allocated_mem_bytes);
}

void CudaMgr::copyDeviceToHost(int8_t* host_ptr,
                               const int8_t* device_ptr,
                               const size_t num_bytes,
                               std::optional<std::string_view> tag,
                               CUstream cuda_stream) {
  auto const cu_device_ptr = reinterpret_cast<CUdeviceptr>(device_ptr);
  int const device_num = getDeviceNumFromDevicePtr(cu_device_ptr, num_bytes);
  copyDeviceToHost(host_ptr, device_ptr, num_bytes, device_num, tag, cuda_stream);
}

void CudaMgr::copyDeviceToHost(int8_t* host_ptr,
                               const int8_t* device_ptr,
                               const size_t num_bytes,
                               const int device_num,
                               std::optional<std::string_view> tag,
                               CUstream cuda_stream) {
  auto const cu_device_ptr = reinterpret_cast<CUdeviceptr>(device_ptr);
  bool jump_buffer_used{false};
  if (jump_buffer_transfer_mgr_->shouldUseForDeviceToHostTransfer(num_bytes)) {
    jump_buffer_used = jump_buffer_transfer_mgr_->copyDeviceToHost(
        host_ptr, device_ptr, num_bytes, device_num, cuda_stream);
  }

  if (!jump_buffer_used) {
    setContext(device_num);
    if (!cuda_stream) {
      checkError(cuMemcpyDtoH(host_ptr, cu_device_ptr, num_bytes));
    } else {
      checkError(cuMemcpyDtoHAsync(host_ptr, cu_device_ptr, num_bytes, cuda_stream));
      checkError(cuStreamSynchronize(cuda_stream));
    }
  }

  if (log_memory_activity_ && tag) {
    VLOG(1) << "CUDA D2H, tag: " << *tag
            << ", host address: " << static_cast<void*>(host_ptr)
            << ", size: " << num_bytes << ", device: " << device_num
            << ", device address: " << static_cast<const void*>(device_ptr);
  }
}

void CudaMgr::enablePeerAccess(const int dest_device_num,
                               const int src_device_num) const {
  CHECK_GE(dest_device_num, 0);
  CHECK_LT(dest_device_num, device_count_);
  CHECK_GE(src_device_num, 0);
  CHECK_LT(src_device_num, device_count_);

  if (dest_device_num == src_device_num) {
    return;
  }

  if (!canAccessPeer(dest_device_num, src_device_num)) {
    throw std::runtime_error("CUDA reports no peer access from device " +
                             std::to_string(src_device_num) + " to device " +
                             std::to_string(dest_device_num));
  }

  setContext(dest_device_num);
  auto const status = cuCtxEnablePeerAccess(device_contexts_[src_device_num], 0);
  if (status != CUDA_SUCCESS && status != CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED) {
    checkError(status);
  }
}

void CudaMgr::ensurePeerAccess(const int dest_device_num,
                               const int src_device_num) const {
  enablePeerAccess(dest_device_num, src_device_num);
}

bool CudaMgr::ensurePeerAccessToDevicePtr(const int dest_device_num,
                                          const int src_device_num,
                                          const int8_t* device_ptr,
                                          const size_t num_bytes) {
  if (num_bytes == 0 || dest_device_num == src_device_num) {
    return true;
  }
  CHECK_GE(dest_device_num, 0);
  CHECK_LT(dest_device_num, device_count_);
  CHECK_GE(src_device_num, 0);
  CHECK_LT(src_device_num, device_count_);
  if (!canAccessPeer(dest_device_num, src_device_num)) {
    return false;
  }
  if (!canAccessPeerMemoryFromKernel(dest_device_num, src_device_num)) {
    return false;
  }

  CUcontext original_context{nullptr};
  checkError(cuCtxGetCurrent(&original_context));
  ScopeGuard restore_context = [original_context] {
    const auto status = cuCtxSetCurrent(original_context);
    if (status != CUDA_SUCCESS && status != CUDA_ERROR_DEINITIALIZED) {
      LOG(ERROR) << "Failed to restore CUDA context after enabling peer access: "
                 << error_message(status);
    }
  };
  const auto cu_device_ptr = reinterpret_cast<CUdeviceptr>(device_ptr);
  std::lock_guard<std::mutex> map_lock(device_mutex_);
  const auto& allocation_map = getDeviceMemoryAllocationMap().getMap();
  auto allocation_it = allocation_map.upper_bound(cu_device_ptr);
  if (allocation_it == allocation_map.begin()) {
    return false;
  }
  --allocation_it;
  const auto allocation_base = allocation_it->first;
  const auto& allocation = allocation_it->second;
  if (cu_device_ptr < allocation_base || allocation.device_num != src_device_num) {
    return false;
  }
  const auto allocation_offset = cu_device_ptr - allocation_base;
  if (allocation_offset > allocation.size ||
      num_bytes > allocation.size - allocation_offset) {
    return false;
  }

  enablePeerAccess(dest_device_num, src_device_num);

  setContext(src_device_num);
  CUmemAccessDesc access_desc{};
  access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access_desc.location.id = device_properties_[dest_device_num].device;
  access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READ;
  const auto status = cuMemSetAccess(
      allocation_base, allocation.size, &access_desc, static_cast<size_t>(1));
  if (status != CUDA_SUCCESS) {
    LOG(WARNING) << "Failed to grant peer access to CUDA VMM allocation: source_device="
                 << src_device_num << " destination_device=" << dest_device_num
                 << " allocation_base=" << reinterpret_cast<void*>(allocation_base)
                 << " allocation_size=" << allocation.size
                 << " status=" << static_cast<int>(status);
    return false;
  }
  return true;
}

bool CudaMgr::canAccessPeerMemoryFromKernel(const int dest_device_num,
                                            const int src_device_num) {
  if (dest_device_num == src_device_num) {
    return true;
  }
  CHECK_GE(dest_device_num, 0);
  CHECK_LT(dest_device_num, device_count_);
  CHECK_GE(src_device_num, 0);
  CHECK_LT(src_device_num, device_count_);

  const auto cache_key = std::make_pair(dest_device_num, src_device_num);
  std::lock_guard<std::mutex> map_lock(device_mutex_);
  if (const auto cached_it = peer_kernel_access_cache_.find(cache_key);
      cached_it != peer_kernel_access_cache_.end()) {
    return cached_it->second;
  }

  int access_supported{0};
  auto status = cuDeviceGetP2PAttribute(&access_supported,
                                        CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED,
                                        device_properties_[dest_device_num].device,
                                        device_properties_[src_device_num].device);
  if (status != CUDA_SUCCESS || !access_supported) {
    peer_kernel_access_cache_[cache_key] = false;
    return false;
  }

  // Peer-copy capability is weaker than safe peer dereference from a consumer
  // kernel. On PCIe-only systems CUDA can report access support while remote SM
  // loads still produce corrupt values, so require the stronger native-atomic
  // capability before exposing remote payload pointers to generated kernels.
  int native_atomic_supported{0};
  status = cuDeviceGetP2PAttribute(&native_atomic_supported,
                                   CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED,
                                   device_properties_[dest_device_num].device,
                                   device_properties_[src_device_num].device);
  if (status != CUDA_SUCCESS || !native_atomic_supported) {
    peer_kernel_access_cache_[cache_key] = false;
    return false;
  }

  peer_kernel_access_cache_[cache_key] = true;
  return true;
}

std::optional<int8_t*> CudaMgr::registerMappedHostMemory(const int8_t* host_ptr,
                                                         const size_t num_bytes,
                                                         const int device_num) {
  if (!host_ptr || num_bytes == 0) {
    return std::nullopt;
  }
  CHECK_GE(device_num, 0);
  CHECK_LT(device_num, device_count_);

  CUcontext original_context{nullptr};
  checkError(cuCtxGetCurrent(&original_context));
  ScopeGuard restore_context = [original_context] {
    const auto status = cuCtxSetCurrent(original_context);
    if (status != CUDA_SUCCESS && status != CUDA_ERROR_DEINITIALIZED) {
      LOG(ERROR) << "Failed to restore CUDA context after mapping host memory: "
                 << error_message(status);
    }
  };
  setContext(device_num);

  const MappedHostMemoryKey key{host_ptr, num_bytes};
  {
    std::lock_guard<std::mutex> lock(mapped_host_memory_mutex_);
    auto [it, inserted] = mapped_host_memory_.try_emplace(key);
    if (inserted) {
      const auto status =
          cuMemHostRegister(const_cast<int8_t*>(host_ptr),
                            num_bytes,
                            CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_DEVICEMAP);
      if (status != CUDA_SUCCESS) {
        mapped_host_memory_.erase(it);
        VLOG(1) << "Failed to register mapped host memory: host_ptr="
                << static_cast<const void*>(host_ptr) << " bytes=" << num_bytes
                << " device=" << device_num << " error=" << error_message(status);
        return std::nullopt;
      }
    }
    ++it->second.ref_count;
  }

  CUdeviceptr device_ptr{0};
  const auto status =
      cuMemHostGetDevicePointer(&device_ptr, const_cast<int8_t*>(host_ptr), 0);
  if (status != CUDA_SUCCESS) {
    unregisterMappedHostMemory(host_ptr, num_bytes);
    VLOG(1) << "Failed to get mapped host device pointer: host_ptr="
            << static_cast<const void*>(host_ptr) << " bytes=" << num_bytes
            << " device=" << device_num << " error=" << error_message(status);
    return std::nullopt;
  }

  return reinterpret_cast<int8_t*>(device_ptr);
}

void CudaMgr::unregisterMappedHostMemory(const int8_t* host_ptr, const size_t num_bytes) {
  if (!host_ptr || num_bytes == 0) {
    return;
  }

  std::lock_guard<std::mutex> lock(mapped_host_memory_mutex_);
  const MappedHostMemoryKey key{host_ptr, num_bytes};
  auto it = mapped_host_memory_.find(key);
  if (it == mapped_host_memory_.end()) {
    return;
  }
  CHECK_GT(it->second.ref_count, size_t(0));
  --it->second.ref_count;
  if (it->second.ref_count != 0) {
    return;
  }
  const auto status = cuMemHostUnregister(const_cast<int8_t*>(host_ptr));
  if (status != CUDA_SUCCESS && status != CUDA_ERROR_DEINITIALIZED) {
    // The host allocation can be released immediately after this call. Do not retain a
    // software entry that could later make a different allocation at the same address
    // look registered and expose a stale device pointer.
    LOG(WARNING) << "Failed to unregister mapped host memory: host_ptr="
                 << static_cast<const void*>(host_ptr) << " bytes=" << num_bytes
                 << " error=" << error_message(status);
  }
  mapped_host_memory_.erase(it);
}

void CudaMgr::copyDeviceToDeviceOnDevice(int8_t* dest_ptr,
                                         const int8_t* src_ptr,
                                         const size_t num_bytes,
                                         const int device_num,
                                         CUstream cuda_stream,
                                         const bool synchronize) const {
  if (num_bytes == 0) {
    return;
  }

  setContext(device_num);
  if (!cuda_stream) {
    checkError(cuMemcpy(reinterpret_cast<CUdeviceptr>(dest_ptr),
                        reinterpret_cast<CUdeviceptr>(src_ptr),
                        num_bytes));
  } else {
    checkError(cuMemcpyAsync(reinterpret_cast<CUdeviceptr>(dest_ptr),
                             reinterpret_cast<CUdeviceptr>(src_ptr),
                             num_bytes,
                             cuda_stream));
    if (synchronize) {
      checkError(cuStreamSynchronize(cuda_stream));
    }
  }
}

int8_t* CudaMgr::allocatePeerCopyableDeviceMem(const size_t num_bytes,
                                               const int device_num) {
  CHECK_GT(num_bytes, size_t(0));
  CHECK_GE(device_num, 0);
  CHECK_LT(device_num, device_count_);

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11020
  std::lock_guard<std::mutex> map_lock(device_mutex_);
  for (int access_device_num = 0; access_device_num < device_count_;
       ++access_device_num) {
    if (access_device_num != device_num && canAccessPeer(access_device_num, device_num)) {
      enablePeerAccess(access_device_num, device_num);
    }
  }
  setContext(device_num);

  CUdeviceptr device_ptr{};
  CUstream stream{};
  checkError(cuStreamCreate(&stream, CU_STREAM_DEFAULT));
  ScopeGuard destroy_stream = [&] {
    if (!stream) {
      return;
    }
    const auto status = cuStreamDestroy(stream);
    if (status != CUDA_SUCCESS && status != CUDA_ERROR_DEINITIALIZED) {
      LOG(ERROR) << "Failed to destroy peer-copy allocation stream: device=" << device_num
                 << " error=" << error_message(status);
    }
  };
  bool allocation_live{false};
  try {
    checkError(cuMemAllocAsync(&device_ptr, num_bytes, stream));
    allocation_live = true;
    checkError(cuStreamSynchronize(stream));
    const auto inserted =
        peer_copyable_device_allocations_
            .emplace(device_ptr, PeerCopyableDeviceAllocation{num_bytes, device_num})
            .second;
    if (!inserted) {
      throw std::runtime_error("Duplicate peer-copyable CUDA allocation address");
    }
    allocation_live = false;
  } catch (...) {
    if (allocation_live) {
      const auto free_status = cuMemFreeAsync(device_ptr, stream);
      if (free_status == CUDA_SUCCESS) {
        const auto sync_status = cuStreamSynchronize(stream);
        if (sync_status != CUDA_SUCCESS && sync_status != CUDA_ERROR_DEINITIALIZED) {
          LOG(ERROR) << "Failed to synchronize peer-copy allocation rollback: device="
                     << device_num << " error=" << error_message(sync_status);
        }
      } else if (free_status != CUDA_ERROR_DEINITIALIZED) {
        LOG(ERROR) << "Failed to roll back peer-copy allocation: device=" << device_num
                   << " error=" << error_message(free_status);
      }
    }
    throw;
  }

  if (log_memory_activity_) {
    VLOG(1) << "Allocate peer-copyable GPU memory: address: "
            << reinterpret_cast<void*>(device_ptr) << ", requested: " << num_bytes
            << " bytes, allocated: " << num_bytes << ", device id: " << device_num;
  }
  return reinterpret_cast<int8_t*>(device_ptr);
#else
  throw std::runtime_error("Peer-copyable GPU staging buffers require CUDA 11.2+");
#endif
}

void CudaMgr::freePeerCopyableDeviceMem(int8_t* device_ptr, const int device_num) {
  if (!device_ptr) {
    return;
  }
  CHECK_GE(device_num, 0);
  CHECK_LT(device_num, device_count_);

  std::lock_guard<std::mutex> map_lock(device_mutex_);
  setContext(device_num);
  auto const cu_device_ptr = reinterpret_cast<CUdeviceptr>(device_ptr);
  auto const it = peer_copyable_device_allocations_.find(cu_device_ptr);
  CHECK(it != peer_copyable_device_allocations_.end());
  CHECK_EQ(it->second.device_num, device_num);
  const auto allocation = it->second;

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11020
  CUstream stream{};
  checkError(cuStreamCreate(&stream, CU_STREAM_DEFAULT));
  ScopeGuard destroy_stream = [&] {
    const auto status = cuStreamDestroy(stream);
    if (status != CUDA_SUCCESS && status != CUDA_ERROR_DEINITIALIZED) {
      LOG(ERROR) << "Failed to destroy peer-copy deallocation stream: device="
                 << device_num << " error=" << error_message(status);
    }
  };
  checkError(cuMemFreeAsync(cu_device_ptr, stream));
  // Once the free is enqueued, ownership has transferred to CUDA. Remove the
  // allocation before waiting so a synchronization error cannot cause a later
  // double-free.
  peer_copyable_device_allocations_.erase(it);
  const auto sync_status = cuStreamSynchronize(stream);
  if (sync_status != CUDA_SUCCESS && sync_status != CUDA_ERROR_DEINITIALIZED) {
    LOG(ERROR) << "Failed to synchronize peer-copy deallocation: device=" << device_num
               << " error=" << error_message(sync_status);
  }
#else
  throw std::runtime_error("Peer-copyable GPU staging buffers require CUDA 11.2+");
#endif

  if (log_memory_activity_) {
    VLOG(1) << "Deallocate peer-copyable GPU memory: address: "
            << static_cast<void*>(device_ptr) << ", size: " << allocation.size
            << ", device id: " << device_num;
  }
}

CudaMgr::PeerCopyStagingBuffers& CudaMgr::getPeerCopyStagingBuffers(
    const int src_device_num,
    const int dest_device_num) {
  CHECK_GE(src_device_num, 0);
  CHECK_LT(src_device_num, device_count_);
  CHECK_GE(dest_device_num, 0);
  CHECK_LT(dest_device_num, device_count_);

  std::lock_guard<std::mutex> map_lock(device_mutex_);
  auto const key = std::make_pair(src_device_num, dest_device_num);
  auto& staging_buffers = peer_copy_staging_buffers_[key];
  if (!staging_buffers) {
    staging_buffers = std::make_unique<PeerCopyStagingBuffers>();
    staging_buffers->src_device_num = src_device_num;
    staging_buffers->dest_device_num = dest_device_num;
  }
  return *staging_buffers;
}

void CudaMgr::ensurePeerCopyStagingBuffers(PeerCopyStagingBuffers& staging_buffers,
                                           const size_t required_size) {
  CHECK_GT(required_size, size_t(0));
  if (staging_buffers.size >= required_size) {
    return;
  }

  if (staging_buffers.src_buffer) {
    freePeerCopyableDeviceMem(staging_buffers.src_buffer, staging_buffers.src_device_num);
    staging_buffers.src_buffer = nullptr;
  }
  if (staging_buffers.dest_buffer) {
    freePeerCopyableDeviceMem(staging_buffers.dest_buffer,
                              staging_buffers.dest_device_num);
    staging_buffers.dest_buffer = nullptr;
  }
  staging_buffers.size = 0;

  int8_t* src_buffer{nullptr};
  int8_t* dest_buffer{nullptr};
  try {
    src_buffer =
        allocatePeerCopyableDeviceMem(required_size, staging_buffers.src_device_num);
    dest_buffer =
        allocatePeerCopyableDeviceMem(required_size, staging_buffers.dest_device_num);
  } catch (...) {
    if (src_buffer) {
      freePeerCopyableDeviceMem(src_buffer, staging_buffers.src_device_num);
    }
    if (dest_buffer) {
      freePeerCopyableDeviceMem(dest_buffer, staging_buffers.dest_device_num);
    }
    throw;
  }

  staging_buffers.src_buffer = src_buffer;
  staging_buffers.dest_buffer = dest_buffer;
  staging_buffers.size = required_size;
}

void CudaMgr::releasePeerCopyStagingBuffers() {
  std::vector<std::pair<int8_t*, int>> buffers_to_free;
  {
    std::lock_guard<std::mutex> map_lock(device_mutex_);
    for (auto& entry : peer_copy_staging_buffers_) {
      auto& staging_buffers = entry.second;
      CHECK(staging_buffers);
      if (staging_buffers->src_buffer) {
        buffers_to_free.emplace_back(staging_buffers->src_buffer,
                                     staging_buffers->src_device_num);
        staging_buffers->src_buffer = nullptr;
      }
      if (staging_buffers->dest_buffer) {
        buffers_to_free.emplace_back(staging_buffers->dest_buffer,
                                     staging_buffers->dest_device_num);
        staging_buffers->dest_buffer = nullptr;
      }
      staging_buffers->size = 0;
    }
    peer_copy_staging_buffers_.clear();
  }

  for (auto const& [buffer, device_num] : buffers_to_free) {
    freePeerCopyableDeviceMem(buffer, device_num);
  }
}

void CudaMgr::copyPeerToPeer(int8_t* dest_ptr,
                             const int8_t* src_ptr,
                             const size_t num_bytes,
                             const int dest_device_num,
                             const int src_device_num,
                             std::optional<std::string_view> tag,
                             CUstream cuda_stream,
                             const bool synchronize) {
  (void)tag;
  CHECK_GT(num_bytes, size_t(0));
  CHECK_GE(dest_device_num, 0);
  CHECK_LT(dest_device_num, device_count_);
  CHECK_GE(src_device_num, 0);
  CHECK_LT(src_device_num, device_count_);

  if (src_device_num == dest_device_num) {
    copyDeviceToDeviceOnDevice(
        dest_ptr, src_ptr, num_bytes, src_device_num, cuda_stream, synchronize);
  } else {
    setContext(dest_device_num);
    if (!cuda_stream) {
      CUstream local_stream{};
      checkError(cuStreamCreate(&local_stream, CU_STREAM_DEFAULT));
      try {
        checkError(cuMemcpyPeerAsync(reinterpret_cast<CUdeviceptr>(dest_ptr),
                                     device_contexts_[dest_device_num],
                                     reinterpret_cast<CUdeviceptr>(src_ptr),
                                     device_contexts_[src_device_num],
                                     num_bytes,
                                     local_stream));
        checkError(cuStreamSynchronize(local_stream));
        checkError(cuStreamDestroy(local_stream));
      } catch (...) {
        cuStreamDestroy(local_stream);
        throw;
      }
    } else {
      checkError(cuMemcpyPeerAsync(reinterpret_cast<CUdeviceptr>(dest_ptr),
                                   device_contexts_[dest_device_num],
                                   reinterpret_cast<CUdeviceptr>(src_ptr),
                                   device_contexts_[src_device_num],
                                   num_bytes,
                                   cuda_stream));
      if (synchronize) {
        checkError(cuStreamSynchronize(cuda_stream));
      }
    }
  }
}

void CudaMgr::copyDeviceToDevice(int8_t* dest_ptr,
                                 int8_t* src_ptr,
                                 const size_t num_bytes,
                                 const int dest_device_num,
                                 const int src_device_num,
                                 std::optional<std::string_view> tag,
                                 CUstream cuda_stream,
                                 const bool synchronize) {
  if (num_bytes == 0) {
    return;
  }

  // dest_device_num and src_device_num are the device numbers relative to start_gpu_
  // (real_device_num - start_gpu_)
  if (src_device_num == dest_device_num) {
    copyDeviceToDeviceOnDevice(
        dest_ptr, src_ptr, num_bytes, src_device_num, cuda_stream, synchronize);
  } else {
    const bool peer_access_available = canAccessPeer(dest_device_num, src_device_num);
    if (g_peer_copy_mode == kPeerCopyModeDirect && peer_access_available) {
      copyPeerToPeer(dest_ptr,
                     src_ptr,
                     num_bytes,
                     dest_device_num,
                     src_device_num,
                     tag,
                     cuda_stream,
                     synchronize);
    } else if (g_peer_copy_mode == kPeerCopyModeStaged && peer_access_available &&
               g_peer_copy_staging_buffer_size > 0) {
      try {
        copyDeviceToDeviceViaPeerStaging(dest_ptr,
                                         src_ptr,
                                         num_bytes,
                                         dest_device_num,
                                         src_device_num,
                                         tag,
                                         cuda_stream);
      } catch (const CudaErrorException& error) {
        if (error.getStatus() != CUDA_ERROR_OUT_OF_MEMORY) {
          throw;
        }
        VLOG(1) << "Peer staging allocation failed; using host staging: source_device="
                << src_device_num << " destination_device=" << dest_device_num
                << " bytes=" << num_bytes << " error=" << error.what();
        copyDeviceToDeviceViaHost(dest_ptr,
                                  src_ptr,
                                  num_bytes,
                                  dest_device_num,
                                  src_device_num,
                                  tag,
                                  cuda_stream);
      } catch (const std::bad_alloc& error) {
        VLOG(1) << "Peer staging bookkeeping allocation failed; using host staging: "
                << "source_device=" << src_device_num
                << " destination_device=" << dest_device_num << " bytes=" << num_bytes
                << " error=" << error.what();
        copyDeviceToDeviceViaHost(dest_ptr,
                                  src_ptr,
                                  num_bytes,
                                  dest_device_num,
                                  src_device_num,
                                  tag,
                                  cuda_stream);
      }
    } else {
      copyDeviceToDeviceViaHost(dest_ptr,
                                src_ptr,
                                num_bytes,
                                dest_device_num,
                                src_device_num,
                                tag,
                                cuda_stream);
    }
    return;
  }
  if (log_memory_activity_ && tag) {
    VLOG(1) << "CUDA D2D, tag: " << *tag << ", source device id: " << src_device_num
            << ", destination device id: " << dest_device_num
            << ", source address: " << static_cast<void*>(src_ptr)
            << ", size: " << num_bytes
            << ", destination address: " << static_cast<void*>(dest_ptr);
  }
}

void CudaMgr::copyDeviceToDeviceViaHost(int8_t* dest_ptr,
                                        int8_t* src_ptr,
                                        const size_t num_bytes,
                                        const int dest_device_num,
                                        const int src_device_num,
                                        std::optional<std::string_view> tag,
                                        CUstream cuda_stream) {
  if (num_bytes == 0) {
    return;
  }

  // A caller may have placed a producer-ready event wait on the destination stream.
  // This bridge reads the source from another context, so honor that wait before
  // leaving the destination stream and starting the synchronous source D2H leg.
  if (cuda_stream) {
    setContext(dest_device_num);
    checkError(cuStreamSynchronize(cuda_stream));
  }

  constexpr size_t default_host_staging_buffer_size{64 * 1024 * 1024};
  const auto staging_buffer_size =
      std::min(num_bytes,
               g_peer_copy_staging_buffer_size > 0 ? g_peer_copy_staging_buffer_size
                                                   : default_host_staging_buffer_size);
  std::vector<int8_t> host_buffer(staging_buffer_size);
  size_t bytes_copied{0};
  while (bytes_copied < num_bytes) {
    const auto copy_size = std::min(staging_buffer_size, num_bytes - bytes_copied);
    // The caller-provided stream belongs to the destination context. The source D2H
    // leg must use a source-context operation before the destination H2D leg starts.
    copyDeviceToHost(host_buffer.data(),
                     src_ptr + bytes_copied,
                     copy_size,
                     src_device_num,
                     tag,
                     /*cuda_stream=*/0);
    copyHostToDevice(dest_ptr + bytes_copied,
                     host_buffer.data(),
                     copy_size,
                     dest_device_num,
                     tag,
                     cuda_stream);
    bytes_copied += copy_size;
  }
}

void CudaMgr::copyDeviceToDeviceViaPeerStaging(int8_t* dest_ptr,
                                               int8_t* src_ptr,
                                               const size_t num_bytes,
                                               const int dest_device_num,
                                               const int src_device_num,
                                               std::optional<std::string_view> tag,
                                               CUstream cuda_stream) {
  if (num_bytes == 0) {
    return;
  }

  CHECK_GE(dest_device_num, 0);
  CHECK_LT(dest_device_num, device_count_);
  CHECK_GE(src_device_num, 0);
  CHECK_LT(src_device_num, device_count_);

  if (src_device_num == dest_device_num) {
    copyDeviceToDeviceOnDevice(dest_ptr, src_ptr, num_bytes, src_device_num, cuda_stream);
    return;
  }

  CHECK(canAccessPeer(dest_device_num, src_device_num));
  CHECK_GT(g_peer_copy_staging_buffer_size, size_t(0));
  const size_t staging_buffer_size = std::min(num_bytes, g_peer_copy_staging_buffer_size);

  // As in the host bridge, the caller stream may carry the only ordering edge from
  // the source producer. The remaining staged operations are synchronous, so drain
  // that edge before switching to source-context work.
  if (cuda_stream) {
    setContext(dest_device_num);
    checkError(cuStreamSynchronize(cuda_stream));
  }

  auto& staging_buffers = getPeerCopyStagingBuffers(src_device_num, dest_device_num);
  std::lock_guard<std::mutex> staging_lock(staging_buffers.mutex);
  ensurePeerCopyStagingBuffers(staging_buffers, staging_buffer_size);
  auto src_staging_buffer = staging_buffers.src_buffer;
  auto dest_staging_buffer = staging_buffers.dest_buffer;
  CHECK(src_staging_buffer);
  CHECK(dest_staging_buffer);

  // Cross-device bridge copies use their own synchronous operations because a CUstream
  // belongs to one CUDA context, while this path touches both source and destination
  // contexts.
  size_t bytes_copied{0};
  while (bytes_copied < num_bytes) {
    const auto copy_size = std::min(staging_buffer_size, num_bytes - bytes_copied);
    copyDeviceToDeviceOnDevice(
        src_staging_buffer, src_ptr + bytes_copied, copy_size, src_device_num, 0);
    copyPeerToPeer(dest_staging_buffer,
                   src_staging_buffer,
                   copy_size,
                   dest_device_num,
                   src_device_num,
                   tag);
    copyDeviceToDeviceOnDevice(
        dest_ptr + bytes_copied, dest_staging_buffer, copy_size, dest_device_num, 0);
    bytes_copied += copy_size;
  }
}

void CudaMgr::loadGpuModuleData(CUmodule* module,
                                const void* image,
                                unsigned int num_options,
                                CUjit_option* options,
                                void** option_vals,
                                const int device_id) const {
  setContext(device_id);
  checkError(cuModuleLoadDataEx(module, image, num_options, options, option_vals));
}

void CudaMgr::unloadGpuModuleData(CUmodule* module, const int device_id) const {
  std::lock_guard<std::mutex> device_lock(device_mutex_);
  CHECK(module);
  setContext(device_id);
  try {
    auto code = cuModuleUnload(*module);
    // If the Cuda driver has already shut down, ignore the resulting errors.
    if (code != CUDA_ERROR_DEINITIALIZED) {
      checkError(code);
    }
  } catch (const std::runtime_error& e) {
    LOG(ERROR) << "CUDA Error: " << e.what();
  }
}

std::vector<CudaMgr::CudaMemoryUsage> CudaMgr::getCudaMemoryUsage() {
  std::vector<CudaMgr::CudaMemoryUsage> m;
  std::lock_guard<std::mutex> map_lock(device_mutex_);
  CUcontext cnow;
  checkError(cuCtxGetCurrent(&cnow));
  for (int device_num = 0; device_num < device_count_; ++device_num) {
    setContext(device_num);
    CudaMemoryUsage usage;
    cuMemGetInfo(&usage.free, &usage.total);
    m.push_back(usage);
  }
  cuCtxSetCurrent(cnow);
  return m;
}

std::string CudaMgr::getCudaMemoryUsageInString() {
  auto const device_mem_status = getCudaMemoryUsage();
  std::ostringstream oss;
  int device_id = 0;
  oss << "{ \"name\": \"GPU Memory Info\", ";
  for (auto& info : device_mem_status) {
    oss << "{\"device_id\": " << device_id++ << ", \"freeMB:\": " << info.free / 1048576.0
        << ", \"totalMB\": " << info.total / 1048576.0 << "} ";
  }
  oss << "}";
  return oss.str();
}

namespace {
bool is_arch_pascal_or_greater(int compute_major) {
  return compute_major >= 6;
}
}  // namespace

void CudaMgr::fillDeviceProperties() {
  device_properties_.resize(device_count_);
  cuDriverGetVersion(&gpu_driver_version_);
  for (int device_num = 0; device_num < device_count_; ++device_num) {
    checkError(
        cuDeviceGet(&device_properties_[device_num].device, device_num + start_gpu_));
    CUuuid cuda_uuid;
    checkError(cuDeviceGetUuid(&cuda_uuid, device_properties_[device_num].device));
    device_properties_[device_num].uuid = heavyai::UUID(cuda_uuid.bytes);
    checkError(cuDeviceGetAttribute(&device_properties_[device_num].computeMajor,
                                    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                                    device_properties_[device_num].device));
    checkError(cuDeviceGetAttribute(&device_properties_[device_num].computeMinor,
                                    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                                    device_properties_[device_num].device));
    checkError(cuDeviceTotalMem(&device_properties_[device_num].globalMem,
                                device_properties_[device_num].device));
    checkError(cuDeviceGetAttribute(&device_properties_[device_num].constantMem,
                                    CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY,
                                    device_properties_[device_num].device));
    checkError(
        cuDeviceGetAttribute(&device_properties_[device_num].sharedMemPerMP,
                             CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
                             device_properties_[device_num].device));
    if (g_enable_gpu_dynamic_smem &&
        is_arch_pascal_or_greater(device_properties_[device_num].computeMajor)) {
      checkError(
          cuDeviceGetAttribute(&device_properties_[device_num].sharedMemPerBlockOptIn,
                               CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
                               device_properties_[device_num].device));
    }
    checkError(cuDeviceGetAttribute(&device_properties_[device_num].sharedMemPerBlock,
                                    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
                                    device_properties_[device_num].device));
    checkError(cuDeviceGetAttribute(&device_properties_[device_num].numMPs,
                                    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                                    device_properties_[device_num].device));
    checkError(cuDeviceGetAttribute(&device_properties_[device_num].warpSize,
                                    CU_DEVICE_ATTRIBUTE_WARP_SIZE,
                                    device_properties_[device_num].device));
    checkError(cuDeviceGetAttribute(&device_properties_[device_num].maxThreadsPerBlock,
                                    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                                    device_properties_[device_num].device));
    checkError(cuDeviceGetAttribute(&device_properties_[device_num].maxRegistersPerBlock,
                                    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK,
                                    device_properties_[device_num].device));
    checkError(cuDeviceGetAttribute(&device_properties_[device_num].maxRegistersPerMP,
                                    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR,
                                    device_properties_[device_num].device));
    checkError(cuDeviceGetAttribute(&device_properties_[device_num].pciBusId,
                                    CU_DEVICE_ATTRIBUTE_PCI_BUS_ID,
                                    device_properties_[device_num].device));
    checkError(cuDeviceGetAttribute(&device_properties_[device_num].pciDeviceId,
                                    CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID,
                                    device_properties_[device_num].device));
    checkError(cuDeviceGetAttribute(&device_properties_[device_num].clockKhz,
                                    CU_DEVICE_ATTRIBUTE_CLOCK_RATE,
                                    device_properties_[device_num].device));
    checkError(cuDeviceGetAttribute(&device_properties_[device_num].memoryClockKhz,
                                    CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,
                                    device_properties_[device_num].device));
    checkError(cuDeviceGetAttribute(&device_properties_[device_num].memoryBusWidth,
                                    CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH,
                                    device_properties_[device_num].device));
    device_properties_[device_num].memoryBandwidthGBs =
        device_properties_[device_num].memoryClockKhz / 1000000.0 / 8.0 *
        device_properties_[device_num].memoryBusWidth;

    // capture memory allocation granularity
    device_properties_[device_num].allocationGranularity = getGranularity(device_num);
  }
  finishDevicePropertiesInitialization();

  min_shared_memory_per_block_for_all_devices =
      computeMinSharedMemoryPerBlockForAllDevices();
  min_num_mps_for_all_devices = computeMinNumMPsForAllDevices();
}

int8_t* CudaMgr::allocateDeviceMem(const size_t num_bytes,
                                   const int device_num,
                                   const bool is_slab) {
  std::lock_guard<std::mutex> map_lock(device_mutex_);
  setContext(device_num);

  CUdeviceptr device_ptr{};
  CUmemGenericAllocationHandle handle{};
  auto granularity = getGranularity(device_num);
  // reserve the actual memory
  auto padded_num_bytes = computePaddedBufferSize(num_bytes, granularity);
  auto status = cuMemAddressReserve(&device_ptr, padded_num_bytes, granularity, 0, 0);

  if (status == CUDA_SUCCESS) {
    // create a handle for the allocation
    CUmemAllocationProp allocation_prop{};
    allocation_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    allocation_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    allocation_prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    allocation_prop.location.id = device_num + start_gpu_;
    status = cuMemCreate(&handle, padded_num_bytes, &allocation_prop, 0);

    if (status == CUDA_SUCCESS) {
      // map the memory
      status = cuMemMap(device_ptr, padded_num_bytes, 0, handle, 0);

      if (status == CUDA_SUCCESS) {
        // set the memory access
        CUmemAccessDesc access_desc{};
        access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access_desc.location.id = device_num + start_gpu_;
        access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        status = cuMemSetAccess(device_ptr, padded_num_bytes, &access_desc, 1);
      }
    }
  }

  if (status != CUDA_SUCCESS) {
    // clean up in reverse order
    if (device_ptr && handle) {
      cuMemUnmap(device_ptr, padded_num_bytes);
    }
    if (handle) {
      cuMemRelease(handle);
    }
    if (device_ptr) {
      cuMemAddressFree(device_ptr, padded_num_bytes);
    }
    throw CudaErrorException(status);
  }
  // emplace in the map
  auto const& device_uuid = getDeviceProperties(device_num)->uuid;
  getDeviceMemoryAllocationMap().addAllocation(
      device_ptr, padded_num_bytes, handle, device_uuid, device_num, is_slab);
  // notify
  getDeviceMemoryAllocationMap().notifyMapChanged(device_uuid, is_slab);
  if (log_memory_activity_) {
    VLOG(1) << "Allocate GPU memory: address: " << reinterpret_cast<void*>(device_ptr)
            << ", requested: " << num_bytes << " bytes, allocated: " << padded_num_bytes
            << ", device id: " << device_num << ", slab? " << is_slab;
  }
  return reinterpret_cast<int8_t*>(device_ptr);
}

void CudaMgr::freeDeviceMem(int8_t* device_ptr) {
  // take lock
  std::lock_guard<std::mutex> map_lock(device_mutex_);
  // fetch and remove from map
  auto const cu_device_ptr = reinterpret_cast<CUdeviceptr>(device_ptr);
  auto allocation = getDeviceMemoryAllocationMap().removeAllocation(cu_device_ptr);
  // attempt to unmap, release, free
  auto status_unmap = cuMemUnmap(cu_device_ptr, allocation.size);
  auto status_release = cuMemRelease(allocation.handle);
  auto status_free = cuMemAddressFree(cu_device_ptr, allocation.size);
  // check for errors
  checkError(status_unmap);
  checkError(status_release);
  checkError(status_free);
  // notify
  getDeviceMemoryAllocationMap().notifyMapChanged(allocation.device_uuid,
                                                  allocation.is_slab);
  if (log_memory_activity_) {
    VLOG(1) << "Deallocate GPU memory: address: " << static_cast<void*>(device_ptr)
            << ", size: " << allocation.size << ", device id: " << allocation.device_num
            << ", slab? " << allocation.is_slab;
  }
}

void CudaMgr::zeroDeviceMem(int8_t* device_ptr,
                            const size_t num_bytes,
                            const int device_num,
                            CUstream cuda_stream) {
  setDeviceMem(device_ptr, 0, num_bytes, device_num, cuda_stream);
}

void CudaMgr::setDeviceMem(int8_t* device_ptr,
                           const unsigned char uc,
                           const size_t num_bytes,
                           const int device_num,
                           CUstream cuda_stream) {
  setContext(device_num);
  if (!cuda_stream) {
    checkError(cuMemsetD8(reinterpret_cast<CUdeviceptr>(device_ptr), uc, num_bytes));
  } else {
    checkError(cuMemsetD8Async(
        reinterpret_cast<CUdeviceptr>(device_ptr), uc, num_bytes, cuda_stream));
    checkError(cuStreamSynchronize(cuda_stream));
  }
}

/**
 * Returns true if all devices have Volta micro-architecture
 * Returns false, if there is any non-Volta device available.
 */
bool CudaMgr::isArchVoltaOrGreaterForAll() const {
  CHECK(device_properties_initialized_);
  for (int i = 0; i < device_count_; i++) {
    if (device_properties_[i].computeMajor < 7) {
      return false;
    }
  }
  return true;
}

/**
 * This function returns the minimum available dynamic shared memory that is available per
 * block for all GPU devices.
 */
size_t CudaMgr::computeMinSharedMemoryPerBlockForAllDevices() const {
  CHECK(device_properties_initialized_);
  int shared_mem_size = 0;
  for (int d = 0; d < device_count_; d++) {
    int size = g_enable_gpu_dynamic_smem ? device_properties_[d].sharedMemPerBlockOptIn
                                         : device_properties_[d].sharedMemPerBlock;
    shared_mem_size = (d == 0) ? size : std::min(shared_mem_size, size);
  }
  return shared_mem_size;
}

/**
 * This function returns the minimum number of multiprocessors (MPs, also known as SMs)
 * per device across all GPU devices
 */
size_t CudaMgr::computeMinNumMPsForAllDevices() const {
  CHECK(device_properties_initialized_);
  int num_mps = device_count_ > 0 ? device_properties_.front().numMPs : 0;
  for (int d = 1; d < device_count_; d++) {
    num_mps = std::min(num_mps, device_properties_[d].numMPs);
  }
  return num_mps;
}
void CudaMgr::createDeviceContexts() {
  CHECK(device_properties_initialized_);
  CHECK_EQ(device_contexts_.size(), size_t(0));
  device_contexts_.resize(device_count_);
  for (int d = 0; d < device_count_; ++d) {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 13000
    CUresult status = cuCtxCreate(
        &device_contexts_[d], nullptr, CU_CTX_MAP_HOST, device_properties_[d].device);
#else
    CUresult status =
        cuCtxCreate(&device_contexts_[d], CU_CTX_MAP_HOST, device_properties_[d].device);
#endif
    if (status != CUDA_SUCCESS) {
      // this is called from destructor so we need
      // to clean up
      // destroy all contexts up to this point
      for (int destroy_id = 0; destroy_id <= d; ++destroy_id) {
        try {
          checkError(cuCtxDestroy(device_contexts_[destroy_id]));
        } catch (const CudaErrorException& e) {
          LOG(ERROR) << "Failed to destroy CUDA context for device ID " << destroy_id
                     << " with " << e.what()
                     << ". CUDA contexts were being destroyed due to an error creating "
                        "CUDA context for device ID "
                     << d << " out of " << device_count_ << " (" << error_message(status)
                     << ").";
        }
      }
      // checkError will translate the message and throw
      checkError(status);
    }
  }
}

void CudaMgr::setContext(const int device_num) const {
  // deviceNum is the device number relative to startGpu (realDeviceNum - startGpu_)
  CHECK_LT(device_num, device_count_);
  set_context(device_contexts_, device_num);
}

int CudaMgr::getContext() const {
  CUcontext cnow;
  checkError(cuCtxGetCurrent(&cnow));
  if (cnow == NULL) {
    throw std::runtime_error("no cuda device context");
  }
  int device_num{0};
  for (auto& c : device_contexts_) {
    if (c == cnow) {
      return device_num;
    }
    ++device_num;
  }
  // TODO(sy): Change device_contexts_ to have O(1) lookup? (Or maybe not worth it.)
  throw std::runtime_error("invalid cuda device context");
}

void CudaMgr::logDeviceProperties() const {
  CHECK(device_properties_initialized_);
  LOG(INFO) << "CUDA Driver version: " << gpu_driver_version_;
  LOG(INFO) << "Using " << device_count_ << " Gpus.";
  if (device_count_ > 1) {
    LOG(INFO) << "CUDA peer access matrix, rows are destination GPUs and columns are "
                 "source GPUs; 1 means cuDeviceCanAccessPeer is available.";
    for (int dest = 0; dest < device_count_; ++dest) {
      std::string row;
      row.reserve(device_count_);
      for (int src = 0; src < device_count_; ++src) {
        row.push_back(dest == src ? '-' : (canAccessPeer(dest, src) ? '1' : '0'));
      }
      LOG(INFO) << "CUDA peer access dst_gpu=" << dest << " src_mask=" << row;
    }
  }
  for (int d = 0; d < device_count_; ++d) {
    VLOG(1) << "Device: " << device_properties_[d].device;
    VLOG(1) << "UUID: " << device_properties_[d].uuid;
    VLOG(1) << "Clock (khz): " << device_properties_[d].clockKhz;
    VLOG(1) << "Compute Major: " << device_properties_[d].computeMajor;
    VLOG(1) << "Compute Minor: " << device_properties_[d].computeMinor;
    VLOG(1) << "PCI bus id: " << device_properties_[d].pciBusId;
    VLOG(1) << "PCI deviceId id: " << device_properties_[d].pciDeviceId;
    VLOG(1) << "Per device global memory: "
            << device_properties_[d].globalMem / 1073741824.0 << " GB";
    VLOG(1) << "Memory clock (khz): " << device_properties_[d].memoryClockKhz;
    VLOG(1) << "Memory bandwidth: " << device_properties_[d].memoryBandwidthGBs
            << " GB/sec";

    VLOG(1) << "Constant Memory: " << device_properties_[d].constantMem;
    VLOG(1) << "Shared memory per multiprocessor: "
            << device_properties_[d].sharedMemPerMP;
    VLOG(1) << "Shared memory per block: " << device_properties_[d].sharedMemPerBlock;
    if (g_enable_gpu_dynamic_smem) {
      VLOG(1) << "Shared memory per block (Dynamic): "
              << device_properties_[d].sharedMemPerBlockOptIn;
    }
    VLOG(1) << "Number of MPs: " << device_properties_[d].numMPs;
    VLOG(1) << "Warp Size: " << device_properties_[d].warpSize;
    VLOG(1) << "Max threads per block: " << device_properties_[d].maxThreadsPerBlock;
    VLOG(1) << "Max registers per block: " << device_properties_[d].maxRegistersPerBlock;
    VLOG(1) << "Max register per MP: " << device_properties_[d].maxRegistersPerMP;
    VLOG(1) << "Memory bus width in bits: " << device_properties_[d].memoryBusWidth;
  }
}

void CudaMgr::checkError(CUresult status) const {
  check_error(status);
}

DeviceMemoryAllocationMap& CudaMgr::getDeviceMemoryAllocationMap() {
  CHECK(device_memory_allocation_map_);
  return *device_memory_allocation_map_;
}

int CudaMgr::exportHandle(const uint64_t handle) const {
  int fd{-1};
  checkError(cuMemExportToShareableHandle(
      &fd, handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));
  return fd;
}

void CudaMgr::enableMemoryActivityLog() {
  log_memory_activity_ = true;
}

bool CudaMgr::logMemoryActivity() const {
  return log_memory_activity_;
}

std::ostream& operator<<(std::ostream& os, NvidiaDeviceArch device_arch) {
  constexpr size_t array_size{8};
  constexpr char const* strings[array_size]{
      "Kepler", "Maxwell", "Pascal", "Volta", "Turing", "Ampere", "Ada", "Hopper"};

  auto index = static_cast<size_t>(device_arch);
  CHECK_LT(index, array_size);
  return os << strings[index];
}
}  // namespace CudaMgr_Namespace

std::string get_cuda_home(void) {
  static const char* CUDA_DEFAULT_PATH = "/usr/local/cuda";
  const char* env = nullptr;

  if (!(env = getenv("CUDA_HOME")) && !(env = getenv("CUDA_DIR"))) {
    // check if the default CUDA directory exists: /usr/local/cuda
    if (boost::filesystem::exists(boost::filesystem::path(CUDA_DEFAULT_PATH))) {
      env = CUDA_DEFAULT_PATH;
    }
  }

  if (env == nullptr) {
    LOG(WARNING) << "Could not find CUDA installation path: environment variables "
                    "CUDA_HOME or CUDA_DIR are not defined";
    return "";
  }

  // check if the CUDA directory is sensible:
  auto cuda_include_dir = env + std::string("/include");
  auto cuda_h_file = cuda_include_dir + "/cuda.h";
  if (!boost::filesystem::exists(boost::filesystem::path(cuda_h_file))) {
    LOG(WARNING) << "cuda.h does not exist in `" << cuda_include_dir << "`. Discarding `"
                 << env << "` as CUDA installation path.";
    return "";
  }

  return std::string(env);
}

std::string get_cuda_libdevice_dir(void) {
  static const char* CUDA_DEFAULT_PATH = "/usr/local/cuda";
  const char* env = nullptr;

  if (!(env = getenv("CUDA_HOME")) && !(env = getenv("CUDA_DIR"))) {
    // check if the default CUDA directory exists: /usr/local/cuda
    if (boost::filesystem::exists(boost::filesystem::path(CUDA_DEFAULT_PATH))) {
      env = CUDA_DEFAULT_PATH;
    }
  }

  if (env == nullptr) {
    LOG(WARNING) << "Could not find CUDA installation path: environment variables "
                    "CUDA_HOME or CUDA_DIR are not defined";
    return "";
  }

  // check if the CUDA directory is sensible:
  auto libdevice_dir = env + std::string("/nvvm/libdevice");
  auto libdevice_bc_file = libdevice_dir + "/libdevice.10.bc";
  if (!boost::filesystem::exists(boost::filesystem::path(libdevice_bc_file))) {
    LOG(WARNING) << "`" << libdevice_bc_file << "` does not exist. Discarding `" << env
                 << "` as CUDA installation path with libdevice.";
    return "";
  }

  return libdevice_dir;
}
