/*
 * Copyright 2018 OmniSci, Inc.
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

#include "CudaMgr.h"
#include <cuda_runtime.h>
#include <glog/logging.h>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <stdexcept>

namespace CudaMgr_Namespace {

CudaMgr::CudaMgr(const int num_gpus, const int start_gpu)
    : start_gpu_(start_gpu), max_shared_memory_for_all_(0) {
  checkError(cuInit(0));
  checkError(cuDeviceGetCount(&device_count_));

  if (num_gpus > 0) {  // numGpus <= 0 will just use number of gpus found
    CHECK_LE(num_gpus + start_gpu_, device_count_);
    device_count_ = std::min(device_count_, num_gpus);
  } else {
    // if we are using all gpus we cannot start on a gpu other than 0
    CHECK_EQ(start_gpu_, 0);
  }
  fillDeviceProperties();
  createDeviceContexts();
  printDeviceProperties();
}

CudaMgr::~CudaMgr() {
  try {
    // We don't want to remove the cudaMgr before all other processes have cleaned up.
    // This should be enforced by the lifetime policies, but take this lock to be safe.
    std::lock_guard<std::mutex> gpu_lock(device_cleanup_mutex_);

    synchronizeDevices();
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

void CudaMgr::synchronizeDevices() const {
  for (int d = 0; d < device_count_; ++d) {
    setContext(d);
    checkError(cuCtxSynchronize());
  }
}

void CudaMgr::copyHostToDevice(int8_t* device_ptr,
                               const int8_t* host_ptr,
                               const size_t num_bytes,
                               const int device_num) {
  setContext(device_num);
  checkError(
      cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(device_ptr), host_ptr, num_bytes));
}

void CudaMgr::copyDeviceToHost(int8_t* host_ptr,
                               const int8_t* device_ptr,
                               const size_t num_bytes,
                               const int device_num) {
  setContext(device_num);
  checkError(
      cuMemcpyDtoH(host_ptr, reinterpret_cast<const CUdeviceptr>(device_ptr), num_bytes));
}

void CudaMgr::copyDeviceToDevice(int8_t* dest_ptr,
                                 int8_t* src_ptr,
                                 const size_t num_bytes,
                                 const int dest_device_num,
                                 const int src_device_num) {
  // dest_device_num and src_device_num are the device numbers relative to start_gpu_
  // (real_device_num - start_gpu_)
  if (src_device_num == dest_device_num) {
    setContext(src_device_num);
    checkError(cuMemcpy(reinterpret_cast<CUdeviceptr>(dest_ptr),
                        reinterpret_cast<CUdeviceptr>(src_ptr),
                        num_bytes));
  } else {
    checkError(cuMemcpyPeer(reinterpret_cast<CUdeviceptr>(dest_ptr),
                            device_contexts_[dest_device_num],
                            reinterpret_cast<CUdeviceptr>(src_ptr),
                            device_contexts_[src_device_num],
                            num_bytes));  // will we always have peer?
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
  std::lock_guard<std::mutex> gpuLock(device_cleanup_mutex_);
  CHECK(module);

  setContext(device_id);
  try {
    checkError(cuModuleUnload(*module));
  } catch (const std::runtime_error& e) {
    LOG(ERROR) << "CUDA Error: " << e.what();
  }
}

void CudaMgr::fillDeviceProperties() {
  device_properties_.resize(device_count_);
  cudaDriverGetVersion(&gpu_driver_version_);
  for (int device_num = 0; device_num < device_count_; ++device_num) {
    checkError(
        cuDeviceGet(&device_properties_[device_num].device, device_num + start_gpu_));
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
  }
  max_shared_memory_for_all_ = computeMaxSharedMemoryForAll();
}

int8_t* CudaMgr::allocatePinnedHostMem(const size_t num_bytes) {
  setContext(0);
  void* host_ptr;
  checkError(cuMemHostAlloc(&host_ptr, num_bytes, CU_MEMHOSTALLOC_PORTABLE));
  return reinterpret_cast<int8_t*>(host_ptr);
}

int8_t* CudaMgr::allocateDeviceMem(const size_t num_bytes, const int device_num) {
  setContext(device_num);
  CUdeviceptr device_ptr;
  checkError(cuMemAlloc(&device_ptr, num_bytes));
  return reinterpret_cast<int8_t*>(device_ptr);
}

void CudaMgr::freePinnedHostMem(int8_t* host_ptr) {
  checkError(cuMemFreeHost(reinterpret_cast<void*>(host_ptr)));
}

void CudaMgr::freeDeviceMem(int8_t* device_ptr) {
  std::lock_guard<std::mutex> gpu_lock(device_cleanup_mutex_);

  checkError(cuMemFree(reinterpret_cast<CUdeviceptr>(device_ptr)));
}

void CudaMgr::zeroDeviceMem(int8_t* device_ptr,
                            const size_t num_bytes,
                            const int device_num) {
  setDeviceMem(device_ptr, 0, num_bytes, device_num);
}

void CudaMgr::setDeviceMem(int8_t* device_ptr,
                           const unsigned char uc,
                           const size_t num_bytes,
                           const int device_num) {
  setContext(device_num);
  checkError(cuMemsetD8(reinterpret_cast<CUdeviceptr>(device_ptr), uc, num_bytes));
}

/**
 * Returns true if all devices have Maxwell micro-architecture, or later.
 * Returns false, if there is any device with compute capability of < 5.0
 */
bool CudaMgr::isArchMaxwellOrLaterForAll() const {
  for (int i = 0; i < device_count_; i++) {
    if (device_properties_[i].computeMajor < 5) {
      return false;
    }
  }
  return true;
}

/**
 * Returns true if all devices have Volta micro-architecture
 * Returns false, if there is any non-Volta device available.
 */
bool CudaMgr::isArchVoltaForAll() const {
  for (int i = 0; i < device_count_; i++) {
    if (device_properties_[i].computeMajor != 7) {
      return false;
    }
  }
  return true;
}

/**
 * This function returns the maximum available dynamic shared memory that is available for
 * all GPU devices (i.e., minimum of all available dynamic shared memory per blocks, for
 * all GPU devices).
 */
size_t CudaMgr::computeMaxSharedMemoryForAll() const {
  int shared_mem_size =
      device_count_ > 0 ? device_properties_.front().sharedMemPerBlock : 0;
  for (int d = 1; d < device_count_; d++) {
    shared_mem_size = std::min(shared_mem_size, device_properties_[d].sharedMemPerBlock);
  }
  return shared_mem_size;
}

void CudaMgr::createDeviceContexts() {
  CHECK_EQ(device_contexts_.size(), size_t(0));
  device_contexts_.resize(device_count_);
  for (int d = 0; d < device_count_; ++d) {
    CUresult status = cuCtxCreate(&device_contexts_[d], 0, device_properties_[d].device);
    if (status != CUDA_SUCCESS) {
      // this is called from destructor so we need
      // to clean up
      // destroy all contexts up to this point
      for (int destroy_id = 0; destroy_id <= d; ++destroy_id) {
        try {
          checkError(cuCtxDestroy(device_contexts_[destroy_id]));
        } catch (const CudaErrorException& e) {
          LOG(ERROR) << "Error destroying context after failed creation for device "
                     << destroy_id;
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
  cuCtxSetCurrent(device_contexts_[device_num]);
}

void CudaMgr::printDeviceProperties() const {
  LOG(INFO) << "Using " << device_count_ << " Gpus.";
  for (int d = 0; d < device_count_; ++d) {
    VLOG(1) << "Device: " << device_properties_[d].device;
    VLOG(1) << "Clock (khz): " << device_properties_[d].clockKhz;
    VLOG(1) << "Compute Major: " << device_properties_[d].computeMajor;
    VLOG(1) << "Compute Minor: " << device_properties_[d].computeMinor;
    VLOG(1) << "PCI bus id: " << device_properties_[d].pciBusId;
    VLOG(1) << "PCI deviceId id: " << device_properties_[d].pciDeviceId;
    VLOG(1) << "Total Global memory: " << device_properties_[d].globalMem / 1073741824.0
            << " GB";
    VLOG(1) << "Memory clock (khz): " << device_properties_[d].memoryClockKhz;
    VLOG(1) << "Memory bandwidth: " << device_properties_[d].memoryBandwidthGBs
            << " GB/sec";

    VLOG(1) << "Constant Memory: " << device_properties_[d].constantMem;
    VLOG(1) << "Shared memory per multiprocessor: "
            << device_properties_[d].sharedMemPerMP;
    VLOG(1) << "Shared memory per block: " << device_properties_[d].sharedMemPerBlock;
    VLOG(1) << "Number of MPs: " << device_properties_[d].numMPs;
    VLOG(1) << "Warp Size: " << device_properties_[d].warpSize;
    VLOG(1) << "Max threads per block: " << device_properties_[d].maxThreadsPerBlock;
    VLOG(1) << "Max registers per block: " << device_properties_[d].maxRegistersPerBlock;
    VLOG(1) << "Max register per MP: " << device_properties_[d].maxRegistersPerMP;
    VLOG(1) << "Memory bus width in bits: " << device_properties_[d].memoryBusWidth;
  }
}

void CudaMgr::checkError(CUresult status) const {
  if (status != CUDA_SUCCESS) {
    throw CudaErrorException(status);
  }
}

}  // namespace CudaMgr_Namespace
