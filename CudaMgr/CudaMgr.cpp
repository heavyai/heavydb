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

#include "CudaMgr/CudaMgr.h"
#include "QueryEngine/NvidiaKernel.h"

#include <algorithm>
#include <boost/stacktrace.hpp>
#include <cassert>
#include <iostream>
#include <stdexcept>

#include "Logger/Logger.h"

namespace CudaMgr_Namespace {

CudaErrorException::CudaErrorException(CUresult status)
    : std::runtime_error(errorMessage(status)), status_(status) {
  // cuda already de-initialized can occur during system shutdown. avoid making calls to
  // the logger to prevent failing during a standard teardown.
  if (status != CUDA_ERROR_DEINITIALIZED) {
    VLOG(1) << errorMessage(status);
    VLOG(1) << boost::stacktrace::stacktrace();
  }
}

std::string errorMessage(CUresult const status) {
  const char* errorString{nullptr};
  cuGetErrorString(status, &errorString);
  return errorString
             ? "CUDA Error (" + std::to_string(status) + "): " + std::string(errorString)
             : "CUDA Driver API error code " + std::to_string(status);
}

CudaMgr::CudaMgr(const int num_gpus, const int start_gpu)
    : start_gpu_(start_gpu)
    , min_shared_memory_per_block_for_all_devices(0)
    , min_num_mps_for_all_devices(0) {
  checkError(cuInit(0));
  checkError(cuDeviceGetCount(&device_count_));

  if (num_gpus > 0) {  // numGpus <= 0 will just use number of gpus found
    device_count_ = std::min(device_count_, num_gpus);
  } else {
    // if we are using all gpus we cannot start on a gpu other than 0
    CHECK_EQ(start_gpu_, 0);
  }
  fillDeviceProperties();
  initDeviceGroup();
  createDeviceContexts();
  printDeviceProperties();

  // warm up the GPU JIT
  LOG(INFO) << "Warming up the GPU JIT Compiler... (this may take several seconds)";
  setContext(0);
  nvidia_jit_warmup();
  LOG(INFO) << "GPU JIT Compiler initialized.";
}

void CudaMgr::initDeviceGroup() {
  for (int device_id = 0; device_id < device_count_; device_id++) {
    device_group_.push_back(
        {device_id, device_id + start_gpu_, device_properties_[device_id].uuid});
  }
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
    auto code = cuModuleUnload(*module);
    // If the Cuda driver has already shut down, ignore the resulting errors.
    if (code != CUDA_ERROR_DEINITIALIZED) {
      checkError(code);
    }
  } catch (const std::runtime_error& e) {
    LOG(ERROR) << "CUDA Error: " << e.what();
  }
}

CudaMgr::CudaMemoryUsage CudaMgr::getCudaMemoryUsage() {
  CudaMemoryUsage usage;
  cuMemGetInfo(&usage.free, &usage.total);
  return usage;
}

void CudaMgr::fillDeviceProperties() {
  device_properties_.resize(device_count_);
  cuDriverGetVersion(&gpu_driver_version_);
  for (int device_num = 0; device_num < device_count_; ++device_num) {
    checkError(
        cuDeviceGet(&device_properties_[device_num].device, device_num + start_gpu_));
    CUuuid cuda_uuid;
    checkError(cuDeviceGetUuid(&cuda_uuid, device_properties_[device_num].device));
    device_properties_[device_num].uuid = omnisci::UUID(cuda_uuid.bytes);
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
  min_shared_memory_per_block_for_all_devices =
      computeMinSharedMemoryPerBlockForAllDevices();
  min_num_mps_for_all_devices = computeMinNumMPsForAllDevices();
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
bool CudaMgr::isArchVoltaOrGreaterForAll() const {
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
  int shared_mem_size =
      device_count_ > 0 ? device_properties_.front().sharedMemPerBlock : 0;
  for (int d = 1; d < device_count_; d++) {
    shared_mem_size = std::min(shared_mem_size, device_properties_[d].sharedMemPerBlock);
  }
  return shared_mem_size;
}

/**
 * This function returns the minimum number of multiprocessors (MPs, also known as SMs)
 * per device across all GPU devices
 */
size_t CudaMgr::computeMinNumMPsForAllDevices() const {
  int num_mps = device_count_ > 0 ? device_properties_.front().numMPs : 0;
  for (int d = 1; d < device_count_; d++) {
    num_mps = std::min(num_mps, device_properties_[d].numMPs);
  }
  return num_mps;
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
          LOG(ERROR) << "Failed to destroy CUDA context for device ID " << destroy_id
                     << " with " << e.what()
                     << ". CUDA contexts were being destroyed due to an error creating "
                        "CUDA context for device ID "
                     << d << " out of " << device_count_ << " (" << errorMessage(status)
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
  cuCtxSetCurrent(device_contexts_[device_num]);
}

void CudaMgr::printDeviceProperties() const {
  LOG(INFO) << "Using " << device_count_ << " Gpus.";
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

std::string get_cuda_home(void) {
  static const char* CUDA_DEFAULT_PATH = "/usr/local/cuda";
  const char* env = nullptr;

  if (!(env = getenv("CUDA_HOME")) && !(env = getenv("CUDA_DIR"))) {
    // check if the default CUDA directory exists: /usr/local/cuda
    if (boost::filesystem::exists(boost::filesystem::path(CUDA_DEFAULT_PATH)))
      env = CUDA_DEFAULT_PATH;
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
