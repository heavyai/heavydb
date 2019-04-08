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

#ifndef CUDAMGR_H
#define CUDAMGR_H

#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>
#ifdef HAVE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#else
#include "../Shared/nocuda.h"
#endif  // HAVE_CUDA

namespace CudaMgr_Namespace {

#ifdef HAVE_CUDA
class CudaErrorException : public std::runtime_error {
 public:
  CudaErrorException(CUresult status)
      : std::runtime_error(processStatus(status)), status_(status) {}

  CUresult getStatus() const { return status_; }

 private:
  CUresult status_;
  std::string processStatus(CUresult status) {
    const char* errorString{nullptr};
    cuGetErrorString(status, &errorString);
    return errorString ? std::string(errorString) : "Unknown error";
  }
};
#endif

struct DeviceProperties {
  CUdevice device;
  int computeMajor;
  int computeMinor;
  size_t globalMem;
  int constantMem;
  int sharedMemPerBlock;
  int sharedMemPerMP;
  int numMPs;
  int warpSize;
  int maxThreadsPerBlock;
  int maxRegistersPerBlock;
  int maxRegistersPerMP;
  int pciBusId;
  int pciDeviceId;
  int memoryClockKhz;
  int memoryBusWidth;  // in bits
  float memoryBandwidthGBs;
  int clockKhz;
  int numCore;
  std::string arch;
};

class CudaMgr {
 public:
  CudaMgr(const int num_gpus, const int start_gpu = 0);
  ~CudaMgr();

  void synchronizeDevices() const;
  int getDeviceCount() const { return device_count_; }

  void copyHostToDevice(int8_t* device_ptr,
                        const int8_t* host_ptr,
                        const size_t num_bytes,
                        const int device_num);
  void copyDeviceToHost(int8_t* host_ptr,
                        const int8_t* device_ptr,
                        const size_t num_bytes,
                        const int device_num);
  void copyDeviceToDevice(int8_t* dest_ptr,
                          int8_t* src_ptr,
                          const size_t num_bytes,
                          const int dest_device_num,
                          const int src_device_num);

  int8_t* allocatePinnedHostMem(const size_t num_bytes);
  int8_t* allocateDeviceMem(const size_t num_bytes, const int device_num);
  void freePinnedHostMem(int8_t* host_ptr);
  void freeDeviceMem(int8_t* device_ptr);
  void zeroDeviceMem(int8_t* device_ptr, const size_t num_bytes, const int device_num);
  void setDeviceMem(int8_t* device_ptr,
                    const unsigned char uc,
                    const size_t num_bytes,
                    const int device_num);

  int getStartGpu() const { return start_gpu_; }
  size_t getMaxSharedMemoryForAll() const { return max_shared_memory_for_all_; }

  const std::vector<DeviceProperties>& getAllDeviceProperties() const {
    return device_properties_;
  }
  const DeviceProperties* getDeviceProperties(const size_t device_num) const {
    // device_num is the device number relative to start_gpu_ (real_device_num -
    // start_gpu_)
    if (device_num < device_properties_.size()) {
      return &device_properties_[device_num];
    }
    throw std::runtime_error("Specified device number " + std::to_string(device_num) +
                             " is out of range of number of devices (" +
                             std::to_string(device_properties_.size()) + ")");
  }
  inline bool isArchMaxwell() const {
    return (getDeviceCount() > 0 && device_properties_[0].computeMajor == 5);
  }
  inline bool isArchMaxwellOrLater() const {
    return (getDeviceCount() > 0 && device_properties_[0].computeMajor >= 5);
  }
  inline bool isArchPascal() const {
    return (getDeviceCount() > 0 && device_properties_[0].computeMajor == 6);
  }
  inline bool isArchPascalOrLater() const {
    return (getDeviceCount() > 0 && device_properties_[0].computeMajor >= 6);
  }
  bool isArchMaxwellOrLaterForAll() const;
  bool isArchVoltaForAll() const;

  void setContext(const int device_num) const;

#ifdef HAVE_CUDA
  void printDeviceProperties() const;

  const std::vector<CUcontext>& getDeviceContexts() const { return device_contexts_; }
  const int getGpuDriverVersion() const { return gpu_driver_version_; }

  void loadGpuModuleData(CUmodule* module,
                         const void* image,
                         unsigned int num_options,
                         CUjit_option* options,
                         void** option_values,
                         const int device_id) const;
  void unloadGpuModuleData(CUmodule* module, const int device_id) const;
#endif

 private:
#ifdef HAVE_CUDA
  void fillDeviceProperties();
  void createDeviceContexts();
  size_t computeMaxSharedMemoryForAll() const;
  void checkError(CUresult cu_result) const;
#endif

  int device_count_;
  int gpu_driver_version_;
  int start_gpu_;
  size_t max_shared_memory_for_all_;
  std::vector<DeviceProperties> device_properties_;
  std::vector<CUcontext> device_contexts_;

  mutable std::mutex device_cleanup_mutex_;
};

}  // Namespace CudaMgr_Namespace

#endif  // CUDAMGR_H
