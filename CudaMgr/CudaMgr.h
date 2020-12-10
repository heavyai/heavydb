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
#pragma once

#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

#include "Logger/Logger.h"
#include "Shared/DeviceGroup.h"

#ifdef HAVE_CUDA
#include <cuda.h>
#else
#include "Shared/nocuda.h"
#endif  // HAVE_CUDA

namespace CudaMgr_Namespace {

enum class NvidiaDeviceArch {
  Kepler,   // compute major = 3
  Maxwell,  // compute major = 5
  Pascal,   // compute major = 6
  Volta,    // compute major = 7, compute minor = 0
  Turing,   // compute major = 7, compute minor = 5
  Ampere    // compute major = 8
};

#ifdef HAVE_CUDA
std::string errorMessage(CUresult const);

class CudaErrorException : public std::runtime_error {
 public:
  CudaErrorException(CUresult status);

  CUresult getStatus() const { return status_; }

 private:
  CUresult const status_;
};
#endif

struct DeviceProperties {
  CUdevice device;
  omnisci::UUID uuid;
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
};

class CudaMgr {
 public:
  CudaMgr(const int num_gpus, const int start_gpu = 0);
  ~CudaMgr();

  void synchronizeDevices() const;
  int getDeviceCount() const { return device_count_; }
  int getStartGpu() const { return start_gpu_; }
  const omnisci::DeviceGroup& getDeviceGroup() const { return device_group_; }

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

  size_t getMinSharedMemoryPerBlockForAllDevices() const {
    return min_shared_memory_per_block_for_all_devices;
  }

  size_t getMinNumMPsForAllDevices() const { return min_num_mps_for_all_devices; }

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
  bool isArchVoltaOrGreaterForAll() const;

  static std::string deviceArchToSM(const NvidiaDeviceArch arch) {
    // Must match ${CUDA_COMPILATION_ARCH} CMAKE flag
    switch (arch) {
      case NvidiaDeviceArch::Kepler:
        return "sm_35";
      case NvidiaDeviceArch::Maxwell:
        return "sm_50";
      case NvidiaDeviceArch::Pascal:
        return "sm_60";
      case NvidiaDeviceArch::Volta:
        return "sm_70";
      case NvidiaDeviceArch::Turing:
        return "sm_75";
      case NvidiaDeviceArch::Ampere:
        return "sm_75";
      default:
        LOG(WARNING) << "Unrecognized Nvidia device architecture, falling back to "
                        "Kepler-compatibility.";
        return "sm_35";
    }
    UNREACHABLE();
    return "";
  }

  NvidiaDeviceArch getDeviceArch() const {
    if (device_properties_.size() > 0) {
      const auto& device_properties = device_properties_.front();
      switch (device_properties.computeMajor) {
        case 3:
          return NvidiaDeviceArch::Kepler;
        case 5:
          return NvidiaDeviceArch::Maxwell;
        case 6:
          return NvidiaDeviceArch::Pascal;
        case 7:
          if (device_properties.computeMinor == 0) {
            return NvidiaDeviceArch::Volta;
          } else {
            return NvidiaDeviceArch::Turing;
          }
        case 8:
          return NvidiaDeviceArch::Ampere;
        default:
          return NvidiaDeviceArch::Kepler;
      }
    } else {
      // always fallback to Kepler if an architecture cannot be detected
      return NvidiaDeviceArch::Kepler;
    }
  }

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

  struct CudaMemoryUsage {
    size_t free;   // available GPU RAM memory on active card in bytes
    size_t total;  // total GPU RAM memory on active card in bytes
  };

  static CudaMemoryUsage getCudaMemoryUsage();
#endif

 private:
#ifdef HAVE_CUDA
  void fillDeviceProperties();
  void initDeviceGroup();
  void createDeviceContexts();
  size_t computeMinSharedMemoryPerBlockForAllDevices() const;
  size_t computeMinNumMPsForAllDevices() const;
  void checkError(CUresult cu_result) const;

  int gpu_driver_version_;
#endif

  int device_count_;
  int start_gpu_;
  size_t min_shared_memory_per_block_for_all_devices;
  size_t min_num_mps_for_all_devices;
  std::vector<DeviceProperties> device_properties_;
  omnisci::DeviceGroup device_group_;
  std::vector<CUcontext> device_contexts_;

  mutable std::mutex device_cleanup_mutex_;
};

}  // Namespace CudaMgr_Namespace

extern std::string get_cuda_home(void);
