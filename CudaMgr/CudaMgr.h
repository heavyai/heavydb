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
  CudaMgr(const int numGpus, const int startGpu = 0);
  ~CudaMgr();
  void setContext(const int deviceNum) const;
  void printDeviceProperties() const;
  DeviceProperties* getDeviceProperties(const size_t deviceNum);
  int8_t* allocatePinnedHostMem(const size_t numBytes);
  int8_t* allocateDeviceMem(const size_t numBytes, const int deviceNum);
  void freePinnedHostMem(int8_t* hostPtr);
  void freeDeviceMem(int8_t* devicePtr);
  void copyHostToDevice(int8_t* devicePtr,
                        const int8_t* hostPtr,
                        const size_t numBytes,
                        const int deviceNum);
  void copyDeviceToHost(int8_t* hostPtr,
                        const int8_t* devicePtr,
                        const size_t numBytes,
                        const int deviceNum);
  void copyDeviceToDevice(int8_t* destPtr,
                          int8_t* srcPtr,
                          const size_t numBytes,
                          const int destDeviceNum,
                          const int srcDeviceNum);
  void zeroDeviceMem(int8_t* devicePtr, const size_t numBytes, const int deviceNum);
  void setDeviceMem(int8_t* devicePtr,
                    const unsigned char uc,
                    const size_t numBytes,
                    const int deviceNum);
  inline int getDeviceCount() const { return deviceCount_; }
  inline int getStartGpu() const {
#ifdef HAVE_CUDA
    return startGpu_;
#else
    return -1;
#endif
  }
  inline bool isArchMaxwell() const {
    return (getDeviceCount() > 0 && deviceProperties[0].computeMajor == 5);
  }
  inline bool isArchMaxwellOrLater() const {
    return (getDeviceCount() > 0 && deviceProperties[0].computeMajor >= 5);
  }
  inline bool isArchPascal() const {
    return (getDeviceCount() > 0 && deviceProperties[0].computeMajor == 6);
  }
  inline bool isArchPascalOrLater() const {
    return (getDeviceCount() > 0 && deviceProperties[0].computeMajor >= 6);
  }

  bool isArchMaxwellOrLaterForAll() const;
  bool isArchVoltaForAll() const;

  std::vector<DeviceProperties> deviceProperties;

  size_t maxSharedMemoryForAll;
  size_t computeMaxSharedMemoryForAll() const;

  const std::vector<CUcontext>& getDeviceContexts() const { return deviceContexts; }

  const int getGpuDriverVersion() { return gpu_driver_version; }

  void synchronizeDevices() const;

#ifdef HAVE_CUDA
  void loadGpuModuleData(CUmodule* module,
                         const void* image,
                         unsigned int num_options,
                         CUjit_option* options,
                         void** option_values,
                         const int device_id) const;
  void unloadGpuModuleData(CUmodule* module, const int device_id) const;
#endif

 private:
  void fillDeviceProperties();
  void createDeviceContexts();
  void checkError(CUresult cuResult) const;

  int deviceCount_;
  int gpu_driver_version;
  mutable std::mutex device_cleanup_mutex_;

#ifdef HAVE_CUDA
  int startGpu_;
#endif  // HAVE_CUDA
  std::vector<CUcontext> deviceContexts;

};  // class CudaMgr

}  // Namespace CudaMgr_Namespace

#endif  // CUDAMGR_H
