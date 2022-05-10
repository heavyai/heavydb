/*
 * Copyright 2021 OmniSci, Inc.
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
#include <cstddef>
#include <cstdint>

enum GpuMgrName { CUDA, L0 };

struct GpuMgr {
  virtual ~GpuMgr() = default;
  virtual void copyHostToDevice(int8_t* device_ptr,
                                const int8_t* host_ptr,
                                const size_t num_bytes,
                                const int device_num) = 0;
  virtual void copyDeviceToHost(int8_t* host_ptr,
                                const int8_t* device_ptr,
                                const size_t num_bytes,
                                const int device_num) = 0;
  virtual void copyDeviceToDevice(int8_t* dest_ptr,
                                  int8_t* src_ptr,
                                  const size_t num_bytes,
                                  const int dest_device_num,
                                  const int src_device_num) = 0;
  virtual void zeroDeviceMem(int8_t* device_ptr,
                             const size_t num_bytes,
                             const int device_num) = 0;
  virtual void setDeviceMem(int8_t* device_ptr,
                            const unsigned char uc,
                            const size_t num_bytes,
                            const int device_num) = 0;
  virtual int8_t* allocateDeviceMem(const size_t num_bytes, const int device_num) = 0;
  virtual void freeDeviceMem(int8_t* device_ptr) = 0;
  // `setContext()` method seems redundant as we already pass an actual context via
  // parameter `device_num` into every manager's method, maybe we should remove
  // `setContext()`?
  virtual void setContext(const int device_num) const = 0;
  virtual void synchronizeDevices() const = 0;
  virtual int getDeviceCount() const = 0;
  virtual GpuMgrName getName() const = 0;
};
