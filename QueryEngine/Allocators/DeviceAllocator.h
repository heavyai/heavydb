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

/**
 * @file    DeviceAllocator.h
 * @author  Alex Baden <alex.baden@omnisci.com>
 * @brief   Abstract class for managing device memory allocations
 */

#pragma once

#include <glog/logging.h>

#ifdef HAVE_CUDA
#include <cuda.h>
#else
#include <Shared/nocuda.h>
#endif

namespace Data_Namespace {
class AbstractBuffer;
class DataMgr;
}  // namespace Data_Namespace

// TODO(adb): RenderAllocator becomes a first class citizen
class RenderAllocator;

class DeviceAllocator {
 public:
  virtual CUdeviceptr alloc(const size_t num_bytes,
                            RenderAllocator* render_allocator) const = 0;

  virtual void free(Data_Namespace::AbstractBuffer* ab) const = 0;

  virtual void copyToDevice(CUdeviceptr dst,
                            const void* src,
                            const size_t num_bytes) const = 0;

  virtual void copyFromDevice(void* dst,
                              const CUdeviceptr src,
                              const size_t num_bytes) const = 0;

  virtual void zeroDeviceMem(int8_t* device_ptr, const size_t num_bytes) const = 0;

  virtual void setDeviceMem(int8_t* device_ptr,
                            unsigned char uc,
                            const size_t num_bytes) const = 0;
};
