/*
 * Copyright 2018 MapD Technologies, Inc.
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
 * @file    CudaAllocator.h
 * @author  Alex Baden <alex.baden@mapd.com>
 * @brief   Allocate GPU memory using GpuBuffers via DataMgr
 */

#ifndef CUDA_ALLOCATOR_H
#define CUDA_ALLOCATOR_H

#ifdef HAVE_CUDA
#include <cuda.h>
#else
#include <Shared/nocuda.h>
#endif

#include <cstdint>
#include <cstdlib>

namespace Data_Namespace {
class AbstractBuffer;
class DataMgr;
}  // namespace Data_Namespace

class RenderAllocator;

class CudaAllocator {
 public:
  CudaAllocator(Data_Namespace::DataMgr* data_mgr, const int device_id);

  CUdeviceptr alloc(const size_t num_bytes,
                    const int device_id,
                    RenderAllocator* render_allocator) const;

  void free(Data_Namespace::AbstractBuffer* ab) const;

  void copyToDevice(CUdeviceptr dst,
                    const void* src,
                    const size_t num_bytes,
                    const int device_id) const;

  void copyFromDevice(void* dst,
                      const CUdeviceptr src,
                      const size_t num_bytes,
                      const int device_id) const;

  void zeroDeviceMem(int8_t* device_ptr,
                     const size_t num_bytes,
                     const int device_id) const;

  void setDeviceMem(int8_t* device_ptr,
                    unsigned char uc,
                    const size_t num_bytes,
                    const int device_id) const;

 private:
  Data_Namespace::AbstractBuffer* allocGpuAbstractBuffer(const size_t num_bytes,
                                                         const int device_id) const;

  Data_Namespace::DataMgr* data_mgr_;
};

#endif  // CUDA_ALLOCATOR_H