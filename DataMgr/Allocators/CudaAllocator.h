/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file    CudaAllocator.h
 * @brief   Allocate GPU memory using GpuBuffers via DataMgr
 */

#pragma once

#include <cstdint>
#include <cstdlib>

#ifdef HAVE_CUDA
#include <cuda.h>
#else
#include <Shared/nocuda.h>
#endif

#include "DataMgr/Allocators/DeviceAllocator.h"

namespace Data_Namespace {
class AbstractBuffer;
class DataMgr;
}  // namespace Data_Namespace

class RenderAllocator;

class CudaAllocator : public DeviceAllocator {
 public:
  CudaAllocator(Data_Namespace::DataMgr* data_mgr,
                const int device_id,
                CUstream cuda_stream);

  ~CudaAllocator() override;

  static Data_Namespace::AbstractBuffer* allocGpuAbstractBuffer(
      Data_Namespace::DataMgr* data_mgr,
      const size_t num_bytes,
      const int device_id);

  static void freeGpuAbstractBuffer(Data_Namespace::DataMgr* data_mgr,
                                    Data_Namespace::AbstractBuffer* ab);

  int8_t* alloc(const size_t num_bytes) override;

  void free(Data_Namespace::AbstractBuffer* ab) const override;

  void copyToDevice(void* device_dst,
                    const void* host_src,
                    const size_t num_bytes,
                    std::optional<std::string_view> tag) const override;

  void copyFromDevice(void* host_dst,
                      const void* device_src,
                      const size_t num_bytes,
                      std::optional<std::string_view> tag) const override;

  void zeroDeviceMem(int8_t* device_ptr, const size_t num_bytes) const override;

  void setDeviceMem(int8_t* device_ptr,
                    unsigned char uc,
                    const size_t num_bytes) const override;

  int getDeviceId() const { return device_id_; }

  Data_Namespace::DataMgr* getDataMgr() const { return data_mgr_; }

 private:
  std::vector<Data_Namespace::AbstractBuffer*> owned_buffers_;

  Data_Namespace::DataMgr* data_mgr_;
  int device_id_;
  CUstream cuda_stream_;
};
