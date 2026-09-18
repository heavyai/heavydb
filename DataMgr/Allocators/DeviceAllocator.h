/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file    DeviceAllocator.h
 * @brief   Abstract class for managing device memory allocations
 */

#pragma once

#include "Logger/Logger.h"

#include <optional>

#ifdef HAVE_CUDA
#include <cuda.h>
#else
#include <Shared/nocuda.h>
#endif

namespace Data_Namespace {
class AbstractBuffer;
class DataMgr;
}  // namespace Data_Namespace

class Allocator {
 public:
  Allocator() {}
  virtual ~Allocator() {}

  virtual int8_t* alloc(const size_t num_bytes) = 0;
};

class DeviceAllocator : public Allocator {
 public:
  virtual void free(Data_Namespace::AbstractBuffer* ab) const = 0;

  virtual void copyToDevice(void* device_dst,
                            const void* host_src,
                            const size_t num_bytes,
                            std::optional<std::string_view> tag) const = 0;

  virtual void copyFromDevice(void* host_dst,
                              const void* device_src,
                              const size_t num_bytes,
                              std::optional<std::string_view> tag) const = 0;

  virtual void zeroDeviceMem(int8_t* device_ptr, const size_t num_bytes) const = 0;

  virtual void setDeviceMem(int8_t* device_ptr,
                            unsigned char uc,
                            const size_t num_bytes) const = 0;
};
