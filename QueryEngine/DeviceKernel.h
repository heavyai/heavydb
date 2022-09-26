/*
 * Copyright 2021 OmniSci, Inc.
 * Copyright (C) 2022 Intel Corporation
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

#include <memory>
#include <vector>

#include "QueryEngine/CompilationContext.h"

class DeviceClock {
 public:
  virtual void start() = 0;
  virtual int stop() = 0;

  virtual ~DeviceClock() = default;
};

struct KernelOptions {
  unsigned int gridDimX = 1;
  unsigned int gridDimY = 1;
  unsigned int gridDimZ = 1;
  unsigned int blockDimX = 1;
  unsigned int blockDimY = 1;
  unsigned int blockDimZ = 1;
  unsigned int sharedMemBytes = 0;
  bool hoistLiterals = true;
};

class DeviceKernel {
 public:
  virtual void launch(const KernelOptions& ko, std::vector<int8_t*>& kernelParams) = 0;

  virtual void initializeDynamicWatchdog(bool could_interrupt,
                                         uint64_t cycle_budget,
                                         size_t time_limit) {}
  virtual void initializeRuntimeInterrupter(){};

  virtual std::unique_ptr<DeviceClock> make_clock() = 0;

  virtual ~DeviceKernel() = default;
};

std::unique_ptr<DeviceKernel> create_device_kernel(const CompilationContext* ctx,
                                                   int device_id);
