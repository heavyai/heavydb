/*
 * Copyright 2022 HEAVY.AI, Inc.
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
#include <ostream>

class CompilationContext;
struct CompilationResult;
struct KernelParamsLog;

class DeviceClock {
 public:
  virtual void start() = 0;
  virtual int stop() = 0;

  virtual ~DeviceClock() = default;
};

class DeviceKernel {
 public:
  virtual void launch(unsigned int grid_dim_x,
                      unsigned int grid_dim_y,
                      unsigned int grid_dim_z,
                      unsigned int block_dim_x,
                      unsigned int block_dim_y,
                      unsigned int block_dim_z,
                      CompilationResult const& compilation_result,
                      void** kernel_params,
                      KernelParamsLog const&,
                      bool optimize_block_and_grid_sizes) = 0;

  virtual void initializeDynamicWatchdog(bool could_interrupt, uint64_t cycle_budget) = 0;
  virtual void initializeRuntimeInterrupter(const int device_id) = 0;
  virtual void resetRuntimeInterrupter(const int device_id) = 0;

  virtual std::unique_ptr<DeviceClock> make_clock() = 0;
  virtual char const* name() const = 0;

  virtual ~DeviceKernel() = default;
};

std::unique_ptr<DeviceKernel> create_device_kernel(const CompilationContext* ctx,
                                                   int device_id);

// Temp struct used to log information on a CUDA error after a kernel launch.
struct DeviceKernelLaunchErrorLogDump {
  CUresult const cu_result;
  DeviceKernel* const device_kernel;
  unsigned int const grid_dim_x;
  unsigned int const grid_dim_y;
  unsigned int const grid_dim_z;
  unsigned int const block_dim_x;
  unsigned int const block_dim_y;
  unsigned int const block_dim_z;
  CompilationResult const& compilation_result;
  CUstream const qe_cuda_stream;
  void** const kernel_params;
  KernelParamsLog const& kernel_params_log;
  bool const optimize_block_and_grid_sizes;
};

std::ostream& operator<<(std::ostream&, DeviceKernelLaunchErrorLogDump const&);
