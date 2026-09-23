/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <ostream>

#include <boost/noncopyable.hpp>

#include "GfxDriver/DeviceLimits.h"
#include "GfxDriver/Enums.h"
#include "GfxDriver/ShaderCompiler/Types.h"
#include "GfxDriver/Types.h"
#include "Shared/uuid.h"

#include "GfxDriver/Commands/CommandExecutionContext.h"
#include "GfxDriver/Commands/CommandExecutor.h"
#include "GfxDriver/Commands/CommandList.h"

namespace gfx {

class DeviceContext : boost::noncopyable {
 public:
  explicit DeviceContext(const BaseDriver& driver);
  DeviceContext() = delete;
  virtual ~DeviceContext() = default;

  // Cuda ID (ignores start_gpu, respects CUDA_VISIBLE_DEVICES)
  virtual const DeviceId getGpuId() const = 0;

  // Physcial device UUID
  virtual const heavyai::UUID& getGpuUUID() const = 0;

  virtual ResourceManager& getResourceManager() const = 0;

  // System accessors
  DriverType getDriverType() const;
  virtual DeviceVendor getVendor() const = 0;

  // Device Limits and Capabilites
  virtual const DeviceLimits& getLimits() const = 0;
  virtual DeviceCapabilityBits getCapabilityBits() const = 0;

  // Command support
  // TODO: move to a command context?
  CommandList& getCommandList() const;
  CommandExecutor& getCommandExecutor() const;
  virtual CommandExecutionContextUqPtr createCommandExecutionContext() const = 0;
  virtual void resetCommandPools() const = 0;

  // Wait until all queues are drained
  // Only valid after logical device has been initialized
  virtual void waitIdle() = 0;

  virtual uint64_t getPeakMemoryUsage() const = 0;
  virtual void logMemoryBudgetInfo(std::ostream& os) const = 0;

  virtual MemoryBudgetInfo getMemoryBudget() const = 0;

 protected:
  const BaseDriver& driver_;
  CommandExecutorUqPtr command_executor_;
  CommandListUqPtr command_list_;

  void destructBase();

 private:
  virtual void createResourceManager(const ShaderManager& shader_mgr) = 0;
  virtual void createCommandList() = 0;

  friend class BaseDriver;
  friend class DriverInstance;
};

}  // namespace gfx
