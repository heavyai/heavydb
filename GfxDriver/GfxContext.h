/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <boost/noncopyable.hpp>

#include "GfxDriver/DeviceLimits.h"
#include "GfxDriver/Enums.h"
#include "GfxDriver/ShaderCompiler/Types.h"
#include "GfxDriver/Types.h"
#include "Shared/DeviceGroup.h"

namespace gfx {

class GfxContext : boost::noncopyable {
 public:
  explicit GfxContext(DriverType primary_driver,
                      GfxUsage usage,
                      LibraryUqPtr shader_library,
                      uint32_t command_timeout_ms,
                      const WindowSystemCreateInfo* wsi_ci = nullptr,
                      bool allow_raytracing_init = true);
  GfxContext() = delete;
  ~GfxContext();

  // Access the primary driver selected at startup
  const DriverInstance& getPrimaryDriver() const;

  // Non-device bound system components
  const ShaderManager& getShaderManager() const;

  // Window system integration
  WindowSystemIntegration* getWSI() const;

  uint32_t getCommandTimeout() const;

  const DeviceLimits& getDeviceLimits() const;
  bool queryDeviceCapabilities(DeviceCapabilityBits capabilities) const;

  void createDeviceContexts(const heavyai::DeviceGroup& device_group);
  bool destroyDeviceContexts();

  // In the event of an OutOfGpuMemoryError or DeviceLostError, all VkDevices must
  // be destroyed and recreated. This function should only be called as part of
  // a full renderer teardown and reinit
  void recreateDevices();  // OOM / device lost handling

  const DeviceContext& getDeviceContext(const heavyai::DeviceIdentifier& di) const;
  const heavyai::DeviceGroup& getDeviceGroup() const;

 private:
  ShaderManagerUqPtr shader_mgr_;
  DriverInstanceUqPtr drivers_[kNumDriverTypes];
  const DriverType primary_driver_;
  uint32_t command_timeout_ms_;

  std::vector<DeviceContextUqPtr> device_contexts_;
  heavyai::DeviceGroup device_group_;
};

}  // namespace gfx
