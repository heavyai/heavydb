/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string_view>
#include <thread>
#include <vector>

#include <boost/noncopyable.hpp>

#include "GfxDriver/DeviceLimits.h"
#include "GfxDriver/Enums.h"
#include "GfxDriver/GfxContext.h"
#include "GfxDriver/Types.h"
#include "Shared/DeviceGroup.h"
#include "Shared/uuid.h"

namespace gfx {

/**
 * Maximum number of render devices currently supported by the GfxDriver api.
 * */
constexpr uint32_t g_max_num_render_devices = 32;

/**
 * Public interface for selecting and configuring the graphics driver,
 * including enumerating physical Gpu device count and configuring logical
 * devices for rendering. In Vulkan a Device corresponds to a VkDevice.
 *
 * The concrete DriverInstance acts as a public fascade on private implementation classes.
 */
class DriverInstance : boost::noncopyable {
 public:
  explicit DriverInstance(DriverType type,
                          GfxUsage usage,
                          const GfxContext& gfx_context,
                          const WindowSystemCreateInfo* wsi_ci,
                          bool allow_raytracing_init);
  DriverInstance() = delete;
  ~DriverInstance() = default;

  uint32_t getNumGpus() const;

  // Get DeviceGroup representing all supported devices in the system
  const heavyai::DeviceGroup& getDeviceGroup() const;

  // Retrieve a vector of all available device UUIDs. This is useful for allowing
  // DeviceContext creation without Cuda and DeviceGroups
  std::vector<heavyai::UUID> getUUIDs() const;

  bool queryCapabilities(DeviceCapabilityBits capabilities) const;
  const DeviceLimits& getLimits() const;

  std::string_view getName() const;
  DriverType getType() const;

  //! Configure device for rendering.
  DeviceContextUqPtr createDeviceContext(
      const heavyai::UUID& uuid,     //!< Hardware UUID for API pairing
      const DeviceId gpu_id) const;  //!< Currently the Cuda ID for fallback pairing

  bool destroyDeviceContext(DeviceContextUqPtr device_ctx) const;

  WindowSystemIntegration* getWSI() const;

  MemoryUsageInfo getPeakMemoryUsage() const;

 private:
  static const std::thread::id main_thread_id_;

  BaseDriverUqPtr driver_;
  const GfxContext& gfx_context_;
  heavyai::DeviceGroup device_group_;
};

/**
 * Base class for Driver implementations. Implementations are private to
 * the GfxDriver library and instantiated by the DriverInstance public class.
 * */
class BaseDriver : boost::noncopyable {
 public:
  BaseDriver() = default;
  virtual ~BaseDriver() = default;

  /**
   * Methods in the public scope are primarily for access by created DeviceContexts
   * and should not change the internal state of the driver implementation.
   * If you can't make it const, it probably shouldn't be here!
   * */
  virtual std::string_view getName() const = 0;
  virtual DriverType getDriverType() const = 0;

  virtual bool queryCapabilities(const DeviceCapabilityBits capabilities) const = 0;
  virtual const DeviceLimits& getLimits() const = 0;
  virtual uint32_t getNumGpus() const = 0;

 private:
  virtual std::vector<heavyai::UUID> getUUIDs() const = 0;

  virtual DeviceContextUqPtr createDeviceContextInternal(const heavyai::UUID& uuid,
                                                         const DeviceId gpu_id) = 0;

  virtual bool destroyDeviceContextInternal(DeviceContextUqPtr device_ctx) = 0;

  virtual WindowSystemIntegration* getWSI() const = 0;

  virtual MemoryUsageInfo getPeakMemoryUsage() const = 0;

  friend class DriverInstance;
};

}  // namespace gfx
