/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/DriverInstance.h"

// DO NOT REMOVE EGL IT BREAKS HEADLESS SESSIONS
#include <EGL/egl.h>

#include "GfxDriver/DeviceContext.h"
#include "GfxDriver/Drivers/Vulkan/VulkanDriver.h"
#include "GfxDriver/RenderError.h"

namespace gfx {

// store the main thread id for comparison as GLDevices
// must be created on the main thread
const std::thread::id DriverInstance::main_thread_id_ = std::this_thread::get_id();

DriverInstance::DriverInstance(DriverType type,
                               GfxUsage usage,
                               const GfxContext& gfx_context,
                               const WindowSystemCreateInfo* wsi_ci,
                               bool allow_raytracing_init)
    : driver_{nullptr}, gfx_context_{gfx_context} {
  // TODO(scb): can we lift this restriction?
  if (std::this_thread::get_id() != main_thread_id_) {
    THROW_RUNTIME_EX("A DriverInstance can only be created on the main thread");
  }

  switch (type) {
    case DriverType::kVulkan:
      LOG(INFO) << "Using GfxDriver: Vulkan";
      driver_ = std::make_unique<VulkanDriver>(
          usage, gfx_context_.getCommandTimeout(), wsi_ci, allow_raytracing_init);
      break;
    default:
      CHECK(false);
      // This function does not need to be called, it just needs to be linked in to
      // ensure EGL statics are initialized. If not, Vulkan init can randomly fail in
      // remote (headless) sessions - because nvidia.
      eglQueryAPI();
  }
  CHECK(driver_ != nullptr);

  CHECK_EQ(static_cast<int>(driver_->getDriverType()), static_cast<int>(type));

  auto uuids = driver_->getUUIDs();
  for (int32_t i = 0; i < static_cast<int32_t>(uuids.size()); ++i) {
    device_group_.push_back({i, i, uuids[i]});
  }
}

bool DriverInstance::queryCapabilities(DeviceCapabilityBits capabilities) const {
  return driver_->queryCapabilities(capabilities);
}

const DeviceLimits& DriverInstance::getLimits() const {
  return driver_->getLimits();
}

std::string_view DriverInstance::getName() const {
  return driver_->getName();
}

DriverType DriverInstance::getType() const {
  return driver_->getDriverType();
}

uint32_t DriverInstance::getNumGpus() const {
  return driver_->getNumGpus();
}

const heavyai::DeviceGroup& DriverInstance::getDeviceGroup() const {
  return device_group_;
}

std::vector<heavyai::UUID> DriverInstance::getUUIDs() const {
  return driver_->getUUIDs();
}

DeviceContextUqPtr DriverInstance::createDeviceContext(const heavyai::UUID& uuid,
                                                       const DeviceId gpu_id) const {
  auto device_context = driver_->createDeviceContextInternal(uuid, gpu_id);

  RUNTIME_EX_ASSERT(device_context != nullptr,
                    "Failed to create DeviceContext for gpu " + std::to_string(gpu_id) +
                        ": " + to_string(uuid));

  device_context->createResourceManager(gfx_context_.getShaderManager());
  device_context->createCommandList();

  return device_context;
}

bool DriverInstance::destroyDeviceContext(DeviceContextUqPtr device_ctx) const {
  return driver_->destroyDeviceContextInternal(std::move(device_ctx));
}

WindowSystemIntegration* DriverInstance::getWSI() const {
  return driver_->getWSI();
}

MemoryUsageInfo DriverInstance::getPeakMemoryUsage() const {
  return driver_->getPeakMemoryUsage();
}

}  // namespace gfx
