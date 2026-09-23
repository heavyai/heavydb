/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/GfxContext.h"

#include "GfxDriver/DeviceContext.h"
#include "GfxDriver/DriverInstance.h"
#include "GfxDriver/RenderDoc/RenderDoc.h"
#include "GfxDriver/ShaderCompiler/ShaderManager.h"
#include "Logger/Logger.h"

namespace gfx {

namespace {
static int max_driver_index = 0;
}

GfxContext::GfxContext(DriverType primary_driver,
                       GfxUsage usage,
                       LibraryUqPtr shader_library,
                       uint32_t command_timeout_ms,
                       const WindowSystemCreateInfo* wsi_ci,
                       bool allow_raytracing_init)
    : primary_driver_{primary_driver}, command_timeout_ms_{command_timeout_ms} {
  auto primary_driver_index = static_cast<int>(primary_driver);
  CHECK_LE(primary_driver_index, max_driver_index)
      << "Invalid primary gfxdriver selected";

  drivers_[primary_driver_index] = std::make_unique<DriverInstance>(
      primary_driver, usage, *this, wsi_ci, allow_raytracing_init);
  CHECK(drivers_[primary_driver_index]);

  CHECK(shader_library);
  shader_mgr_ = std::make_unique<ShaderManager>(std::move(shader_library));
  CHECK(shader_mgr_);

#if ENABLE_RENDERDOC
  renderdoc::load();
#endif
}

GfxContext::~GfxContext() {
#if ENABLE_RENDERDOC
  renderdoc::unload();
#endif

  destroyDeviceContexts();  // does this need to be done separately?

  drivers_[static_cast<int>(DriverType::kVulkan)] = nullptr;
  shader_mgr_ = nullptr;
}

const DriverInstance& GfxContext::getPrimaryDriver() const {
  return *drivers_[static_cast<int>(primary_driver_)];
}

WindowSystemIntegration* GfxContext::getWSI() const {
  return getPrimaryDriver().getWSI();
}

uint32_t GfxContext::getCommandTimeout() const {
  return command_timeout_ms_;
}

const DeviceLimits& GfxContext::getDeviceLimits() const {
  return getPrimaryDriver().getLimits();
}

bool GfxContext::queryDeviceCapabilities(DeviceCapabilityBits capabilities) const {
  return getPrimaryDriver().queryCapabilities(capabilities);
}

const ShaderManager& GfxContext::getShaderManager() const {
  return *shader_mgr_;
}

void GfxContext::createDeviceContexts(const heavyai::DeviceGroup& device_group) {
  auto const& driver = getPrimaryDriver();
  CHECK_EQ(device_contexts_.size(), 0ULL);
  device_contexts_.resize(device_group.size());
  for (auto const& di : device_group) {
    CHECK_GE(di.index, 0);
    CHECK_LT(di.index, static_cast<int32_t>(device_contexts_.size()));
    auto dc = driver.createDeviceContext(di.uuid, di.gpu_id);
    CHECK(dc);
    device_contexts_[di.index] = std::move(dc);
    device_group_.push_back(di);
  }
}

bool GfxContext::destroyDeviceContexts() {
  bool did_leak_resources{false};
  auto const& driver = getPrimaryDriver();
  for (auto& dc : device_contexts_) {
    did_leak_resources |= driver.destroyDeviceContext(std::move(dc));
  }
  device_contexts_.clear();
  device_group_.clear();
  return did_leak_resources;
}

void GfxContext::recreateDevices() {
  auto orig_device_group = device_group_;
  LOG(INFO) << "  Destroying Vulkan Device Contexts";
  destroyDeviceContexts();
  LOG(INFO) << "  Recreating Vulkan Device Contexts";
  createDeviceContexts(orig_device_group);
  LOG(INFO) << "  Vulkan Device Contexts recreated";
}

const DeviceContext& GfxContext::getDeviceContext(
    const heavyai::DeviceIdentifier& di) const {
  CHECK_GE(di.index, 0);
  CHECK_LT(di.index, static_cast<int32_t>(device_contexts_.size()));
  return *device_contexts_[di.index];
}

const heavyai::DeviceGroup& GfxContext::getDeviceGroup() const {
  return device_group_;
}

}  // namespace gfx
