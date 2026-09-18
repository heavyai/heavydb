/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <limits>
#include <memory>

namespace gfx {

class GfxContext;

class DriverInstance;
using DriverInstanceUqPtr = std::unique_ptr<DriverInstance>;

class BaseDriver;
using BaseDriverUqPtr = std::unique_ptr<BaseDriver>;

/** Device handle. This should be used to identify a logical device
 *  and treated as an opaque type. Currently it is enforced that this
 *  matches the Cuda ID for the device when using Cuda for queries **/
using DeviceId = uint32_t;
constexpr DeviceId NullDeviceId = std::numeric_limits<DeviceId>::max();

/** Logical device and associated data structures configured for rendering.
 *  Usually a single physical device but may be multiple devices depending
 *  on which driver is selected.
 *  For Vulkan this wraps the VkDevice object, the active Queues, StagingContext,
 *  and CommandContexts **/
class DeviceContext;
using DeviceContextUqPtr = std::unique_ptr<DeviceContext>;

class WindowSystemIntegration;
struct WindowSystemCreateInfo;

class ResourceManager;
using ResourceManagerUqPtr = std::unique_ptr<ResourceManager>;

struct BaseTypeGLSL;
using TypeGLSLUqPtr = std::unique_ptr<BaseTypeGLSL>;
using TypeGLSLShPtr = std::shared_ptr<BaseTypeGLSL>;

struct MemoryUsageInfo {
  DeviceId device;
  uint64_t size;
};

struct MemoryBudgetInfo {
  uint64_t total;
  uint64_t used;
  uint64_t available;
};

}  // namespace gfx
