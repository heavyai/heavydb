/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <ostream>

#include "Shared/EnumBitmaskOps.h"

namespace gfx {

/**
 * Driver API in use
 **/

enum class DriverType { kVulkan };
constexpr int32_t kNumDriverTypes = 1;

enum class GfxUsage {
  kCudaInterop,
  kSingleGpuPreferDiscreet,
  kSingleGpuPreferIntegrated,
  kCpuEmulation
};

enum class DeviceVendor { kNvidia, kAMD, kIntel, kMesa, kOther };

enum class DeviceType { kDiscreetGpu, kIntegratedGpu, kVirtualGpu, kCPU, kOther };

std::ostream& operator<<(std::ostream& os, const GfxUsage& usage);
std::ostream& operator<<(std::ostream& os, const DeviceVendor& vendor);
std::ostream& operator<<(std::ostream& os, const DeviceType& type);

/**
 * Bit flags for querying device capabilities, at the device or driver level.
 * Querying the driver will return the homogenous capabilities across all
 * usable devices
 * These are intended to simplify coalescing Vulkan device capabilities across
 * all devices in the system, and for querying individual devices in
 * experimental tests
 * */

enum class DeviceCapabilityBits : uint32_t {
  kNone = 0,
  kBufferMemoryExport = 1 << 0,
  kImageMemoryExport = 1 << 1,
  kSemaphoreExport = 1 << 2,
  kMemoryBudget = 1 << 3,
  kSubgroupVote = 1 << 4,
  kSubgroupBallot = 1 << 5,
  kSubgroupArithmetic = 1 << 6,
  kSubgroupExtendedTypes = 1 << 7,
  kMeshShaders = 1 << 8,
  kTaskShaders = 1 << 9,
  kCanPresent = 1 << 10,
  kFragmentShaderPixelInterlock = 1 << 11,
  kFragmentShaderSampleInterlock = 1 << 12,
  kBufferDeviceAddress = 1 << 13,
  kRaytracing = 1 << 14,
  kRayQuery = 1 << 15,
  kBufferMemoryImport = 1 << 16,
  kImageMemoryImport = 1 << 17,
  kSemaphoreImport = 1 << 18,
  kAll = 0xFFFFFFFF
};

}  // namespace gfx

ENABLE_BITMASK_OPS(::gfx::DeviceCapabilityBits);
