/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Enums.h"

namespace gfx {

std::ostream& operator<<(std::ostream& os, const GfxUsage& usage) {
  switch (usage) {
    case GfxUsage::kCudaInterop:
      os << "Cuda Interop";
      break;
    case GfxUsage::kSingleGpuPreferDiscreet:
      os << "Single GPU (prefer discreet)";
      break;
    case GfxUsage::kSingleGpuPreferIntegrated:
      os << "Single GPU (prefer integrated)";
      break;
    case GfxUsage::kCpuEmulation:
      os << "CPU Emulation";
      break;
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const DeviceVendor& vendor) {
  switch (vendor) {
    case DeviceVendor::kNvidia:
      os << "Nvidia";
      break;
    case DeviceVendor::kAMD:
      os << "AMD";
      break;
    case DeviceVendor::kIntel:
      os << "Intel";
      break;
    case DeviceVendor::kMesa:
      os << "Mesa";
      break;
    case DeviceVendor::kOther:
      os << "Unknown";
      break;
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const DeviceType& type) {
  switch (type) {
    case DeviceType::kDiscreetGpu:
      os << "Discreet GPU";
      break;
    case DeviceType::kIntegratedGpu:
      os << "Integrated GPU";
      break;
    case DeviceType::kVirtualGpu:
      os << "Virtual GPU";
      break;
    case DeviceType::kCPU:
      os << "CPU";
      break;
    case DeviceType::kOther:
      os << "Unknown";
      break;
  }
  return os;
}

}  // namespace gfx
