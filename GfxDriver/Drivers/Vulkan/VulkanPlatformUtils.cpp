/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Drivers/Vulkan/VulkanPlatformUtils.h"

#include <sstream>

#include <vulkan/vulkan.h>

#include "Logger/Logger.h"

namespace gfx {

std::string vulkan_version_to_string(uint32_t version) {
  std::stringstream stream;
  stream << VK_VERSION_MAJOR(version) << "." << VK_VERSION_MINOR(version) << "."
         << VK_VERSION_PATCH(version);
  return stream.str();
}

NvidiaDriverVersion unpack_nvidia_driver_version(uint32_t packed) {
  return {(packed >> 22) & 0x3ff, (packed >> 14) & 0x0ff, (packed >> 6) & 0x0ff};
}

std::string nvidia_driver_version_to_string(uint32_t version) {
  std::stringstream stream;
  auto [major, minor, patch] = unpack_nvidia_driver_version(version);
  stream << major << '.' << +minor << '.' << +patch;
  return stream.str();
}

std::string vk_driver_id_to_string(VkDriverId id) {
  using e = VkDriverId;
  switch (id) {
    case e::VK_DRIVER_ID_AMD_PROPRIETARY:
      return "AMD Proprietary";
    case e::VK_DRIVER_ID_AMD_OPEN_SOURCE:
      return "AMD Open Source";
    case e::VK_DRIVER_ID_MESA_RADV:
      return "Mesa RadV";
    case e::VK_DRIVER_ID_NVIDIA_PROPRIETARY:
      return "Nvidia Proprietary";
    case e::VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS:
      return "Intel Proprietary Windows";
    case e::VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA:
      return "Intel Open Source Mesa";
    case e::VK_DRIVER_ID_IMAGINATION_PROPRIETARY:
      return "Imagination Proprietary";
    case e::VK_DRIVER_ID_QUALCOMM_PROPRIETARY:
      return "Qualcomm Proprietary";
    case e::VK_DRIVER_ID_ARM_PROPRIETARY:
      return "Arm Proprietary";
    case e::VK_DRIVER_ID_GOOGLE_SWIFTSHADER:
      return "Google Swiftshader";
    case e::VK_DRIVER_ID_GGP_PROPRIETARY:
      return "GGP Proprietary";
    case e::VK_DRIVER_ID_BROADCOM_PROPRIETARY:
      return "Broadcomm Proprietary";
    case e::VK_DRIVER_ID_MESA_LLVMPIPE:
      return "Mesa LLVM Pipe";
    case e::VK_DRIVER_ID_MOLTENVK:
      return "MoltenVk";
    default:
      return "Unknown";
  }
  UNREACHABLE();
  return {};
}

struct VkStructChainBuilder::VkStructureHeader {
  VkStructureType sType;
  VkStructureHeader* pNext;
};

VkStructChainBuilder::VkStructChainBuilder(VkStructureType type, void* header)
    : next_ptr_{nullptr} {
  VkStructureHeader* s = static_cast<VkStructChainBuilder::VkStructureHeader*>(header);
  s->sType = type;
  next_ptr_ = &s->pNext;
}

void VkStructChainBuilder::add(VkStructureType type, void* header) {
  VkStructureHeader* s = static_cast<VkStructChainBuilder::VkStructureHeader*>(header);
  s->sType = type;
  if (next_ptr_) {
    *next_ptr_ = s;
  }
  next_ptr_ = &s->pNext;
}

}  // namespace gfx
