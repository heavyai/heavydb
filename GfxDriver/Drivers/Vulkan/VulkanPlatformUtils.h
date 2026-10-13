/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>

#include <vulkan/vulkan.h>

namespace gfx {

/**
 * Vulkan API versions
 *
 * These are two separate decisions that happen to share a value. The instance
 * version is what we advertise to the loader, and raising it is a deployment
 * choice because it raises the driver version customers need. The device
 * minimum is what the code actually depends on: VulkanPhysicalDevice chains the
 * Vulkan 1.3 property and feature structs unconditionally, so a device below
 * that cannot be probed, let alone used.
 *
 * Keep them separate even while they agree, so that moving one does not
 * silently move the other.
 **/

constexpr uint32_t kVulkanInstanceApiVersion = VK_API_VERSION_1_3;
constexpr uint32_t kMinVulkanDeviceApiVersion = VK_API_VERSION_1_3;

/**
 * version and driver id stringifiers
 **/

std::string vulkan_version_to_string(uint32_t version);
std::string vk_driver_id_to_string(VkDriverId id);

using NvidiaDriverVersion = std::tuple<uint16_t, uint8_t, uint8_t>;
NvidiaDriverVersion unpack_nvidia_driver_version(uint32_t packed_version);

std::string nvidia_driver_version_to_string(uint32_t version);

/**
 *  Struct chain builder (inspired by vulkaninfo)
 *  Vulkan chainable structs are POD, with the first 2 fields matching the following
 *  layout, which enables automated tool and conformance test generation from spec XML
 *
 *  type: struct identifier (eg. VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES)
 *  header: pointer to the struct to populate (eg. VkPhysicalDeviceIDProperties*)
 **/
class VkStructChainBuilder {
 public:
  explicit VkStructChainBuilder(VkStructureType type, void* header);
  void add(VkStructureType type, void* header);

 private:
  struct VkStructureHeader;
  VkStructureHeader** next_ptr_;
};

}  // namespace gfx
