/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>

#include <vulkan/vulkan.h>

namespace gfx {

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
