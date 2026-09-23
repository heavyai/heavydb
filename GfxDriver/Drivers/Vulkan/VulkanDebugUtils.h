/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>

#include <vulkan/vulkan.h>

namespace gfx {

class VulkanPlatform;

class VulkanDebugUtils {
 public:
  explicit VulkanDebugUtils(const VulkanPlatform& platform, VkInstance vk_instance);
  ~VulkanDebugUtils();
  VulkanDebugUtils() = delete;

  void nameVulkanObject(const VkDevice vk_device,
                        const uint32_t gpu_id,
                        const VkObjectType object_type,
                        const void* object_handle,
                        const std::string& object_name);

  void insertCmdLabel(VkCommandBuffer cmd_buffer, const std::string_view name) const;
  void beginCmdLabel(VkCommandBuffer cmd_buffer, const std::string_view name) const;
  void endCmdLabel(VkCommandBuffer cmd_buffer) const;

  void beginQueueLabel(VkQueue queue, const std::string_view name) const;
  void endQueueLabel(VkQueue queue) const;

 private:
  const VulkanPlatform& platform_;
  VkInstance vk_instance_;

  VkDebugUtilsMessengerEXT debug_utils_messenger_;

  PFN_vkCreateDebugUtilsMessengerEXT vkCreateDebugUtilsMessengerEXT_;
  PFN_vkDestroyDebugUtilsMessengerEXT vkDestroyDebugUtilsMessengerEXT_;
  PFN_vkSetDebugUtilsObjectNameEXT vkSetDebugUtilsObjectNameEXT_;

  PFN_vkQueueBeginDebugUtilsLabelEXT vkQueueBeginDebugUtilsLabelEXT_;
  PFN_vkQueueEndDebugUtilsLabelEXT vkQueueEndDebugUtilsLabelEXT_;
  PFN_vkQueueInsertDebugUtilsLabelEXT vkQueueInsertDebugUtilsLabelEXT_;

  PFN_vkCmdBeginDebugUtilsLabelEXT vkCmdBeginDebugUtilsLabelEXT_;
  PFN_vkCmdEndDebugUtilsLabelEXT vkCmdEndDebugUtilsLabelEXT_;
  PFN_vkCmdInsertDebugUtilsLabelEXT vkCmdInsertDebugUtilsLabelEXT_;

  void init();
  void shutdown();
};

}  // namespace gfx
