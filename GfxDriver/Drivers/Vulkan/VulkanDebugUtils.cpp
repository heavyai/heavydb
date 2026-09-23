/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Drivers/Vulkan/VulkanDebugUtils.h"

#include <string>

#include "GfxDriver/Drivers/Vulkan/VulkanPlatform.h"
#include "GfxDriver/Drivers/Vulkan/VulkanResult.h"

// these become env vars when resolved with BE-6022
#define TREAT_VALIDATION_MESSAGES_AS_ERRORS true
#define FAIL_ON_VALIDATION_ERROR true

// Output all validation messages to std::cerr
// @TODO(se) should this be an env var too?
#define OUTPUT_VALIDATION_TO_STDERR 0
#if OUTPUT_VALIDATION_TO_STDERR == 1
#include <iostream>
#endif

// Log image info (including layout) on all contexts on Validation Error
// @TODO(se) still unconvinced of the usefulness of this
// image layouts are often changed even by operations which fail,
// meaning that the previous layout still cannot be determined
#define LOG_IMAGE_INFO_ON_VALIDATION_ERROR 0

namespace gfx {

VulkanDebugUtils::VulkanDebugUtils(const VulkanPlatform& platform, VkInstance vk_instance)
    : platform_{platform}
    , vk_instance_{vk_instance}
    , debug_utils_messenger_{VK_NULL_HANDLE}
    , vkCreateDebugUtilsMessengerEXT_{nullptr}
    , vkDestroyDebugUtilsMessengerEXT_{nullptr} {
  init();
}

VulkanDebugUtils::~VulkanDebugUtils() {
  shutdown();
}

static VKAPI_ATTR VkBool32 VKAPI_CALL debug_utils_messenger_callback_fn(
    VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
    VkDebugUtilsMessageTypeFlagsEXT message_type,
    const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
    void* user_data) {
  CHECK(callback_data)
      << "Vulkan Debug Utils Messenger Callback has invalid callback_data";
  CHECK(user_data) << "Vulkan Debug Utils Messenger Callback has invalid user_data";

  // ignore messages if suppressed
  if (VulkanPlatform::areValidationMessagesSuppressed()) {
    return VK_FALSE;
  }

  if (callback_data->messageIdNumber == (int)0xfd5d8e3f) {
    // Suppress UNASSIGNED-BestPractices-PushConstants
    // The 1.3.275 update started throwing these out in code that looks correct and has
    // been working for years. This suppression is temporary to allow moving forward with
    // the 275 SDK update while investigation continues
    // See [GFX-357]
    return VK_FALSE;
  } else if (callback_data->messageIdNumber == (int)0xc093a791) {
    // Validation gets confused when mixing Vulkan->Cuda and Cuda->Vulkan memory
    // allocation export/import, as it doesn't see Cuda consuming the file
    // descriptors. See [GFX-355] for fix task
    // Log a brief description since this *may* be an actual error (unlike the best
    // practices suppression above)
    LOG(ERROR) << "Suppressed memory handle import size mismatch. See [GFX-355]";
    return VK_FALSE;
  } else if (callback_data->messageIdNumber == (int)0x7bc61184) {
    // Suppress UNASSIGNED-BestPractices-SpirvDeprecated_WorkgroupSize
    // See [GFX-437]
    return VK_FALSE;
  }

  // evaluate severity
  auto const fail_always =
      VulkanPlatform::getValidationMode() == VulkanPlatform::ValidationMode::kFailAlways;
  auto const fail_if_error =
      VulkanPlatform::getValidationMode() == VulkanPlatform::ValidationMode::kFailIfError;
  std::string severity;
  bool fail{false};
  if (message_severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
    severity = "ERROR";
    fail = fail_if_error || fail_always;
  } else if (message_severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
    severity = "WARNING";
    fail = fail_always;
  } else if (message_severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) {
    severity = "INFO";
    fail = fail_always;
  } else if (message_severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT) {
    severity = "DEBUG";
    fail = fail_always;
  }

  // build message
  std::stringstream ss;
  ss << "Vulkan Debug Utils Message:" << std::endl;
  ss << "  Severity  : " << severity << std::endl;
  ss << "  ID Name   : "
     << (callback_data->pMessageIdName ? callback_data->pMessageIdName : "NULL")
     << std::endl;
  ss << "  ID Number : " << std::hex << callback_data->messageIdNumber << std::endl;
  ss << "  Message   : " << (callback_data->pMessage ? callback_data->pMessage : "NULL")
     << std::endl;
  ss << "  Queues    : " << callback_data->queueLabelCount << std::endl;
  for (int i = 0; i < static_cast<int>(callback_data->queueLabelCount); i++) {
    auto const* n = callback_data->pQueueLabels[i].pLabelName;
    ss << "              " << (n ? n : "NULL") << std::endl;
  }
  ss << "  CmdBufs   : " << callback_data->cmdBufLabelCount << std::endl;
  for (int i = 0; i < static_cast<int>(callback_data->cmdBufLabelCount); i++) {
    auto const* n = callback_data->pCmdBufLabels[i].pLabelName;
    ss << "              " << (n ? n : "NULL") << std::endl;
  }
  ss << "  Objects   : " << callback_data->objectCount << std::endl;
  for (int i = 0; i < static_cast<int>(callback_data->objectCount); i++) {
    auto const h = callback_data->pObjects[i].objectHandle;
    auto const t = vulkan_object_type_to_str(callback_data->pObjects[i].objectType);
    auto const* n = callback_data->pObjects[i].pObjectName;
    ss << "              0x" << std::hex << h << std::dec << " " << t << " '"
       << (n ? n : "NULL") << "'" << std::endl;
  }
  auto message = ss.str();

  // output message
  if (fail) {
    LOG(ERROR) << message;
  } else {
    LOG(WARNING) << message;
  }
#if OUTPUT_VALIDATION_TO_STDERR == 1
  std::cerr << message;
#endif

  // call platform debug callback
  auto const* platform = static_cast<const VulkanPlatform*>(user_data);
  platform->debugCallback();

  return fail ? VK_TRUE : VK_FALSE;
}

void VulkanDebugUtils::init() {
  //
  // get instance function pointers
  //

  vkCreateDebugUtilsMessengerEXT_ =
      (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
          vk_instance_, "vkCreateDebugUtilsMessengerEXT");
  CHECK(vkCreateDebugUtilsMessengerEXT_);

  vkDestroyDebugUtilsMessengerEXT_ =
      (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
          vk_instance_, "vkDestroyDebugUtilsMessengerEXT");
  CHECK(vkDestroyDebugUtilsMessengerEXT_);

  vkSetDebugUtilsObjectNameEXT_ = (PFN_vkSetDebugUtilsObjectNameEXT)vkGetInstanceProcAddr(
      vk_instance_, "vkSetDebugUtilsObjectNameEXT");
  CHECK(vkSetDebugUtilsObjectNameEXT_);

  vkQueueBeginDebugUtilsLabelEXT_ =
      (PFN_vkQueueBeginDebugUtilsLabelEXT)vkGetInstanceProcAddr(
          vk_instance_, "vkQueueBeginDebugUtilsLabelEXT");
  CHECK(vkQueueBeginDebugUtilsLabelEXT_);

  vkQueueEndDebugUtilsLabelEXT_ = (PFN_vkQueueEndDebugUtilsLabelEXT)vkGetInstanceProcAddr(
      vk_instance_, "vkQueueEndDebugUtilsLabelEXT");
  CHECK(vkQueueEndDebugUtilsLabelEXT_);

  vkQueueInsertDebugUtilsLabelEXT_ =
      (PFN_vkQueueInsertDebugUtilsLabelEXT)vkGetInstanceProcAddr(
          vk_instance_, "vkQueueInsertDebugUtilsLabelEXT");
  CHECK(vkQueueInsertDebugUtilsLabelEXT_);

  vkCmdBeginDebugUtilsLabelEXT_ = (PFN_vkCmdBeginDebugUtilsLabelEXT)vkGetInstanceProcAddr(
      vk_instance_, "vkCmdBeginDebugUtilsLabelEXT");
  CHECK(vkCmdBeginDebugUtilsLabelEXT_);

  vkCmdEndDebugUtilsLabelEXT_ = (PFN_vkCmdEndDebugUtilsLabelEXT)vkGetInstanceProcAddr(
      vk_instance_, "vkCmdEndDebugUtilsLabelEXT");
  CHECK(vkCmdEndDebugUtilsLabelEXT_);

  vkCmdInsertDebugUtilsLabelEXT_ =
      (PFN_vkCmdInsertDebugUtilsLabelEXT)vkGetInstanceProcAddr(
          vk_instance_, "vkCmdInsertDebugUtilsLabelEXT");
  CHECK(vkCmdInsertDebugUtilsLabelEXT_);

  //
  // create messenger
  //

  VkDebugUtilsMessengerCreateInfoEXT create_info = {};
  create_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
  create_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
  create_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                            VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
  create_info.pfnUserCallback = debug_utils_messenger_callback_fn;
  create_info.pUserData = const_cast<void*>(static_cast<const void*>(&platform_));

  CHECK_VKRESULT(vkCreateDebugUtilsMessengerEXT_(
                     vk_instance_, &create_info, nullptr, &debug_utils_messenger_),
                 "Failed to create VkDebugUtilsMessengerEXT");
}

void VulkanDebugUtils::shutdown() {
  //
  // destroy messenger
  //

  if (debug_utils_messenger_ != VK_NULL_HANDLE) {
    CHECK(vkDestroyDebugUtilsMessengerEXT_);
    vkDestroyDebugUtilsMessengerEXT_(vk_instance_, debug_utils_messenger_, nullptr);
    debug_utils_messenger_ = VK_NULL_HANDLE;
  }
}

void VulkanDebugUtils::nameVulkanObject(const VkDevice vk_device,
                                        const uint32_t gpu_id,
                                        const VkObjectType object_type,
                                        const void* object_handle,
                                        const std::string& object_name) {
  CHECK(vkSetDebugUtilsObjectNameEXT_);
  auto object_name_with_gpu_id = object_name + " (GPU " + std::to_string(gpu_id) + ")";
  VkDebugUtilsObjectNameInfoEXT object_name_info{
      VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
      nullptr,
      object_type,
      reinterpret_cast<uint64_t>(object_handle),
      object_name_with_gpu_id.c_str()};
  auto error_msg =
      "Error naming Vulkan Object '" + std::string(object_name_with_gpu_id) + "'";
  CHECK_VKRESULT(vkSetDebugUtilsObjectNameEXT_(vk_device, &object_name_info), error_msg);
}

static float kCmdColor[4] = {0.3f, 0.3f, 0.8f, 1.0f};
static std::string kEmptyLabel{"no label"};

void VulkanDebugUtils::insertCmdLabel(VkCommandBuffer cmd_buffer,
                                      const std::string_view name) const {
  CHECK(vkCmdInsertDebugUtilsLabelEXT_);
  VkDebugUtilsLabelEXT label_info = {
      VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT,
      nullptr,
      name.empty() ? kEmptyLabel.c_str() : name.data(),
      {kCmdColor[0], kCmdColor[1], kCmdColor[2], kCmdColor[3]}};
  vkCmdInsertDebugUtilsLabelEXT_(cmd_buffer, &label_info);
}

void VulkanDebugUtils::beginCmdLabel(VkCommandBuffer cmd_buffer,
                                     const std::string_view name) const {
  CHECK(vkCmdBeginDebugUtilsLabelEXT_);
  VkDebugUtilsLabelEXT label_info = {
      VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT,
      nullptr,
      name.empty() ? kEmptyLabel.c_str() : name.data(),
      {kCmdColor[0], kCmdColor[1], kCmdColor[2], kCmdColor[3]}};
  vkCmdBeginDebugUtilsLabelEXT_(cmd_buffer, &label_info);
}

void VulkanDebugUtils::endCmdLabel(const VkCommandBuffer cmd_buffer) const {
  CHECK(vkCmdEndDebugUtilsLabelEXT_);
  vkCmdEndDebugUtilsLabelEXT_(cmd_buffer);
}

void VulkanDebugUtils::beginQueueLabel(VkQueue queue, const std::string_view name) const {
  CHECK(vkQueueBeginDebugUtilsLabelEXT_);
  VkDebugUtilsLabelEXT label_info = {
      VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT,
      nullptr,
      name.empty() ? kEmptyLabel.c_str() : name.data(),
      {kCmdColor[0], kCmdColor[1], kCmdColor[2], kCmdColor[3]}};
  vkQueueBeginDebugUtilsLabelEXT_(queue, &label_info);
}

void VulkanDebugUtils::endQueueLabel(const VkQueue queue) const {
  CHECK(vkQueueEndDebugUtilsLabelEXT_);
  vkQueueEndDebugUtilsLabelEXT_(queue);
}

}  // namespace gfx
