/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "GfxDriver/Drivers/Vulkan/VulkanResult.h"

#include <sstream>

#include <vulkan/vk_enum_string_helper.h>

#include "GfxDriver/DeviceContext.h"
#include "GfxDriver/RenderError.h"
#include "GfxDriver/Resources/ResourceManager.h"

namespace gfx {

void handle_vulkan_error(VkResult result,
                         std::string_view msg,
                         const char* file,
                         int line) {
  std::stringstream ss;
  ss << msg << ": Vulkan Error: " << vulkan_result_to_string(result) << " - " << file
     << ":" << line;
  if (result == VK_ERROR_OUT_OF_DEVICE_MEMORY) {
    throw OutOfGpuMemoryError(ss.str());
  } else {
    THROW_RUNTIME_EX(ss.str());
  }
}

void check_vulkan_oom(VkResult result,
                      std::string_view msg,
                      std::string_view resource_name,
                      uint64_t size,
                      const DeviceContext& device,
                      std::optional<LoggingCallback> callback,
                      const char* file,
                      int line) {
  std::stringstream exception_ss;
  exception_ss << msg << ": Vulkan Error: " << vulkan_result_to_string(result) << " - "
               << file << ":" << line;
  if (result == VK_ERROR_OUT_OF_DEVICE_MEMORY) {
    std::stringstream log_ss;
    // Log current render memory state
    log_ss << "Out of device memory error allocating " << size << " bytes on GPU "
           << device.getGpuId();
    log_ss << "\nResource: \"" << resource_name << "\"";
    log_ss << "\n\n";
    device.logMemoryBudgetInfo(log_ss);
    log_ss << "\n";
    device.getResourceManager().logMemorySummary(log_ss);
    if (callback) {
      log_ss << "\n";
      (*callback)(log_ss);
    }
    LOG(ERROR) << log_ss.str();

    throw OutOfGpuMemoryError(exception_ss.str());
  } else {
    THROW_RUNTIME_EX(exception_ss.str());
  }
}

std::string_view vulkan_result_to_string(VkResult result) {
  return string_VkResult(result);
}

std::string_view vulkan_object_type_to_str(VkObjectType type) {
  return string_VkObjectType(type);
}

}  // namespace gfx
